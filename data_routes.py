import os
import shutil
from flask import Blueprint, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
from backend.config import Config
from backend.ml.preprocessing import (
    load_csv, validate_data, handle_missing_values, 
    handle_invalid_values, calculate_energy_consumption, 
    calculate_carbon_emission, aggregate_data
)
from backend.services.data_service import DataService
from backend.database.models import (
    db, EnergyRecord, Dataset, Prediction, Cluster, Anomaly, Recommendation, ModelMetric, TrainedModel
)
from backend.routes.auth import get_current_user_id
from datetime import datetime, timedelta

data_bp = Blueprint('data', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def clear_user_data_internal(user_id):
    """
    Clears all user-specific energy records, ML predictions, clusters, anomalies,
    recommendations, and original uploaded CSV file on disk, but PRESERVES trained models.
    """
    # 1. Find and delete dataset file from disk
    dataset = Dataset.query.filter_by(user_id=user_id).first()
    if dataset and dataset.file_path and os.path.exists(dataset.file_path):
        try:
            os.remove(dataset.file_path)
            print(f"Deleted old dataset file for user {user_id} at {dataset.file_path}")
        except Exception as e:
            print(f"Failed to delete old dataset file for user {user_id}: {e}")
            
    # 2. DO NOT delete user-specific trained models directory from disk
    # This ensures pre-trained models persist across uploads.

    # 3. Delete database records (excluding TrainedModel and ModelMetric)
    EnergyRecord.query.filter_by(user_id=user_id).delete()
    Prediction.query.filter_by(user_id=user_id).delete()
    Cluster.query.filter_by(user_id=user_id).delete()
    Anomaly.query.filter_by(user_id=user_id).delete()
    Recommendation.query.filter_by(user_id=user_id).delete()
    
    if dataset:
        db.session.delete(dataset)
        
    db.session.commit()

@data_bp.route('/upload', methods=['POST'])
def upload_file():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized. Please log in.'}), 401
        
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in the request.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Invalid file format. Only CSV files are allowed.'}), 400
        
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    if file_length > Config.MAX_CONTENT_LENGTH:
        return jsonify({'success': False, 'message': 'File size exceeds the 50MB limit.'}), 400
    file.seek(0)
    
    temp_filepath = None
    try:
        # Create directories if they do not exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.UPLOAD_DATA_FOLDER, exist_ok=True)
        
        # Save temp file
        temp_filename = f"temp_user_{user_id}_{secure_filename(file.filename)}"
        temp_filepath = os.path.join(Config.UPLOAD_FOLDER, temp_filename)
        file.save(temp_filepath)
        
        # Load and validate with chunking/early exit optimization
        sample_size = int(request.form.get('sample_size', 50000))
        df = load_csv(temp_filepath, sample_size=sample_size)
        
        is_valid, err_msg = validate_data(df)
        if not is_valid:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({'success': False, 'message': f'Validation failed: {err_msg}'}), 400
            
        original_record_count = len(df)
        
        df = handle_missing_values(df)
        df = handle_invalid_values(df)
        df = calculate_energy_consumption(df)
        df = calculate_carbon_emission(df, Config.EMISSION_FACTOR)
        df_hourly = aggregate_data(df, freq='h')
        
        processed_record_count = len(df_hourly)
        
        if len(df_hourly) > sample_size:
            df_hourly = df_hourly.iloc[:sample_size].copy()
            
        # Clean up existing data for this user to prevent duplication
        clear_user_data_internal(user_id)
            
        records_loaded = DataService.add_records_batch(user_id, df_hourly)
        
        # Now permanently store the original CSV file
        stored_filename = f"user_{user_id}_dataset.csv"
        filepath = os.path.join(Config.UPLOAD_DATA_FOLDER, stored_filename)
        
        # Rename/Move temp file to permanent storage
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)
        
        # Save metadata in DB
        new_dataset = Dataset(
            user_id=user_id,
            original_filename=secure_filename(file.filename),
            stored_filename=stored_filename,
            file_path=filepath,
            file_size=file_length,
            upload_timestamp=datetime.utcnow(),
            record_count=original_record_count,
            processed_record_count=processed_record_count,
            status='processed'
        )
        db.session.add(new_dataset)
        db.session.commit()
        
        # Invalidate previous trained models for this user so models must be retrained for the new active dataset
        tms = TrainedModel.query.filter_by(user_id=user_id).all()
        for tm in tms:
            tm.status = 'idle'
            tm.dataset_id = new_dataset.id
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully. Please click Train Models to train models for this dataset.',
            'records_loaded': records_loaded,
            'dataset_id': new_dataset.id
        }), 200
        
    except Exception as e:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
        db.session.rollback()
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Data processing error: {str(e)}'}), 500


@data_bp.route('/summary', methods=['GET'])
def get_summary():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        stats = DataService.get_summary_statistics(user_id)
        dataset = Dataset.query.filter_by(user_id=user_id).first()
        if dataset:
            stats['dataset'] = {
                'original_filename': dataset.original_filename,
                'file_size': dataset.file_size,
                'upload_timestamp': dataset.upload_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'record_count': dataset.record_count,
                'status': dataset.status
            }
        else:
            stats['dataset'] = None
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@data_bp.route('/records', methods=['GET'])
def get_records():
    """
    Returns a paginated list of energy records for the logged-in user.
    """
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    query = EnergyRecord.query.filter_by(user_id=user_id)
    
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            query = query.filter(EnergyRecord.timestamp >= start_date)
        except ValueError:
            pass
            
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(EnergyRecord.timestamp < end_date)
        except ValueError:
            pass
            
    # Order chronologically
    query = query.order_by(EnergyRecord.timestamp)
    
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    records_list = []
    for r in pagination.items:
        records_list.append({
            'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'global_active_power': r.global_active_power,
            'global_reactive_power': r.global_reactive_power,
            'voltage': r.voltage,
            'global_intensity': r.global_intensity,
            'sub_metering_1': r.sub_metering_1,
            'sub_metering_2': r.sub_metering_2,
            'sub_metering_3': r.sub_metering_3,
            'energy_consumption': r.energy_consumption,
            'carbon_emission': r.carbon_emission
        })
        
    return jsonify({
        'success': True,
        'records': records_list,
        'total_records': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    }), 200


@data_bp.route('/export', methods=['GET'])
def export_data():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        df = DataService.get_user_records_df(user_id)
        if df.empty:
            return jsonify({'success': False, 'message': 'No processed data available to download.'}), 404
            
        csv_data = df.to_csv(index=False)
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=processed_energy_records_{user_id}.csv"}
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to export data: {str(e)}'}), 500


@data_bp.route('/download-original', methods=['GET'])
def download_original():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        dataset = Dataset.query.filter_by(user_id=user_id).first()
        if not dataset:
            return jsonify({'success': False, 'message': 'No uploaded dataset found.'}), 404
            
        filepath = dataset.file_path
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Dataset file not found on disk.'}), 404
            
        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=dataset.original_filename
        )
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to download original dataset: {str(e)}'}), 500


@data_bp.route('/clear', methods=['DELETE'])
def clear_data():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        clear_user_data_internal(user_id)
        return jsonify({'success': True, 'message': 'All user data and associated models cleared successfully.'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to clear data: {str(e)}'}), 500

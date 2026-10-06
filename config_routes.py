import os
from flask import Blueprint, request, jsonify
from backend.config import Config
from backend.database.models import EnergyRecord, Dataset, TrainedModel
from backend.routes.auth import get_current_user_id

config_bp = Blueprint('config', __name__)

@config_bp.route('/config/emission-factor', methods=['GET'])
def get_emission_factor():
    return jsonify({
        'emission_factor': Config.EMISSION_FACTOR,
        'unit': 'kg CO2 per kWh'
    }), 200

@config_bp.route('/config/emission-factor', methods=['POST'])
def update_emission_factor():
    data = request.get_json() or {}
    factor = data.get('emission_factor')
    
    if factor is None:
        return jsonify({'success': False, 'message': 'Missing emission_factor.'}), 400
        
    try:
        Config.EMISSION_FACTOR = float(factor)
        return jsonify({
            'success': True,
            'emission_factor': Config.EMISSION_FACTOR
        }), 200
    except ValueError:
        return jsonify({'success': False, 'message': 'Emission factor must be a valid number.'}), 400

@config_bp.route('/config/status', methods=['GET'])
def get_app_status():
    user_id = get_current_user_id()
    
    dataset_loaded = False
    models_status = {
        'random_forest': False,
        'xgboost': False,
        'linear_regression': False,
        'lstm': False,
        'kmeans': False,
        'isolation_forest': False
    }
    active_dataset_id = None
    
    if user_id:
        # 1. Resolve active dataset for current user (fallback to user 1 pre-trained dataset)
        dataset = Dataset.query.filter_by(user_id=user_id).first()
        using_fallback_dataset = False
        if not dataset:
            dataset = Dataset.query.filter_by(user_id=1).first()
            using_fallback_dataset = True
            
        count = EnergyRecord.query.filter_by(user_id=user_id).count()
        if count == 0 and dataset:
            count = EnergyRecord.query.filter_by(user_id=dataset.user_id).count()
            
        dataset_loaded = count > 0
        active_dataset_id = dataset.id if dataset else None
        
        user_model_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}")
        fallback_model_dir = os.path.join(Config.MODEL_SAVE_PATH, "user_1")
        
        # 2. Check trained status for each required model
        for model_name in models_status.keys():
            tm = TrainedModel.query.filter_by(user_id=user_id, model_name=model_name).first()
            m_dir = user_model_dir
            
            if not tm:
                tm = TrainedModel.query.filter_by(user_id=1, model_name=model_name).first()
                m_dir = fallback_model_dir
                
            # File check on disk
            if model_name == 'random_forest':
                file_exists = os.path.exists(os.path.join(m_dir, 'random_forest.pkl'))
            elif model_name == 'xgboost':
                file_exists = os.path.exists(os.path.join(m_dir, 'xgboost.pkl'))
            elif model_name == 'linear_regression':
                file_exists = os.path.exists(os.path.join(m_dir, 'linear_regression.pkl'))
            elif model_name == 'lstm':
                file_exists = os.path.exists(os.path.join(m_dir, 'lstm.keras'))
            elif model_name == 'kmeans':
                file_exists = os.path.exists(os.path.join(m_dir, 'kmeans.pkl'))
            elif model_name == 'isolation_forest':
                file_exists = os.path.exists(os.path.join(m_dir, 'isolation_forest.pkl'))
            else:
                file_exists = False
                
            # Validation criteria: file exists AND status is 'trained' AND dataset matches active dataset
            # (or pre-trained fallback model for default dataset)
            model_is_trained = tm and (tm.status == 'trained')
            dataset_matches = tm and (
                tm.dataset_id == active_dataset_id or 
                tm.dataset_id is None or 
                using_fallback_dataset or 
                tm.user_id == 1
            )
            
            if file_exists and model_is_trained and dataset_matches:
                models_status[model_name] = True
            else:
                # Model missing or outdated on disk/DB
                models_status[model_name] = False

    # Auto-repair/train if dataset is loaded but any model is reported untrained
    if dataset_loaded and not all(models_status.values()) and user_id:
        from backend.services.auto_training_service import ensure_models_trained, is_model_valid
        print(f"[CONFIG-STATUS] Auto-training missing models for user_{user_id}...")
        ensure_models_trained(user_id)
        
        # Re-evaluate statuses
        for mname in models_status.keys():
            models_status[mname] = is_model_valid(user_id, mname) or is_model_valid(1, mname)
            
    models_trained = all(models_status.values()) and dataset_loaded
    
    return jsonify({
        'dataset_loaded': dataset_loaded,
        'models_trained': models_trained,
        'models': models_status,
        'active_dataset_id': active_dataset_id
    }), 200

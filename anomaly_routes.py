import os
import pickle
import numpy as np
from flask import Blueprint, request, jsonify
from backend.config import Config
from backend.database.models import db, Anomaly, ModelMetric, Dataset, TrainedModel
from backend.services.data_service import DataService
from backend.ml.isolation_forest import train_isolation_forest, detect_anomalies, get_anomaly_statistics
from backend.routes.auth import get_current_user_id
from sqlalchemy import func
import json
from datetime import datetime

anomaly_bp = Blueprint('anomaly', __name__)

def load_isolation_forest_model(user_id):
    path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'isolation_forest.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

@anomaly_bp.route('/anomaly/detect', methods=['POST'])
def detect_anomalies_endpoint():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 10:
        return jsonify({'success': False, 'message': 'Insufficient data. At least 10 records are required.'}), 400
        
    data = request.get_json() or {}
    contamination = float(data.get('contamination', Config.ISOLATION_FOREST_CONTAMINATION))
    
    try:
        # Prepare features for anomaly detection:
        # 'energy_consumption' and 'voltage' and 'global_intensity'
        features_list = ['energy_consumption', 'voltage', 'global_intensity']
        
        # Make case-insensitive column mapping matching dataframe
        df_cols_lower = [c.lower() for c in df.columns]
        actual_features = []
        for feat in features_list:
            if feat.lower() in df_cols_lower:
                idx = df_cols_lower.index(feat.lower())
                actual_features.append(df.columns[idx])
            else:
                actual_features.append(feat)
                
        X = df[actual_features].values
        
        # Load or Train Isolation Forest
        model = load_isolation_forest_model(user_id)
        if not model or model.contamination != contamination:
            print(f"Training Isolation Forest with contamination={contamination} for user {user_id}...")
            model = train_isolation_forest(X, contamination=contamination, user_id=user_id)
            
        # Detect anomalies
        is_anomaly, scores = detect_anomalies(model, X)
        
        # Clear old anomalies
        Anomaly.query.filter_by(user_id=user_id).delete()
        
        # Save to database
        db_anomalies = []
        for i, row in df.iterrows():
            anomaly_record = Anomaly(
                user_id=user_id,
                timestamp=row['Datetime'],
                energy_consumption=float(row['energy_consumption']),
                anomaly_score=float(scores[i]),
                is_anomaly=bool(is_anomaly[i])
            )
            db_anomalies.append(anomaly_record)
            db.session.add(anomaly_record)
            
        # Update Model Metric in DB
        metric = ModelMetric.query.filter_by(user_id=user_id, model_name='isolation_forest', metric_name='Anomalies Count').first()
        anomalies_detected = int(np.sum(is_anomaly))
        if not metric:
            metric = ModelMetric(
                user_id=user_id,
                model_name='isolation_forest',
                metric_name='Anomalies Count',
                metric_value=float(anomalies_detected)
            )
            db.session.add(metric)
        else:
            metric.metric_value = float(anomalies_detected)
            
        # Save to TrainedModel
        dataset = Dataset.query.filter_by(user_id=user_id).first()
        dataset_id = dataset.id if dataset else None
        
        tm = TrainedModel.query.filter_by(user_id=user_id, model_name='isolation_forest').first()
        if not tm:
            tm = TrainedModel(user_id=user_id, model_name='isolation_forest')
            db.session.add(tm)
        tm.dataset_id = dataset_id
        tm.training_timestamp = datetime.utcnow()
        tm.status = 'trained'
        tm.dataset_size = len(df)
        tm.metrics_json = json.dumps({'Anomalies Count': anomalies_detected})
            
        db.session.commit()
        
        # Retrieve anomalies for response
        stats = get_anomaly_statistics(is_anomaly, scores)
        
        anomalies_list = []
        for r in db_anomalies:
            if r.is_anomaly:
                anomalies_list.append({
                    'timestamp': r.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'energy': r.energy_consumption,
                    'score': r.anomaly_score,
                    'status': 'Anomalous'
                })
                
        return jsonify({
            'success': True,
            'total_records': stats['total_records'],
            'anomalies_detected': stats['anomalies_detected'],
            'percentage': stats['percentage'],
            'anomalies': anomalies_list
        }), 200
        
    except Exception as e:
        db.session.rollback()
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Anomaly detection execution failed: {str(e)}'}), 500


@anomaly_bp.route('/anomaly/timeline', methods=['GET'])
def get_anomaly_timeline():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    anomalies = Anomaly.query.filter_by(user_id=user_id).order_by(Anomaly.timestamp).all()
    
    timeline = []
    for a in anomalies:
        timeline.append({
            'timestamp': a.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'energy': a.energy_consumption,
            'score': a.anomaly_score,
            'is_anomaly': a.is_anomaly
        })
        
    return jsonify({'timeline': timeline}), 200


@anomaly_bp.route('/anomaly/statistics', methods=['GET'])
def get_anomaly_stats():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    anomalies_query = Anomaly.query.filter_by(user_id=user_id)
    total = anomalies_query.count()
    if total == 0:
        return jsonify({
            'total_anomalies': 0,
            'percentage': 0.0,
            'mean_score': 0.0,
            'anomaly_score_distribution': {}
        }), 200
        
    anomalies_detected = anomalies_query.filter_by(is_anomaly=True).count()
    percentage = (anomalies_detected / total) * 100
    
    mean_score_row = db.session.query(func.avg(Anomaly.anomaly_score)).filter_by(user_id=user_id).first()
    mean_score = float(mean_score_row[0]) if mean_score_row[0] is not None else 0.0
    
    high_normal = anomalies_query.filter(Anomaly.anomaly_score >= 0.1).count()
    borderline = anomalies_query.filter((Anomaly.anomaly_score >= 0.0) & (Anomaly.anomaly_score < 0.1)).count()
    anomalous = anomalies_query.filter(Anomaly.anomaly_score < 0.0).count()
    
    return jsonify({
        'total_anomalies': anomalies_detected,
        'percentage': percentage,
        'mean_score': mean_score,
        'anomaly_score_distribution': {
            'Normal (>=0.1)': high_normal,
            'Borderline (0.0 to 0.1)': borderline,
            'Anomalous (<0.0)': anomalous
        }
    }), 200

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from backend.config import Config
from backend.database.models import db, Prediction, ModelMetric, Dataset, TrainedModel
from backend.services.data_service import DataService
from backend.ml.feature_engineering import run_feature_engineering
from backend.ml.preprocessing import prepare_lstm_sequences
from backend.ml.random_forest import get_feature_importance
from backend.ml.xgboost_model import get_feature_importance_xgb
from backend.routes.auth import get_current_user_id

# Set TF log level to suppress output during inference
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from backend.services.auto_training_service import ensure_models_trained, is_model_valid

prediction_bp = Blueprint('prediction', __name__)

FEATURE_COLS = [
    'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'hour_category',
    'energy_consumption_lag_1', 'energy_consumption_lag_2', 'energy_consumption_lag_3', 'energy_consumption_lag_24',
    'energy_consumption_roll_mean_3', 'energy_consumption_roll_std_3',
    'energy_consumption_roll_mean_6', 'energy_consumption_roll_std_6'
]

MODEL_CACHE = {}

def check_model_trained(user_id, model_name):
    """
    Checks if the model is registered as trained for the user,
    and checks if the file exists on disk. If missing, attempts automatic auto-training.
    """
    if is_model_valid(user_id, model_name) or is_model_valid(1, model_name):
        return True
        
    print(f"[PREDICT-CHECK] Model '{model_name}' missing for user_{user_id}. Triggering auto-training...")
    ensure_models_trained(user_id)
    
    return is_model_valid(user_id, model_name) or is_model_valid(1, model_name)

def get_cached_model(user_id, filename):
    cache_key = f"{user_id}_{filename}"
    if cache_key not in MODEL_CACHE:
        path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", filename)
        if not os.path.exists(path):
            path = os.path.join(Config.MODEL_SAVE_PATH, "user_1", filename)
            cache_key = f"1_{filename}"
            if cache_key in MODEL_CACHE:
                return MODEL_CACHE[cache_key]
                
        if not os.path.exists(path):
            return None
            
        if filename.endswith('.keras'):
            try:
                print(f"[CACHE] Loading LSTM model from disk at {path}...")
                MODEL_CACHE[cache_key] = tf.keras.models.load_model(path)
            except Exception as e:
                print(f"Error loading Keras model at {path}: {e}")
                return None
        else:
            print(f"[CACHE] Loading pickle model {filename} from disk at {path}...")
            with open(path, 'rb') as f:
                MODEL_CACHE[cache_key] = pickle.load(f)
    return MODEL_CACHE[cache_key]

def get_metrics_from_db(user_id, model_name):
    metrics = ModelMetric.query.filter_by(user_id=user_id, model_name=model_name).all()
    metrics_dict = {}
    for m in metrics:
        metrics_dict[m.metric_name] = m.metric_value
    return metrics_dict

def prepare_next_hour_features(df):
    """
    Appends a placeholder row for the next hour and runs feature engineering.
    Returns the feature vector for the next hour.
    """
    if df.empty or len(df) < 24:
        return None, None
        
    latest_row = df.iloc[-1]
    next_time = latest_row['Datetime'] + timedelta(hours=1)
    
    next_row = pd.DataFrame([{
        'Datetime': next_time,
        'Global_active_power': latest_row['Global_active_power'],
        'Global_reactive_power': latest_row['Global_reactive_power'],
        'Voltage': latest_row['Voltage'],
        'Global_intensity': latest_row['Global_intensity'],
        'Sub_metering_1': latest_row['Sub_metering_1'],
        'Sub_metering_2': latest_row['Sub_metering_2'],
        'Sub_metering_3': latest_row['Sub_metering_3'],
        'energy_consumption': latest_row['energy_consumption'],
        'carbon_emission': latest_row['carbon_emission']
    }])
    
    combined_df = pd.concat([df, next_row], ignore_index=True)
    engineered_df = run_feature_engineering(combined_df, target_col='energy_consumption')
    
    next_hour_feats = engineered_df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    return next_hour_feats, next_time

@prediction_bp.route('/predict/random-forest', methods=['POST'])
def predict_rf():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized. Please log in.'}), 401
        
    if not check_model_trained(user_id, 'random_forest'):
        return jsonify({'success': False, 'message': 'Random Forest model not trained or outdated. Please train models first.'}), 400
        
    model = get_cached_model(user_id, 'random_forest.pkl')
    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 24:
        return jsonify({'success': False, 'message': 'At least 24 historical records required.'}), 400
        
    try:
        scaler_path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'scalers', 'rf_scaler.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(Config.MODEL_SAVE_PATH, "user_1", 'scalers', 'rf_scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({'success': False, 'message': 'Model scaler is missing.'}), 400
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        feats, next_time = prepare_next_hour_features(df)
        feats_scaled = scaler.transform(feats)
        
        pred_scaled = model.predict(feats_scaled)[0]
        
        target_scaler_path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'scalers', 'target_scaler.pkl')
        if not os.path.exists(target_scaler_path):
            target_scaler_path = os.path.join(Config.MODEL_SAVE_PATH, "user_1", 'scalers', 'target_scaler.pkl')
        if os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                t_scaler = pickle.load(f)
            pred_energy = float(t_scaler.inverse_transform([[pred_scaled]])[0][0])
        else:
            pred_energy = float(pred_scaled)
            
        pred_carbon = pred_energy * Config.EMISSION_FACTOR
        
        metrics = get_metrics_from_db(user_id, 'random_forest')
        
        # Save prediction to DB
        prediction_record = Prediction(
            user_id=user_id,
            timestamp=next_time,
            model_name='random_forest',
            predicted_energy=pred_energy,
            predicted_carbon=pred_carbon,
            mae=metrics.get('MAE'),
            rmse=metrics.get('RMSE'),
            r2=metrics.get('R2')
        )
        db.session.add(prediction_record)
        db.session.commit()
        
        # Get feature importances
        importances = get_feature_importance(model, FEATURE_COLS)
        
        return jsonify({
            'success': True,
            'predicted_energy': pred_energy,
            'predicted_carbon': pred_carbon,
            'mae': metrics.get('MAE', 0.0),
            'rmse': metrics.get('RMSE', 0.0),
            'r2': metrics.get('R2', 0.0),
            'importances': importances
        }), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'}), 500

@prediction_bp.route('/predict/xgboost', methods=['POST'])
def predict_xgb():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized. Please log in.'}), 401
        
    if not check_model_trained(user_id, 'xgboost'):
        return jsonify({'success': False, 'message': 'XGBoost model not trained or outdated. Please train models first.'}), 400
        
    model = get_cached_model(user_id, 'xgboost.pkl')
    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 24:
        return jsonify({'success': False, 'message': 'At least 24 records required.'}), 400
        
    try:
        scaler_path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'scalers', 'xgb_scaler.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(Config.MODEL_SAVE_PATH, "user_1", 'scalers', 'xgb_scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({'success': False, 'message': 'Model scaler is missing.'}), 400
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        feats, next_time = prepare_next_hour_features(df)
        feats_scaled = scaler.transform(feats)
        
        pred_scaled = model.predict(feats_scaled)[0]
        
        target_scaler_path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'scalers', 'target_scaler.pkl')
        if not os.path.exists(target_scaler_path):
            target_scaler_path = os.path.join(Config.MODEL_SAVE_PATH, "user_1", 'scalers', 'target_scaler.pkl')
        if os.path.exists(target_scaler_path):
            with open(target_scaler_path, 'rb') as f:
                t_scaler = pickle.load(f)
            pred_energy = float(t_scaler.inverse_transform([[pred_scaled]])[0][0])
        else:
            pred_energy = float(pred_scaled)
            
        pred_carbon = pred_energy * Config.EMISSION_FACTOR
        
        metrics = get_metrics_from_db(user_id, 'xgboost')
        
        prediction_record = Prediction(
            user_id=user_id,
            timestamp=next_time,
            model_name='xgboost',
            predicted_energy=pred_energy,
            predicted_carbon=pred_carbon,
            mae=metrics.get('MAE'),
            rmse=metrics.get('RMSE'),
            r2=metrics.get('R2')
        )
        db.session.add(prediction_record)
        db.session.commit()
        
        importances = get_feature_importance_xgb(model, FEATURE_COLS)
        
        return jsonify({
            'success': True,
            'predicted_energy': pred_energy,
            'predicted_carbon': pred_carbon,
            'mae': metrics.get('MAE', 0.0),
            'rmse': metrics.get('RMSE', 0.0),
            'r2': metrics.get('R2', 0.0),
            'importances': importances
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction failed: {str(e)}'}), 500

@prediction_bp.route('/forecast/lstm', methods=['POST'])
def forecast_lstm_endpoint():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    if not check_model_trained(user_id, 'lstm'):
        return jsonify({'success': False, 'message': 'LSTM model not trained or outdated. Please train models first.'}), 400
        
    model = get_cached_model(user_id, 'lstm.keras')
    df = DataService.get_user_records_df(user_id)
    seq_len = Config.LSTM_SEQUENCE_LENGTH
    
    if df.empty or len(df) < seq_len:
        return jsonify({'success': False, 'message': f'At least {seq_len} records are required for LSTM.'}), 400
        
    data = request.get_json() or {}
    steps = int(data.get('steps', 24))
    
    try:
        scaler_path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'scalers', 'lstm_scaler.pkl')
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(Config.MODEL_SAVE_PATH, "user_1", 'scalers', 'lstm_scaler.pkl')
        if not os.path.exists(scaler_path):
            return jsonify({'success': False, 'message': 'LSTM scalers are missing.'}), 400
            
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        features_list = [
            'Global_active_power', 'Global_reactive_power', 'Voltage', 
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        
        # Ensure column names map correctly to dataframe
        df_cols_lower = [c.lower() for c in df.columns]
        actual_features = []
        for feat in features_list:
            if feat.lower() in df_cols_lower:
                idx = df_cols_lower.index(feat.lower())
                actual_features.append(df.columns[idx])
            else:
                actual_features.append(feat)
        
        latest_data = df[actual_features].tail(seq_len).copy()
        # Rename back to expected by scaler in case of case variance
        latest_data.columns = features_list
        
        scaled_data = scaler.transform(latest_data)
        
        forecast_scaled = []
        current_seq = scaled_data.copy().reshape(1, seq_len, len(features_list))
        
        for _ in range(steps):
            pred = model.predict(current_seq, verbose=0)[0][0]
            forecast_scaled.append(float(pred))
            
            next_row = current_seq[0, -1, :].copy()
            next_row[0] = pred  # Update Global_active_power
            
            new_seq = np.empty_like(current_seq)
            new_seq[0, :-1, :] = current_seq[0, 1:, :]
            new_seq[0, -1, :] = next_row
            current_seq = new_seq
            
        forecast_kwh = []
        for val in forecast_scaled:
            dummy_row = np.zeros((1, len(features_list)))
            dummy_row[0, 0] = val
            inv_row = scaler.inverse_transform(dummy_row)
            kwh = float(inv_row[0, 0])
            forecast_kwh.append(max(0.0, kwh))
            
        latest_time = df['Datetime'].iloc[-1]
        timestamps = []
        for i in range(1, steps + 1):
            next_time = latest_time + timedelta(hours=i)
            timestamps.append(next_time.strftime('%Y-%m-%d %H:%M:%S'))
            
        metrics = get_metrics_from_db(user_id, 'lstm')
        
        # Save predictions in DB
        for t, kwh in zip(timestamps, forecast_kwh):
            parsed_t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            prediction_record = Prediction(
                user_id=user_id,
                timestamp=parsed_t,
                model_name='lstm',
                predicted_energy=kwh,
                predicted_carbon=kwh * Config.EMISSION_FACTOR,
                mae=metrics.get('MAE'),
                rmse=metrics.get('RMSE')
            )
            db.session.add(prediction_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'forecast': forecast_kwh,
            'timestamps': timestamps,
            'mae': metrics.get('MAE', 0.0),
            'rmse': metrics.get('RMSE', 0.0)
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Forecasting failed: {str(e)}'}), 500

@prediction_bp.route('/predict/comparison', methods=['GET'])
def get_comparison():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        rf_metrics = get_metrics_from_db(user_id, 'random_forest')
        xgb_metrics = get_metrics_from_db(user_id, 'xgboost')
        lstm_metrics = get_metrics_from_db(user_id, 'lstm')
        
        return jsonify({
            'models': {
                'random_forest': rf_metrics,
                'xgboost': xgb_metrics,
                'lstm': lstm_metrics
            }
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

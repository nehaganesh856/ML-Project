import os
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Add the parent folder to sys.path so we can import backend packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import Config
from backend.app import app
from backend.database.models import db, ModelMetric, Cluster, Anomaly, User, TrainedModel, Dataset, EnergyRecord
from backend.services.data_service import DataService

# Import ML model steps
from backend.ml.preprocessing import (
    load_csv, handle_missing_values, handle_invalid_values,
    calculate_energy_consumption, calculate_carbon_emission,
    aggregate_data, split_train_test, prepare_lstm_sequences
)
from backend.ml.feature_engineering import run_feature_engineering
from backend.ml.random_forest import train_random_forest, evaluate_random_forest
from backend.ml.xgboost_model import train_xgboost, evaluate_xgboost
from backend.ml.kmeans_model import train_kmeans, evaluate_kmeans, get_cluster_statistics
from backend.ml.isolation_forest import train_isolation_forest, detect_anomalies
from backend.ml.lstm_model import train_lstm, evaluate_lstm

def main():
    parser = argparse.ArgumentParser(description="Train all 5 ML models on household power consumption data.")
    parser.add_argument('--sample-size', type=int, default=50000, 
                        help='Number of records to sample from CSV for training efficiency.')
    parser.add_argument('--lstm-epochs', type=int, default=10, 
                        help='Number of training epochs for the LSTM model.')
    parser.add_argument('--csv-path', type=str, default=None,
                        help='Custom path to the household power consumption CSV file.')
    
    args = parser.parse_args()
    
    # 1. Resolve CSV file path
    csv_file = args.csv_path
    if not csv_file:
        csv_file = os.path.join(Config.DATA_PATH, 'household_power_consumption.csv')
        
    print(f"=== Starting Model Training Pipeline ===")
    print(f"Dataset path: {csv_file}")
    print(f"Sample size: {args.sample_size}")
    
    if not os.path.exists(csv_file):
        print(f"\n[ERROR] Dataset CSV file not found at: {csv_file}")
        print("Please place the UCI 'household_power_consumption.csv' file in 'backend/data/'")
        print("or upload it through the application web dashboard.")
        sys.exit(1)
        
    # Get a user ID from database to associate training records (default to 'demo' user)
    with app.app_context():
        demo_user = User.query.filter_by(username='demo').first()
        if not demo_user:
            print("[ERROR] Default demo user not found in database. Start the Flask application first.")
            sys.exit(1)
        user_id = demo_user.id

        # Resolve or create Dataset record
        dataset = Dataset.query.filter_by(user_id=user_id).first()
        if not dataset:
            # Seed dataset record
            dataset = Dataset(
                user_id=user_id,
                original_filename='household_power_consumption.csv',
                stored_filename=f'user_{user_id}_dataset.csv',
                file_path=csv_file,
                file_size=os.path.getsize(csv_file) if os.path.exists(csv_file) else 0,
                upload_timestamp=datetime.utcnow(),
                status='processed'
            )
            db.session.add(dataset)
            db.session.commit()
            
        dataset_id = dataset.id
        dataset_size = args.sample_size

        def update_trained_model(name, status, metrics={}):
            tm = TrainedModel.query.filter_by(user_id=user_id, model_name=name).first()
            if not tm:
                tm = TrainedModel(user_id=user_id, model_name=name)
                db.session.add(tm)
            tm.dataset_id = dataset_id
            tm.training_timestamp = datetime.utcnow()
            tm.status = status
            tm.dataset_size = dataset_size
            tm.metrics_json = json.dumps(metrics)
            db.session.commit()

        # 2. Loading & Preprocessing
        print("\n[1/7] Loading and cleaning dataset...")
        try:
            df = load_csv(csv_file, sample_size=args.sample_size) # Apply sample size directly
            df = handle_missing_values(df)
            df = handle_invalid_values(df)
            df = calculate_energy_consumption(df)
            df = calculate_carbon_emission(df, Config.EMISSION_FACTOR)
            df_hourly = aggregate_data(df, freq='h')
            
            # Apply sampling if hourly is large
            if len(df_hourly) > args.sample_size:
                df_hourly = df_hourly.iloc[:args.sample_size].copy()
                
            dataset_size = len(df_hourly)
            print(f"Data loading complete. Loaded {len(df_hourly)} hourly aggregated records.")
            
            # Sync dataset size to metadata
            dataset.record_count = len(df)
            dataset.processed_record_count = len(df_hourly)
            db.session.commit()
            
            # Save these records to the DB for the demo user
            print("Syncing data to SQLite database...")
            # Pre-clear old records to avoid duplication
            EnergyRecord.query.filter_by(user_id=user_id).delete()
            db.session.commit()
            DataService.add_records_batch(user_id, df_hourly)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            sys.exit(1)
            
        # 3. K-Means Clustering
        print("\n[2/7] Training K-Means model...")
        try:
            X_clust = df_hourly[['energy_consumption', 'carbon_emission']].values
            km_model = train_kmeans(X_clust, Config.KMEANS_CLUSTERS, user_id=user_id)
            km_preds = km_model.predict(X_clust)
            km_sil = evaluate_kmeans(X_clust, km_preds)
            
            # Save metrics
            ModelMetric.query.filter_by(user_id=user_id, model_name='kmeans').delete()
            db.session.add(ModelMetric(user_id=user_id, model_name='kmeans', metric_name='Silhouette Score', metric_value=km_sil))
            
            # Save clusters
            Cluster.query.filter_by(user_id=user_id).delete()
            km_stats = get_cluster_statistics(km_model, X_clust, km_preds, ['energy_consumption', 'carbon_emission'])
            for cid, info in km_stats.items():
                db.session.add(Cluster(
                    user_id=user_id, cluster_id=info['cluster_id'], cluster_label=info['cluster_label'],
                    centroid_energy=info['centroid_energy'], centroid_carbon=info['centroid_carbon'], cluster_size=info['cluster_size']
                ))
            db.session.commit()
            update_trained_model('kmeans', 'trained', {'Silhouette Score': km_sil})
            print(f"K-Means complete. Silhouette Score: {km_sil:.4f}")
        except Exception as e:
            print(f"[ERROR] K-Means training failed: {e}")
            db.session.rollback()
            
        # 4. Anomaly Detection
        print("\n[3/7] Training Isolation Forest model...")
        try:
            X_anom = df_hourly[['energy_consumption', 'Voltage', 'Global_intensity']].values
            if_model = train_isolation_forest(X_anom, Config.ISOLATION_FOREST_CONTAMINATION, user_id=user_id)
            is_anomaly, anom_scores = detect_anomalies(if_model, X_anom)
            
            # Save anomalies
            Anomaly.query.filter_by(user_id=user_id).delete()
            anomalies_detected = int(np.sum(is_anomaly))
            for i, row in df_hourly.iterrows():
                db.session.add(Anomaly(
                    user_id=user_id, timestamp=row['Datetime'],
                    energy_consumption=float(row['energy_consumption']),
                    anomaly_score=float(anom_scores[i]), is_anomaly=bool(is_anomaly[i])
                ))
                
            ModelMetric.query.filter_by(user_id=user_id, model_name='isolation_forest').delete()
            db.session.add(ModelMetric(user_id=user_id, model_name='isolation_forest', metric_name='Anomalies Count', metric_value=float(anomalies_detected)))
            db.session.commit()
            update_trained_model('isolation_forest', 'trained', {'Anomalies Count': anomalies_detected})
            print(f"Anomaly detection complete. Identified {anomalies_detected} anomalies ({anomalies_detected/len(df_hourly)*100:.2f}%)")
        except Exception as e:
            print(f"[ERROR] Isolation Forest failed: {e}")
            db.session.rollback()

        # 5. Feature Engineering and Split (NO DATA LEAKAGE)
        print("\n[4/7] Engineering regression features...")
        try:
            df_feat = run_feature_engineering(df_hourly, target_col='energy_consumption')
            feature_cols = [
                'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'hour_category',
                'energy_consumption_lag_1', 'energy_consumption_lag_2', 'energy_consumption_lag_3', 'energy_consumption_lag_24',
                'energy_consumption_roll_mean_3', 'energy_consumption_roll_std_3',
                'energy_consumption_roll_mean_6', 'energy_consumption_roll_std_6'
            ]
            
            X_reg_raw = df_feat[feature_cols].values
            y_reg_raw = df_feat[['energy_consumption']].values
            
            # Split chronologically BEFORE scaling
            X_train_df, X_test_df = split_train_test(pd.DataFrame(X_reg_raw), time_series=True)
            y_train_df, y_test_df = split_train_test(pd.DataFrame(y_reg_raw), time_series=True)
            X_train_raw, X_test_raw = X_train_df.values, X_test_df.values
            y_train_raw, y_test_raw = y_train_df.values, y_test_df.values
            
            # Fit scalers only on training data
            scaler_rf = MinMaxScaler()
            X_train = scaler_rf.fit_transform(X_train_raw)
            X_test = scaler_rf.transform(X_test_raw)
            
            t_scaler = MinMaxScaler()
            y_train = t_scaler.fit_transform(y_train_raw).ravel()
            y_test = t_scaler.transform(y_test_raw).ravel()
            
            # Save Scalers to isolated folder
            user_scaler_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", "scalers")
            os.makedirs(user_scaler_dir, exist_ok=True)
            with open(os.path.join(user_scaler_dir, 'rf_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler_rf, f)
            with open(os.path.join(user_scaler_dir, 'xgb_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler_rf, f)
            with open(os.path.join(user_scaler_dir, 'target_scaler.pkl'), 'wb') as f:
                pickle.dump(t_scaler, f)
                
        except Exception as e:
            print(f"[ERROR] Feature engineering failed: {e}")
            sys.exit(1)

        # 6. Random Forest Training
        print("\n[5/7] Training Random Forest Regressor...")
        try:
            rf_model = train_random_forest(X_train, y_train, Config.RF_N_ESTIMATORS, Config.RF_MAX_DEPTH, user_id=user_id)
            rf_eval = evaluate_random_forest(rf_model, X_test, y_test)
            
            # Save RF metrics
            ModelMetric.query.filter_by(user_id=user_id, model_name='random_forest').delete()
            for mkey, mval in rf_eval.items():
                db.session.add(ModelMetric(user_id=user_id, model_name='random_forest', metric_name=mkey, metric_value=mval))
            db.session.commit()
            update_trained_model('random_forest', 'trained', rf_eval)
            print(f"Random Forest complete. R2 Score: {rf_eval['R2']:.4f}, MAE: {rf_eval['MAE']:.4f}")
        except Exception as e:
            print(f"[ERROR] Random Forest failed: {e}")
            db.session.rollback()

        # 7. XGBoost Training
        print("\n[6/7] Training XGBoost Regressor...")
        try:
            xgb_model = train_xgboost(X_train, y_train, Config.XGB_LEARNING_RATE, Config.XGB_N_ESTIMATORS, user_id=user_id)
            xgb_eval = evaluate_xgboost(xgb_model, X_test, y_test)
            
            # Save XGBoost metrics
            ModelMetric.query.filter_by(user_id=user_id, model_name='xgboost').delete()
            for mkey, mval in xgb_eval.items():
                db.session.add(ModelMetric(user_id=user_id, model_name='xgboost', metric_name=mkey, metric_value=mval))
            db.session.commit()
            update_trained_model('xgboost', 'trained', xgb_eval)
            print(f"XGBoost complete. R2 Score: {xgb_eval['R2']:.4f}, MAE: {xgb_eval['MAE']:.4f}")
        except Exception as e:
            print(f"[ERROR] XGBoost failed: {e}")
            db.session.rollback()

        # 8. LSTM Training
        print("\n[7/7] Training LSTM model (may take some time)...")
        try:
            lstm_features = [
                'Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
            ]
            
            # Chronological split first
            split_idx = int(len(df_hourly) * 0.8)
            train_df = df_hourly.iloc[:split_idx].copy()
            test_df = df_hourly.iloc[split_idx:].copy()
            
            # Fit scaler ONLY on train data features
            scaler_lstm = MinMaxScaler()
            train_df[lstm_features] = scaler_lstm.fit_transform(train_df[lstm_features])
            test_df[lstm_features] = scaler_lstm.transform(test_df[lstm_features])
            
            # Save LSTM Scaler
            with open(os.path.join(user_scaler_dir, 'lstm_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler_lstm, f)
                
            # Prepare sequences on scaled subsets
            X_seq_train, y_seq_train, _ = prepare_lstm_sequences(
                train_df, feature_cols=lstm_features, target_col='Global_active_power', sequence_length=Config.LSTM_SEQUENCE_LENGTH
            )
            X_seq_test, y_seq_test, _ = prepare_lstm_sequences(
                test_df, feature_cols=lstm_features, target_col='Global_active_power', sequence_length=Config.LSTM_SEQUENCE_LENGTH
            )
            
            # LSTM Split
            X_seq_train, X_seq_test = X_seq_train, X_seq_test
            y_seq_train, y_seq_test = y_seq_train, y_seq_test
            
            # Val split
            val_split = int(len(X_seq_train) * 0.9)
            X_seq_tr, X_seq_val = X_seq_train[:val_split], X_seq_train[val_split:]
            y_seq_tr, y_seq_val = y_seq_train[:val_split], y_seq_train[val_split:]
            
            # Train LSTM
            lstm_model, history = train_lstm(
                X_seq_tr, y_seq_tr, X_seq_val, y_seq_val,
                epochs=args.lstm_epochs,
                batch_size=Config.LSTM_BATCH_SIZE,
                user_id=user_id
            )
            
            lstm_eval = evaluate_lstm(lstm_model, X_seq_test, y_seq_test)
            
            # Save LSTM metrics
            ModelMetric.query.filter_by(user_id=user_id, model_name='lstm').delete()
            for mkey, mval in lstm_eval.items():
                db.session.add(ModelMetric(user_id=user_id, model_name='lstm', metric_name=mkey, metric_value=mval))
            db.session.commit()
            update_trained_model('lstm', 'trained', lstm_eval)
            print(f"LSTM complete. Test MAE: {lstm_eval['MAE']:.4f}, RMSE: {lstm_eval['RMSE']:.4f}")
        except Exception as e:
            print(f"[ERROR] LSTM training failed: {e}")
            db.session.rollback()

    print("\n=== Model Training Complete! All models are successfully saved. ===")

if __name__ == '__main__':
    main()

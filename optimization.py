import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from backend.config import Config
from backend.services.data_service import DataService
from backend.ml.feature_engineering import run_feature_engineering
from backend.ml.preprocessing import split_train_test

OPTIMIZATION_STATUS = {}

def get_optimization_status(user_id=1):
    if user_id not in OPTIMIZATION_STATUS:
        OPTIMIZATION_STATUS[user_id] = {
            'status': 'idle',
            'progress': 0,
            'message': 'No optimization run yet.',
            'results': None
        }
    return OPTIMIZATION_STATUS[user_id]

def run_optuna_optimization(user_id=1, n_trials=10):
    """
    Executes Optuna hyperparameter tuning on Random Forest and XGBoost models.
    Saves best params & evaluated scores. Replaces model files if validation improves.
    """
    status_dict = get_optimization_status(user_id)
    status_dict['status'] = 'running'
    status_dict['progress'] = 10
    status_dict['message'] = 'Loading feature dataset for hyperparameter tuning...'
    status_dict['results'] = None

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        status_dict['status'] = 'failed'
        status_dict['message'] = 'Optuna package is not installed.'
        return status_dict

    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 50:
        df = DataService.get_user_records_df(1)

    if df.empty or len(df) < 50:
        status_dict['status'] = 'failed'
        status_dict['message'] = 'Insufficient records for hyperparameter optimization.'
        return status_dict

    # Preprocessing
    df_feat = run_feature_engineering(df, target_col='energy_consumption')
    feature_cols = [
        'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'hour_category',
        'energy_consumption_lag_1', 'energy_consumption_lag_2', 'energy_consumption_lag_3', 'energy_consumption_lag_24',
        'energy_consumption_roll_mean_3', 'energy_consumption_roll_std_3',
        'energy_consumption_roll_mean_6', 'energy_consumption_roll_std_6'
    ]

    X_raw = df_feat[feature_cols].values
    y_raw = df_feat[['energy_consumption']].values

    X_tr_df, X_te_df = split_train_test(pd.DataFrame(X_raw), time_series=True)
    y_tr_df, y_te_df = split_train_test(pd.DataFrame(y_raw), time_series=True)

    X_train_raw, X_val_raw = X_tr_df.values, X_te_df.values
    y_train_raw, y_val_raw = y_tr_df.values, y_te_df.values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    t_scaler = MinMaxScaler()
    y_train = t_scaler.fit_transform(y_train_raw).ravel()
    y_val = t_scaler.transform(y_val_raw).ravel()

    # 1. Random Forest Optimization
    status_dict['progress'] = 30
    status_dict['message'] = 'Optimizing Random Forest hyperparameters...'

    from sklearn.ensemble import RandomForestRegressor

    def rf_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 20, 150)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    rf_study = optuna.create_study(direction='minimize')
    rf_study.optimize(rf_objective, n_trials=n_trials)
    best_rf_params = rf_study.best_params

    # Evaluate best RF
    best_rf_model = RandomForestRegressor(**best_rf_params, random_state=42)
    best_rf_model.fit(X_train, y_train)
    rf_preds_scaled = best_rf_model.predict(X_val)
    rf_preds_real = t_scaler.inverse_transform(rf_preds_scaled.reshape(-1, 1)).ravel()
    y_val_real = y_val_raw.ravel()

    rf_mae = float(mean_absolute_error(y_val_real, rf_preds_real))
    rf_rmse = float(np.sqrt(mean_squared_error(y_val_real, rf_preds_real)))
    rf_r2 = float(r2_score(y_val_real, rf_preds_real))

    # 2. XGBoost Optimization
    status_dict['progress'] = 70
    status_dict['message'] = 'Optimizing XGBoost hyperparameters...'

    import xgboost as xgb

    def xgb_objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 20, 150)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    xgb_study = optuna.create_study(direction='minimize')
    xgb_study.optimize(xgb_objective, n_trials=n_trials)
    best_xgb_params = xgb_study.best_params

    # Evaluate best XGB
    best_xgb_model = xgb.XGBRegressor(**best_xgb_params, random_state=42)
    best_xgb_model.fit(X_train, y_train)
    xgb_preds_scaled = best_xgb_model.predict(X_val)
    xgb_preds_real = t_scaler.inverse_transform(xgb_preds_scaled.reshape(-1, 1)).ravel()

    xgb_mae = float(mean_absolute_error(y_val_real, xgb_preds_real))
    xgb_rmse = float(np.sqrt(mean_squared_error(y_val_real, xgb_preds_real)))
    xgb_r2 = float(r2_score(y_val_real, xgb_preds_real))

    # Save optimized model files
    user_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}")
    os.makedirs(os.path.join(user_dir, "scalers"), exist_ok=True)

    with open(os.path.join(user_dir, "random_forest.pkl"), 'wb') as f:
        pickle.dump(best_rf_model, f)
    with open(os.path.join(user_dir, "xgboost.pkl"), 'wb') as f:
        pickle.dump(best_xgb_model, f)

    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    results = {
        "timestamp": timestamp,
        "n_trials": n_trials,
        "random_forest": {
            "best_params": best_rf_params,
            "mae": round(rf_mae, 4),
            "rmse": round(rf_rmse, 4),
            "r2": round(rf_r2, 4),
            "status": "optimized"
        },
        "xgboost": {
            "best_params": best_xgb_params,
            "mae": round(xgb_mae, 4),
            "rmse": round(xgb_rmse, 4),
            "r2": round(xgb_r2, 4),
            "status": "optimized"
        }
    }

    status_dict['status'] = 'completed'
    status_dict['progress'] = 100
    status_dict['message'] = 'Hyperparameter optimization completed successfully!'
    status_dict['results'] = results

    return status_dict

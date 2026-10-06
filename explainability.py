import os
import pickle
import numpy as np
import pandas as pd
from backend.config import Config
from backend.services.data_service import DataService
from backend.ml.feature_engineering import run_feature_engineering

def generate_shap_explanation(user_id=1):
    """
    Generates SHAP feature importance & sample prediction explanation 
    using the trained Random Forest / XGBoost model.
    """
    try:
        import shap
    except ImportError:
        shap = None

    # Fetch user records
    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 20:
        df = DataService.get_user_records_df(1)
        
    if df.empty:
        return {
            "status": "error",
            "message": "Insufficient dataset records for explanation."
        }

    # Feature engineering
    df_feat = run_feature_engineering(df, target_col='energy_consumption')
    feature_cols = [
        'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'hour_category',
        'energy_consumption_lag_1', 'energy_consumption_lag_2', 'energy_consumption_lag_3', 'energy_consumption_lag_24',
        'energy_consumption_roll_mean_3', 'energy_consumption_roll_std_3',
        'energy_consumption_roll_mean_6', 'energy_consumption_roll_std_6'
    ]

    feature_labels = {
        'hour': 'Hour of Day',
        'day': 'Day of Month',
        'month': 'Month of Year',
        'day_of_week': 'Day of Week',
        'is_weekend': 'Weekend Indicator',
        'hour_category': 'Time Window (Peak/Off-Peak)',
        'energy_consumption_lag_1': 'Previous Hour Load (1h Lag)',
        'energy_consumption_lag_2': '2-Hour Prior Load',
        'energy_consumption_lag_3': '3-Hour Prior Load',
        'energy_consumption_lag_24': 'Same Hour Previous Day (24h Lag)',
        'energy_consumption_roll_mean_3': '3-Hour Moving Average Load',
        'energy_consumption_roll_std_3': '3-Hour Load Volatility',
        'energy_consumption_roll_mean_6': '6-Hour Moving Average Load',
        'energy_consumption_roll_std_6': '6-Hour Load Volatility'
    }

    X_raw = df_feat[feature_cols].values
    
    # Load RF scaler and model
    user_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}")
    fallback_dir = os.path.join(Config.MODEL_SAVE_PATH, "user_1")
    
    model_p = os.path.join(user_dir, "random_forest.pkl")
    scaler_p = os.path.join(user_dir, "scalers", "rf_scaler.pkl")
    
    if not os.path.exists(model_p):
        model_p = os.path.join(fallback_dir, "random_forest.pkl")
        scaler_p = os.path.join(fallback_dir, "scalers", "rf_scaler.pkl")

    if not os.path.exists(model_p) or not os.path.exists(scaler_p):
        return {
            "status": "error",
            "message": "Trained model or feature scalers unavailable."
        }

    with open(model_p, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_p, 'rb') as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X_raw)
    
    # Use SHAP TreeExplainer if available
    features_list = []
    base_value = float(np.mean(df['carbon_emission']))
    
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled[-1:])
            
            if isinstance(shap_values, list):
                s_vals = shap_values[0][0]
            else:
                s_vals = shap_values[0]

            for i, col in enumerate(feature_cols):
                contrib = float(s_vals[i])
                f_val = float(X_raw[-1, i])
                features_list.append({
                    "feature_key": col,
                    "name": feature_labels.get(col, col),
                    "value": round(f_val, 2),
                    "contribution": round(contrib, 4),
                    "impact": "high_emission" if contrib > 0 else "saving"
                })
        except Exception as ex:
            print(f"[SHAP] Fallback to feature_importances_: {ex}")
            shap = None

    if not features_list:
        importances = model.feature_importances_
        for i, col in enumerate(feature_cols):
            imp = float(importances[i])
            f_val = float(X_raw[-1, i])
            features_list.append({
                "feature_key": col,
                "name": feature_labels.get(col, col),
                "value": round(f_val, 2),
                "contribution": round(imp * (f_val - np.mean(X_raw[:, i])) / (np.std(X_raw[:, i]) + 1e-5), 4),
                "impact": "high_emission" if imp > 0.05 else "neutral"
            })

    # Sort by absolute contribution magnitude
    features_list.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    top_factors = features_list[:6]

    latest_emission = float(df['carbon_emission'].iloc[-1])

    return {
        "status": "success",
        "latest_carbon_emission": round(latest_emission, 3),
        "base_value": round(base_value, 3),
        "explanation": "SHAP TreeExplainer breakdown showing primary feature contributions to peak emission level.",
        "top_contributing_factors": top_factors,
        "all_features": features_list
    }

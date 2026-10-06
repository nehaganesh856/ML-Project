import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from backend.config import Config

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=15, user_id=None):
    """
    Train a Random Forest Regressor and save it.
    """
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Save model
        save_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}") if user_id else Config.MODEL_SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'random_forest.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Random Forest model saved successfully to {model_path}")
        return model
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        raise e

def evaluate_random_forest(model, X_test, y_test, scaler=None):
    """
    Evaluate the Random Forest model and return metrics.
    """
    y_pred = model.predict(X_test)
    
    if scaler is not None:
        y_test = scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).ravel()
        y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).ravel()
        
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    try:
        mask = np.abs(y_test) > 1e-3
        if np.any(mask):
            mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
        else:
            mape = float(mean_absolute_percentage_error(y_test, y_pred) * 100)
    except Exception:
        mape = 0.0
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape)
    }

def predict_random_forest(model, X_input):
    """
    Make predictions using the Random Forest model.
    """
    # X_input can be single row or multiple rows
    if len(X_input.shape) == 1:
        X_input = X_input.reshape(1, -1)
    predictions = model.predict(X_input)
    return predictions

def get_feature_importance(model, feature_names):
    """
    Extract, sort, and return feature importance scores.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    importance_list = []
    for i in indices:
        importance_list.append({
            'feature': feature_names[i],
            'importance': float(importances[i])
        })
        
    return importance_list

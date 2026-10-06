import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from backend.config import Config

def train_linear_regression(X_train, y_train, user_id=None):
    """
    Train a Linear Regression model and save it.
    """
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        save_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}") if user_id else Config.MODEL_SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'linear_regression.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Linear Regression model saved successfully to {model_path}")
        return model
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
        raise e

def evaluate_linear_regression(model, X_test, y_test, scaler=None):
    """
    Evaluate Linear Regression model and return metrics (MAE, MSE, RMSE, R2, MAPE).
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

def predict_linear_regression(model, X_input):
    """
    Make predictions using Linear Regression.
    """
    if len(X_input.shape) == 1:
        X_input = X_input.reshape(1, -1)
    return model.predict(X_input)

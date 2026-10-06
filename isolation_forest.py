import os
import pickle
import numpy as np
from sklearn.ensemble import IsolationForest
from backend.config import Config

def train_isolation_forest(X, contamination=0.05, user_id=None):
    """
    Train an Isolation Forest model and save it.
    """
    try:
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X)
        
        # Save model
        save_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}") if user_id else Config.MODEL_SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'isolation_forest.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Isolation Forest model saved successfully to {model_path}")
        return model
    except Exception as e:
        print(f"Error training Isolation Forest: {e}")
        raise e

def detect_anomalies(model, X):
    """
    Run anomaly detection on dataset.
    Returns:
        is_anomaly: boolean array (True if anomaly, False if normal)
        scores: array of anomaly scores (lower = more anomalous)
    """
    # sklearn IsolationForest predicts: -1 for anomaly, 1 for normal
    preds = model.predict(X)
    is_anomaly = preds == -1
    
    # decision_function gives anomaly scores. Lower scores mean more anomalous
    scores = model.decision_function(X)
    
    return is_anomaly, scores

def predict_anomaly(model, X_input):
    """
    Predict anomaly for a single input record.
    Returns boolean (is_anomaly) and score.
    """
    if len(X_input.shape) == 1:
        X_input = X_input.reshape(1, -1)
        
    pred = model.predict(X_input)[0]
    score = model.decision_function(X_input)[0]
    
    return bool(pred == -1), float(score)

def get_anomaly_statistics(is_anomaly, scores):
    """
    Calculate summary stats of anomaly predictions.
    """
    total = len(is_anomaly)
    anomalies_count = int(np.sum(is_anomaly))
    percentage = float(anomalies_count / total * 100) if total > 0 else 0.0
    mean_score = float(np.mean(scores)) if total > 0 else 0.0
    
    return {
        'total_records': total,
        'anomalies_detected': anomalies_count,
        'percentage': percentage,
        'mean_score': mean_score
    }

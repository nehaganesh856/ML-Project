import os
import json
from datetime import datetime

def log_experiment_run(model_name, params=None, metrics=None, user_id=1):
    """
    Logs model training parameters & metrics into MLflow.
    Gracefully handles missing MLflow / exceptions without breaking model execution.
    """
    try:
        import mlflow

        mlflow_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{mlflow_dir.replace('\\', '/')}")
        mlflow.set_experiment("EcoSense_Carbon_Monitoring")

        with mlflow.start_run(run_name=f"{model_name}_user_{user_id}_{datetime.utcnow().strftime('%M%S')}"):
            mlflow.log_param("user_id", user_id)
            mlflow.log_param("model_name", model_name)
            
            if params and isinstance(params, dict):
                for k, v in params.items():
                    mlflow.log_param(k, str(v))
                    
            if metrics and isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k.replace(' ', '_').lower(), float(v))

        print(f"[MLFLOW] Successfully logged experiment run for '{model_name}'.")
        return True
    except Exception as ex:
        print(f"[MLFLOW] Experiment logging skipped ({ex}).")
        return False

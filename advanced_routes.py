import threading
from flask import Blueprint, jsonify, request
from backend.routes.auth import get_current_user_id
from backend.ml.explainability import generate_shap_explanation
from backend.ml.prophet_forecasting import forecast_carbon_emissions
from backend.ml.optimization import run_optuna_optimization, get_optimization_status

advanced_bp = Blueprint('advanced', __name__)

@advanced_bp.route('/explainability/shap', methods=['GET'])
def get_shap_explainability():
    user_id = get_current_user_id() or 1
    try:
        explanation = generate_shap_explanation(user_id=user_id)
        return jsonify(explanation), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Explainability service error: {str(e)}"
        }), 500

@advanced_bp.route('/forecast/carbon', methods=['GET'])
def get_carbon_forecast():
    user_id = get_current_user_id() or 1
    days = request.args.get('days', 7, type=int)
    try:
        forecast_res = forecast_carbon_emissions(user_id=user_id, forecast_days=days)
        return jsonify(forecast_res), 200
    except Exception as e:
        return jsonify({
            "status": "not_available",
            "reason": f"Forecast service error: {str(e)}"
        }), 500

@advanced_bp.route('/models/optimize', methods=['POST'])
def trigger_optimization():
    user_id = get_current_user_id() or 1
    status = get_optimization_status(user_id)
    if status['status'] == 'running':
        return jsonify({
            "status": "error",
            "message": "Optimization is already running in background."
        }), 400

    n_trials = 10
    if request.is_json and request.json:
        n_trials = request.json.get('n_trials', 10)

    thread = threading.Thread(target=run_optuna_optimization, args=(user_id, n_trials))
    thread.daemon = True
    thread.start()

    return jsonify({
        "status": "success",
        "message": "Hyperparameter optimization started in background.",
        "optimization_status": get_optimization_status(user_id)
    }), 200

@advanced_bp.route('/models/optimization-status', methods=['GET'])
def get_opt_status():
    user_id = get_current_user_id() or 1
    return jsonify({
        "status": "success",
        "optimization_status": get_optimization_status(user_id)
    }), 200

@advanced_bp.route('/realtime/status', methods=['GET'])
def get_realtime_status():
    return jsonify({
        "status": "success",
        "websocket_active": True,
        "mode": "Flask-SocketIO + REST Fallback",
        "channel": "carbon_update"
    }), 200

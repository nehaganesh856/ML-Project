from flask import Blueprint, jsonify
from backend.config import Config
from backend.database.models import db, EnergyRecord, Prediction, Anomaly, Cluster, Recommendation, ModelMetric
from backend.services.data_service import DataService
from sqlalchemy import func
from datetime import datetime
from backend.routes.auth import get_current_user_id

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        # Fetch stats from EnergyRecord
        total_energy_row = db.session.query(func.sum(EnergyRecord.energy_consumption)).filter_by(user_id=user_id).first()
        total_carbon_row = db.session.query(func.sum(EnergyRecord.carbon_emission)).filter_by(user_id=user_id).first()
        
        total_energy = float(total_energy_row[0]) if total_energy_row[0] is not None else 0.0
        total_carbon = float(total_carbon_row[0]) if total_carbon_row[0] is not None else 0.0
        
        latest_record = EnergyRecord.query.filter_by(user_id=user_id).order_by(EnergyRecord.timestamp.desc()).first()
        latest_consumption = latest_record.energy_consumption if latest_record else 0.0
        last_updated = latest_record.timestamp.strftime('%Y-%m-%d %H:%M:%S') if latest_record else None
        
        # Calculate avg daily consumption
        # Find number of unique days
        num_days_row = db.session.query(func.count(func.distinct(func.date(EnergyRecord.timestamp)))).filter_by(user_id=user_id).scalar()
        num_days = int(num_days_row) if num_days_row else 1
        avg_daily = total_energy / num_days if num_days > 0 else 0.0
        
        # Anomalies count
        anomalies_count = Anomaly.query.filter_by(user_id=user_id, is_anomaly=True).count()
        
        # Latest prediction
        latest_pred = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).first()
        pred_energy = latest_pred.predicted_energy if latest_pred else 0.0
        pred_carbon = latest_pred.predicted_carbon if latest_pred else 0.0
        
        # Current cluster
        latest_cluster = Cluster.query.filter_by(user_id=user_id).order_by(Cluster.created_at.desc()).first()
        current_cluster_data = {
            'id': latest_cluster.cluster_id if latest_cluster else -1,
            'label': latest_cluster.cluster_label if latest_cluster else 'Not clustered'
        }
        consumption_category = latest_cluster.cluster_label if latest_cluster else 'Unknown'
        
        # Recommendation
        latest_rec = Recommendation.query.filter_by(user_id=user_id).order_by(Recommendation.created_at.desc()).first()
        recommendation_text = latest_rec.recommendation_text if latest_rec else "Upload electricity consumption data to get recommendations."
        
        # Model performance metrics injection for summary
        model_metrics = {}
        for mname in ['random_forest', 'xgboost', 'linear_regression', 'lstm', 'kmeans', 'isolation_forest']:
            model_metrics[mname] = {}
            metrics = ModelMetric.query.filter_by(user_id=user_id, model_name=mname).all()
            if not metrics:
                metrics = ModelMetric.query.filter_by(user_id=1, model_name=mname).all()
            for m in metrics:
                model_metrics[mname][m.metric_name] = m.metric_value
                model_metrics[mname][m.metric_name.lower()] = m.metric_value
                if m.metric_name == 'Silhouette Score':
                    model_metrics[mname]['silhouette'] = m.metric_value
                    model_metrics[mname]['silhouette_score'] = m.metric_value

        return jsonify({
            'total_energy': total_energy,
            'total_carbon': total_carbon,
            'avg_daily_consumption': avg_daily,
            'latest_consumption': latest_consumption,
            'consumption_category': consumption_category,
            'anomalies_count': anomalies_count,
            'predicted_energy': pred_energy,
            'predicted_carbon': pred_carbon,
            'current_cluster': current_cluster_data,
            'recommendation': recommendation_text,
            'emission_factor': Config.EMISSION_FACTOR,
            'last_updated': last_updated,
            'random_forest': model_metrics.get('random_forest', {}),
            'xgboost': model_metrics.get('xgboost', {}),
            'linear_regression': model_metrics.get('linear_regression', {}),
            'lstm': model_metrics.get('lstm', {}),
            'kmeans': model_metrics.get('kmeans', {}),
            'isolation_forest': model_metrics.get('isolation_forest', {}),
            'model_metrics': model_metrics
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500


@dashboard_bp.route('/dashboard/charts', methods=['GET'])
def get_dashboard_charts():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        # 1. Consumption & Emission timelines (limit to latest 100 hourly records)
        timeline_records = EnergyRecord.query.filter_by(user_id=user_id).order_by(EnergyRecord.timestamp.desc()).limit(100).all()
        # Reverse to chronological order for charts
        timeline_records.reverse()
        
        consumption_timeline = []
        emission_timeline = []
        for r in timeline_records:
            t_str = r.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            consumption_timeline.append({'timestamp': t_str, 'energy': r.energy_consumption})
            emission_timeline.append({'timestamp': t_str, 'carbon': r.carbon_emission})
            
        # 2. Hourly average consumption (0-23)
        hourly_avg = DataService.get_hourly_consumption(user_id)
        
        # 3. Monthly total consumption (1-12)
        monthly_total = DataService.get_monthly_consumption(user_id)
        
        # 4. Cluster Distribution (sizes from database)
        db_clusters = Cluster.query.filter_by(user_id=user_id).all()
        cluster_distribution = {}
        for c in db_clusters:
            cluster_distribution[c.cluster_label] = c.cluster_size
            
        # 5. Model Comparison
        rf_r2 = db.session.query(ModelMetric.metric_value).filter_by(user_id=user_id, model_name='random_forest', metric_name='R2').scalar()
        xgb_r2 = db.session.query(ModelMetric.metric_value).filter_by(user_id=user_id, model_name='xgboost', metric_name='R2').scalar()
        lstm_mae = db.session.query(ModelMetric.metric_value).filter_by(user_id=user_id, model_name='lstm', metric_name='MAE').scalar()
        
        model_comparison = {
            'random_forest': float(rf_r2) if rf_r2 is not None else 0.0,
            'xgboost': float(xgb_r2) if xgb_r2 is not None else 0.0,
            'lstm_mae': float(lstm_mae) if lstm_mae is not None else 0.0
        }
        
        return jsonify({
            'consumption_timeline': consumption_timeline,
            'emission_timeline': emission_timeline,
            'hourly_consumption': hourly_avg,
            'monthly_consumption': monthly_total,
            'cluster_distribution': cluster_distribution,
            'model_comparison': model_comparison
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

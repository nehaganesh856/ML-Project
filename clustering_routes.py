import os
import pickle
import numpy as np
from flask import Blueprint, request, jsonify
from backend.config import Config
from backend.database.models import db, Cluster, ModelMetric, Dataset, TrainedModel
from backend.services.data_service import DataService
from backend.ml.kmeans_model import train_kmeans, get_cluster_statistics, get_cluster_label_mapping, evaluate_kmeans
from backend.routes.auth import get_current_user_id
import json
from datetime import datetime

clustering_bp = Blueprint('clustering', __name__)

def load_kmeans_model(user_id):
    path = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}", 'kmeans.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

@clustering_bp.route('/cluster/run', methods=['POST'])
def run_clustering():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 10:
        return jsonify({'success': False, 'message': 'Insufficient data. At least 10 records are required.'}), 400
        
    data = request.get_json() or {}
    n_clusters = int(data.get('n_clusters', Config.KMEANS_CLUSTERS))
    
    try:
        # Prepare features for clustering:
        # 'energy_consumption' and 'carbon_emission'
        features_list = ['energy_consumption', 'carbon_emission']
        
        # Case insensitive mapping
        df_cols_lower = [c.lower() for c in df.columns]
        actual_features = []
        for feat in features_list:
            if feat.lower() in df_cols_lower:
                idx = df_cols_lower.index(feat.lower())
                actual_features.append(df.columns[idx])
            else:
                actual_features.append(feat)
                
        X = df[actual_features].values
        
        # Load or Train K-Means
        model = load_kmeans_model(user_id)
        if not model or model.n_clusters != n_clusters:
            print(f"Training K-Means with {n_clusters} clusters for user {user_id}...")
            model = train_kmeans(X, n_clusters=n_clusters, user_id=user_id)
            
        # Predict clusters for all records
        preds = model.predict(X)
        
        # Calculate silhouette score
        silhouette = evaluate_kmeans(X, preds)
        
        # Get statistics
        stats = get_cluster_statistics(model, X, preds, actual_features)
        
        # Clear old clusters from database
        Cluster.query.filter_by(user_id=user_id).delete()
        
        # Store new cluster summaries in database
        db_clusters = []
        for cid, info in stats.items():
            cluster_record = Cluster(
                user_id=user_id,
                cluster_id=info['cluster_id'],
                cluster_label=info['cluster_label'],
                centroid_energy=info['centroid_energy'],
                centroid_carbon=info['centroid_carbon'],
                cluster_size=info['cluster_size']
            )
            db_clusters.append(cluster_record)
            db.session.add(cluster_record)
            
        # Update Silhouette Score metric in DB
        metric = ModelMetric.query.filter_by(user_id=user_id, model_name='kmeans', metric_name='Silhouette Score').first()
        if not metric:
            metric = ModelMetric(
                user_id=user_id,
                model_name='kmeans',
                metric_name='Silhouette Score',
                metric_value=silhouette
            )
            db.session.add(metric)
        else:
            metric.metric_value = silhouette
            
        # Save to TrainedModel
        dataset = Dataset.query.filter_by(user_id=user_id).first()
        dataset_id = dataset.id if dataset else None
        
        tm = TrainedModel.query.filter_by(user_id=user_id, model_name='kmeans').first()
        if not tm:
            tm = TrainedModel(user_id=user_id, model_name='kmeans')
            db.session.add(tm)
        tm.dataset_id = dataset_id
        tm.training_timestamp = datetime.utcnow()
        tm.status = 'trained'
        tm.dataset_size = len(df)
        tm.metrics_json = json.dumps({'Silhouette Score': silhouette})
            
        db.session.commit()
        
        # Format response
        clusters_resp = []
        for record in db_clusters:
            clusters_resp.append({
                'id': record.cluster_id,
                'label': record.cluster_label,
                'centroid_energy': record.centroid_energy,
                'centroid_carbon': record.centroid_carbon,
                'size': record.cluster_size
            })
            
        return jsonify({
            'success': True,
            'clusters': clusters_resp,
            'silhouette_score': silhouette
        }), 200
        
    except Exception as e:
        db.session.rollback()
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Clustering execution failed: {str(e)}'}), 500


@clustering_bp.route('/cluster/current', methods=['GET'])
def get_current_cluster():
    """
    Classifies the user's overall/latest average consumption pattern into one of the clusters.
    """
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    model = load_kmeans_model(user_id)
    if not model:
        return jsonify({'success': False, 'message': 'K-Means model not trained yet. Run clustering first.'}), 400
        
    df = DataService.get_user_records_df(user_id)
    if df.empty:
        return jsonify({'success': False, 'message': 'No energy data found.'}), 400
        
    try:
        # Calculate user average energy and carbon
        avg_energy = float(df['energy_consumption'].mean())
        avg_carbon = float(df['carbon_emission'].mean())
        
        features_list = ['energy_consumption', 'carbon_emission']
        user_vector = np.array([[avg_energy, avg_carbon]])
        
        # Predict cluster ID
        cluster_id = int(model.predict(user_vector)[0])
        
        # Retrieve label mapping
        mapping = get_cluster_label_mapping(model, features_list)
        label = mapping.get(cluster_id, f"Cluster {cluster_id}")
        
        # Find stats from DB
        db_cluster = Cluster.query.filter_by(user_id=user_id, cluster_id=cluster_id).first()
        centroid_energy = db_cluster.centroid_energy if db_cluster else avg_energy
        centroid_carbon = db_cluster.centroid_carbon if db_cluster else avg_carbon
        
        return jsonify({
            'cluster_id': cluster_id,
            'cluster_label': label,
            'characteristics': {
                'avg_energy': avg_energy,
                'avg_carbon': avg_carbon,
                'centroid_energy': centroid_energy,
                'centroid_carbon': centroid_carbon
            }
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@clustering_bp.route('/cluster/distribution', methods=['GET'])
def get_cluster_distribution():
    """
    Returns the distribution of user records across clusters.
    """
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    db_clusters = Cluster.query.filter_by(user_id=user_id).all()
    if not db_clusters:
        return jsonify({'cluster_distribution': {}}), 200
        
    dist = {}
    for c in db_clusters:
        dist[c.cluster_label] = c.cluster_size
        
    return jsonify({'cluster_distribution': dist}), 200

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from backend.config import Config

def train_kmeans(X, n_clusters=3, user_id=None):
    """
    Train a K-Means clustering model and save it.
    """
    try:
        # Fit model
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        model.fit(X)
        
        # Save model
        save_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}") if user_id else Config.MODEL_SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'kmeans.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"K-Means model saved successfully to {model_path}")
        return model
    except Exception as e:
        print(f"Error training K-Means: {e}")
        raise e

def get_cluster_label_mapping(model, feature_names):
    """
    Creates a mapping of cluster ID to human-readable label:
    'Low Consumption', 'Moderate Consumption', 'High Consumption'.
    Determined dynamically based on the energy consumption feature in the centroids.
    """
    centroids = model.cluster_centers_
    
    # Try to find the index of the energy consumption column
    energy_idx = 0
    for idx, name in enumerate(feature_names):
        if 'energy' in name.lower() or 'active' in name.lower():
            energy_idx = idx
            break
            
    # Get energy value of each centroid
    energy_values = centroids[:, energy_idx]
    
    # Sort centroids index by energy value
    sorted_cluster_indices = np.argsort(energy_values)
    
    mapping = {}
    if len(sorted_cluster_indices) == 3:
        mapping[sorted_cluster_indices[0]] = "Low Consumption"
        mapping[sorted_cluster_indices[1]] = "Moderate Consumption"
        mapping[sorted_cluster_indices[2]] = "High Consumption"
    else:
        # Fallback for other cluster counts
        for i, idx in enumerate(sorted_cluster_indices):
            if i == 0:
                mapping[idx] = "Low Consumption"
            elif i == len(sorted_cluster_indices) - 1:
                mapping[idx] = "High Consumption"
            else:
                mapping[idx] = f"Moderate Consumption Level {i}"
                
    return mapping

def predict_clusters(model, X_input, feature_names):
    """
    Predict the cluster for a given input and return the cluster ID and its label.
    """
    if len(X_input.shape) == 1:
        X_input = X_input.reshape(1, -1)
    
    cluster_ids = model.predict(X_input)
    mapping = get_cluster_label_mapping(model, feature_names)
    
    labels = [mapping.get(cid, "Unknown") for cid in cluster_ids]
    return cluster_ids, labels

def get_cluster_statistics(model, X, y_clusters, feature_names):
    """
    Calculate statistical averages for each cluster.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['cluster'] = y_clusters
    
    mapping = get_cluster_label_mapping(model, feature_names)
    
    # Try to find energy and carbon columns to return their true centroids
    energy_col = next((c for c in feature_names if 'energy' in c.lower() or 'active' in c.lower()), feature_names[0])
    carbon_col = next((c for c in feature_names if 'carbon' in c.lower() or 'emission' in c.lower()), None)
    
    stats = {}
    for cid in range(model.n_clusters):
        cluster_df = df[df['cluster'] == cid]
        size = len(cluster_df)
        
        avg_energy = float(cluster_df[energy_col].mean()) if size > 0 else 0.0
        avg_carbon = float(cluster_df[carbon_col].mean()) if (carbon_col and size > 0) else (avg_energy * Config.EMISSION_FACTOR)
        
        stats[cid] = {
            'cluster_id': int(cid),
            'cluster_label': mapping.get(cid, f"Cluster {cid}"),
            'centroid_energy': avg_energy,
            'centroid_carbon': avg_carbon,
            'cluster_size': int(size)
        }
        
    return stats

def evaluate_kmeans(X, clusters):
    """
    Evaluate K-Means using Silhouette Score.
    Limits data size for silhouette score to 10k rows to avoid freeze/memory crash.
    """
    if len(X) < 2:
        return 0.0
        
    # Sample data if too large
    if len(X) > 10000:
        indices = np.random.choice(len(X), size=10000, replace=False)
        X_sampled = X[indices]
        clusters_sampled = clusters[indices]
    else:
        X_sampled = X
        clusters_sampled = clusters
        
    try:
        score = silhouette_score(X_sampled, clusters_sampled)
        return float(score)
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return 0.0

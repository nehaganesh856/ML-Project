from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

# We will initialize db here or import it from database.py.
# To avoid circular imports, we can declare the db object here and import it in database.py and app.py.
db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    records = db.relationship('EnergyRecord', backref='user', lazy=True, cascade="all, delete-orphan")
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade="all, delete-orphan")
    clusters = db.relationship('Cluster', backref='user', lazy=True, cascade="all, delete-orphan")
    anomalies = db.relationship('Anomaly', backref='user', lazy=True, cascade="all, delete-orphan")
    recommendations = db.relationship('Recommendation', backref='user', lazy=True, cascade="all, delete-orphan")
    metrics = db.relationship('ModelMetric', backref='user', lazy=True, cascade="all, delete-orphan")
    dataset = db.relationship('Dataset', backref='user', uselist=False, lazy=True, cascade="all, delete-orphan")
    trained_models = db.relationship('TrainedModel', backref='user', lazy=True, cascade="all, delete-orphan")


class EnergyRecord(db.Model):
    __tablename__ = 'energy_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    global_active_power = db.Column(db.Float)
    global_reactive_power = db.Column(db.Float)
    voltage = db.Column(db.Float)
    global_intensity = db.Column(db.Float)
    sub_metering_1 = db.Column(db.Float)
    sub_metering_2 = db.Column(db.Float)
    sub_metering_3 = db.Column(db.Float)
    energy_consumption = db.Column(db.Float)  # in kWh (computed)
    carbon_emission = db.Column(db.Float)     # in kg CO2 (computed)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=True)  # For forecast timestamps
    model_name = db.Column(db.String(50), nullable=False)  # 'random_forest', 'xgboost', 'lstm'
    predicted_energy = db.Column(db.Float, nullable=False)
    predicted_carbon = db.Column(db.Float, nullable=False)
    actual_energy = db.Column(db.Float, nullable=True)
    mae = db.Column(db.Float, nullable=True)
    rmse = db.Column(db.Float, nullable=True)
    r2 = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Cluster(db.Model):
    __tablename__ = 'clusters'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    cluster_id = db.Column(db.Integer, nullable=False)
    cluster_label = db.Column(db.String(50), nullable=False)  # 'Low Consumption', etc.
    centroid_energy = db.Column(db.Float, nullable=False)
    centroid_carbon = db.Column(db.Float, nullable=False)
    cluster_size = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Anomaly(db.Model):
    __tablename__ = 'anomalies'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    energy_consumption = db.Column(db.Float, nullable=False)
    anomaly_score = db.Column(db.Float, nullable=False)
    is_anomaly = db.Column(db.Boolean, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Recommendation(db.Model):
    __tablename__ = 'recommendations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    recommendation_text = db.Column(db.String(500), nullable=False)
    recommendation_type = db.Column(db.String(50), nullable=False)  # 'consumption', 'carbon', 'anomaly', 'general'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ModelMetric(db.Model):
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    metric_name = db.Column(db.String(50), nullable=False)  # 'MAE', 'RMSE', 'R2', 'Silhouette Score'
    metric_value = db.Column(db.Float, nullable=False)
    training_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Dataset(db.Model):
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, unique=True)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    record_count = db.Column(db.Integer, nullable=True)
    processed_record_count = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(50), default='uploaded')


class TrainedModel(db.Model):
    __tablename__ = 'trained_models'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    model_name = db.Column(db.String(50), nullable=False)  # 'random_forest', 'xgboost', 'lstm', 'kmeans', 'isolation_forest'
    model_version = db.Column(db.String(20), default='1.0')
    training_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='idle')  # 'idle', 'running', 'trained', 'failed'
    dataset_size = db.Column(db.Integer, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)  # JSON string of metrics

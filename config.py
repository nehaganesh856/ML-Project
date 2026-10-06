import os
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load environment variables from root or backend directory if they exist
load_dotenv(os.path.join(BASE_DIR, '.env'))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), '.env'))

class Config:
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    # Flask Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", 3600))
    
    # Database Settings
    # SQLite file will be placed in the root directory under database/
    DATABASE_PATH = os.path.join(ROOT_DIR, "database")
    db_file = os.path.join(DATABASE_PATH, "app.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.abspath(db_file)}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File Upload Settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"csv"}

    # ML Paths
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "trained_models")
    DATA_PATH = os.path.join(BASE_DIR, "data")
    UPLOAD_DATA_FOLDER = os.path.join(BASE_DIR, "uploads", "data")

    # ML Parameters
    EMISSION_FACTOR = float(os.getenv("EMISSION_FACTOR", 0.5))  # kg CO2 per kWh
    LSTM_SEQUENCE_LENGTH = int(os.getenv("LSTM_SEQUENCE_LENGTH", 24))  # hours
    KMEANS_CLUSTERS = int(os.getenv("KMEANS_CLUSTERS", 3))
    ISOLATION_FOREST_CONTAMINATION = float(os.getenv("ISOLATION_FOREST_CONTAMINATION", 0.05))

    # Model Hyperparameters
    RF_N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", 100))
    RF_MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", 15))
    XGB_N_ESTIMATORS = int(os.getenv("XGB_N_ESTIMATORS", 100))
    XGB_LEARNING_RATE = float(os.getenv("XGB_LEARNING_RATE", 0.1))
    LSTM_EPOCHS = int(os.getenv("LSTM_EPOCHS", 50))
    LSTM_BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", 32))

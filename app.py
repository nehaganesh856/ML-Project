import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from backend.config import Config
from backend.database.models import db
from backend.database.database import init_db

# Import blueprints
from backend.routes.auth_routes import auth_bp
from backend.routes.data_routes import data_bp
from backend.routes.prediction_routes import prediction_bp
from backend.routes.clustering_routes import clustering_bp
from backend.routes.anomaly_routes import anomaly_bp
from backend.routes.dashboard_routes import dashboard_bp
from backend.routes.recommendation_routes import recommendation_bp
from backend.routes.config_routes import config_bp
from backend.routes.performance_routes import performance_bp
from backend.routes.advanced_routes import advanced_bp

try:
    from flask_socketio import SocketIO
    socketio = SocketIO(cors_allowed_origins="*")
except ImportError:
    socketio = None

def create_app():
    # Configure Flask to serve static files from the frontend/ folder
    app = Flask(__name__, 
                static_folder=os.path.join(Config.ROOT_DIR, 'frontend'),
                static_url_path='')
                
    # Load settings
    app.config.from_object(Config)
    
    # Enable CORS for frontend requests (vital for local testing)
    CORS(app, supports_credentials=True)
    
    if socketio:
        try:
            socketio.init_app(app)
        except Exception as e:
            print(f"[SOCKETIO] Failed to init socketio: {e}")

    # Initialize DB & Seed Demo user
    init_db(app)
    
    # Register API Blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(data_bp, url_prefix='/api/data')
    app.register_blueprint(prediction_bp, url_prefix='/api')
    app.register_blueprint(clustering_bp, url_prefix='/api')
    app.register_blueprint(anomaly_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp, url_prefix='/api')
    app.register_blueprint(recommendation_bp, url_prefix='/api')
    app.register_blueprint(config_bp, url_prefix='/api')
    app.register_blueprint(performance_bp, url_prefix='/api')
    app.register_blueprint(advanced_bp, url_prefix='/api')
    
    # Catch-all route to serve the React SPA index.html for any client-side route
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def catch_all(path):
        # Prevent intercepting API routes
        if path.startswith('api/'):
            return jsonify({'success': False, 'message': 'Endpoint not found.'}), 404
        # Serve static assets directly if they exist
        static_file_path = os.path.join(app.static_folder, path)
        if path and os.path.exists(static_file_path):
            return app.send_static_file(path)
        # Fallback to index.html for React Router
        return app.send_static_file('index.html')
        
    @app.errorhandler(500)
    def internal_server_error(e):
        return jsonify({'success': False, 'message': 'Internal server error occurred.'}), 500
        
    @app.errorhandler(413) # Payload Too Large
    def file_too_large(e):
        return jsonify({'success': False, 'message': 'File too large. Maximum size is 50MB.'}), 413
        
    return app

app = create_app()

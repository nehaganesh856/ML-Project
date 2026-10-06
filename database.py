import os
from werkzeug.security import generate_password_hash
from backend.database.models import db, User

def init_db(app):
    """
    Initialize the database, create tables, and seed default user.
    """
    # Ensure database folder exists
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI')
    if db_uri and db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
                print(f"Created database directory: {db_dir}")
            except Exception as e:
                print(f"Error creating database directory {db_dir}: {e}")
                
    try:
        db.init_app(app)
        with app.app_context():
            db.create_all()
            print("Database tables checked/created successfully.")
            
            # Check if default demo user exists, if not create one
            demo_user = User.query.filter_by(username='demo').first()
            if not demo_user:
                hashed_pw = generate_password_hash('demo123')
                new_user = User(
                    username='demo',
                    email='demo@example.com',
                    password_hash=hashed_pw
                )
                db.session.add(new_user)
                db.session.commit()
                demo_user = new_user
                print("Default demo user ('demo':'demo123') seeded successfully.")

            # Automatically verify and ensure all ML models are trained and loadable on startup
            from backend.services.auto_training_service import ensure_models_trained
            ensure_models_trained(demo_user.id)
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise e

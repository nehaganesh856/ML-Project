from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from backend.database.models import db, User
from backend.routes.auth import generate_token, get_current_user_id

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'Missing required fields: username, email, password.'}), 400
        
    # Check if user already exists with specific error messages
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already taken.'}), 400
        
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered.'}), 400
        
    try:
        hashed_pw = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_pw
        )
        db.session.add(new_user)
        db.session.commit()
        
        # Log user in automatically and generate token
        token = generate_token(new_user.id)
        session['user_id'] = new_user.id
        session.permanent = True
        
        return jsonify({
            'success': True,
            'message': 'Registration successful.',
            'token': token,
            'user_id': new_user.id,
            'username': new_user.username
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing fields: username and password.'}), 400
        
    user = User.query.filter((User.username == username) | (User.email == username)).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401
        
    token = generate_token(user.id)
    session['user_id'] = user.id
    session.permanent = True
    
    return jsonify({
        'success': True,
        'message': 'Login successful.',
        'token': token,
        'user_id': user.id,
        'username': user.username
    }), 200


@auth_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logged out successfully.'}), 200


@auth_bp.route('/session', methods=['GET'])
def get_session():
    """
    Checks if a user is currently logged in and returns session details.
    """
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'logged_in': False}), 200
        
    user = User.query.get(user_id)
    if not user:
        session.pop('user_id', None)
        return jsonify({'logged_in': False}), 200
        
    return jsonify({
        'logged_in': True,
        'user_id': user.id,
        'username': user.username,
        'email': user.email
    }), 200


@auth_bp.route('/me', methods=['GET'])
def get_current_user():
    """
    Validates token and returns current user details.
    """
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Invalid or expired token.'}), 401
        
    user = User.query.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': 'User not found.'}), 404
        
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email
        }
    }), 200


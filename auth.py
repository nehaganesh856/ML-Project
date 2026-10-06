import os
from flask import request
from itsdangerous import URLSafeTimedSerializer
from backend.config import Config

def generate_token(user_id):
    serializer = URLSafeTimedSerializer(Config.SECRET_KEY)
    return serializer.dumps({'user_id': user_id}, salt='auth-salt')

def get_current_user_id():
    # 1. Check Authorization header
    auth_header = request.headers.get('Authorization')
    token = None
    if auth_header:
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        else:
            token = auth_header
            
    # 2. Check X-Auth-Token header
    if not token:
        token = request.headers.get('X-Auth-Token')
        
    # 3. Check query param/body as fallback
    if not token:
        token = request.args.get('token')
        
    # 4. Check flask session as ultimate fallback
    if not token:
        from flask import session
        return session.get('user_id')
        
    try:
        serializer = URLSafeTimedSerializer(Config.SECRET_KEY)
        data = serializer.loads(token, salt='auth-salt', max_age=Config.SESSION_TIMEOUT)
        return data.get('user_id')
    except Exception as e:
        print(f"Token validation failed: {e}")
        return None

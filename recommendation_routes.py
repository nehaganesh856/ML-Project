from flask import Blueprint, jsonify
from backend.database.models import db, Recommendation
from backend.services.recommendation_engine import RecommendationEngine
from datetime import datetime
from backend.routes.auth import get_current_user_id

recommendation_bp = Blueprint('recommendation', __name__)

@recommendation_bp.route('/recommendations', methods=['GET'])
def get_recommendations():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized. Please log in.'}), 401
        
    try:
        recs = Recommendation.query.filter_by(user_id=user_id).order_by(Recommendation.created_at.desc()).all()
        
        # If no recommendations exist, generate them automatically
        if not recs:
            generated = RecommendationEngine.generate_recommendations(user_id)
            return jsonify({
                'success': True,
                'recommendations': [{
                    'text': r['recommendation_text'],
                    'type': r['recommendation_type'],
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                } for r in generated]
            }), 200
            
        recs_list = []
        for r in recs:
            recs_list.append({
                'text': r.recommendation_text,
                'type': r.recommendation_type,
                'timestamp': r.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        return jsonify({
            'success': True,
            'recommendations': recs_list
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to retrieve recommendations: {str(e)}'}), 500


@recommendation_bp.route('/recommendations/generate', methods=['POST'])
def generate_recommendations():
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({'success': False, 'message': 'Unauthorized.'}), 401
        
    try:
        generated = RecommendationEngine.generate_recommendations(user_id)
        
        recs_list = []
        for r in generated:
            recs_list.append({
                'text': r['recommendation_text'],
                'type': r['recommendation_type']
            })
            
        return jsonify({
            'success': True,
            'new_recommendations': recs_list
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': f'Failed to generate recommendations: {str(e)}'}), 500

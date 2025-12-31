"""
Intelligent Chatbot Backend API
Flask server with NLP, RAG, and MongoDB storage
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import models
from models.nlp_trainer import NLPChatbot
from models.rag_engine import RAGEngine

# Import utilities
from utils.conversation_storage import ConversationStorage
from utils.preprocessor import TextPreprocessor
from utils.response_generator import ResponseGenerator
from utils.similarity import SimilarityCalculator

# Initialize Flask app
app = Flask(__name__)

# CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False
    }
})

# Initialize components
print("\n" + "="*60)
print("Initializing Chatbot Components...")
print("="*60 + "\n")

nlp_chatbot = None
rag_engine = None
preprocessor = TextPreprocessor()
response_generator = ResponseGenerator()
similarity_calculator = SimilarityCalculator()

# Conversation storage (MongoDB + JSON fallback)
conversation_storage = ConversationStorage(use_mongodb=True)

# Conversation history for this session
conversation_history = []


def initialize_models():
    """Initialize or load trained models"""
    global nlp_chatbot, rag_engine
    
    try:
        # Initialize NLP Chatbot
        print("Loading NLP Chatbot...")
        nlp_chatbot = NLPChatbot()
        print("✓ NLP Chatbot initialized\n")
        
        # Initialize RAG Engine
        print("Loading RAG Engine...")
        rag_engine = RAGEngine()
        print("✓ RAG Engine initialized\n")
        
        # Check if models are trained
        nlp_stats = nlp_chatbot.get_stats()
        rag_stats = rag_engine.get_stats()
        
        print("Model Status:")
        print("-"*60)
        print(f"NLP Model:")
        print(f"  - Patterns loaded: {nlp_stats['num_patterns']}")
        print(f"  - Model trained: {nlp_stats['model_loaded']}")
        print(f"  - spaCy available: {nlp_stats['spacy_available']}")
        
        print(f"\nRAG Model:")
        print(f"  - Conversations indexed: {rag_stats['num_conversations']}")
        print(f"  - Embeddings loaded: {rag_stats['embeddings_loaded']}")
        print(f"  - Intelligent generation: {rag_stats.get('intelligent_generation', False)}")
        print(f"  - Model: {rag_stats['model_name']}")
        
        print(f"\nStorage:")
        storage_stats = conversation_storage.get_statistics()
        print(f"  - Type: {storage_stats.get('storage_type', 'Unknown')}")
        print(f"  - Connected: {storage_stats.get('mongodb_connected', False)}")
        print(f"  - Total conversations: {storage_stats.get('total_conversations', 0)}")
        print(f"  - Total users: {storage_stats.get('total_users', 0)}")
        print("-"*60 + "\n")
        
        if nlp_stats['num_patterns'] == 0 or rag_stats['num_conversations'] == 0:
            print("⚠️  Warning: Models not trained. Please run train_model.py first.")
        else:
            print("✅ All models loaded successfully!\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing models: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# Initialize models on startup
initialize_models()


@app.after_request
def after_request(response):
    """Add CORS headers to every response"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


@app.route('/', methods=['GET'])
@app.route('/health-check', methods=['GET'])
def home():
    """Health check endpoint"""
    rag_stats = rag_engine.get_stats() if rag_engine else {}
    storage_stats = conversation_storage.get_statistics()
    
    return jsonify({
        'status': 'running',
        'message': 'Chatbot API is online',
        'version': '2.0.0',
        'intelligent_generation': rag_stats.get('intelligent_generation', False),
        'storage_type': storage_stats.get('storage_type', 'JSON'),
        'mongodb_connected': storage_stats.get('mongodb_connected', False),
        'endpoints': {
            '/': 'GET - Health check',
            '/chat': 'POST - Send message to chatbot',
            '/train': 'POST - Train models with new data',
            '/stats': 'GET - Get model statistics',
            '/history': 'GET - Get conversation history',
            '/clear': 'POST - Clear conversation history',
            '/profile/<user_id>': 'GET - Get user profile',
            '/export-conversations': 'POST - Export stored conversations',
            '/retrain-from-storage': 'POST - Retrain models from storage',
            '/storage-stats': 'GET - Get storage statistics',
            '/backup': 'POST - Create backup of conversations',
            '/search': 'POST - Search similar conversations',
            '/similar': 'POST - Check text similarity',
            '/intent': 'POST - Detect message intent'
        }
    }), 200


@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Enhanced chat endpoint with intelligent response generation"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Log incoming request
        print(f"\n📨 Incoming chat request from {request.remote_addr}")
        
        data = request.json
        user_message = data.get('message', '').strip()
        approach = data.get('approach', 'hybrid')
        user_id = data.get('user_id', 'anonymous')
        
        print(f"   User: {user_id}")
        print(f"   Message: {user_message[:50]}...")
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get user profile for context
        user_context = conversation_storage.get_user_context(user_id)
        user_profile = conversation_storage.get_user_profile(user_id)
        
        # Detect intent
        intent = response_generator.detect_intent(user_message)
        print(f"   Intent: {intent}")
        
        # Preprocess user input
        processed_input = preprocessor.preprocess_for_inference(user_message)
        
        # Get responses based on approach
        response_text = ""
        source = ""
        confidence = 0.0
        
        if approach == 'nlp':
            # Get NLP response
            nlp_result = nlp_chatbot.get_response(processed_input)
            
            if isinstance(nlp_result, tuple) and len(nlp_result) == 2:
                response_text, nlp_metadata = nlp_result
                confidence = nlp_metadata.get('confidence', 0.5)
            else:
                response_text = nlp_result
                confidence = 0.5
            
            source = 'nlp'
            
        elif approach == 'rag':
            # RAG response with intelligent generation
            response_text, similarity = rag_engine.get_similar_response(
                processed_input, 
                top_k=3,
                intent=intent
            )
            confidence = similarity
            source = 'rag'
            
        else:  # hybrid approach (default and recommended)
            # Get both responses
            nlp_result = nlp_chatbot.get_response(processed_input)
            
            if isinstance(nlp_result, tuple) and len(nlp_result) == 2:
                nlp_response, nlp_metadata = nlp_result
                nlp_conf = nlp_metadata.get('confidence', 0.5)
            else:
                nlp_response = nlp_result
                nlp_conf = 0.5
            
            # RAG response
            rag_response, rag_sim = rag_engine.get_similar_response(
                processed_input,
                top_k=3,
                intent=intent
            )
            
            # Select best response
            response_text, source = response_generator.select_best_response(
                nlp_response=nlp_response,
                nlp_confidence=nlp_conf,
                rag_response=rag_response,
                rag_similarity=rag_sim,
                user_input=user_message,
                user_name=user_profile.get('name') if user_profile else None
            )
            
            confidence = nlp_conf if source.startswith('nlp') else rag_sim
        
        # Format response with personality
        formatted_response = response_generator.format_response(
            response_text,
            add_personality=True,
            intent=intent
        )
        
        # Make response context-aware
        formatted_response = response_generator.get_context_aware_response(
            formatted_response,
            user_message
        )
        
        # Add emoji based on intent
        formatted_response = response_generator.enhance_response_with_emojis(
            formatted_response,
            intent
        )
        
        # Save conversation with metadata
        conversation_storage.save_conversation(
            user_id=user_id,
            user_message=user_message,
            bot_response=formatted_response,
            metadata={
                'source': source,
                'confidence': float(confidence),
                'approach': approach,
                'intent': intent,
                'context_used': user_context
            }
        )
        
        # Add to session history
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user_message,
            'bot': formatted_response,
            'source': source,
            'confidence': float(confidence),
            'approach': approach,
            'intent': intent
        }
        conversation_history.append(conversation_entry)
        
        # Update context
        response_generator.add_context(user_message, formatted_response)
        
        # Generate follow-up suggestions
        suggestions = response_generator.generate_follow_up_suggestions(
            formatted_response,
            user_message
        )
        
        # Extract URL reference if present
        url_match = re.search(r'https?://[^\s]+', formatted_response)
        reference_url = url_match.group(0) if url_match else None
        
        # Build response
        result = {
            'response': formatted_response,
            'source': source,
            'confidence': float(confidence),
            'intent': intent,
            'approach_used': approach,
            'suggestions': suggestions,
            'reference_url': reference_url,
            'timestamp': conversation_entry['timestamp'],
            'user_profile': {
                'name': user_profile.get('name') if user_profile else None,
                'conversation_count': user_profile.get('conversation_count', 0) if user_profile else 0,
                'interests': user_profile.get('interests', []) if user_profile else []
            }
        }
        
        print(f"   ✅ Response: {formatted_response[:50]}...")
        print(f"   Source: {source}, Confidence: {confidence:.2f}, Intent: {intent}\n")
        
        # Return JSON response
        response_obj = make_response(jsonify(result))
        response_obj.headers['Content-Type'] = 'application/json'
        return response_obj, 200
        
    except Exception as e:
        print(f"❌ Error in /chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'An error occurred processing your message',
            'details': str(e)
        }), 500


@app.route('/profile/<user_id>', methods=['GET'])
def get_profile(user_id):
    """Get user profile"""
    try:
        profile = conversation_storage.get_user_profile(user_id)
        
        if not profile:
            return jsonify({
                'user_id': user_id,
                'name': None,
                'conversation_count': 0,
                'interests': [],
                'recent_conversations': []
            }), 200
        
        recent_convs = conversation_storage.get_recent_conversations(user_id, limit=5)
        
        return jsonify({
            'profile': profile,
            'recent_conversations': recent_convs
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export-conversations', methods=['POST'])
def export_conversations():
    """Export stored conversations for training"""
    try:
        data = request.json or {}
        output_file = data.get('output_file', 'data/exported_conversations.json')
        min_quality = data.get('min_quality', 0)
        
        count = conversation_storage.export_for_training(output_file, min_quality)
        
        return jsonify({
            'status': 'success',
            'conversations_exported': count,
            'output_file': output_file
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/backup', methods=['POST'])
def backup_conversations():
    """Create backup of all conversations"""
    try:
        backup_dir = request.json.get('backup_dir', 'data/backups') if request.json else 'data/backups'
        
        backup_files = conversation_storage.backup(backup_dir)
        
        return jsonify({
            'status': 'success',
            'message': 'Backup created successfully',
            'backup_files': backup_files
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/retrain-from-storage', methods=['POST'])
def retrain_from_storage():
    """Retrain models using stored conversations"""
    try:
        export_file = 'data/exported_conversations.json'
        count = conversation_storage.export_for_training(export_file)
        
        if count < 10:
            return jsonify({
                'error': 'Not enough conversations to retrain',
                'count': count,
                'minimum_required': 10
            }), 400
        
        with open(export_file, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
        
        training_file = 'data/chat_history.json'
        if os.path.exists(training_file):
            with open(training_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        all_data = existing_data + new_data
        seen = set()
        unique_data = []
        for conv in all_data:
            key = conv['user'].lower().strip()
            if key not in seen:
                seen.add(key)
                unique_data.append(conv)
        
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n🔄 Retraining models with {len(unique_data)} conversations...")
        
        nlp_chatbot.train(unique_data)
        rag_engine.index_conversations(unique_data)
        
        print("✅ Retraining complete!\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Models retrained successfully',
            'new_conversations': count,
            'total_training_data': len(unique_data),
            'previous_count': len(existing_data)
        }), 200
        
    except Exception as e:
        print(f"Error in /retrain-from-storage: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/storage-stats', methods=['GET'])
def storage_stats():
    """Get conversation storage statistics"""
    try:
        stats = conversation_storage.get_statistics()
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    """Train models with new data"""
    try:
        data = request.json
        chat_data = data.get('chat_data', [])
        
        if not chat_data:
            data_file = 'data/chat_history.json'
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)
            else:
                return jsonify({'error': 'No training data provided or found'}), 400
        
        print(f"\n🔄 Training models with {len(chat_data)} conversations...")
        
        nlp_chatbot.train(chat_data)
        rag_engine.index_conversations(chat_data)
        
        nlp_stats = nlp_chatbot.get_stats()
        rag_stats = rag_engine.get_stats()
        
        print("✅ Training complete!\n")
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'nlp_patterns': nlp_stats['num_patterns'],
            'rag_conversations': rag_stats['num_conversations'],
            'intelligent_generation': rag_stats.get('intelligent_generation', False)
        }), 200
        
    except Exception as e:
        print(f"Error in /train: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Training failed', 'details': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    try:
        nlp_stats = nlp_chatbot.get_stats()
        rag_stats = rag_engine.get_stats()
        generator_stats = response_generator.get_stats()
        storage_stats_data = conversation_storage.get_statistics()
        
        return jsonify({
            'nlp_model': nlp_stats,
            'rag_model': rag_stats,
            'response_generator': generator_stats,
            'conversations_this_session': len(conversation_history),
            'storage': storage_stats_data,
            'intelligent_generation_enabled': rag_stats.get('intelligent_generation', False)
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get statistics', 'details': str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        limit = request.args.get('limit', default=50, type=int)
        user_id = request.args.get('user_id', default=None, type=str)
        
        if user_id:
            user_history = conversation_storage.get_recent_conversations(user_id, limit=limit)
            return jsonify({
                'history': user_history,
                'total': len(user_history),
                'user_id': user_id
            }), 200
        else:
            return jsonify({
                'history': conversation_history[-limit:],
                'total': len(conversation_history)
            }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to get history', 'details': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    try:
        global conversation_history
        conversation_history = []
        response_generator.context_history = []
        
        return jsonify({
            'status': 'success',
            'message': 'Conversation history cleared'
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to clear history', 'details': str(e)}), 500


@app.route('/search', methods=['POST'])
def search_similar():
    """Search for similar conversations"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 5)
        threshold = data.get('threshold', 0.5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        matches = rag_engine.get_top_matches(query, top_k=top_k)
        filtered_matches = [m for m in matches if m['similarity'] >= threshold]
        
        return jsonify({
            'query': query,
            'matches': filtered_matches,
            'count': len(filtered_matches)
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Search failed', 'details': str(e)}), 500


@app.route('/similar', methods=['POST'])
def check_similarity():
    """Check similarity between two texts"""
    try:
        data = request.json
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        method = data.get('method', 'cosine')
        
        if not text1 or not text2:
            return jsonify({'error': 'Both text1 and text2 are required'}), 400
        
        if method == 'combined':
            score = similarity_calculator.combined_similarity(text1, text2)
        elif method == 'jaccard':
            score = similarity_calculator.jaccard_similarity(text1, text2)
        elif method == 'dice':
            score = similarity_calculator.dice_coefficient(text1, text2)
        elif method == 'levenshtein':
            score = similarity_calculator.levenshtein_similarity(text1, text2)
        else:
            score = similarity_calculator.cosine_similarity(text1, text2)
        
        return jsonify({
            'text1': text1,
            'text2': text2,
            'method': method,
            'similarity': float(score)
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Similarity calculation failed', 'details': str(e)}), 500


@app.route('/intent', methods=['POST'])
def detect_intent():
    """Detect intent from user message"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        intent = response_generator.detect_intent(message)
        
        return jsonify({
            'message': message,
            'intent': intent
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Intent detection failed', 'details': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


@app.before_request
def log_request():
    """Log incoming requests"""
    if request.path not in ['/health-check', '/']:
        print(f"📥 Request: {request.method} {request.path}")


@app.after_request
def log_response(response):
    """Log outgoing responses"""
    if request.path not in ['/health-check', '/']:
        print(f"📤 Response: {response.status_code}")
    return response


# Run the app
if __name__ == '__main__':
    import os
    
    # CRITICAL: Get port from environment (Render sets this automatically)
    port = int(os.environ.get('PORT', 10000))
    
    print("\n" + "="*60)
    print("🤖 Starting Intelligent Chatbot Server...")
    print("="*60)
    print(f"\n🌐 Port: {port}")
    print(f"🌐 Binding to: 0.0.0.0:{port}")
    print("✨ Architecture:")
    print("  • ResponseGenerator: Intent detection, formatting, suggestions")
    print("  • IntelligentResponseGenerator: Pattern learning inside RAG")
    print("  • MongoDB + JSON: Dual storage for persistence")
    print("\n📡 Ready to chat!")
    print("="*60 + "\n")
    
    # Run Flask app - MUST bind to 0.0.0.0 and PORT
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )

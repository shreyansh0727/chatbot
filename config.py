import os
from pathlib import Path
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # ===== Application Settings =====
    APP_NAME = "Personal Chatbot"
    VERSION = "1.0.0"
    DEBUG = True
    TESTING = False
    
    # ===== Server Settings =====
    HOST = '0.0.0.0'
    PORT = 5000
    
    # ===== Paths =====
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / 'data'
    MODELS_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Training data
    CHAT_HISTORY_FILE = DATA_DIR / 'chat_history.json'
    WHATSAPP_PARSED_FILE = DATA_DIR / 'whatsapp_parsed.json'
    TELEGRAM_PARSED_FILE = DATA_DIR / 'telegram_parsed.json'
    
    # Model files
    NLP_MODEL_FILE = MODELS_DIR / 'chat_model.pkl'
    RAG_EMBEDDINGS_DIR = MODELS_DIR / 'vectorstore'
    
    # ===== Model Settings =====
    
    # NLP Model
    NLP_CONFIDENCE_THRESHOLD = 0.5
    NLP_TFIDF_MAX_FEATURES = 1000
    NLP_NGRAM_RANGE = (1, 2)  # unigrams and bigrams
    NLP_MIN_DF = 1
    
    # spaCy Model
    SPACY_MODEL = 'en_core_web_md'  # Options: en_core_web_sm, en_core_web_md, en_core_web_lg
    SPACY_FALLBACK_MODEL = 'en_core_web_sm'
    
    # RAG Model
    RAG_MODEL_NAME = 'all-MiniLM-L6-v2'  # Sentence transformer model
    # Alternative models:
    # - 'all-mpnet-base-v2' (higher quality, slower)
    # - 'paraphrase-MiniLM-L6-v2' (paraphrase detection)
    # - 'multi-qa-MiniLM-L6-cos-v1' (question answering)
    
    RAG_SIMILARITY_THRESHOLD = 0.5
    RAG_TOP_K = 3  # Number of similar conversations to retrieve
    RAG_BATCH_SIZE = 32
    
    # ===== Response Generation Settings =====
    HYBRID_NLP_WEIGHT = 0.5
    HYBRID_RAG_WEIGHT = 0.5
    
    RESPONSE_MIN_CONFIDENCE = 0.3
    RESPONSE_ADD_PERSONALITY = True
    RESPONSE_ADD_EMOJIS = True
    RESPONSE_EMOJI_PROBABILITY = 0.3
    
    # Context settings
    MAX_CONTEXT_HISTORY = 10
    USE_CONTEXT_AWARE = True
    
    # ===== Preprocessing Settings =====
    REMOVE_STOPWORDS_TRAINING = False
    REMOVE_EMOJIS_TRAINING = False
    EXPAND_CONTRACTIONS = True
    NORMALIZE_UNICODE = True
    
    # ===== Similarity Settings =====
    SIMILARITY_METHOD = 'cosine'  # Options: jaccard, cosine, dice, levenshtein, combined
    
    # Weights for combined similarity
    SIMILARITY_WEIGHTS = {
        'jaccard': 0.25,
        'cosine': 0.30,
        'dice': 0.20,
        'levenshtein': 0.15,
        'ngram': 0.10
    }
    
    # ===== Training Settings =====
    AUTO_TRAIN_ON_STARTUP = False
    SAVE_TRAINING_LOGS = True
    
    MIN_TRAINING_SAMPLES = 10
    TRAIN_TEST_SPLIT = 0.8
    
    # ===== API Settings =====
    ENABLE_CORS = True
    CORS_ORIGINS = '*'  # In production, specify exact origins
    
    MAX_MESSAGE_LENGTH = 1000
    MAX_HISTORY_LIMIT = 100
    
    # Rate limiting (requests per minute)
    RATE_LIMIT_ENABLED = False
    RATE_LIMIT = 60
    
    # ===== Session Settings =====
    SESSION_TIMEOUT = timedelta(hours=24)
    SAVE_CONVERSATIONS = True
    CONVERSATIONS_FILE = DATA_DIR / 'conversations_log.json'
    
    # ===== Logging Settings =====
    LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = LOGS_DIR / 'chatbot.log'
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    
    # ===== Feature Flags =====
    ENABLE_INTENT_DETECTION = True
    ENABLE_FOLLOW_UP_SUGGESTIONS = True
    ENABLE_SEMANTIC_SEARCH = True
    ENABLE_SIMILARITY_ENDPOINT = True
    
    # ===== Performance Settings =====
    USE_GPU = True  # Use GPU for sentence transformers if available
    NUM_WORKERS = 4
    CACHE_EMBEDDINGS = True
    
    # ===== User Settings (Customize these!) =====
    YOUR_NAME = "Shreyansh"
    BOT_NAME = "Shreyansh Bot"
    BOT_DESCRIPTION = "A personal chatbot trained on your conversation style"
    
    # Personality settings
    PERSONALITY_TRAITS = {
        'casual': True,
        'technical': True,
        'friendly': True,
        'professional': False
    }
    
    # ===== Fallback Responses =====
    ENABLE_CUSTOM_FALLBACKS = True
    CUSTOM_FALLBACK_RESPONSES = {
        'greeting': [
            "Hey! What's up?",
            "Hi there! How's it going?",
            "Hello! How can I help?"
        ],
        'goodbye': [
            "See you later! Take care!",
            "Bye! Have a great day!",
            "Later! Feel free to reach out anytime."
        ],
        'thanks': [
            "No problem! Happy to help.",
            "Anytime! Glad I could help.",
            "You're welcome!"
        ]
    }
    
    # ===== Export Settings =====
    EXPORT_FORMAT = 'json'  # json, csv, txt
    EXPORT_DIR = BASE_DIR / 'exports'
    
    @classmethod
    def init_app(cls):
        """Initialize application directories"""
        # Create necessary directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.RAG_EMBEDDINGS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.EXPORT_DIR.mkdir(exist_ok=True)
        
        print(f"✓ Initialized directories:")
        print(f"  - Data: {cls.DATA_DIR}")
        print(f"  - Models: {cls.MODELS_DIR}")
        print(f"  - Logs: {cls.LOGS_DIR}")
    
    @classmethod
    def get_config_summary(cls):
        """Get configuration summary"""
        return {
            'app': {
                'name': cls.APP_NAME,
                'version': cls.VERSION,
                'debug': cls.DEBUG
            },
            'models': {
                'nlp_threshold': cls.NLP_CONFIDENCE_THRESHOLD,
                'rag_model': cls.RAG_MODEL_NAME,
                'rag_threshold': cls.RAG_SIMILARITY_THRESHOLD,
                'spacy_model': cls.SPACY_MODEL
            },
            'features': {
                'intent_detection': cls.ENABLE_INTENT_DETECTION,
                'follow_up_suggestions': cls.ENABLE_FOLLOW_UP_SUGGESTIONS,
                'semantic_search': cls.ENABLE_SEMANTIC_SEARCH
            },
            'performance': {
                'use_gpu': cls.USE_GPU,
                'cache_embeddings': cls.CACHE_EMBEDDINGS,
                'batch_size': cls.RAG_BATCH_SIZE
            }
        }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'
    
    # More verbose logging in development
    SAVE_TRAINING_LOGS = True
    
    # Lower thresholds for testing
    NLP_CONFIDENCE_THRESHOLD = 0.3
    RAG_SIMILARITY_THRESHOLD = 0.3


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'
    
    # Stricter settings for production
    NLP_CONFIDENCE_THRESHOLD = 0.6
    RAG_SIMILARITY_THRESHOLD = 0.6
    
    # Security
    CORS_ORIGINS = ['http://yourdomain.com', 'https://yourdomain.com']
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT = 30  # 30 requests per minute
    
    # Performance
    USE_GPU = True
    CACHE_EMBEDDINGS = True


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    
    # Use small models for faster testing
    SPACY_MODEL = 'en_core_web_sm'
    RAG_MODEL_NAME = 'all-MiniLM-L6-v2'
    
    # Lower thresholds
    MIN_TRAINING_SAMPLES = 5
    
    # Disable some features for testing
    ENABLE_FOLLOW_UP_SUGGESTIONS = False
    RESPONSE_ADD_EMOJIS = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name='default'):
    """
    Get configuration by name
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configuration class
    """
    return config.get(config_name, config['default'])


# Environment-based configuration
def get_active_config():
    """Get configuration based on environment variable"""
    env = os.getenv('CHATBOT_ENV', 'development')
    return get_config(env)


# Test the config
if __name__ == "__main__":
    print("="*60)
    print("Configuration Test")
    print("="*60 + "\n")
    
    # Initialize directories
    Config.init_app()
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print("-"*60)
    
    summary = Config.get_config_summary()
    
    for category, settings in summary.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    # Show file paths
    print("\n" + "="*60)
    print("File Paths:")
    print("-"*60)
    print(f"Chat History: {Config.CHAT_HISTORY_FILE}")
    print(f"NLP Model: {Config.NLP_MODEL_FILE}")
    print(f"RAG Embeddings: {Config.RAG_EMBEDDINGS_DIR}")
    print(f"Logs: {Config.LOG_FILE}")
    
    # Test different environments
    print("\n" + "="*60)
    print("Environment Configurations:")
    print("-"*60)
    
    for env_name in ['development', 'production', 'testing']:
        env_config = get_config(env_name)
        print(f"\n{env_name.upper()}:")
        print(f"  Debug: {env_config.DEBUG}")
        print(f"  Log Level: {env_config.LOG_LEVEL}")
        print(f"  NLP Threshold: {env_config.NLP_CONFIDENCE_THRESHOLD}")
        print(f"  RAG Threshold: {env_config.RAG_SIMILARITY_THRESHOLD}")

import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class NLPChatbot:
    def __init__(self, model_path='models/chat_model.pkl'):
        """Initialize NLP Chatbot with enhanced capabilities"""
        self.model_path = model_path
        self.nlp = None
        self.spacy_available = False
        self.training_data = []
        self.responses = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        self.confidence_score = 0.0
        self.tfidf_matrix = None
        self.patterns = []
        self.pattern_to_response = {}
        
        # Try to load spaCy model (optional but recommended)
        self._load_spacy()
        
        # Try to load existing model
        self.load_model()
    
    def _load_spacy(self):
        """Try to load spaCy model with fallback options"""
        try:
            import spacy
            
            # Try transformer model (best accuracy but slower)
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
                print("✓ spaCy model loaded: en_core_web_sm")
                return
            except:
                pass
            
            print("⚠️  No spaCy model found. Using basic processing.")
            print("   Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            self.spacy_available = False
            
        except ImportError:
            print("⚠️  spaCy not installed. Using basic processing.")
            print("   Install with: pip install spacy")
            self.nlp = None
            self.spacy_available = False
    
    def preprocess_text(self, text):
        """
        Enhanced text preprocessing with spaCy or fallback
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text string
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        if self.nlp and self.spacy_available:
            # Advanced preprocessing with spaCy
            try:
                doc = self.nlp(text.lower())
                
                # Lemmatize and remove stopwords, punctuation
                tokens = [
                    token.lemma_ for token in doc 
                    if not token.is_stop 
                    and not token.is_punct 
                    and not token.is_space
                    and len(token.text) > 1
                ]
                
                return ' '.join(tokens)
            except Exception as e:
                print(f"⚠️  spaCy preprocessing error: {e}")
                # Fallback to basic preprocessing
                return self._basic_preprocess(text)
        else:
            # Basic preprocessing without spaCy
            return self._basic_preprocess(text)
    
    def _basic_preprocess(self, text):
        """Basic preprocessing without spaCy"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple stopword removal
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        words = text.split()
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        return ' '.join(filtered_words)
    
    def train(self, chat_data):
        """
        Train on personal chat history with enhanced processing
        
        Args:
            chat_data: List of conversation dictionaries
        """
        print("\n" + "="*60)
        print("Training NLP Model...")
        print("="*60)
        
        if not chat_data:
            print("❌ No training data provided")
            return
        
        patterns = []
        responses = []
        pattern_to_response = {}
        
        print(f"Processing {len(chat_data)} conversations...")
        
        for i, conversation in enumerate(chat_data):
            if i % 100 == 0 and i > 0:
                print(f"  Progress: {i}/{len(chat_data)}")
            
            # Handle both 'bot' and 'assistant' keys
            user_msg = conversation.get('user', '')
            bot_msg = conversation.get('bot') or conversation.get('assistant', '')
            
            if user_msg and bot_msg:
                # Preprocess user message
                processed_msg = self.preprocess_text(user_msg)
                
                # Skip if preprocessing resulted in empty string
                if not processed_msg or len(processed_msg) < 3:
                    continue
                
                patterns.append(processed_msg)
                responses.append(bot_msg)
                
                # Store mapping (handle duplicates)
                if processed_msg in pattern_to_response:
                    # If duplicate, keep the longer response
                    if len(bot_msg) > len(pattern_to_response[processed_msg]):
                        pattern_to_response[processed_msg] = bot_msg
                else:
                    pattern_to_response[processed_msg] = bot_msg
        
        print(f"\n✓ Processed {len(patterns)} valid conversation pairs")
        print(f"✓ Unique patterns: {len(pattern_to_response)}")
        
        if len(patterns) < 5:
            print("❌ Not enough valid training data (minimum 5 required)")
            return
        
        # Create TF-IDF vectors
        print("Creating TF-IDF vectors...")
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(patterns)
            self.patterns = patterns
            self.pattern_to_response = pattern_to_response
            print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        except Exception as e:
            print(f"❌ Error creating TF-IDF vectors: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Save model
        self.save_model()
        print("✓ NLP model training complete!\n")
    
    def get_response(self, user_input):
        """
        Find most similar pattern and return response with confidence
        
        Args:
            user_input: User's message
            
        Returns:
            Tuple of (response, metadata) or just response string
        """
        if not self.patterns or self.tfidf_matrix is None:
            return "Model not trained yet. Please train the model first.", {
                'confidence': 0.0,
                'source': 'error'
            }
        
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        if not processed_input or len(processed_input) < 2:
            return "I didn't quite understand that. Could you rephrase?", {
                'confidence': 0.0,
                'source': 'fallback'
            }
        
        try:
            # Transform user input to TF-IDF vector
            user_vec = self.vectorizer.transform([processed_input])
            
            # Calculate cosine similarity with all training patterns
            similarities = cosine_similarity(user_vec, self.tfidf_matrix)[0]
            
            # Get top 3 matches
            top_indices = np.argsort(similarities)[-3:][::-1]
            top_similarities = similarities[top_indices]
            
            # Get best match
            best_match_idx = top_indices[0]
            self.confidence_score = float(top_similarities[0])
            
            # Adaptive threshold based on query complexity
            threshold = 0.25 if len(processed_input.split()) > 3 else 0.3
            
            # Return response if confidence is above threshold
            if self.confidence_score >= threshold:
                matched_pattern = self.patterns[best_match_idx]
                response = self.pattern_to_response.get(
                    matched_pattern,
                    "I'm not sure how to respond to that."
                )
                
                return response, {
                    'confidence': self.confidence_score,
                    'source': 'nlp',
                    'matched_pattern': matched_pattern,
                    'top_matches': len([s for s in top_similarities if s > 0.2])
                }
            else:
                # Smart fallback based on confidence
                if self.confidence_score > 0.15:
                    # Low confidence but some match
                    fallback_responses = [
                        "I'm not entirely sure, but let me try to help.",
                        "That's interesting. Could you tell me more?",
                        "Hmm, I'm not quite sure about that one.",
                        "Can you rephrase that? I want to make sure I understand."
                    ]
                else:
                    # Very low confidence
                    fallback_responses = [
                        "I'm not sure how to respond to that. Can you rephrase?",
                        "Could you say that differently? I didn't quite get it.",
                        "Not sure I understand. Can you elaborate?",
                        "That's a new one for me. Can you explain more?"
                    ]
                
                return np.random.choice(fallback_responses), {
                    'confidence': self.confidence_score,
                    'source': 'fallback',
                    'reason': 'low_confidence'
                }
        
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Sorry, I encountered an error processing that.", {
                'confidence': 0.0,
                'source': 'error',
                'error': str(e)
            }
    
    def get_confidence(self):
        """Return confidence score of last prediction"""
        return self.confidence_score
    
    def save_model(self):
        """Save trained model to disk with error handling"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            model_data = {
                'vectorizer': self.vectorizer,
                'patterns': self.patterns,
                'pattern_to_response': self.pattern_to_response,
                'tfidf_matrix': self.tfidf_matrix,
                'spacy_available': self.spacy_available
            }
            
            # Use protocol 4 for better compatibility
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=4)
            
            # Verify save was successful
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path)
                print(f"✓ Model saved to: {self.model_path}")
                print(f"  File size: {file_size / 1024:.2f} KB")
            else:
                print(f"⚠️  Model file not found after save attempt")
                
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def load_model(self):
        """Load trained model from disk with error handling"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.vectorizer = model_data['vectorizer']
                self.patterns = model_data['patterns']
                
                # Handle both old and new format
                if 'pattern_to_response' in model_data:
                    self.pattern_to_response = model_data['pattern_to_response']
                elif 'responses' in model_data:
                    # Legacy format
                    self.pattern_to_response = model_data['responses']
                
                self.tfidf_matrix = model_data['tfidf_matrix']
                
                print(f"✓ Loaded existing model from: {self.model_path}")
                print(f"  - {len(self.patterns)} patterns loaded")
                print(f"  - {len(self.pattern_to_response)} unique responses")
                
                return True
                
            except Exception as e:
                print(f"⚠️  Could not load model: {str(e)}")
                print(f"  Model will be retrained on next training call")
                
                # Delete corrupted file
                try:
                    os.remove(self.model_path)
                    print(f"  Removed corrupted model file")
                except:
                    pass
                
                return False
        
        print("ℹ️  No existing model found. Will train on first training call.")
        return False
    
    def get_stats(self):
        """Get detailed model statistics"""
        return {
            'num_patterns': len(self.patterns),
            'num_responses': len(self.pattern_to_response),
            'model_loaded': self.tfidf_matrix is not None,
            'spacy_available': self.spacy_available,
            'spacy_model': self.nlp.meta.get('name') if self.nlp else None,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0,
            'tfidf_features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'model_file_exists': os.path.exists(self.model_path)
        }
    
    def get_similar_patterns(self, user_input, top_k=5):
        """
        Get top-k similar patterns for debugging/analysis
        
        Args:
            user_input: User's message
            top_k: Number of similar patterns to return
            
        Returns:
            List of (pattern, similarity, response) tuples
        """
        if not self.patterns or self.tfidf_matrix is None:
            return []
        
        processed_input = self.preprocess_text(user_input)
        
        try:
            user_vec = self.vectorizer.transform([processed_input])
            similarities = cosine_similarity(user_vec, self.tfidf_matrix)[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                pattern = self.patterns[idx]
                similarity = similarities[idx]
                response = self.pattern_to_response.get(pattern, "")
                results.append((pattern, float(similarity), response))
            
            return results
            
        except Exception as e:
            print(f"Error in get_similar_patterns: {e}")
            return []


# Test the model
if __name__ == "__main__":
    print("Testing Enhanced NLP Chatbot Model\n")
    
    # Initialize
    chatbot = NLPChatbot()
    
    # Load training data
    data_file = '../data/chat_history.json'
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Train
        chatbot.train(chat_data)
        
        # Test with sample queries
        test_queries = [
            "Hey, what are you working on?",
            "How's the project going?",
            "What tech stack do you use?",
            "Tell me about yourself",
            "What do you know about Firebase?",
            "Can you help me with React?"
        ]
        
        print("\n" + "="*60)
        print("Testing Model with Sample Queries")
        print("="*60)
        
        for query in test_queries:
            response, metadata = chatbot.get_response(query)
            print(f"\nUser: {query}")
            print(f"Bot:  {response[:100]}...")
            print(f"Confidence: {metadata['confidence']:.3f}")
            print(f"Source: {metadata['source']}")
        
        # Show model stats
        print("\n" + "="*60)
        print("Model Statistics")
        print("="*60)
        stats = chatbot.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All tests completed!")
    else:
        print(f"❌ Training data not found at: {data_file}")

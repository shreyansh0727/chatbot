from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import os
import pickle
import torch


class RAGEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', embeddings_path='models/vectorstore'):
        """
        Initialize RAG Engine for Retrieval-Augmented Generation
        
        Args:
            model_name: Sentence transformer model to use
            embeddings_path: Path to save/load embeddings
        """
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.conversations = []
        self.embeddings = None
        self.model = None
        self.intelligent_generator = None
        
        # Create embeddings directory
        os.makedirs(embeddings_path, exist_ok=True)
        
        # Load sentence transformer model
        print(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
        
        # Initialize intelligent generator
        try:
            from .response_generator_ai import IntelligentResponseGenerator
            self.intelligent_generator = IntelligentResponseGenerator()
            print("✓ Intelligent response generator initialized")
        except ImportError:
            print("⚠️  Warning: Could not load IntelligentResponseGenerator")
            print("   Bot will use direct matching instead of intelligent generation")
            self.intelligent_generator = None
        
        # Try to load existing embeddings
        self.load_embeddings()
    
    def index_conversations(self, chat_data):
        """
        Create vector embeddings from chat history AND learn patterns
        
        Args:
            chat_data: List of conversation dicts with 'user' and 'bot'/'assistant' keys
        """
        print("\n" + "="*60)
        print("Building RAG Vector Store...")
        print("="*60)
        
        if not chat_data:
            print("❌ No chat data provided")
            return
        
        # Normalize conversation format (handle both 'bot' and 'assistant' keys)
        normalized_convs = []
        for conv in chat_data:
            normalized_convs.append({
                'user': conv.get('user', ''),
                'bot': conv.get('bot') or conv.get('assistant', ''),
                'metadata': conv.get('metadata', {})
            })
        
        self.conversations = normalized_convs
        
        # Extract all user messages for embedding
        user_messages = [conv['user'] for conv in self.conversations]
        
        print(f"Encoding {len(user_messages)} messages...")
        
        try:
            # Create embeddings (this may take a minute)
            self.embeddings = self.model.encode(
                user_messages, 
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=32
            )
            
            print(f"✓ Created embeddings with shape: {self.embeddings.shape}")
            
            # Save embeddings
            self.save_embeddings()
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            raise
        
        print("✓ RAG indexing complete!")
        
        # NEW: Learn patterns from data for intelligent generation
        if self.intelligent_generator:
            print("\n" + "="*60)
            print("🧠 Teaching AI to Understand Patterns...")
            print("="*60)
            try:
                self.intelligent_generator.learn_from_data(normalized_convs)
                print("✓ AI learning complete!\n")
            except Exception as e:
                print(f"⚠️  Warning: AI learning failed: {e}")
                print("   Continuing with direct matching mode\n")
        else:
            print("\n⚠️  Intelligent generation disabled - using direct matching\n")
    
    def get_response(self, user_input, top_k=3):
        """
        Retrieve most similar conversation and return response
        
        Args:
            user_input: User's message
            top_k: Number of similar conversations to consider
            
        Returns:
            Bot response from the most similar conversation
        """
        if len(self.conversations) == 0 or self.embeddings is None:
            return "No training data available. Please train the model first."
        
        try:
            # Encode user input
            query_embedding = self.model.encode(
                user_input, 
                convert_to_tensor=True
            )
            
            # Calculate cosine similarity with all stored conversations
            similarities = util.cos_sim(query_embedding, self.embeddings)[0]
            
            # Get top-k matches
            top_results = torch.topk(similarities, k=min(top_k, len(similarities)))
            
            # Get best match
            best_match_idx = top_results.indices[0].item()
            similarity_score = top_results.values[0].item()
            
            # Return response if similarity is above threshold
            if similarity_score > 0.5:  # Threshold for relevance
                return self.conversations[best_match_idx]['bot']
            else:
                # If no good match, try to return something relevant
                if similarity_score > 0.3:
                    return self.conversations[best_match_idx]['bot']
                else:
                    # Fallback responses
                    fallback_responses = [
                        "I'm not quite sure about that.",
                        "Hmm, that's interesting. Tell me more?",
                        "I don't have a good answer for that right now.",
                        "Not sure I can help with that one."
                    ]
                    return np.random.choice(fallback_responses)
        
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "Sorry, I encountered an error processing that."
    
    def get_top_matches(self, user_input, top_k=5):
        """
        Get top-k most similar conversations with scores
        
        Args:
            user_input: User's message
            top_k: Number of results to return
            
        Returns:
            List of dicts with conversation data and similarity scores
        """
        if len(self.conversations) == 0 or self.embeddings is None:
            return []
        
        try:
            # Encode user input
            query_embedding = self.model.encode(
                user_input, 
                convert_to_tensor=True
            )
            
            # Calculate similarities
            similarities = util.cos_sim(query_embedding, self.embeddings)[0]
            
            # Get top-k results
            top_results = torch.topk(similarities, k=min(top_k, len(similarities)))
            
            # Prepare results
            matches = []
            for idx, score in zip(top_results.indices, top_results.values):
                idx = idx.item()
                score = score.item()
                conv = self.conversations[idx]
                
                matches.append({
                    'user': conv['user'],
                    'response': conv['bot'],  # Use 'response' for consistency
                    'bot': conv['bot'],  # Keep 'bot' for backward compatibility
                    'assistant': conv['bot'],  # Keep 'assistant' for backward compatibility
                    'similarity': score,
                    'metadata': conv.get('metadata', {})
                })
            
            return matches
        
        except Exception as e:
            print(f"Error in get_top_matches: {str(e)}")
            return []
    
    def get_similar_response(self, query: str, top_k: int = 3, 
                            threshold: float = 0.5, intent: str = None):
        """
        Get intelligent response with URL reference support
        Now uses AI generation instead of just retrieval
        
        Args:
            query: User query
            top_k: Number of top matches to consider
            threshold: Minimum similarity threshold
            intent: Optional detected intent for better generation
            
        Returns:
            Tuple of (response_text, similarity_score)
        """
        if not self.conversations or self.embeddings is None:
            return "I don't have enough training data yet.", 0.0
        
        try:
            # Get similar matches from vector store
            matches = self.get_top_matches(query, top_k=top_k)
            
            if not matches:
                return "I'm not sure about that. Can you ask differently?", 0.0
            
            best_match = matches[0]
            best_similarity = best_match['similarity']
            
            # If very high similarity (>0.92) and it's a simple/direct question,
            # use the exact match
            query_words = len(query.split())
            if best_similarity >= 0.92 and query_words <= 5:
                response = best_match['response']
                metadata = best_match.get('metadata', {})
                url = metadata.get('url')
                
                if url and url not in response:
                    response = f"{response}\n\n🔗 Learn more: {url}"
                
                return response, best_similarity
            
            # Use intelligent generation if available
            if self.intelligent_generator and best_similarity >= 0.4:
                try:
                    generated_response, gen_metadata = self.intelligent_generator.generate_response(
                        query=query,
                        rag_matches=matches,
                        intent=intent
                    )
                    return generated_response, best_similarity
                except Exception as e:
                    print(f"⚠️  Intelligent generation failed: {e}")
                    # Fallback to direct match
                    pass
            
            # Fallback: Use direct match if similarity is good enough
            if best_similarity >= threshold:
                response = best_match['response']
                metadata = best_match.get('metadata', {})
                url = metadata.get('url')
                
                if url and url not in response:
                    response = f"{response}\n\n🔗 Learn more: {url}"
                
                return response, best_similarity
            
            # Below threshold
            if best_similarity > 0.3:
                return best_match['response'], best_similarity
            
            return "I'm not sure about that. Can you ask differently?", 0.0
            
        except Exception as e:
            print(f"Error in get_similar_response: {e}")
            import traceback
            traceback.print_exc()
            return "Sorry, I encountered an error.", 0.0
    
    def save_embeddings(self):
        """Save embeddings and conversations to disk"""
        try:
            embeddings_file = os.path.join(self.embeddings_path, 'embeddings.pkl')
            
            # Convert tensor to numpy for storage
            embeddings_np = self.embeddings.cpu().numpy()
            
            save_data = {
                'embeddings': embeddings_np,
                'conversations': self.conversations,
                'model_name': self.model_name
            }
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"✓ Embeddings saved to: {embeddings_file}")
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {str(e)}")
    
    def load_embeddings(self):
        """Load embeddings and conversations from disk"""
        embeddings_file = os.path.join(self.embeddings_path, 'embeddings.pkl')
        
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    save_data = pickle.load(f)
                
                # Convert numpy back to tensor
                embeddings_np = save_data['embeddings']
                self.embeddings = torch.tensor(embeddings_np)
                
                # Normalize old conversation format
                raw_conversations = save_data['conversations']
                self.conversations = []
                
                for conv in raw_conversations:
                    normalized_conv = {
                        'user': conv.get('user', ''),
                        'bot': conv.get('bot') or conv.get('assistant', ''),
                        'metadata': conv.get('metadata', {})
                    }
                    self.conversations.append(normalized_conv)
                
                # Verify model name matches
                if save_data['model_name'] != self.model_name:
                    print(f"⚠ Warning: Saved model ({save_data['model_name']}) differs from current ({self.model_name})")
                
                print(f"✓ Loaded existing embeddings from: {embeddings_file}")
                print(f"  - {len(self.conversations)} conversations")
                print(f"  - Embedding shape: {self.embeddings.shape}")
                
                return True
            
            except Exception as e:
                print(f"⚠ Error loading embeddings: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        return False
    
    def get_stats(self):
        """Get RAG engine statistics"""
        return {
            'num_conversations': len(self.conversations),
            'embeddings_loaded': self.embeddings is not None,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': self.model_name,
            'intelligent_generation': self.intelligent_generator is not None
        }
    
    def semantic_search(self, query, threshold=0.5):
        """
        Perform semantic search on conversations
        
        Args:
            query: Search query
            threshold: Minimum similarity score
            
        Returns:
            List of matching conversations above threshold
        """
        matches = self.get_top_matches(query, top_k=10)
        return [m for m in matches if m['similarity'] >= threshold]


# Test the RAG engine
if __name__ == "__main__":
    print("Testing RAG Engine\n")
    
    # Initialize
    rag = RAGEngine()
    
    # Load training data
    data_file = '../data/chat_history.json'
    if os.path.exists(data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # Index conversations
        rag.index_conversations(chat_data)
        
        # Test with sample queries
        test_queries = [
            "Hey, what are you working on?",
            "How's the project going?",
            "What tech stack do you use?",
            "Tell me about Firebase",
            "What projects do you have?",
            "How do I setup authentication?"
        ]
        
        print("\n" + "="*60)
        print("Testing RAG with Sample Queries")
        print("="*60)
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"User: {query}")
            print("-"*60)
            
            # Get response with intelligent generation
            response, similarity = rag.get_similar_response(query)
            print(f"Bot Response:\n{response}")
            print(f"\nSimilarity: {similarity:.3f}")
            
            # Get top matches with scores
            matches = rag.get_top_matches(query, top_k=3)
            print(f"\nTop 3 Similar Conversations:")
            for i, match in enumerate(matches, 1):
                print(f"  [{i}] Similarity: {match['similarity']:.3f}")
                print(f"      Q: {match['user'][:50]}...")
                print(f"      A: {match['response'][:50]}...")
        
        # Print stats
        print("\n" + "="*60)
        print("RAG Engine Statistics")
        print("="*60)
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✅ All tests completed!")
    
    else:
        print(f"❌ Training data not found at: {data_file}")

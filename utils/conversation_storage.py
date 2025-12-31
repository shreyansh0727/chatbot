"""
Conversation Storage with MongoDB + JSON Fallback
Automatically uses MongoDB if available, falls back to JSON if not
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
from dotenv import load_dotenv

load_dotenv()

class ConversationStorage:
    """
    Store and manage conversation history with user profiles
    Supports both MongoDB (production) and JSON (local/fallback)
    """
    
    def __init__(self, storage_file='data/conversation_logs.json', 
                 profiles_file='data/user_profiles.json',
                 use_mongodb=True):
        self.storage_file = storage_file
        self.profiles_file = profiles_file
        self.use_mongodb = use_mongodb
        
        # MongoDB connection
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_conversations = None
        self.mongo_users = None
        
        # Fallback storage
        self.conversations = []
        self.user_profiles = {}
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Try MongoDB first
        if self.use_mongodb:
            self._init_mongodb()
        
        # Load JSON as fallback
        if self.mongo_client is None:
            print("ℹ️  Using JSON file storage")
            self.load_conversations()
            self.load_profiles()
    
    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            from pymongo import MongoClient
            
            # Get MongoDB URI from environment
            mongo_uri = os.environ.get(
                'MONGODB_URI',
                'mongodb://localhost:27017/'  # Local fallback
            )
            
            self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.mongo_client.admin.command('ping')
            
            # Set up database and collections
            self.mongo_db = self.mongo_client['chatbot_db']
            self.mongo_conversations = self.mongo_db['conversations']
            self.mongo_users = self.mongo_db['users']
            
            # Create indexes
            self.mongo_conversations.create_index([('user_id', 1), ('timestamp', -1)])
            self.mongo_conversations.create_index('timestamp')
            self.mongo_users.create_index('user_id', unique=True)
            
            print("✅ Connected to MongoDB successfully")
            print(f"   Database: chatbot_db")
            print(f"   Collections: conversations, users")
            
        except ImportError:
            print("⚠️  pymongo not installed. Using JSON storage.")
            print("   Install with: pip install pymongo dnspython")
            self.mongo_client = None
        except Exception as e:
            print(f"⚠️  MongoDB connection failed: {e}")
            print("   Falling back to JSON file storage")
            self.mongo_client = None
    
    def load_conversations(self):
        """Load stored conversations from JSON"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.conversations = json.load(f)
                print(f"✓ Loaded {len(self.conversations)} stored conversations from JSON")
            except Exception as e:
                print(f"⚠️  Could not load conversations: {str(e)}")
                self.conversations = []
        else:
            print("ℹ️  No existing conversation logs found, starting fresh")
            self.conversations = []
    
    def load_profiles(self):
        """Load user profiles from JSON"""
        if os.path.exists(self.profiles_file):
            try:
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    self.user_profiles = json.load(f)
                print(f"✓ Loaded {len(self.user_profiles)} user profiles from JSON")
            except Exception as e:
                print(f"⚠️  Could not load profiles: {str(e)}")
                self.user_profiles = {}
        else:
            print("ℹ️  No existing user profiles found, starting fresh")
            self.user_profiles = {}
    
    def save_conversation(self, user_id: str, user_message: str, 
                         bot_response: str, metadata: Dict = None):
        """Save a conversation exchange (MongoDB or JSON)"""
        conversation = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'user': user_message,
            'assistant': bot_response,
            'metadata': metadata or {}
        }
        
        # Save to MongoDB if available
        if self.mongo_conversations is not None:
            try:
                self.mongo_conversations.insert_one(conversation.copy())
            except Exception as e:
                print(f"⚠️  MongoDB save failed: {e}")
        
        # Also save to JSON (backup)
        self.conversations.append(conversation)
        
        # Extract and update user info
        self.extract_user_info(user_id, user_message, bot_response)
        
        # Save to disk periodically
        if len(self.conversations) % 10 == 0:  # Every 10 conversations
            self.save_to_disk()
        
        return conversation
    
    def extract_user_info(self, user_id: str, user_message: str, bot_response: str):
        """Extract user information from conversation"""
        
        # MongoDB path
        if self.mongo_users is not None:
            self._extract_user_info_mongo(user_id, user_message, bot_response)
        else:
            # JSON path
            self._extract_user_info_json(user_id, user_message, bot_response)
    
    def _extract_user_info_mongo(self, user_id: str, user_message: str, bot_response: str):
        """Extract and save user info to MongoDB"""
        try:
            # Check if user exists
            existing_user = self.mongo_users.find_one({'user_id': user_id})
            
            # Extract information
            extracted = self._extract_info_from_text(user_message, bot_response)
            
            if existing_user:
                # Update existing user
                update_ops = {
                    '$inc': {'conversation_count': 1},
                    '$set': {
                        'last_seen': datetime.utcnow()
                    }
                }
                
                # Update name if extracted
                if extracted['name']:
                    update_ops['$set']['name'] = extracted['name']
                
                # Add new interests
                if extracted['interests']:
                    update_ops['$addToSet'] = {
                        'interests': {'$each': extracted['interests']}
                    }
                
                # Add new topics
                if extracted['topics']:
                    if '$addToSet' not in update_ops:
                        update_ops['$addToSet'] = {}
                    update_ops['$addToSet']['topics_discussed'] = {'$each': extracted['topics']}
                
                # Add preferences
                for category, items in extracted['preferences'].items():
                    if items:
                        if '$addToSet' not in update_ops:
                            update_ops['$addToSet'] = {}
                        update_ops['$addToSet'][f'preferences.{category}'] = {'$each': items}
                
                self.mongo_users.update_one({'user_id': user_id}, update_ops)
                
            else:
                # Create new user
                new_user = {
                    'user_id': user_id,
                    'name': extracted['name'],
                    'preferences': extracted['preferences'],
                    'interests': extracted['interests'],
                    'context': {},
                    'conversation_count': 1,
                    'first_seen': datetime.utcnow(),
                    'last_seen': datetime.utcnow(),
                    'topics_discussed': extracted['topics']
                }
                self.mongo_users.insert_one(new_user)
                
                if extracted['name']:
                    print(f"   💡 Learned user name: {extracted['name']}")
                    
        except Exception as e:
            print(f"⚠️  Error updating MongoDB user: {e}")
    
    def _extract_user_info_json(self, user_id: str, user_message: str, bot_response: str):
        """Extract and save user info to JSON"""
        # Initialize profile if doesn't exist
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'name': None,
                'preferences': {},
                'interests': [],
                'context': {},
                'conversation_count': 0,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'topics_discussed': []
            }
        
        profile = self.user_profiles[user_id]
        profile['conversation_count'] += 1
        profile['last_seen'] = datetime.now().isoformat()
        
        # Extract information
        extracted = self._extract_info_from_text(user_message, bot_response)
        
        # Update profile
        if extracted['name'] and not profile['name']:
            profile['name'] = extracted['name']
            print(f"   💡 Learned user name: {extracted['name']}")
        
        # Add interests
        for interest in extracted['interests']:
            if interest not in profile['interests']:
                profile['interests'].append(interest)
        
        # Add topics
        for topic in extracted['topics']:
            if topic not in profile['topics_discussed']:
                profile['topics_discussed'].append(topic)
        
        # Add preferences
        for category, items in extracted['preferences'].items():
            if category not in profile['preferences']:
                profile['preferences'][category] = []
            for item in items:
                if item not in profile['preferences'][category]:
                    profile['preferences'][category].append(item)
    
    def _extract_info_from_text(self, user_message: str, bot_response: str) -> Dict:
        """Extract information from text"""
        result = {
            'name': None,
            'interests': [],
            'topics': [],
            'preferences': {}
        }
        
        message_lower = user_message.lower()
        
        # Extract name
        name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
            r"name's (\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(1).capitalize()
                if self._is_valid_name(name):
                    result['name'] = name
                    break
        
        # Extract preferences
        preference_patterns = [
            (r"i (like|love|prefer|enjoy) ([\w\s]+?)(?:\.|,|!|\?|$)", 'likes'),
            (r"i (hate|dislike|don't like) ([\w\s]+?)(?:\.|,|!|\?|$)", 'dislikes'),
            (r"i'm (interested in|into) ([\w\s]+?)(?:\.|,|!|\?|$)", 'interests'),
            (r"i (work with|use|develop|build) ([\w\s]+?)(?:\.|,|!|\?|$)", 'tools'),
            (r"i'm (a|an) ([\w\s]+?)(?:\.|,|!|\?|$)", 'profession')
        ]
        
        for pattern, category in preference_patterns:
            matches = re.finditer(pattern, message_lower)
            for match in matches:
                item = match.group(2).strip()
                item = re.sub(r'\s+', ' ', item)
                if 2 < len(item) < 50:
                    if category not in result['preferences']:
                        result['preferences'][category] = []
                    result['preferences'][category].append(item)
        
        # Extract tech interests
        tech_keywords = [
            'python', 'javascript', 'react', 'firebase', 'arduino', 
            'ai', 'ml', 'machine learning', 'deep learning', 'iot', 
            'web development', 'mobile', 'backend', 'frontend', 'fullstack',
            'node', 'express', 'fastapi', 'flask', 'django',
            'nextjs', 'react native', 'flutter', 'kotlin', 'swift',
            'database', 'mongodb', 'postgresql', 'mysql', 'redis',
            'docker', 'kubernetes', 'aws', 'gcp', 'azure',
            'devops', 'ci/cd', 'git', 'github', 'api'
        ]
        
        for keyword in tech_keywords:
            if keyword in message_lower:
                result['interests'].append(keyword)
        
        # Extract topics
        topic_keywords = {
            'coding': ['code', 'programming', 'develop', 'build', 'create'],
            'help': ['help', 'issue', 'problem', 'error', 'fix'],
            'learning': ['learn', 'tutorial', 'how to', 'teach', 'explain'],
            'project': ['project', 'app', 'application', 'system', 'platform'],
            'greeting': ['hi', 'hello', 'hey', 'good morning', 'good evening']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                result['topics'].append(topic)
        
        return result
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate extracted name"""
        if not name or len(name) < 2 or len(name) > 20:
            return False
        if not name.isalpha():
            return False
        
        # Avoid common false positives
        false_positives = [
            'about', 'from', 'here', 'there', 'where', 'what', 'when',
            'good', 'bad', 'sure', 'fine', 'okay', 'yeah', 'yes', 'no'
        ]
        return name.lower() not in false_positives
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile (MongoDB or JSON)"""
        if self.mongo_users is not None:
            try:
                profile = self.mongo_users.find_one({'user_id': user_id})
                if profile:
                    profile['_id'] = str(profile['_id'])  # Convert ObjectId to string
                    if 'first_seen' in profile and isinstance(profile['first_seen'], datetime):
                        profile['first_seen'] = profile['first_seen'].isoformat()
                    if 'last_seen' in profile and isinstance(profile['last_seen'], datetime):
                        profile['last_seen'] = profile['last_seen'].isoformat()
                return profile
            except Exception as e:
                print(f"⚠️  Error fetching MongoDB profile: {e}")
        
        return self.user_profiles.get(user_id)
    
    def update_user_profile(self, user_id: str, updates: Dict):
        """Manually update user profile"""
        if self.mongo_users is not None:
            try:
                self.mongo_users.update_one(
                    {'user_id': user_id},
                    {'$set': updates},
                    upsert=True
                )
            except Exception as e:
                print(f"⚠️  MongoDB update failed: {e}")
        
        # Also update JSON
        if user_id not in self.user_profiles:
            self.extract_user_info(user_id, "", "")
        
        self.user_profiles[user_id].update(updates)
        self.save_to_disk()
    
    def get_user_context(self, user_id: str) -> str:
        """Generate context string for personalized responses"""
        profile = self.get_user_profile(user_id)
        
        if not profile:
            return ""
        
        context_parts = []
        
        if profile.get('name'):
            context_parts.append(f"User's name: {profile['name']}")
        
        if profile.get('conversation_count'):
            context_parts.append(f"Conversations: {profile['conversation_count']}")
        
        if profile.get('preferences'):
            for category, items in profile['preferences'].items():
                if items:
                    items_list = items if isinstance(items, list) else []
                    if items_list:
                        context_parts.append(f"{category.capitalize()}: {', '.join(items_list[:3])}")
        
        if profile.get('interests'):
            interests = profile['interests'][:5]
            context_parts.append(f"Interests: {', '.join(interests)}")
        
        if profile.get('topics_discussed'):
            topics = profile['topics_discussed'][:3]
            context_parts.append(f"Topics: {', '.join(topics)}")
        
        return " | ".join(context_parts)
    
    def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversations for a user"""
        if self.mongo_conversations is not None:
            try:
                convs = self.mongo_conversations.find(
                    {'user_id': user_id}
                ).sort('timestamp', -1).limit(limit)
                
                result = []
                for conv in convs:
                    conv['_id'] = str(conv['_id'])
                    if isinstance(conv.get('timestamp'), datetime):
                        conv['timestamp'] = conv['timestamp'].isoformat()
                    result.append(conv)
                return result
            except Exception as e:
                print(f"⚠️  Error fetching MongoDB conversations: {e}")
        
        # Fallback to JSON
        user_convs = [c for c in self.conversations if c['user_id'] == user_id]
        return user_convs[-limit:]
    
    def save_to_disk(self):
        """Save conversations and profiles to JSON disk"""
        try:
            # Save conversations (keep only last 10000)
            conversations_to_save = self.conversations[-10000:]
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(conversations_to_save, f, indent=2, ensure_ascii=False)
            
            # Save profiles
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_profiles, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"❌ Error saving data: {str(e)}")
    
    def get_conversation_history(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Get conversation history within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if self.mongo_conversations is not None:
            try:
                convs = self.mongo_conversations.find({
                    'user_id': user_id,
                    'timestamp': {'$gte': cutoff_time.isoformat()}
                }).sort('timestamp', 1)
                
                result = []
                for conv in convs:
                    conv['_id'] = str(conv['_id'])
                    result.append(conv)
                return result
            except Exception as e:
                print(f"⚠️  MongoDB query failed: {e}")
        
        # Fallback to JSON
        user_convs = []
        for conv in self.conversations:
            if conv['user_id'] == user_id:
                conv_time = datetime.fromisoformat(conv['timestamp'])
                if conv_time >= cutoff_time:
                    user_convs.append(conv)
        return user_convs
    
    def export_for_training(self, output_file: str = 'data/exported_conversations.json', 
                           min_quality: float = 0) -> int:
        """Export conversations for model training"""
        training_data = []
        
        # Get from MongoDB if available
        if self.mongo_conversations is not None:
            try:
                query = {}
                if min_quality > 0:
                    query['metadata.confidence'] = {'$gte': min_quality}
                
                convs = self.mongo_conversations.find(query).limit(10000)
                
                for conv in convs:
                    training_data.append({
                        'user': conv['user'],
                        'assistant': conv['assistant'],
                        'metadata': conv.get('metadata', {})
                    })
            except Exception as e:
                print(f"⚠️  MongoDB export failed: {e}")
        
        # Fallback to JSON
        if not training_data:
            for conv in self.conversations:
                quality = conv.get('metadata', {}).get('quality', 5)
                confidence = conv.get('metadata', {}).get('confidence', 1.0)
                
                if quality >= min_quality and confidence >= 0.5:
                    training_data.append({
                        'user': conv['user'],
                        'assistant': conv['assistant'],
                        'metadata': conv.get('metadata', {})
                    })
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported {len(training_data)} conversations to {output_file}")
        return len(training_data)
    
    def get_statistics(self) -> Dict:
        """Get storage statistics"""
        stats = {
            'storage_type': 'MongoDB' if self.mongo_client is not None else 'JSON',
            'mongodb_connected': self.mongo_client is not None
        }
        
        if self.mongo_conversations is not None and self.mongo_users is not None:
            try:
                stats['total_conversations'] = self.mongo_conversations.count_documents({})
                stats['total_users'] = self.mongo_users.count_documents({})
                stats['users_with_names'] = self.mongo_users.count_documents({'name': {'$ne': None}})
                
                # Get most active user
                pipeline = [
                    {'$group': {'_id': '$user_id', 'count': {'$sum': 1}}},
                    {'$sort': {'count': -1}},
                    {'$limit': 1}
                ]
                most_active = list(self.mongo_conversations.aggregate(pipeline))
                if most_active:
                    stats['max_conversations_single_user'] = most_active[0]['count']
                
            except Exception as e:
                print(f"⚠️  Error getting MongoDB stats: {e}")
        else:
            # JSON stats
            stats['total_conversations'] = len(self.conversations)
            stats['total_users'] = len(self.user_profiles)
            stats['users_with_names'] = len([p for p in self.user_profiles.values() if p.get('name')])
            
            if self.user_profiles:
                conversation_counts = [p.get('conversation_count', 0) for p in self.user_profiles.values()]
                stats['max_conversations_single_user'] = max(conversation_counts) if conversation_counts else 0
        
        # Time-based metrics
        if self.conversations:
            first_conv = datetime.fromisoformat(self.conversations[0]['timestamp'])
            last_conv = datetime.fromisoformat(self.conversations[-1]['timestamp'])
            stats['first_conversation'] = first_conv.strftime('%Y-%m-%d %H:%M')
            stats['last_conversation'] = last_conv.strftime('%Y-%m-%d %H:%M')
            
            days_active = (last_conv - first_conv).days + 1
            stats['days_active'] = days_active
            stats['conversations_per_day'] = round(len(self.conversations) / max(days_active, 1), 2)
        
        return stats
    
    def backup(self, backup_dir: str = 'data/backups'):
        """Create a backup of conversations and profiles"""
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Always backup JSON
        backup_conv_file = os.path.join(backup_dir, f'conversations_{timestamp}.json')
        backup_profile_file = os.path.join(backup_dir, f'profiles_{timestamp}.json')
        
        with open(backup_conv_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        
        with open(backup_profile_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_profiles, f, indent=2, ensure_ascii=False)
        
        # Also backup from MongoDB if available
        if self.mongo_conversations is not None and self.mongo_users is not None:
            try:
                mongo_backup_conv = os.path.join(backup_dir, f'mongodb_conversations_{timestamp}.json')
                mongo_backup_users = os.path.join(backup_dir, f'mongodb_users_{timestamp}.json')
                
                convs = list(self.mongo_conversations.find({}, {'_id': 0}))
                users = list(self.mongo_users.find({}, {'_id': 0}))
                
                # Convert datetime objects to strings
                for conv in convs:
                    if isinstance(conv.get('timestamp'), datetime):
                        conv['timestamp'] = conv['timestamp'].isoformat()
                
                for user in users:
                    if isinstance(user.get('first_seen'), datetime):
                        user['first_seen'] = user['first_seen'].isoformat()
                    if isinstance(user.get('last_seen'), datetime):
                        user['last_seen'] = user['last_seen'].isoformat()
                
                with open(mongo_backup_conv, 'w', encoding='utf-8') as f:
                    json.dump(convs, f, indent=2, ensure_ascii=False)
                
                with open(mongo_backup_users, 'w', encoding='utf-8') as f:
                    json.dump(users, f, indent=2, ensure_ascii=False)
                
                print(f"✓ MongoDB backup created in {backup_dir}")
            except Exception as e:
                print(f"⚠️  MongoDB backup failed: {e}")
        
        print(f"✓ JSON backup created in {backup_dir}")
        return backup_conv_file, backup_profile_file
    
    def close(self):
        """Close MongoDB connection"""
        if self.mongo_client is not None:
            self.mongo_client.close()
            print("✓ MongoDB connection closed")

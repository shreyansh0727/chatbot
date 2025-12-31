import random
from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime


class ResponseGenerator:
    """
    Advanced response generation utilities for chatbot
    Handles response selection, formatting, and personalization
    """
    
    def __init__(self):
        self.fallback_responses = self._load_fallback_responses()
        self.greeting_patterns = self._load_greeting_patterns()
        self.question_patterns = self._load_question_patterns()
        self.context_history = []
        self.max_history = 10
        self.personality_mode = 'friendly'  # Options: friendly, professional, casual
    
    def _load_fallback_responses(self) -> Dict[str, List[str]]:
        """Load fallback responses for different scenarios"""
        return {
            'low_confidence': [
                "I'm not quite sure about that. Could you rephrase?",
                "Hmm, I don't have a clear answer for that one.",
                "Not sure I understand. Can you elaborate?",
                "That's interesting, but I'm not sure how to respond.",
                "I don't have enough context to answer that well.",
                "Let me think... actually, I'm not 100% sure about that.",
                "Good question! But I'd need more details to answer properly."
            ],
            'greeting': [
                "Hey! What's up?",
                "Hi there! How's it going?",
                "Hello! How can I help?",
                "Hey! What can I do for you?",
                "Hi! What's on your mind?",
                "Yo! What brings you here?",
                "Hey there! Ready to chat?",
                "Hello! Great to see you!"
            ],
            'thanks': [
                "No problem! Happy to help.",
                "Anytime! Glad I could help.",
                "You're welcome!",
                "No worries! Let me know if you need anything else.",
                "Happy to help! 😊",
                "My pleasure!",
                "Glad to be of help!",
                "You got it! 👍"
            ],
            'goodbye': [
                "See you later! Take care!",
                "Bye! Have a great day!",
                "Later! Feel free to reach out anytime.",
                "Catch you later!",
                "See ya! 👋",
                "Take it easy!",
                "Until next time!",
                "Adios! Stay awesome!"
            ],
            'confusion': [
                "I didn't quite get that. Can you say it differently?",
                "Sorry, I'm a bit confused. Could you clarify?",
                "Not sure what you mean. Can you explain more?",
                "I'm having trouble understanding. Can you rephrase?",
                "Could you break that down for me?",
                "Hmm, lost me there. Try explaining it differently?"
            ],
            'general': [
                "That's cool!",
                "Interesting!",
                "I see what you mean.",
                "Makes sense!",
                "Got it!",
                "Nice!",
                "Awesome!",
                "Fair enough!",
                "I hear you!",
                "Sounds good!"
            ],
            'enthusiasm': [
                "That's awesome! 🔥",
                "Love it! 🚀",
                "Super cool!",
                "That's amazing!",
                "Wow, impressive!",
                "That rocks! 🎸",
                "Brilliant! ✨"
            ],
            'empathy': [
                "I understand how that feels.",
                "That must be frustrating.",
                "I can see why that's challenging.",
                "That's tough, but you'll figure it out!",
                "Hang in there!"
            ],
            'encouragement': [
                "You've got this! 💪",
                "Keep pushing! You're doing great!",
                "Don't give up! Progress is progress.",
                "Every step counts!",
                "You're on the right track!"
            ]
        }
    
    def _load_greeting_patterns(self) -> List[str]:
        """Load greeting detection patterns"""
        return [
            r'\b(hi|hey|hello|sup|yo|howdy|hiya|heya)\b',
            r'\bhow are you\b',
            r'\bwhat\'?s up\b',
            r'\bhow\'?s it going\b',
            r'\bgood (morning|afternoon|evening|night)\b',
            r'\bhello there\b',
            r'\bgreetings\b'
        ]
    
    def _load_question_patterns(self) -> Dict[str, List[str]]:
        """Load question type detection patterns"""
        return {
            'what': [r'\bwhat\b', r'\bwhat\'?s\b'],
            'how': [r'\bhow\b', r'\bhow\'?s\b', r'\bhow to\b', r'\bhow do\b'],
            'why': [r'\bwhy\b'],
            'when': [r'\bwhen\b'],
            'where': [r'\bwhere\b'],
            'who': [r'\bwho\b', r'\bwho\'?s\b'],
            'which': [r'\bwhich\b'],
            'can': [r'\bcan you\b', r'\bcan i\b', r'\bcan we\b'],
            'should': [r'\bshould\b', r'\bshould i\b'],
            'is': [r'\bis it\b', r'\bis this\b', r'\bis that\b'],
            'are': [r'\bare you\b', r'\bare there\b', r'\bare we\b'],
            'do': [r'\bdo you\b', r'\bdo i\b', r'\bdoes it\b']
        }
    
    def detect_intent(self, user_input: str) -> str:
        """
        Detect user intent from input
        
        Args:
            user_input: User's message
            
        Returns:
            Detected intent
        """
        text_lower = user_input.lower().strip()
        
        # Check for greetings
        for pattern in self.greeting_patterns:
            if re.search(pattern, text_lower):
                return 'greeting'
        
        # Check for thanks
        if re.search(r'\b(thanks|thank you|thx|tysm|ty|appreciate|grateful)\b', text_lower):
            return 'thanks'
        
        # Check for goodbye
        if re.search(r'\b(bye|goodbye|see you|later|gotta go|gtg|cya|farewell)\b', text_lower):
            return 'goodbye'
        
        # Check for help requests
        if re.search(r'\b(help|assist|support|guide|stuck|issue|problem|error)\b', text_lower):
            return 'help_request'
        
        # Check for excitement/enthusiasm
        if re.search(r'\b(awesome|amazing|great|excellent|perfect|wonderful|fantastic)\b', text_lower):
            return 'enthusiasm'
        
        # Check for frustration/problems
        if re.search(r'\b(frustrated|annoyed|stuck|confused|difficult|hard|trouble)\b', text_lower):
            return 'frustration'
        
        # Check for questions
        for question_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return f'question_{question_type}'
        
        # Check if it ends with question mark
        if text_lower.endswith('?'):
            return 'question_general'
        
        # Check for statements about projects/work
        if re.search(r'\b(working on|building|developing|creating|making)\b', text_lower):
            return 'project_discussion'
        
        return 'statement'
    
    def get_fallback_response(self, intent: str = 'low_confidence', 
                             user_name: Optional[str] = None) -> str:
        """
        Get a fallback response based on intent
        
        Args:
            intent: Detected intent
            user_name: Optional user name for personalization
            
        Returns:
            Fallback response
        """
        if intent in self.fallback_responses:
            response = random.choice(self.fallback_responses[intent])
        else:
            response = random.choice(self.fallback_responses['low_confidence'])
        
        # Personalize with name if available
        if user_name and random.random() < 0.3:
            response = f"{user_name}, {response.lower()}"
        
        return response
    
    def format_response(self, response: str, add_personality: bool = True,
                       intent: Optional[str] = None) -> str:
        """
        Format and personalize response
        
        Args:
            response: Raw response text
            add_personality: Whether to add personality touches
            intent: Optional detected intent for context
            
        Returns:
            Formatted response
        """
        if not response:
            return self.get_fallback_response()
        
        # Basic formatting
        response = response.strip()
        
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response)
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Add personality touches if enabled
        if add_personality:
            response = self._add_personality(response, intent)
        
        # Ensure proper punctuation
        if response and not response[-1] in '.!?':
            # Add period if it looks like a statement
            if not re.search(r'[😊👍👋🚀💻✨🔥]$', response):
                response += '.'
        
        return response
    
    def _add_personality(self, response: str, intent: Optional[str] = None) -> str:
        """
        Add personality touches to response
        
        Args:
            response: Response text
            intent: Optional intent for context
            
        Returns:
            Response with personality
        """
        response_lower = response.lower()
        
        # Add enthusiasm for project discussions
        if intent == 'project_discussion' and random.random() < 0.15:
            additions = [
                " That sounds exciting!",
                " Keep it up!",
                " Love the progress!",
                " Can't wait to see it!"
            ]
            if not any(add.lower() in response_lower for add in additions):
                response += random.choice(additions)
        
        # Add encouragement for help requests
        if intent == 'help_request' and random.random() < 0.2:
            encouragement = [
                " You'll figure it out!",
                " Don't worry, we'll sort it out!",
                " Happy to guide you through it!"
            ]
            if not any(enc.lower() in response_lower for enc in encouragement):
                response += random.choice(encouragement)
        
        # Randomly add casual phrases (8% chance)
        if random.random() < 0.08:
            casual_additions = [
                " Pretty cool, right?",
                " It's been great!",
                " Super excited about it!",
                " Really enjoying the journey!",
                " Making solid progress!"
            ]
            addition = random.choice(casual_additions)
            if addition.lower() not in response_lower:
                response += addition
        
        return response
    
    def select_best_response(self, 
                            nlp_response: str, 
                            nlp_confidence: float,
                            rag_response: str,
                            rag_similarity: float,
                            user_input: str,
                            threshold: float = 0.5,
                            user_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Select best response from NLP and RAG models
        
        Args:
            nlp_response: Response from NLP model
            nlp_confidence: NLP confidence score
            rag_response: Response from RAG model
            rag_similarity: RAG similarity score
            user_input: Original user input
            threshold: Minimum confidence threshold
            user_name: Optional user name for personalization
            
        Returns:
            Tuple of (selected_response, source)
        """
        # Detect intent
        intent = self.detect_intent(user_input)
        
        # Handle specific intents with fallback responses
        if intent in ['greeting', 'thanks', 'goodbye']:
            return self.get_fallback_response(intent, user_name), 'fallback'
        
        # Boost RAG for factual questions
        if intent.startswith('question_'):
            rag_similarity += 0.1  # Give RAG a boost for questions
        
        # Boost NLP for conversational statements
        if intent in ['statement', 'project_discussion']:
            nlp_confidence += 0.05
        
        # Compare model scores
        if nlp_confidence >= threshold and rag_similarity >= threshold:
            # Both models confident
            score_diff = abs(nlp_confidence - rag_similarity)
            
            # If scores are very close, prefer RAG for questions, NLP for statements
            if score_diff < 0.15:
                if intent.startswith('question_'):
                    return rag_response, 'rag'
                else:
                    return nlp_response, 'nlp'
            
            # Otherwise prefer higher score
            if nlp_confidence > rag_similarity:
                return nlp_response, 'nlp'
            else:
                return rag_response, 'rag'
        
        elif nlp_confidence >= threshold:
            # Only NLP confident
            return nlp_response, 'nlp'
        
        elif rag_similarity >= threshold:
            # Only RAG confident
            return rag_response, 'rag'
        
        else:
            # Neither confident - use best available or fallback
            if rag_similarity > nlp_confidence and rag_similarity > 0.3:
                return rag_response, 'rag_low'
            elif nlp_confidence > 0.3:
                return nlp_response, 'nlp_low'
            else:
                # Check for specific intents to provide better fallbacks
                if intent == 'frustration':
                    return self.get_fallback_response('empathy', user_name), 'fallback'
                elif intent == 'enthusiasm':
                    return self.get_fallback_response('enthusiasm', user_name), 'fallback'
                else:
                    return self.get_fallback_response('low_confidence', user_name), 'fallback'
    
    def add_context(self, user_input: str, response: str):
        """
        Add conversation to context history
        
        Args:
            user_input: User's message
            response: Bot's response
        """
        self.context_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last N conversations
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]
    
    def get_context_summary(self) -> str:
        """Get a summary of recent conversation context"""
        if not self.context_history:
            return "No prior context"
        
        recent = self.context_history[-3:]  # Last 3 exchanges
        topics = []
        
        for conv in recent:
            # Extract key topics from user input
            words = conv['user'].lower().split()
            tech_words = ['firebase', 'python', 'react', 'api', 'backend', 'frontend', 
                         'database', 'deployment', 'project', 'code', 'error']
            
            for word in tech_words:
                if word in words and word not in topics:
                    topics.append(word)
        
        if topics:
            return f"Recent topics: {', '.join(topics)}"
        return "General conversation"
    
    def get_context_aware_response(self, 
                                   response: str, 
                                   user_input: str) -> str:
        """
        Make response context-aware based on conversation history
        
        Args:
            response: Generated response
            user_input: User's current input
            
        Returns:
            Context-aware response
        """
        # Check if user is referring to previous context
        referring_words = ['it', 'that', 'this', 'also', 'too', 'same', 'them', 'those']
        
        has_reference = any(word in user_input.lower().split() for word in referring_words)
        
        if has_reference and len(self.context_history) > 0:
            # User might be referring to previous topic
            last_topic = self.context_history[-1]['user']
            
            # Optionally add context reference (5% chance)
            if random.random() < 0.05:
                response = f"About {last_topic[:30]}... {response}"
        
        return response
    
    def generate_follow_up_suggestions(self, response: str, 
                                      user_input: Optional[str] = None) -> List[str]:
        """
        Generate follow-up question suggestions
        
        Args:
            response: Current response
            user_input: Optional user input for context
            
        Returns:
            List of follow-up suggestions
        """
        suggestions = []
        
        response_lower = response.lower()
        
        # Detect topics and suggest follow-ups
        if 'project' in response_lower or 'working' in response_lower:
            suggestions.extend([
                "What features are you building?",
                "What tech stack are you using?",
                "When do you plan to launch?"
            ])
        
        if 'firebase' in response_lower:
            suggestions.extend([
                "How's Firebase working for you?",
                "Any issues with Firebase?",
                "Which Firebase services are you using?"
            ])
        
        if 'backend' in response_lower or 'api' in response_lower:
            suggestions.extend([
                "What backend framework are you using?",
                "How's the API development going?",
                "Any performance concerns?"
            ])
        
        if 'frontend' in response_lower or 'ui' in response_lower or 'react' in response_lower:
            suggestions.extend([
                "What UI framework are you using?",
                "How's the user experience?",
                "Any design challenges?"
            ])
        
        if 'deployment' in response_lower or 'deployed' in response_lower or 'hosting' in response_lower:
            suggestions.extend([
                "Where did you deploy it?",
                "Any deployment issues?",
                "How's the performance after deployment?"
            ])
        
        if 'error' in response_lower or 'issue' in response_lower or 'problem' in response_lower:
            suggestions.extend([
                "What error message are you getting?",
                "Have you tried debugging it?",
                "Need help troubleshooting?"
            ])
        
        if 'arduino' in response_lower or 'iot' in response_lower or 'hardware' in response_lower:
            suggestions.extend([
                "What sensors are you using?",
                "How's the hardware integration?",
                "Any circuit design challenges?"
            ])
        
        # Generic follow-ups if no specific topic detected
        if not suggestions:
            suggestions = [
                "Tell me more about that",
                "What else are you working on?",
                "How's everything going?",
                "Any challenges you're facing?",
                "What are your next steps?"
            ]
        
        # Shuffle and return max 3 suggestions
        random.shuffle(suggestions)
        return suggestions[:3]
    
    def enhance_response_with_emojis(self, response: str, intent: str) -> str:
        """
        Optionally add emojis to response based on context
        
        Args:
            response: Response text
            intent: Detected intent
            
        Returns:
            Response with emojis (if appropriate)
        """
        # Don't add emoji if response already has one
        if re.search(r'[😊👍👋🚀💻✨🔥💪🎉✅❌⚠️]', response):
            return response
        
        # Emoji mappings for different contexts
        emoji_map = {
            'greeting': ['👋', '😊', '🙂'],
            'goodbye': ['👋', '✌️', '🙂'],
            'thanks': ['😊', '🙏', '💯'],
            'project_discussion': ['💻', '🚀', '⚡'],
            'enthusiasm': ['🎉', '✨', '🔥'],
            'help_request': ['🤔', '💡', '🔧'],
            'frustration': ['😅', '💪'],
        }
        
        # Random chance to add emoji (25%)
        if random.random() < 0.25:
            if intent in emoji_map:
                emoji = random.choice(emoji_map[intent])
                response += f" {emoji}"
        
        return response
    
    def get_stats(self) -> Dict:
        """Get response generator statistics"""
        return {
            'context_history_size': len(self.context_history),
            'fallback_categories': len(self.fallback_responses),
            'total_fallback_responses': sum(len(v) for v in self.fallback_responses.values()),
            'personality_mode': self.personality_mode,
            'max_history': self.max_history
        }


# Test the response generator
if __name__ == "__main__":
    print("Testing Response Generator\n")
    print("="*60)
    
    generator = ResponseGenerator()
    
    # Test intent detection
    test_inputs = [
        "Hey! How are you?",
        "What are you working on?",
        "Thanks for the help!",
        "Gotta go, bye!",
        "How do I setup Firebase?",
        "Can you help me with this?",
        "I'm stuck on this error",
        "This is awesome!",
        "I'm so frustrated with this bug"
    ]
    
    print("Intent Detection Tests:")
    print("-"*60)
    for text in test_inputs:
        intent = generator.detect_intent(text)
        print(f"Input: {text}")
        print(f"Intent: {intent}\n")
    
    # Test response selection
    print("\n" + "="*60)
    print("Response Selection Tests:")
    print("-"*60)
    
    test_cases = [
        {
            'nlp_response': "Working on a Firebase project",
            'nlp_confidence': 0.8,
            'rag_response': "Building an attendance system with Firebase",
            'rag_similarity': 0.6,
            'user_input': "What are you working on?"
        },
        {
            'nlp_response': "Not sure about that",
            'nlp_confidence': 0.3,
            'rag_response': "Using FastAPI for backend",
            'rag_similarity': 0.7,
            'user_input': "What backend do you use?"
        },
        {
            'nlp_response': "I don't know",
            'nlp_confidence': 0.2,
            'rag_response': "Maybe try checking the docs",
            'rag_similarity': 0.25,
            'user_input': "How do I fix this error?"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}]")
        print(f"User: {case['user_input']}")
        print(f"NLP: {case['nlp_response']} (confidence: {case['nlp_confidence']})")
        print(f"RAG: {case['rag_response']} (similarity: {case['rag_similarity']})")
        
        response, source = generator.select_best_response(
            case['nlp_response'],
            case['nlp_confidence'],
            case['rag_response'],
            case['rag_similarity'],
            case['user_input']
        )
        
        formatted = generator.format_response(response, intent=generator.detect_intent(case['user_input']))
        print(f"Selected: {formatted} (source: {source})")
    
    # Test follow-up suggestions
    print("\n" + "="*60)
    print("Follow-up Suggestions:")
    print("-"*60)
    
    responses = [
        "Working on a Firebase integration project",
        "Just deployed the backend to GCP",
        "Having some issues with the API",
        "Building an Arduino-based IoT device"
    ]
    
    for response in responses:
        suggestions = generator.generate_follow_up_suggestions(response)
        print(f"\nResponse: {response}")
        print(f"Suggestions: {suggestions}")
    
    # Stats
    print("\n" + "="*60)
    print("Response Generator Stats:")
    print("-"*60)
    stats = generator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ All tests completed!")

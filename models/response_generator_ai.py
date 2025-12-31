"""
AI Response Generator with Integration
Generates intelligent responses by understanding patterns, not just copying
"""

from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Tuple
import re
import random


class IntelligentResponseGenerator:
    """
    Generates contextual responses by understanding the training data,
    not just copying it
    """
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.topics = {}
        self.patterns = {}
        
    def learn_from_data(self, training_data: List[Dict]):
        """
        Learn patterns and extract knowledge from training data
        """
        print("🧠 Learning patterns from training data...")
        
        for conv in training_data:
            user_msg = conv.get('user', '')
            bot_msg = conv.get('bot', '')
            
            # Extract key information
            topic = self._extract_topic(user_msg)
            intent = self._detect_intent(user_msg)
            key_facts = self._extract_facts(bot_msg)
            
            # Store knowledge
            knowledge_entry = {
                'topic': topic,
                'intent': intent,
                'facts': key_facts,
                'example_question': user_msg,
                'original_response': bot_msg,
                'source_url': conv.get('metadata', {}).get('url'),
                'keywords': self._extract_keywords(user_msg + ' ' + bot_msg)
            }
            
            self.knowledge_base.append(knowledge_entry)
            
            # Group by topic
            if topic not in self.topics:
                self.topics[topic] = []
            self.topics[topic].append(knowledge_entry)
        
        print(f"✓ Learned {len(self.knowledge_base)} knowledge entries")
        print(f"✓ Identified {len(self.topics)} topics")
    
    def generate_response(self, query: str, rag_matches: List[Dict], 
                         intent: str = None) -> Tuple[str, Dict]:
        """
        Generate an intelligent response based on learned knowledge
        
        Returns:
            Tuple of (response, metadata)
        """
        # Get intent if not provided
        if not intent:
            intent = self._detect_intent(query)
        
        topic = self._extract_topic(query)
        
        # Find relevant knowledge
        relevant_knowledge = self._find_relevant_knowledge(query, top_k=5)
        
        # Generate response based on intent
        if intent in ['what_is', 'question_what']:
            response = self._generate_definition(topic, relevant_knowledge, rag_matches)
        
        elif intent in ['how_to', 'question_how']:
            response = self._generate_explanation(topic, relevant_knowledge, rag_matches)
        
        elif intent in ['list', 'multiple', 'question_which']:
            response = self._generate_list_response(topic, relevant_knowledge, rag_matches)
        
        elif intent == 'comparison':
            response = self._generate_comparison(topic, relevant_knowledge, rag_matches)
        
        elif intent == 'project_discussion':
            response = self._generate_project_response(topic, relevant_knowledge, rag_matches)
        
        else:
            response = self._generate_conversational_response(topic, relevant_knowledge, rag_matches)
        
        # Add metadata
        metadata = {
            'knowledge_used': len(relevant_knowledge),
            'topic': topic,
            'intent': intent,
            'source_url': relevant_knowledge[0].get('source_url') if relevant_knowledge else None
        }
        
        return response, metadata
    
    def _generate_definition(self, topic: str, knowledge: List[Dict], 
                            rag_matches: List[Dict]) -> str:
        """Generate a definition-style response"""
        if not knowledge:
            if rag_matches and rag_matches[0]['similarity'] > 0.5:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return f"I don't have detailed information about {topic} yet."
        
        # Combine unique facts
        all_facts = []
        for k in knowledge[:3]:
            all_facts.extend(k['facts'][:2])
        
        # Remove duplicates while preserving order
        unique_facts = []
        seen = set()
        for fact in all_facts:
            fact_lower = fact.lower()
            if fact_lower not in seen and len(fact) > 15:
                unique_facts.append(fact)
                seen.add(fact_lower)
        
        if not unique_facts:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return f"I'm still learning about {topic}."
        
        # Generate natural definition
        response = unique_facts[0]
        if not response.endswith('.'):
            response += '.'
        
        # Add supporting facts
        if len(unique_facts) > 1:
            response += " " + unique_facts[1]
            if not response.endswith('.'):
                response += '.'
        
        # Add more context if available
        if len(unique_facts) > 2 and random.random() > 0.5:
            response += " " + unique_facts[2]
            if not response.endswith('.'):
                response += '.'
        
        # Add source
        if knowledge[0].get('source_url'):
            response += f"\n\n🔗 Learn more: {knowledge[0]['source_url']}"
        
        return response
    
    def _generate_explanation(self, topic: str, knowledge: List[Dict], 
                             rag_matches: List[Dict]) -> str:
        """Generate how-to/explanation response"""
        if not knowledge:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return f"I don't have specific guidance on {topic} yet."
        
        # Collect unique facts
        all_facts = []
        for k in knowledge[:4]:
            all_facts.extend(k['facts'][:2])
        
        # Remove duplicates
        unique_facts = list(dict.fromkeys(all_facts))[:4]
        
        if not unique_facts:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return f"I'm still learning about {topic}."
        
        # Create natural explanation
        if len(unique_facts) == 1:
            response = unique_facts[0]
        else:
            response = f"Here's what I know about {topic}:\n\n"
            for fact in unique_facts:
                response += f"• {fact}\n"
        
        # Add source
        if knowledge[0].get('source_url'):
            response += f"\n🔗 More details: {knowledge[0]['source_url']}"
        
        return response.strip()
    
    def _generate_list_response(self, topic: str, knowledge: List[Dict], 
                               rag_matches: List[Dict]) -> str:
        """Generate list-style response"""
        if not knowledge:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return "I don't have enough information for a comprehensive list."
        
        # Collect diverse facts
        facts = set()
        for k in knowledge[:5]:
            for fact in k['facts'][:2]:
                if len(fact) > 20:
                    facts.add(fact)
        
        facts = list(facts)[:5]
        
        if not facts:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return f"I'm building my knowledge about {topic}."
        
        # Format as list
        response = f"Regarding {topic}:\n\n"
        for fact in facts:
            response += f"• {fact}\n"
        
        return response.strip()
    
    def _generate_comparison(self, topic: str, knowledge: List[Dict], 
                            rag_matches: List[Dict]) -> str:
        """Generate comparison response"""
        return self._generate_conversational_response(topic, knowledge, rag_matches)
    
    def _generate_project_response(self, topic: str, knowledge: List[Dict], 
                                   rag_matches: List[Dict]) -> str:
        """Generate project-related response"""
        if not knowledge:
            if rag_matches and rag_matches[0]['similarity'] > 0.5:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return "I'm working on multiple projects right now!"
        
        # Get project-related facts
        facts = []
        for k in knowledge[:3]:
            facts.extend(k['facts'][:2])
        
        if not facts:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return "Got several projects in the pipeline!"
        
        # Create engaging response
        response = facts[0]
        if len(facts) > 1:
            response += " " + facts[1]
        
        # Add enthusiasm
        endings = [
            " Pretty exciting stuff!",
            " It's coming along nicely!",
            " Making good progress on it!",
            ""
        ]
        response += random.choice(endings)
        
        # Add URL if available
        if knowledge[0].get('source_url'):
            response += f"\n\n🔗 Check it out: {knowledge[0]['source_url']}"
        
        return response
    
    def _generate_conversational_response(self, topic: str, knowledge: List[Dict], 
                                         rag_matches: List[Dict]) -> str:
        """Generate natural conversational response"""
        if not knowledge:
            if rag_matches and rag_matches[0]['similarity'] > 0.5:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return "That's an interesting question. Tell me more?"
        
        # Get best facts
        facts = []
        for k in knowledge[:2]:
            facts.extend(k['facts'][:2])
        
        if not facts:
            if rag_matches:
                return self._paraphrase_response(rag_matches[0].get('response', ''))
            return f"I'm learning more about {topic} every day!"
        
        # Create natural response
        response = facts[0]
        if len(facts) > 1 and random.random() > 0.3:
            response += " " + facts[1]
        
        # Add URL if available
        if knowledge[0].get('source_url') and random.random() > 0.5:
            response += f"\n\n🔗 Source: {knowledge[0]['source_url']}"
        
        return response
    
    def _paraphrase_response(self, response: str) -> str:
        """
        Lightly paraphrase a response to make it more natural
        """
        # Simple paraphrasing techniques
        replacements = {
            r'\bI am\b': 'I\'m',
            r'\bdo not\b': 'don\'t',
            r'\bcannot\b': 'can\'t',
            r'\bwill not\b': 'won\'t',
            r'\bshould not\b': 'shouldn\'t',
            r'\bwould not\b': 'wouldn\'t',
        }
        
        for pattern, replacement in replacements.items():
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        return response
    
    def _find_relevant_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find relevant knowledge entries using semantic search"""
        if not self.knowledge_base:
            return []
        
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        similarities = []
        for knowledge in self.knowledge_base:
            # Create searchable text
            knowledge_text = (
                f"{knowledge['topic']} "
                f"{' '.join(knowledge['facts'])} "
                f"{' '.join(knowledge['keywords'])}"
            )
            
            knowledge_embedding = self.model.encode(knowledge_text, convert_to_tensor=True)
            
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                knowledge_embedding.unsqueeze(0)
            ).item()
            
            similarities.append((knowledge, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [k for k, s in similarities[:top_k] if s > 0.3]
    
    def _extract_topic(self, text: str) -> str:
        """Extract main topic from text"""
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'do', 'does', 
                     'can', 'you', 'i', 'me', 'my', 'tell', 'about', 'your'}
        words = text.lower().split()
        topic_words = [w for w in words if w not in stop_words and len(w) > 2]
        
        return ' '.join(topic_words[:3]) if topic_words else 'general'
    
    def _detect_intent(self, text: str) -> str:
        """Detect user intent"""
        text_lower = text.lower()
        
        if re.search(r'\b(what is|what are|define|meaning)\b', text_lower):
            return 'what_is'
        
        if re.search(r'\b(how to|how do|how can|steps|guide)\b', text_lower):
            return 'how_to'
        
        if re.search(r'\b(list|all|multiple|different|which)\b', text_lower):
            return 'list'
        
        if re.search(r'\b(compare|difference|vs|versus|better)\b', text_lower):
            return 'comparison'
        
        if re.search(r'\b(working on|building|developing|creating|project)\b', text_lower):
            return 'project_discussion'
        
        if re.search(r'\bwhat\b', text_lower):
            return 'question_what'
        
        if re.search(r'\bhow\b', text_lower):
            return 'question_how'
        
        if re.search(r'\bwhich\b', text_lower):
            return 'question_which'
        
        return 'general'
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from text"""
        # Split into sentences
        sentences = re.split(r'[.!?\n]+', text)
        
        facts = []
        for sent in sentences:
            sent = sent.strip()
            # Filter out URLs, short fragments, and overly long text
            if (20 < len(sent) < 250 and 
                not sent.startswith('http') and 
                not sent.startswith('🔗')):
                facts.append(sent)
        
        return facts[:5]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        return list(set(keywords))[:15]

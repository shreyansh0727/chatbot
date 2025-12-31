import re
import string
import unicodedata
from typing import List, Optional

class TextPreprocessor:
    """
    Text preprocessing utilities for chatbot
    Handles cleaning, normalization, and tokenization
    """
    
    def __init__(self):
        self.punctuation = string.punctuation
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load common English stopwords"""
        # Common English stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
            'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this',
            'but', 'they', 'have', 'had', 'what', 'when', 'where', 'who',
            'which', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
            'will', 'just', 'should', 'now'
        }
        return stopwords
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\+?[\d\s\-\(\)]{10,}', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_special_characters(self, text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep basic punctuation
            
        Returns:
            Text without special characters
        """
        if keep_punctuation:
            # Keep alphanumeric and basic punctuation
            pattern = r'[^a-zA-Z0-9\s\.\,\?\!\'\-]'
        else:
            # Keep only alphanumeric and spaces
            pattern = r'[^a-zA-Z0-9\s]'
        
        text = re.sub(pattern, '', text)
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize unicode to ASCII
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('utf-8')
        return text
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emoji characters
        
        Args:
            text: Input text
            
        Returns:
            Text without emojis
        """
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"
            u"\u3030"
            "]+", 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand common contractions
        
        Args:
            text: Input text
            
        Returns:
            Text with expanded contractions
        """
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "could've": "could have",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "might've": "might have",
            "must've": "must have",
            "needn't": "need not",
            "should've": "should have",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
        
        # Create pattern for whole words only
        for contraction, expansion in contractions.items():
            text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Input text
            
        Returns:
            Text without stopwords
        """
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple word tokenization
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def preprocess_for_training(self, text: str, 
                                remove_stopwords: bool = False,
                                remove_emojis: bool = False) -> str:
        """
        Full preprocessing pipeline for training data
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            remove_emojis: Whether to remove emojis
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Optionally remove emojis
        if remove_emojis:
            text = self.remove_emojis(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove special characters but keep basic punctuation
        text = self.remove_special_characters(text, keep_punctuation=True)
        
        # Optionally remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        # Final cleanup
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess_for_inference(self, text: str) -> str:
        """
        Lighter preprocessing for user input during inference
        
        Args:
            text: User input text
            
        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = self.clean_text(text)
        
        # Expand contractions
        text = self.expand_contractions(text)
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Keep emojis and most punctuation for better context
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract important keywords from text
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Tokenize and remove stopwords
        tokens = self.tokenize(text)
        keywords = [token for token in tokens if token not in self.stopwords]
        
        # Get unique keywords, sorted by frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    def calculate_similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate simple word-based similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0


# Test the preprocessor
if __name__ == "__main__":
    print("Testing Text Preprocessor\n")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    # Test cases
    test_texts = [
        "Hey! What's up? 😊 Check this out: https://example.com",
        "I'm working on Firebase integration, it's really cool!",
        "Can't believe how easy it was to setup the backend 🚀",
        "Contact me at test@email.com or call +91-9876543210",
        "The project's going great... shouldn't have any issues!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}]")
        print(f"Original: {text}")
        
        # Clean
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned:  {cleaned}")
        
        # Full preprocessing
        processed = preprocessor.preprocess_for_training(text, remove_emojis=True)
        print(f"Full Preprocess: {processed}")
        
        # Keywords
        keywords = preprocessor.extract_keywords(text)
        print(f"Keywords: {keywords}")
        print("-"*60)
    
    # Test similarity
    print("\n" + "="*60)
    print("Testing Similarity")
    print("="*60)
    
    text_a = "Working on Firebase and React Native project"
    text_b = "Building a React Native app with Firebase backend"
    
    similarity = preprocessor.calculate_similarity_score(text_a, text_b)
    print(f"\nText A: {text_a}")
    print(f"Text B: {text_b}")
    print(f"Similarity: {similarity:.3f}")

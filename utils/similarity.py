import numpy as np
from typing import List, Tuple, Optional, Union
import re
from collections import Counter
import math

class SimilarityCalculator:
    """
    Various similarity calculation methods for text comparison
    Used for chatbot response matching and relevance scoring
    """
    
    def __init__(self):
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load common English stopwords"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
            'that', 'the', 'to', 'was', 'will', 'with', 'but', 'they',
            'have', 'had', 'what', 'when', 'where', 'who', 'which',
            'this', 'these', 'those', 'i', 'you', 'we', 'my', 'your'
        }
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and extract words
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        return tokens
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts
        Jaccard = |A ∩ B| / |A ∪ B|
        
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
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        Uses word frequency vectors
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Create word frequency vectors
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        # Get all unique words
        all_words = set(freq1.keys()).union(set(freq2.keys()))
        
        # Create vectors
        vec1 = np.array([freq1.get(word, 0) for word in all_words])
        vec2 = np.array([freq2.get(word, 0) for word in all_words])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def dice_coefficient(self, text1: str, text2: str) -> float:
        """
        Calculate Dice coefficient (Sørensen–Dice coefficient)
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        
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
        
        intersection = tokens1.intersection(tokens2)
        
        return 2 * len(intersection) / (len(tokens1) + len(tokens2))
    
    def overlap_coefficient(self, text1: str, text2: str) -> float:
        """
        Calculate overlap coefficient (Szymkiewicz–Simpson coefficient)
        Overlap = |A ∩ B| / min(|A|, |B|)
        
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
        
        intersection = tokens1.intersection(tokens2)
        
        return len(intersection) / min(len(tokens1), len(tokens2))
    
    def levenshtein_distance(self, text1: str, text2: str) -> int:
        """
        Calculate Levenshtein distance (edit distance)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Edit distance (lower is more similar)
        """
        if len(text1) < len(text2):
            return self.levenshtein_distance(text2, text1)
        
        if len(text2) == 0:
            return len(text1)
        
        previous_row = range(len(text2) + 1)
        
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def levenshtein_similarity(self, text1: str, text2: str) -> float:
        """
        Normalized Levenshtein similarity (0-1)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        distance = self.levenshtein_distance(text1.lower(), text2.lower())
        max_len = max(len(text1), len(text2))
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """
        Calculate n-gram based similarity
        
        Args:
            text1: First text
            text2: Second text
            n: Size of n-grams (default: 2 for bigrams)
            
        Returns:
            Similarity score (0-1)
        """
        def get_ngrams(text: str, n: int) -> List[str]:
            """Generate n-grams from text"""
            text = text.lower()
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        ngrams1 = set(get_ngrams(text1, n))
        ngrams2 = set(get_ngrams(text2, n))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1.intersection(ngrams2)
        union = ngrams1.union(ngrams2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def tfidf_similarity(self, text1: str, text2: str, corpus: List[str] = None) -> float:
        """
        Calculate TF-IDF weighted similarity
        
        Args:
            text1: First text
            text2: Second text
            corpus: List of documents for IDF calculation (optional)
            
        Returns:
            Similarity score (0-1)
        """
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Simple TF-IDF without full corpus
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        all_words = set(freq1.keys()).union(set(freq2.keys()))
        
        # If corpus provided, calculate IDF
        if corpus:
            idf = {}
            num_docs = len(corpus)
            for word in all_words:
                doc_count = sum(1 for doc in corpus if word in doc.lower())
                idf[word] = math.log(num_docs / (doc_count + 1))
        else:
            # Use simple IDF approximation
            idf = {word: 1.0 for word in all_words}
        
        # Calculate TF-IDF vectors
        vec1 = np.array([freq1.get(word, 0) * idf[word] for word in all_words])
        vec2 = np.array([freq2.get(word, 0) * idf[word] for word in all_words])
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def combined_similarity(self, text1: str, text2: str, 
                           weights: Optional[dict] = None) -> float:
        """
        Calculate combined similarity using multiple methods
        
        Args:
            text1: First text
            text2: Second text
            weights: Dictionary of method weights (default: equal weights)
            
        Returns:
            Combined similarity score (0-1)
        """
        if weights is None:
            weights = {
                'jaccard': 0.25,
                'cosine': 0.30,
                'dice': 0.20,
                'levenshtein': 0.15,
                'ngram': 0.10
            }
        
        scores = {
            'jaccard': self.jaccard_similarity(text1, text2),
            'cosine': self.cosine_similarity(text1, text2),
            'dice': self.dice_coefficient(text1, text2),
            'levenshtein': self.levenshtein_similarity(text1, text2),
            'ngram': self.ngram_similarity(text1, text2)
        }
        
        # Calculate weighted average
        total_score = sum(scores[method] * weight 
                         for method, weight in weights.items())
        
        return total_score
    
    def find_most_similar(self, query: str, 
                         candidates: List[str],
                         method: str = 'cosine',
                         top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts from a list of candidates
        
        Args:
            query: Query text
            candidates: List of candidate texts
            method: Similarity method to use
            top_k: Number of results to return
            
        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        similarity_methods = {
            'jaccard': self.jaccard_similarity,
            'cosine': self.cosine_similarity,
            'dice': self.dice_coefficient,
            'overlap': self.overlap_coefficient,
            'levenshtein': self.levenshtein_similarity,
            'ngram': self.ngram_similarity,
            'combined': self.combined_similarity
        }
        
        if method not in similarity_methods:
            method = 'cosine'
        
        similarity_func = similarity_methods[method]
        
        # Calculate similarities
        results = []
        for candidate in candidates:
            score = similarity_func(query, candidate)
            results.append((candidate, score))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def is_similar(self, text1: str, text2: str, 
                   threshold: float = 0.7,
                   method: str = 'cosine') -> bool:
        """
        Check if two texts are similar above a threshold
        
        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold (0-1)
            method: Similarity method to use
            
        Returns:
            True if similar, False otherwise
        """
        similarity_methods = {
            'jaccard': self.jaccard_similarity,
            'cosine': self.cosine_similarity,
            'dice': self.dice_coefficient,
            'levenshtein': self.levenshtein_similarity,
            'combined': self.combined_similarity
        }
        
        if method not in similarity_methods:
            method = 'cosine'
        
        score = similarity_methods[method](text1, text2)
        return score >= threshold


# Test the similarity calculator
if __name__ == "__main__":
    print("Testing Similarity Calculator\n")
    print("="*60)
    
    calculator = SimilarityCalculator()
    
    # Test pairs
    test_pairs = [
        ("Working on Firebase integration", "Building Firebase project"),
        ("How are you doing today?", "How's it going?"),
        ("React Native mobile app", "Mobile application with React Native"),
        ("Python backend with FastAPI", "Using FastAPI for backend"),
        ("Deployed to Google Cloud", "Hosted on AWS")
    ]
    
    print("Similarity Comparisons:\n")
    
    for text1, text2 in test_pairs:
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print("-"*60)
        
        print(f"Jaccard:     {calculator.jaccard_similarity(text1, text2):.3f}")
        print(f"Cosine:      {calculator.cosine_similarity(text1, text2):.3f}")
        print(f"Dice:        {calculator.dice_coefficient(text1, text2):.3f}")
        print(f"Overlap:     {calculator.overlap_coefficient(text1, text2):.3f}")
        print(f"Levenshtein: {calculator.levenshtein_similarity(text1, text2):.3f}")
        print(f"N-gram:      {calculator.ngram_similarity(text1, text2):.3f}")
        print(f"Combined:    {calculator.combined_similarity(text1, text2):.3f}")
        print("="*60 + "\n")
    
    # Test finding most similar
    print("\nFinding Most Similar:\n")
    print("="*60)
    
    query = "What tech stack do you use?"
    candidates = [
        "I use React Native and Firebase",
        "Working with Python and FastAPI",
        "Building with Next.js and MongoDB",
        "Tech stack is React, Node, and PostgreSQL",
        "Using Flutter for mobile development"
    ]
    
    print(f"Query: {query}\n")
    print("Candidates:")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c}")
    
    print("\nTop 3 Most Similar (Cosine):")
    results = calculator.find_most_similar(query, candidates, method='cosine', top_k=3)
    for i, (text, score) in enumerate(results, 1):
        print(f"  {i}. {text}")
        print(f"     Score: {score:.3f}")
    
    # Test similarity threshold
    print("\n" + "="*60)
    print("Similarity Threshold Test:")
    print("-"*60)
    
    test_text1 = "Working on Firebase project"
    test_text2 = "Building Firebase application"
    
    is_sim = calculator.is_similar(test_text1, test_text2, threshold=0.5)
    score = calculator.cosine_similarity(test_text1, test_text2)
    
    print(f"Text 1: {test_text1}")
    print(f"Text 2: {test_text2}")
    print(f"Cosine Score: {score:.3f}")
    print(f"Similar (threshold=0.5): {is_sim}")

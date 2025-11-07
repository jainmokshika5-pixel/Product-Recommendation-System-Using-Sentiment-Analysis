"""
NLP utilities for text processing and feature extraction
Handles query analysis, category detection, and feature extraction
"""

import re
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class NLPProcessor:
    def __init__(self):
        self.download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced product categories matching your datasets
        self.categories = {
            'Laptops': ['laptop', 'laptops', 'computer', 'computers', 'pc', 'notebook', 'notebooks', 'macbook', 'chromebook', 'gaming laptop'],
            'Smartphones': ['smartphone', 'smartphones', 'phone', 'phones', 'mobile phone', 'mobile phones', 'mobile', 'mobiles', 'iphone', 'android phone', 'cell phone', 'oneplus', 'samsung phone'],
            'Wearables': ['watch', 'watches', 'smartwatch', 'smartwatches', 'wearable', 'wearables', 'fitness tracker', 'apple watch', 'fitness band'],
            'Audio': ['speaker', 'speakers', 'headphone', 'headphones', 'earphone', 'earphones', 'audio', 'sound', 'bluetooth speaker', 'wireless speaker', 'earbuds', 'headset', 'headsets', 'wireless headphone', 'wireless headphones'],
            'Smart Home': ['tv', 'television', 'smart tv', 'streaming', 'home automation', 'smart speaker', 'alexa'],
            'Appliances': ['cooler', 'ac', 'air conditioner', 'refrigerator', 'washing machine', 'microwave'],
            'Other': ['accessories', 'charger', 'cable', 'case', 'power bank', 'mount', 'stand']
        }
        
        # Feature keywords for extraction
        self.features = {
            'battery': ['battery', 'power', 'charging', 'long-lasting', 'battery life'],
            'performance': ['fast', 'performance', 'speed', 'quick', 'processor', 'ram', 'cpu'],
            'camera': ['camera', 'photo', 'picture', 'video', 'photography', 'megapixel'],
            'display': ['screen', 'display', 'resolution', 'hd', '4k', 'brightness', 'color'],
            'audio': ['sound', 'audio', 'music', 'noise', 'bass', 'treble', 'volume'],
            'design': ['design', 'style', 'look', 'appearance', 'color', 'build', 'material'],
            'price': ['cheap', 'expensive', 'budget', 'affordable', 'cost', 'price', 'value'],
            'storage': ['storage', 'memory', 'gb', 'tb', 'space', 'capacity'],
            'connectivity': ['wifi', 'bluetooth', 'network', 'internet', 'connection', '5g', '4g']
        }
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Clean and preprocess text
        Returns list of cleaned tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def detect_category(self, query: str) -> tuple[str, float]:
        """
        Detect product category from query
        Returns (category, confidence_score)
        """
        query_lower = query.lower().strip()
        category_scores = {}
        
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches like "phone" in "earphones"
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, query_lower):
                    # Exact match gets higher score
                    if keyword == query_lower:
                        score += 2.0
                    else:
                        score += 1.0
            
            if score > 0:
                category_scores[category] = score
        
        if not category_scores:
            return "General", 0.5
        
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]
        confidence = min(max_score / 2.0, 1.0)  # Normalize confidence
        
        return best_category, confidence
    
    def extract_features(self, query: str) -> List[str]:
        """
        Extract relevant features from query
        Returns list of detected features
        """
        query_lower = query.lower()
        detected_features = []
        
        for feature, keywords in self.features.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if feature not in detected_features:
                        detected_features.append(feature)
                    break
        
        return detected_features
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Complete query analysis
        Returns category, features, and confidence
        """
        # Detect category
        category, confidence = self.detect_category(query)
        
        # Extract features
        features = self.extract_features(query)
        
        # If no features detected, add some default ones based on category
        if not features:
            default_features = {
                'Laptop': ['performance', 'battery', 'display'],
                'Smartphone': ['camera', 'battery', 'performance'],
                'Smartwatch': ['battery', 'design', 'connectivity'],
                'Speakers': ['audio', 'design', 'connectivity']
            }
            features = default_features.get(category, ['performance', 'design'])
        
        return {
            'category': category,
            'features': features,
            'confidence': confidence
        }
    
    def get_categories(self) -> List[str]:
        """Get list of available categories"""
        return list(self.categories.keys())
    
    def get_feature_keywords(self, feature: str) -> List[str]:
        """Get keywords for a specific feature"""
        return self.features.get(feature, [])
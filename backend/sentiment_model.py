"""
Sentiment Analysis using CNN+BiLSTM model
Handles sentiment classification of product reviews
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import pickle
import os
import re

class CNNBiLSTMSentimentModel(nn.Module):
    """CNN + BiLSTM model for sentiment analysis"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=3):
        super(CNNBiLSTMSentimentModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(64, hidden_dim, bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # CNN layers
        conv_input = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        conv1_out = F.relu(self.conv1(conv_input))
        conv2_out = F.relu(self.conv2(conv1_out))  # Take conv1 output as input
        conv3_out = F.relu(self.conv3(conv2_out))  # Take conv2 output as input
        
        # Use final conv layer output
        conv_out = conv3_out
        conv_out = conv_out.transpose(1, 2)  # (batch, seq_len, 64)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(conv_out)  # (batch, seq_len, hidden_dim*2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim*2)
        
        # Classification
        output = self.dropout(attended)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.max_length = 100
        self.vocab_size = 8261
        self.is_model_loaded = False
        
        # Initialize with comprehensive rule-based fallback for all categories
        self.positive_words = {
            # General positive
            'excellent', 'amazing', 'great', 'good', 'fantastic', 'wonderful', 
            'awesome', 'perfect', 'love', 'best', 'outstanding', 'brilliant',
            'superb', 'impressive', 'satisfied', 'happy', 'pleased', 'recommend',
            'nice', 'fabulous', 'classy', 'simply', 'worth', 'mind-blowing',
            'terrific', 'super', 'blazing', 'incredible', 'refreshing',
            
            # Performance related
            'fast', 'smooth', 'responsive', 'efficient', 'powerful', 'reliable',
            'stable', 'durable', 'comfortable', 'convenient', 'easy', 'clear',
            
            # Quality related
            'premium', 'quality', 'sturdy', 'solid', 'robust', 'sleek', 'elegant',
            'beautiful', 'stunning', 'crystal', 'sharp', 'vivid', 'bright'
        }
        
        self.negative_words = {
            # General negative
            'terrible', 'awful', 'bad', 'horrible', 'worst', 'hate', 'disappointing',
            'poor', 'useless', 'broken', 'defective', 'failed', 'disappointed',
            'frustrated', 'annoying', 'waste', 'regret', 'problem', 'issue',
            'horrible', 'pathetic', 'garbage', 'trash', 'worthless', 'overpriced',
            
            # Performance related
            'slow', 'laggy', 'unresponsive', 'freezing', 'crashing', 'heating',
            'overheating', 'draining', 'loud', 'noisy', 'uncomfortable', 'heavy',
            
            # Quality related
            'cheap', 'flimsy', 'fragile', 'unstable', 'unreliable', 'scratched',
            'damaged', 'faulty', 'malfunctioning', 'stopped', 'not working'
        }
    
    def create_simple_tokenizer(self):
        """Create a simple tokenizer for demo purposes"""
        # This would normally be trained on your dataset
        # For demo, we'll use a simple word-to-index mapping
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        # Add common words covering all categories
        common_words = [
            # Basic words
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'it', 'is', 'was', 'are', 'were',
            'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'will', 'would',
            
            # Sentiment words
            'good', 'bad', 'great', 'excellent', 'poor', 'amazing', 'terrible',
            'love', 'hate', 'like', 'dislike', 'best', 'worst', 'awesome', 'wonderful',
            'fantastic', 'horrible', 'perfect', 'disappointing', 'satisfied', 'frustrated',
            
            # Electronics/Mobile
            'phone', 'mobile', 'smartphone', 'iphone', 'android', 'camera', 'battery',
            'screen', 'display', 'charger', 'charging', 'oneplus', 'samsung', 'apple',
            
            # Laptop/Computer
            'laptop', 'computer', 'pc', 'processor', 'ram', 'storage', 'ssd', 'keyboard',
            'trackpad', 'performance', 'cooling', 'ports', 'gaming', 'hp', 'dell', 'asus',
            
            # Audio/Headphones
            'headphones', 'earbuds', 'speaker', 'sound', 'audio', 'music', 'bass',
            'noise', 'cancellation', 'microphone', 'bluetooth', 'wireless', 'boat',
            
            # Smart Home/Wearables
            'smartwatch', 'watch', 'fitness', 'tracking', 'health', 'heart', 'sleep',
            'steps', 'smart', 'home', 'alexa', 'google', 'wifi', 'connection',
            
            # TV/Entertainment
            'tv', 'television', 'streaming', 'netflix', 'remote', 'channel', 'volume',
            'resolution', 'picture', 'video',
            
            # General product attributes
            'quality', 'price', 'value', 'money', 'design', 'build', 'material',
            'fast', 'slow', 'easy', 'hard', 'cheap', 'expensive', 'comfortable',
            'durable', 'reliable', 'working', 'broken', 'defective', 'recommend'
        ]
        
        for i, word in enumerate(common_words, 2):
            vocab[word] = i
        
        return vocab
    
    def load_model(self):
        """Load or create sentiment model"""
        model_path = "models/sentiment_model.pth"
        tokenizer_path = "models/tokenizer.pkl"
        
        try:
            # Try to load existing model
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                self.model = CNNBiLSTMSentimentModel(self.vocab_size)
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                
                self.model.eval()
                self.is_model_loaded = True
                print("✅ Sentiment model loaded successfully!")
            else:
                # Create simple tokenizer for demo
                self.tokenizer = self.create_simple_tokenizer()
                print("⚠️ Using rule-based sentiment analysis (ML model not found)")
                
        except Exception as e:
            print(f"⚠️ Could not load sentiment model: {e}")
            self.tokenizer = self.create_simple_tokenizer()
    
    def preprocess_text(self, text: str) -> List[int]:
        """Convert text to token indices"""
        # Simple preprocessing
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Convert to indices
        indices = []
        for word in words[:self.max_length]:
            indices.append(self.tokenizer.get(word, 1))  # 1 is <UNK>
        
        # Pad sequence
        while len(indices) < self.max_length:
            indices.append(0)  # 0 is <PAD>
        
        return indices[:self.max_length]
    
    def rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple rule-based sentiment analysis as fallback"""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = min(0.6 + (positive_count - negative_count) * 0.1, 0.95)
            score = 6.0 + (positive_count - negative_count) * 0.5
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = min(0.6 + (negative_count - positive_count) * 0.1, 0.95)
            score = 4.0 - (negative_count - positive_count) * 0.5
        else:
            sentiment = 'neutral'
            confidence = 0.5
            score = 5.0
        
        # Ensure score is within bounds
        score = max(1.0, min(10.0, score))
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'score': score
        }
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment of given text"""
        if not text.strip():
            return {'sentiment': 'neutral', 'confidence': 0.5, 'score': 5.0}
        
        if self.is_model_loaded and self.model is not None:
            try:
                # Use ML model
                indices = self.preprocess_text(text)
                input_tensor = torch.LongTensor([indices])
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = F.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class].item()
                
                sentiment = self.label_map[predicted_class]
                
                # Convert to 1-10 score
                if sentiment == 'positive':
                    score = 6.0 + confidence * 4.0  # 6-10 range
                elif sentiment == 'negative':
                    score = 1.0 + confidence * 3.0  # 1-4 range
                else:
                    score = 4.0 + confidence * 2.0  # 4-6 range
                
                return {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'score': float(score)
                }
                
            except Exception as e:
                print(f"ML model prediction failed: {e}, falling back to rules")
        
        # Fallback to rule-based
        return self.rule_based_sentiment(text)
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_model_loaded
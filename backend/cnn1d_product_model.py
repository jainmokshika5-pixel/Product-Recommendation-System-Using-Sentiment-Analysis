"""
1D CNN-based Product Review Analysis Model
Specialized for electronics_balanced_10k.csv dataset with multi-task learning:
- Sentiment classification
- Feature extraction
- Rating prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple
import pickle
import os
import re
from collections import Counter
import matplotlib.pyplot as plt

class ProductReviewDataset(Dataset):
    """Custom dataset for product reviews"""
    
    def __init__(self, texts, sentiments, ratings, features, vocab_to_idx, max_length=128):
        self.texts = texts
        self.sentiments = sentiments
        self.ratings = ratings
        self.features = features
        self.vocab_to_idx = vocab_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        sentiment = self.sentiments[idx]
        rating = self.ratings[idx]
        feature = self.features[idx]
        
        # Convert text to indices
        tokens = self.tokenize(text)
        indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([self.vocab_to_idx['<PAD>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
            
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'rating': torch.tensor(int(rating) - 1, dtype=torch.long),  # Convert 1-5 to 0-4
            'feature': torch.tensor(feature, dtype=torch.long)
        }
    
    def tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()

class MultiTask1DCNNModel(nn.Module):
    """1D CNN model for product review analysis"""
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], 
                 num_sentiments=3, num_ratings=5, num_features=50, dropout=0.3):
        super(MultiTask1DCNNModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Multiple 1D CNN layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, padding='same')
            for fs in filter_sizes
        ])
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        
        # Additional conv layers for deeper feature extraction
        self.conv_deep = nn.Conv1d(num_filters * len(filter_sizes), num_filters * 2, kernel_size=3, padding='same')
        self.batch_norm_deep = nn.BatchNorm1d(num_filters * 2)
        
        # Global pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature dimension after pooling
        pooled_dim = num_filters * 2 * 2  # max + avg pooling
        
        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_sentiments)
        )
        
        self.rating_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_ratings)
        )
        
        self.feature_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_features)
        )
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # Apply multiple conv layers with different filter sizes
        conv_outputs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            conv_out = F.relu(bn(conv(embedded)))  # (batch, num_filters, seq_len)
            # Ensure all outputs have the same sequence length
            if i == 0:
                target_seq_len = conv_out.size(2)
            else:
                # Adjust sequence length if needed
                if conv_out.size(2) != target_seq_len:
                    conv_out = F.adaptive_avg_pool1d(conv_out, target_seq_len)
            conv_outputs.append(conv_out)
        
        # Concatenate all conv outputs
        concat_conv = torch.cat(conv_outputs, dim=1)  # (batch, num_filters*len(filter_sizes), seq_len)
        
        # Deep conv layer
        deep_conv = F.relu(self.batch_norm_deep(self.conv_deep(concat_conv)))
        
        # Global pooling
        max_pooled = self.global_max_pool(deep_conv).squeeze(-1)  # (batch, num_filters*2)
        avg_pooled = self.global_avg_pool(deep_conv).squeeze(-1)  # (batch, num_filters*2)
        
        # Combine pooled features
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)  # (batch, num_filters*2*2)
        
        # Shared features
        shared_features = self.shared_fc(pooled)
        
        # Task-specific predictions
        sentiment_logits = self.sentiment_head(shared_features)
        rating_logits = self.rating_head(shared_features)
        feature_logits = self.feature_head(shared_features)
        
        return {
            'sentiment': sentiment_logits,
            'rating': rating_logits,
            'feature': feature_logits
        }

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class ProductReview1DCNNAnalyzer:
    """1D CNN-based analyzer for product reviews"""
    
    def __init__(self, model_path="cnn1d_product_model.pth"):
        self.model_path = model_path
        self.model = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.sentiment_encoder = None
        self.feature_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def build_vocabulary(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Create vocabulary
        vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count >= min_freq]
        self.vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_vocab = {idx: word for word, idx in self.vocab_to_idx.items()}
        
        return len(vocab)
    
    def tokenize(self, text):
        """Simple tokenization"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def prepare_data(self, csv_path="data/electronics_balanced_10k.csv"):
        """Load and prepare data from CSV"""
        print(f"📊 Loading dataset from: {csv_path}")
        
        # Check if file exists and print info
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            # Try alternative path
            alt_path = csv_path.replace("data/", "../data/")
            if os.path.exists(alt_path):
                csv_path = alt_path
                print(f"✅ Found file at: {csv_path}")
            else:
                raise FileNotFoundError(f"Dataset not found at {csv_path} or {alt_path}")
        
        df = pd.read_csv(csv_path)
        print(f"📋 Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Map column names to expected format
        column_mapping = {
            'Review': 'review_text',
            'Sentiment': 'sentiment', 
            'Rate': 'rating',
            'category': 'feature_mentioned'  # Use category as feature
        }
        
        # Only rename columns that exist
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        print(f"📋 After column mapping: {list(df.columns)}")
        
        # Clean and prepare data
        df = df.dropna(subset=['review_text', 'sentiment', 'rating'])
        
        # Remove duplicate reviews to prevent data leakage
        print(f"📋 Original dataset size: {len(df)}")
        df = df.drop_duplicates(subset=['review_text'], keep='first')
        print(f"📋 After removing duplicates: {len(df)}")
        
        # Filter out very short reviews
        df = df[df['review_text'].str.len() > 5]
        print(f"📋 After filtering short reviews: {len(df)}")
        
        # Check dataset size
        if len(df) < 100:
            print("⚠️ Warning: Very small dataset! Results may not be reliable.")
        
        # Print data distribution
        print(f"📊 Sentiment distribution:")
        for sentiment, count in df['sentiment'].value_counts().items():
            print(f"   {sentiment}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"📊 Category distribution:")
        if 'feature_mentioned' in df.columns:
            try:
                category_counts = df['feature_mentioned'].value_counts().head()
                for category, count in category_counts.items():
                    print(f"   {category}: {count}")
            except Exception as e:
                print(f"   Error displaying categories: {e}")
        else:
            print("   No feature_mentioned column found")
        
        # Encode labels
        self.sentiment_encoder = LabelEncoder()
        df['sentiment_encoded'] = self.sentiment_encoder.fit_transform(df['sentiment'])
        
        self.feature_encoder = LabelEncoder()
        df['feature_encoded'] = self.feature_encoder.fit_transform(df['feature_mentioned'])
        
        # Build vocabulary with adaptive min_freq
        min_freq = max(2, len(df) // 500)  # Adaptive min_freq
        vocab_size = self.build_vocabulary(df['review_text'].tolist(), min_freq=min_freq)
        
        print(f"✅ Dataset prepared: {len(df)} unique reviews")
        print(f"📝 Vocabulary size: {vocab_size} (min_freq: {min_freq})")
        print(f"🎯 Sentiment classes: {len(self.sentiment_encoder.classes_)}")
        print(f"🔧 Feature classes: {len(self.feature_encoder.classes_)}")
        
        return df, vocab_size
    
    def create_data_loaders(self, df, batch_size=32, test_size=0.2):
        """Create train and test data loaders with proper validation"""
        print(f"📊 Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        
        # Adjust parameters based on dataset size
        if len(df) < 500:
            test_size = 0.3
            batch_size = min(batch_size, 16)
            print(f"⚠️ Small dataset detected. Using test_size={test_size}, batch_size={batch_size}")
        elif len(df) > 1000:
            batch_size = max(batch_size, 32)
            print(f"📈 Large dataset detected. Using batch_size={batch_size}")
        
        # Split data with stratification
        try:
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=42, 
                stratify=df['sentiment'],
                shuffle=True
            )
        except ValueError:
            # Fallback if stratification fails
            print("⚠️ Stratification failed, using random split")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
        
        print(f"📈 Train set: {len(train_df)} samples")
        print(f"📉 Test set: {len(test_df)} samples")
        
        # Create datasets
        train_dataset = ProductReviewDataset(
            train_df['review_text'].tolist(),
            train_df['sentiment_encoded'].tolist(),
            train_df['rating'].tolist(),
            train_df['feature_encoded'].tolist(),
            self.vocab_to_idx
        )
        
        test_dataset = ProductReviewDataset(
            test_df['review_text'].tolist(),
            test_df['sentiment_encoded'].tolist(),
            test_df['rating'].tolist(),
            test_df['feature_encoded'].tolist(),
            self.vocab_to_idx
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, csv_path="data/electronics_balanced_10k.csv", epochs=30, batch_size=32, lr=0.001, 
                   patience=5, min_delta=0.005):
        """Train the 1D CNN model with early stopping"""
        print("🚀 Starting 1D CNN model training with early stopping...")
        
        # Prepare data
        df, vocab_size = self.prepare_data(csv_path)
        train_loader, test_loader = self.create_data_loaders(df, batch_size)
        
        # Initialize model
        self.model = MultiTask1DCNNModel(
            vocab_size=vocab_size,
            num_sentiments=len(self.sentiment_encoder.classes_),
            num_features=len(self.feature_encoder.classes_)
        ).to(self.device)
        
        print(f"🧠 Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Loss functions and optimizer with L2 regularization
        criterion_sentiment = nn.CrossEntropyLoss()
        criterion_rating = nn.CrossEntropyLoss()
        criterion_feature = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        
        # Training loop
        train_losses = []
        test_accuracies = []
        best_epoch = 0
        
        print(f"📊 Training with early stopping (patience={patience}, min_delta={min_delta})")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch['text'].to(self.device))
                
                # Calculate losses
                loss_sentiment = criterion_sentiment(outputs['sentiment'], batch['sentiment'].to(self.device))
                loss_rating = criterion_rating(outputs['rating'], batch['rating'].to(self.device))
                loss_feature = criterion_feature(outputs['feature'], batch['feature'].to(self.device))
                
                # Combined loss with weights
                total_batch_loss = loss_sentiment + 0.5 * loss_rating + 0.3 * loss_feature
                
                # Backward pass
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            # Validation
            test_acc = self.evaluate_model(test_loader)
            train_losses.append(total_loss / len(train_loader))
            test_accuracies.append(test_acc)
            
            scheduler.step(test_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Test Acc: {test_acc:.4f}")
            
            # Early stopping check
            if early_stopping(test_acc, self.model):
                best_epoch = epoch + 1 - patience
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"✨ Best model was at epoch {best_epoch} with accuracy {early_stopping.best_score:.4f}")
                break
            
            # Update best epoch if this is the best so far
            if test_acc == early_stopping.best_score:
                best_epoch = epoch + 1
        
        # Save model (early stopping already restored best weights)
        self.save_model()
        
        # Plot training curves
        self.plot_training_curves(train_losses, test_accuracies, best_epoch)
        
        print(f"✅ Training completed! Best epoch: {best_epoch}, Best accuracy: {max(test_accuracies):.4f}")
        
    def evaluate_model(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model(batch['text'].to(self.device))
                predicted = torch.argmax(outputs['sentiment'], dim=1)
                total += batch['sentiment'].size(0)
                correct += (predicted == batch['sentiment'].to(self.device)).sum().item()
        
        return correct / total
    
    def predict(self, text):
        """Predict sentiment, rating, and features for a single text"""
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        # Tokenize and convert to indices
        tokens = self.tokenize(text)
        indices = [self.vocab_to_idx.get(token, self.vocab_to_idx['<UNK>']) for token in tokens]
        
        # Pad or truncate
        max_length = 128
        if len(indices) < max_length:
            indices.extend([self.vocab_to_idx['<PAD>']] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]
        
        # Convert to tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Get predictions
            sentiment_pred = torch.argmax(outputs['sentiment'], dim=1).cpu().numpy()[0]
            rating_pred = torch.argmax(outputs['rating'], dim=1).cpu().numpy()[0] + 1  # Convert back to 1-5
            feature_pred = torch.argmax(outputs['feature'], dim=1).cpu().numpy()[0]
            
            # Get confidence scores
            sentiment_conf = torch.softmax(outputs['sentiment'], dim=1).max().cpu().numpy()
            rating_conf = torch.softmax(outputs['rating'], dim=1).max().cpu().numpy()
            feature_conf = torch.softmax(outputs['feature'], dim=1).max().cpu().numpy()
        
        return {
            'sentiment': self.sentiment_encoder.inverse_transform([sentiment_pred])[0],
            'sentiment_confidence': float(sentiment_conf),
            'predicted_rating': int(rating_pred),
            'rating_confidence': float(rating_conf),
            'predicted_feature': self.feature_encoder.inverse_transform([feature_pred])[0],
            'feature_confidence': float(feature_conf)
        }
    
    def save_model(self):
        """Save the trained model and encoders"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab_to_idx': self.vocab_to_idx,
            'sentiment_encoder': self.sentiment_encoder,
            'feature_encoder': self.feature_encoder
        }, self.model_path)
        
        print(f"✅ Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.vocab_to_idx = checkpoint['vocab_to_idx']
        self.sentiment_encoder = checkpoint['sentiment_encoder']
        self.feature_encoder = checkpoint['feature_encoder']
        
        # Initialize model with correct parameters
        vocab_size = len(self.vocab_to_idx)
        num_sentiments = len(self.sentiment_encoder.classes_)
        num_features = len(self.feature_encoder.classes_)
        
        self.model = MultiTask1DCNNModel(
            vocab_size=vocab_size,
            num_sentiments=num_sentiments,
            num_features=num_features
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def plot_training_curves(self, train_losses, test_accuracies, best_epoch=None):
        """Plot training curves with early stopping indicator"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        if best_epoch:
            plt.axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        plt.title('1D CNN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies, label='Test Accuracy')
        if best_epoch:
            plt.axvline(x=best_epoch-1, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        plt.title('1D CNN Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cnn1d_training_curves.png', dpi=300, bbox_inches='tight')
        print("📊 Training curves saved as 'cnn1d_training_curves.png'")
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ProductReview1DCNNAnalyzer()
    
    # Train model with early stopping (optimized for electronics dataset)
    analyzer.train_model(
        csv_path="data/electronics_balanced_10k.csv",  # Use the larger dataset
        epochs=40,  # More epochs for CNN
        batch_size=64,  # Larger batch size for CNN
        lr=0.001,  # Learning rate
        patience=7,  # Patience for early stopping
        min_delta=0.005  # Minimum improvement threshold
    )
    
    # Test predictions
    test_reviews = [
        "This smartphone has amazing battery life and great camera quality!",
        "Poor build quality, the screen broke after one week",
        "Average product, nothing special but works fine",
        "Excellent headphones with superb sound quality",
        "Terrible customer service and defective product"
    ]
    
    print("\n🧪 Testing 1D CNN predictions:")
    for review in test_reviews:
        result = analyzer.predict(review)
        print(f"\nReview: {review}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.3f})")
        print(f"Predicted Rating: {result['predicted_rating']}/5 (confidence: {result['rating_confidence']:.3f})")
        print(f"Key Feature: {result['predicted_feature']} (confidence: {result['feature_confidence']:.3f})")
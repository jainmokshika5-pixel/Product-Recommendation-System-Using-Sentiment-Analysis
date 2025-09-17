"""
Training script for CNN+BiLSTM sentiment analysis model
This script would train the model on your dataset and save it for inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from collections import Counter
import re
from sentiment_model import CNNBiLSTMSentimentModel

class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        return torch.LongTensor(tokens), torch.LongTensor([label])
    
    def preprocess_text(self, text):
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

def create_tokenizer(texts, vocab_size=10000):
    """Create tokenizer from texts"""
    # Count word frequencies
    word_freq = Counter()
    for text in texts:
        text = re.sub(r'[^\w\s]', ' ', str(text).lower())
        words = text.split()
        word_freq.update(words)
    
    # Create vocabulary
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab

def load_and_preprocess_data():
    """Load and preprocess training data"""
    data_path = "data/Dataset-SA.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Training data not found at {data_path}")
        print("📁 Please place Dataset-SA.csv in the data/ folder")
        print("   Expected columns: review_text, sentiment")
        return None, None, None, None, None
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        
        if 'review_text' not in df.columns or 'sentiment' not in df.columns:
            print("❌ Required columns 'review_text' and 'sentiment' not found")
            return None, None, None, None, None
        
        # Clean data
        df = df.dropna(subset=['review_text', 'sentiment'])
        
        # Map sentiment labels to numbers
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        if not all(sentiment in sentiment_map for sentiment in df['sentiment'].unique()):
            print("❌ Sentiment labels should be 'positive', 'neutral', 'negative'")
            return None, None, None, None, None
        
        df['sentiment_label'] = df['sentiment'].map(sentiment_map)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['review_text'].values,
            df['sentiment_label'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['sentiment_label'].values
        )
        
        # Create tokenizer
        tokenizer = create_tokenizer(X_train)
        
        print(f"✅ Loaded {len(df)} samples")
        print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"📝 Vocabulary size: {len(tokenizer)}")
        
        return X_train, X_test, y_train, y_test, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None, None

def train_model():
    """Train the sentiment analysis model"""
    print("🚀 Starting model training...")
    
    # Load data
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data()
    
    if X_train is None:
        print("❌ Cannot proceed without training data")
        return False
    
    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = CNNBiLSTMSentimentModel(vocab_size=len(tokenizer))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"🔥 Training on {device}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.squeeze().to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    # Save model and tokenizer
    os.makedirs('models', exist_ok=True)
    
    torch.save(model.state_dict(), 'models/sentiment_model.pth')
    
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("✅ Model training completed!")
    print("💾 Model saved to models/sentiment_model.pth")
    print("💾 Tokenizer saved to models/tokenizer.pkl")
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.squeeze().to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    label_names = ['Negative', 'Neutral', 'Positive']
    print("\n📊 Final Model Performance:")
    print(classification_report(all_labels, all_predictions, target_names=label_names))
    
    return True

if __name__ == "__main__":
    success = train_model()
    if success:
        print("\n🎉 Training completed successfully!")
        print("   Run the FastAPI server with: uvicorn main:app --reload")
    else:
        print("\n❌ Training failed. Please check your data and try again.")
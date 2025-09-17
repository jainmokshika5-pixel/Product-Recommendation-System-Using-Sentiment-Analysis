#!/usr/bin/env python3
"""
Test script to verify the new datasets are loaded correctly
"""

import pandas as pd
import os
import sys

def test_datasets():
    """Test loading and basic statistics of the datasets"""
    print("🧪 Testing dataset loading...")
    
    datasets = []
    
    # Test electronics_balanced_10k.csv
    electronics_path = "data/electronics_balanced_10k.csv"
    if os.path.exists(electronics_path):
        df_electronics = pd.read_csv(electronics_path)
        print(f"\n📱 Electronics Dataset:")
        print(f"   Rows: {len(df_electronics)}")
        print(f"   Columns: {list(df_electronics.columns)}")
        print(f"   Categories: {sorted(df_electronics['category'].unique())}")
        print(f"   Sentiments: {sorted(df_electronics['Sentiment'].unique())}")
        
        # Sample data
        print(f"\n   Sample review:")
        sample = df_electronics.iloc[0]
        print(f"   Product: {sample['product_name'][:50]}...")
        print(f"   Category: {sample['category']}")
        print(f"   Review: {sample['Review']}")
        print(f"   Summary: {sample['Summary']}")
        print(f"   Sentiment: {sample['Sentiment']}")
        
        datasets.append("electronics")
    else:
        print(f"❌ Electronics dataset not found at {electronics_path}")
    
    # Test product_reviews.csv
    reviews_path = "data/product_reviews.csv"
    if os.path.exists(reviews_path):
        df_reviews = pd.read_csv(reviews_path)
        print(f"\n📝 Product Reviews Dataset:")
        print(f"   Rows: {len(df_reviews)}")
        print(f"   Columns: {list(df_reviews.columns)}")
        print(f"   Categories: {sorted(df_reviews['category'].unique())}")
        print(f"   Sentiments: {sorted(df_reviews['sentiment'].unique())}")
        
        # Sample data
        print(f"\n   Sample review:")
        sample = df_reviews.iloc[0]
        print(f"   Product: {sample['product']}")
        print(f"   Category: {sample['category']}")
        print(f"   Review: {sample['review_text']}")
        print(f"   Sentiment: {sample['sentiment']}")
        
        datasets.append("reviews")
    else:
        print(f"❌ Product reviews dataset not found at {reviews_path}")
    
    # Test sample dataset
    sample_path = "data/sample_Dataset-SA.csv"
    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path)
        print(f"\n📊 Sample Dataset:")
        print(f"   Rows: {len(df_sample)}")
        print(f"   Columns: {list(df_sample.columns)}")
        print(f"   Sentiments: {sorted(df_sample['sentiment'].unique())}")
        datasets.append("sample")
    else:
        print(f"❌ Sample dataset not found at {sample_path}")
    
    if datasets:
        print(f"\n✅ Found {len(datasets)} datasets: {', '.join(datasets)}")
        print("🚀 Ready for training!")
        return True
    else:
        print("\n❌ No datasets found for training")
        return False

if __name__ == "__main__":
    success = test_datasets()
    sys.exit(0 if success else 1)
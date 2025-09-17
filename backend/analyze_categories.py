#!/usr/bin/env python3
"""
Analyze categories across all datasets to ensure comprehensive coverage
"""

import pandas as pd
import os
from collections import Counter

def analyze_categories():
    """Analyze categories and sentiments across all datasets"""
    print("🔍 Analyzing categories across datasets...")
    
    all_categories = set()
    all_sentiments = set()
    datasets_info = {}
    
    # Analyze electronics_balanced_10k.csv
    electronics_path = "data/electronics_balanced_10k.csv"
    if os.path.exists(electronics_path):
        df = pd.read_csv(electronics_path)
        categories = set(df['category'].unique())
        sentiments = set(df['Sentiment'].str.lower().unique())
        
        all_categories.update(categories)
        all_sentiments.update(sentiments)
        
        datasets_info['electronics'] = {
            'rows': len(df),
            'categories': categories,
            'sentiments': sentiments,
            'category_dist': df['category'].value_counts().to_dict(),
            'sentiment_dist': df['Sentiment'].str.lower().value_counts().to_dict()
        }
        
        print(f"\n📱 Electronics Dataset ({len(df)} samples):")
        print(f"   Categories ({len(categories)}): {sorted(categories)}")
        print(f"   Category distribution:")
        for cat, count in df['category'].value_counts().head(10).items():
            print(f"      {cat}: {count}")
    
    # Analyze product_reviews.csv
    reviews_path = "data/product_reviews.csv"
    if os.path.exists(reviews_path):
        df = pd.read_csv(reviews_path)
        categories = set(df['category'].unique())
        sentiments = set(df['sentiment'].str.lower().unique())
        
        all_categories.update(categories)
        all_sentiments.update(sentiments)
        
        datasets_info['reviews'] = {
            'rows': len(df),
            'categories': categories,
            'sentiments': sentiments,
            'category_dist': df['category'].value_counts().to_dict(),
            'sentiment_dist': df['sentiment'].str.lower().value_counts().to_dict()
        }
        
        print(f"\n📝 Product Reviews Dataset ({len(df)} samples):")
        print(f"   Categories ({len(categories)}): {sorted(categories)}")
        print(f"   Category distribution:")
        for cat, count in df['category'].value_counts().items():
            print(f"      {cat}: {count}")
    
    # Combined analysis
    print(f"\n🌐 Combined Analysis:")
    print(f"   Total unique categories: {len(all_categories)}")
    print(f"   All categories: {sorted(all_categories)}")
    print(f"   All sentiments: {sorted(all_sentiments)}")
    
    # Category mapping for consistency
    category_mapping = {
        # Electronics dataset categories
        'Mobile': 'Smartphones',
        'Laptop': 'Laptops', 
        'Headphone': 'Audio',
        'Speaker': 'Audio',
        'TV': 'Smart Home',
        'Watch': 'Wearables',
        'Cooler/AC': 'Appliances',
        'Other': 'Other',
        
        # Product reviews categories (already normalized)
        'Smartphones': 'Smartphones',
        'Laptops': 'Laptops',
        'Audio': 'Audio',
        'Smart Home': 'Smart Home',
        'Wearables': 'Wearables'
    }
    
    print(f"\n🗂️ Suggested category mapping:")
    for original, mapped in category_mapping.items():
        print(f"   {original} → {mapped}")
    
    return datasets_info, all_categories, all_sentiments

if __name__ == "__main__":
    datasets_info, categories, sentiments = analyze_categories()
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Ready to train on {len(categories)} categories")
    print(f"💭 Sentiment classes: {sorted(sentiments)}")
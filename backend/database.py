"""
Database management for products and reviews
Handles CSV loading and provides data access methods
"""

import pandas as pd
import sqlite3
import os
from typing import List, Dict, Any, Optional
import logging

class DatabaseManager:
    def __init__(self, db_path="database.db"):
        self.db_path = db_path
        self.data_folder = "data"
        self.products_file = "product_reviews.csv"
        self.reviews_file = "Dataset-SA.csv"
        self.is_data_loaded = False
        
        # Create database connection
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                description TEXT
            )
        ''')
        
        # Reviews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                review_text TEXT NOT NULL,
                sentiment TEXT,
                sentiment_score REAL,
                FOREIGN KEY (product_id) REFERENCES products (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_sample_data(self):
        """Load sample data if CSV files are not available"""
        print("📁 CSV files not found, loading sample data...")
        
        # Sample products
        sample_products = [
            {
                'id': 1,
                'name': 'Apple iPhone 15 Pro (128GB) - Natural Titanium',
                'category': 'Smartphone',
                'price': 134900.0,
                'description': 'A17 Pro chip with 6-core GPU, Advanced camera system, Action Button'
            },
            {
                'id': 2,
                'name': 'Samsung Galaxy S24+ 5G (256GB) - Onyx Black',
                'category': 'Smartphone',
                'price': 89999.0,
                'description': '50MP Triple Camera, 6.7-inch Dynamic AMOLED 2X Display, 4900mAh Battery'
            },
            {
                'id': 3,
                'name': 'Apple MacBook Pro (14-inch, M3, 8GB RAM, 512GB SSD)',
                'category': 'Laptop',
                'price': 169900.0,
                'description': 'M3 chip with 8-core CPU, 10-core GPU, 14-inch Liquid Retina XDR display'
            },
            {
                'id': 4,
                'name': 'ASUS Vivobook 15 (Intel Core i5-12th Gen, 8GB RAM, 512GB SSD)',
                'category': 'Laptop',
                'price': 54990.0,
                'description': '15.6-inch FHD Display, Windows 11 Home, MS Office 2021, Silver'
            },
            {
                'id': 5,
                'name': 'OnePlus Buds Pro 2 True Wireless Earbuds',
                'category': 'Speakers',
                'price': 11999.0,
                'description': 'Spatial Audio, Smart Adaptive Noise Cancellation, 39H Playtime'
            },
            {
                'id': 6,
                'name': 'Sony WH-1000XM5 Wireless Noise Canceling Headphones',
                'category': 'Speakers',
                'price': 29990.0,
                'description': 'Industry Leading Noise Cancellation, 30Hr Battery, Quick Charge'
            }
        ]
        
        # Sample reviews with realistic sentiment
        sample_reviews = [
            # iPhone 15 Pro reviews
            {'product_id': 1, 'review_text': 'Amazing camera quality and battery life is excellent. Love the titanium design!', 'sentiment': 'positive', 'sentiment_score': 8.5},
            {'product_id': 1, 'review_text': 'Great phone but very expensive. Camera is top notch though.', 'sentiment': 'neutral', 'sentiment_score': 6.5},
            {'product_id': 1, 'review_text': 'Best iPhone ever! Fast performance and beautiful display.', 'sentiment': 'positive', 'sentiment_score': 9.0},
            
            # Samsung Galaxy S24+ reviews
            {'product_id': 2, 'review_text': 'Incredible camera with great night mode. Display is vibrant and smooth.', 'sentiment': 'positive', 'sentiment_score': 8.8},
            {'product_id': 2, 'review_text': 'Good phone but battery drains quickly with heavy usage.', 'sentiment': 'neutral', 'sentiment_score': 6.0},
            {'product_id': 2, 'review_text': 'Love the curved display and 100W charging. Very impressive value!', 'sentiment': 'positive', 'sentiment_score': 8.2},
            
            # MacBook Pro M3 reviews
            {'product_id': 3, 'review_text': 'Blazing fast performance with M3 chip. Display is stunning and battery lasts all day.', 'sentiment': 'positive', 'sentiment_score': 9.2},
            {'product_id': 3, 'review_text': 'Excellent build quality and performance but quite expensive for Indian market.', 'sentiment': 'neutral', 'sentiment_score': 7.0},
            {'product_id': 3, 'review_text': 'Perfect for video editing and development work. Highly recommended!', 'sentiment': 'positive', 'sentiment_score': 8.9},
            
            # ASUS Vivobook reviews
            {'product_id': 4, 'review_text': 'Great laptop for the price. Good performance and comes with MS Office.', 'sentiment': 'positive', 'sentiment_score': 7.5},
            {'product_id': 4, 'review_text': 'Nice laptop but fan gets loud under heavy load. Build quality is decent.', 'sentiment': 'neutral', 'sentiment_score': 6.8},
            {'product_id': 4, 'review_text': 'Solid choice for students and office work. Great value for money.', 'sentiment': 'positive', 'sentiment_score': 7.8},
            
            # OnePlus Buds Pro 2 reviews
            {'product_id': 5, 'review_text': 'Excellent sound quality and ANC. Great alternative to AirPods Pro.', 'sentiment': 'positive', 'sentiment_score': 8.7},
            {'product_id': 5, 'review_text': 'Good earbuds but fit could be better for smaller ears.', 'sentiment': 'neutral', 'sentiment_score': 6.8},
            {'product_id': 5, 'review_text': 'Love the spatial audio and long battery life. Worth the price!', 'sentiment': 'positive', 'sentiment_score': 8.4},
            
            # Sony WH-1000XM5 reviews
            {'product_id': 6, 'review_text': 'Best noise canceling headphones! Sound quality is incredible and comfortable to wear.', 'sentiment': 'positive', 'sentiment_score': 9.1},
            {'product_id': 6, 'review_text': 'Great sound but price is high for Indian market. Noise canceling works very well.', 'sentiment': 'neutral', 'sentiment_score': 7.2},
            {'product_id': 6, 'review_text': 'Amazing for music and calls. Battery life is excellent too!', 'sentiment': 'positive', 'sentiment_score': 8.6}
        ]
        
        # Insert sample data into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM reviews')
        cursor.execute('DELETE FROM products')
        
        # Insert products
        for product in sample_products:
            cursor.execute('''
                INSERT INTO products (id, name, category, price, description)
                VALUES (?, ?, ?, ?, ?)
            ''', (product['id'], product['name'], product['category'], 
                  product['price'], product['description']))
        
        # Insert reviews
        for review in sample_reviews:
            cursor.execute('''
                INSERT INTO reviews (product_id, review_text, sentiment, sentiment_score)
                VALUES (?, ?, ?, ?)
            ''', (review['product_id'], review['review_text'], 
                  review['sentiment'], review['sentiment_score']))
        
        conn.commit()
        conn.close()
        
        self.is_data_loaded = True
        print(f"✅ Loaded {len(sample_products)} sample products and {len(sample_reviews)} sample reviews")
    
    def load_data_from_csv(self):
        """Load data from CSV files if available"""
        products_path = os.path.join(self.data_folder, self.products_file)
        reviews_path = os.path.join(self.data_folder, self.reviews_file)
        
        if not os.path.exists(products_path) or not os.path.exists(reviews_path):
            return False
        
        try:
            # Load products CSV
            products_df = pd.read_csv(products_path)
            required_product_cols = ['id', 'name', 'category', 'price', 'description']
            if not all(col in products_df.columns for col in required_product_cols):
                print(f"❌ Products CSV missing required columns: {required_product_cols}")
                return False
            
            # Load reviews CSV
            reviews_df = pd.read_csv(reviews_path)
            required_review_cols = ['product_id', 'review_text', 'sentiment']
            if not all(col in reviews_df.columns for col in required_review_cols):
                print(f"❌ Reviews CSV missing required columns: {required_review_cols}")
                return False
            
            # Insert data into database
            conn = sqlite3.connect(self.db_path)
            
            # Clear existing data
            conn.execute('DELETE FROM reviews')
            conn.execute('DELETE FROM products')
            
            # Insert products
            products_df.to_sql('products', conn, if_exists='append', index=False)
            
            # Process reviews and add sentiment scores
            if 'sentiment_score' not in reviews_df.columns:
                # Generate sentiment scores based on sentiment labels
                sentiment_score_map = {'positive': 8.0, 'neutral': 5.0, 'negative': 2.5}
                reviews_df['sentiment_score'] = reviews_df['sentiment'].map(sentiment_score_map)
            
            # Insert reviews
            reviews_df.to_sql('reviews', conn, if_exists='append', index=False)
            
            conn.commit()
            conn.close()
            
            self.is_data_loaded = True
            print(f"✅ Loaded {len(products_df)} products and {len(reviews_df)} reviews from CSV files")
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return False
    
    def load_data(self):
        """Load data from CSV files or use sample data as fallback"""
        # Try to load from CSV first
        if not self.load_data_from_csv():
            # Fallback to sample data
            self.load_sample_data()
    
    def get_all_products(self) -> List[Dict[str, Any]]:
        """Get all products from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM products')
        rows = cursor.fetchall()
        
        products = []
        for row in rows:
            products.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'price': row[3],
                'description': row[4]
            })
        
        conn.close()
        return products
    
    def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get products filtered by category"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM products WHERE LOWER(category) = LOWER(?)', (category,))
        rows = cursor.fetchall()
        
        products = []
        for row in rows:
            products.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'price': row[3],
                'description': row[4]
            })
        
        conn.close()
        return products
    
    def get_reviews_by_product(self, product_id: int) -> List[Dict[str, Any]]:
        """Get all reviews for a specific product"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, product_id, review_text, sentiment, sentiment_score 
            FROM reviews WHERE product_id = ?
        ''', (product_id,))
        rows = cursor.fetchall()
        
        reviews = []
        for row in rows:
            reviews.append({
                'id': row[0],
                'product_id': row[1],
                'review_text': row[2],
                'sentiment': row[3],
                'sentiment_score': row[4] or 5.0
            })
        
        conn.close()
        return reviews
    
    def is_loaded(self) -> bool:
        """Check if data is loaded"""
        return self.is_data_loaded
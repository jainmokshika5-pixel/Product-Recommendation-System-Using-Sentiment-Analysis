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
        """Load data from the comprehensive datasets"""
        datasets_loaded = False
        
        # Load electronics_balanced_10k.csv
        electronics_path = "data/electronics_balanced_10k.csv"
        if os.path.exists(electronics_path):
            try:
                df = pd.read_csv(electronics_path)
                print(f"📱 Found electronics dataset with {len(df)} products")
                
                # Process products from electronics dataset
                products_data = []
                reviews_data = []
                
                for idx, row in df.iterrows():
                    # Create product entry
                    product_id = idx + 1
                    
                    # Map categories to standardized names
                    category_map = {
                        'Mobile': 'Smartphones',
                        'Laptop': 'Laptops',
                        'Headphone': 'Audio',
                        'Speaker': 'Audio', 
                        'TV': 'Smart Home',
                        'Watch': 'Wearables',
                        'Cooler/AC': 'Appliances',
                        'Other': 'Other'
                    }
                    
                    standardized_category = category_map.get(row['category'], row['category'])
                    
                    products_data.append({
                        'id': product_id,
                        'name': str(row['product_name'])[:100],  # Truncate long names
                        'category': standardized_category,
                        'price': float(row['product_price']),
                        'description': str(row['Summary']) if pd.notna(row['Summary']) else 'No description available'
                    })
                    
                    # Create review entry
                    review_text = f"{row['Review']} {row['Summary']}" if pd.notna(row['Summary']) else str(row['Review'])
                    
                    # Map sentiment and calculate score
                    sentiment = str(row['Sentiment']).lower()
                    if sentiment == 'positive':
                        base_score = 7.0 + (float(row['Rate']) - 3) * 0.5  # 7-9 range
                    elif sentiment == 'negative':
                        base_score = 2.0 + (float(row['Rate']) - 1) * 0.5  # 2-4 range  
                    else:
                        base_score = 4.5 + (float(row['Rate']) - 3) * 0.3  # 4.5-6 range
                    
                    sentiment_score = max(1.0, min(10.0, base_score))
                    
                    reviews_data.append({
                        'product_id': product_id,
                        'review_text': review_text,
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score
                    })
                
                datasets_loaded = True
                print(f"✅ Processed {len(products_data)} products from electronics dataset")
                
            except Exception as e:
                print(f"⚠️ Error loading electronics dataset: {e}")
        
        # Load product_reviews.csv 
        reviews_path = "data/product_reviews.csv"
        if os.path.exists(reviews_path):
            try:
                df_reviews = pd.read_csv(reviews_path)
                print(f"📝 Found product reviews dataset with {len(df_reviews)} reviews")
                
                # Get unique products from reviews dataset
                unique_products = df_reviews.drop_duplicates(subset=['product'])
                start_id = len(products_data) + 1 if datasets_loaded else 1
                
                # Add products from reviews dataset
                for idx, row in unique_products.iterrows():
                    product_id = start_id + len([p for p in unique_products.itertuples() if p.Index <= idx])
                    
                    products_data.append({
                        'id': product_id,
                        'name': str(row['product']),
                        'category': str(row['category']),
                        'price': 50000.0,  # Default price since not provided
                        'description': f"High-quality {row['category'].lower()} product with advanced features"
                    })
                    
                    # Add all reviews for this product
                    product_reviews = df_reviews[df_reviews['product'] == row['product']]
                    for _, review_row in product_reviews.iterrows():
                        sentiment = str(review_row['sentiment']).lower()
                        rating = float(review_row['rating'])
                        
                        # Calculate sentiment score from rating
                        if sentiment == 'positive':
                            sentiment_score = 6.0 + rating * 0.8  # 6-10 range
                        elif sentiment == 'negative':
                            sentiment_score = 1.0 + rating * 0.6  # 1-4 range
                        else:
                            sentiment_score = 4.0 + rating * 0.4  # 4-6 range
                        
                        reviews_data.append({
                            'product_id': product_id,
                            'review_text': str(review_row['review_text']),
                            'sentiment': sentiment,
                            'sentiment_score': max(1.0, min(10.0, sentiment_score))
                        })
                
                datasets_loaded = True
                print(f"✅ Added {len(unique_products)} products from reviews dataset")
                
            except Exception as e:
                print(f"⚠️ Error loading product reviews dataset: {e}")
        
        if not datasets_loaded:
            print("❌ No comprehensive datasets found")
            return False
        
        try:
            # Insert data into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute('DELETE FROM reviews')
            cursor.execute('DELETE FROM products')
            
            # Insert products
            for product in products_data:
                cursor.execute('''
                    INSERT INTO products (id, name, category, price, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (product['id'], product['name'], product['category'], 
                      product['price'], product['description']))
            
            # Insert reviews
            for review in reviews_data:
                cursor.execute('''
                    INSERT INTO reviews (product_id, review_text, sentiment, sentiment_score)
                    VALUES (?, ?, ?, ?)
                ''', (review['product_id'], review['review_text'], 
                      review['sentiment'], review['sentiment_score']))
            
            conn.commit()
            conn.close()
            
            self.is_data_loaded = True
            print(f"✅ Database loaded successfully!")
            print(f"📊 Total products: {len(products_data)}")
            print(f"📝 Total reviews: {len(reviews_data)}")
            
            # Print category breakdown
            category_counts = {}
            for product in products_data:
                cat = product['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print(f"📈 Categories loaded:")
            for category, count in sorted(category_counts.items()):
                print(f"   {category}: {count} products")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inserting data into database: {e}")
            return False
    
    def load_data(self):
        """Load data from comprehensive datasets or use sample data as fallback"""
        # Try to load from comprehensive datasets first
        if not self.load_data_from_csv():
            # Fallback to sample data only if comprehensive datasets fail
            print("📁 Comprehensive datasets not available, using sample data...")
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
"""
Database management for products and reviews
Handles CSV loading and provides data access methods
"""

import pandas as pd
import sqlite3
import os
from typing import List, Dict, Any

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
    
    # Sample data loading function removed - using only CSV data
    
    def load_data_from_csv(self):
        """Load data from the comprehensive datasets"""
        datasets_loaded = False
        products_data = []
        reviews_data = []
        
        # Load product_reviews.csv FIRST (has actual smartphones)
        reviews_path = "../data/product_reviews.csv"
        if os.path.exists(reviews_path):
            try:
                df_reviews = pd.read_csv(reviews_path)
                print(f"📝 Found product reviews dataset with {len(df_reviews)} reviews")
                
                # Get unique products from reviews dataset
                unique_products = df_reviews.drop_duplicates(subset=['product'])
                
                # Add products from reviews dataset
                for idx, row in unique_products.iterrows():
                    product_id = len(products_data) + 1
                    
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
        
        # Load electronics_balanced_10k.csv SECOND (has accessories)
        electronics_path = "../data/electronics_balanced_10k.csv"
        if os.path.exists(electronics_path):
            try:
                df = pd.read_csv(electronics_path)
                print(f"📱 Found electronics dataset with {len(df)} products")
                
                for idx, row in df.iterrows():
                    # Create product entry
                    product_id = len(products_data) + 1
                    
                    # Map categories to standardized names with better filtering
                    product_name = str(row['product_name']).lower()
                    
                    # Determine category based on product name for more accuracy
                    # Check accessories FIRST to avoid misclassifying mobile accessories as smartphones
                    if 'charger' in product_name or 'holder' in product_name or 'case' in product_name or 'cable' in product_name or 'adapter' in product_name or 'stand' in product_name:
                        standardized_category = 'Accessories'
                    elif 'mobile' in product_name or 'phone' in product_name or 'smartphone' in product_name or 'iphone' in product_name or 'samsung' in product_name or 'oneplus' in product_name:
                        standardized_category = 'Smartphones'
                    elif 'laptop' in product_name or 'notebook' in product_name or 'macbook' in product_name or 'computer' in product_name:
                        standardized_category = 'Laptops'
                    elif 'headphone' in product_name or 'earphone' in product_name or 'earbud' in product_name or 'airpods' in product_name:
                        standardized_category = 'Audio'
                    elif 'speaker' in product_name or 'sound' in product_name:
                        standardized_category = 'Audio'
                    elif 'watch' in product_name or 'smartwatch' in product_name or 'fitness' in product_name:
                        standardized_category = 'Wearables'
                    elif 'wifi' in product_name or 'router' in product_name or 'extender' in product_name or 'network' in product_name:
                        standardized_category = 'Smart Home'
                    elif 'tv' in product_name or 'television' in product_name or 'monitor' in product_name:
                        standardized_category = 'Smart Home'
                    elif 'cooler' in product_name or 'ac' in product_name or 'air conditioner' in product_name:
                        standardized_category = 'Appliances'
                    else:
                        # Fallback to original category mapping
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
                        standardized_category = category_map.get(row['category'], 'Other')
                    
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
        """Load data from comprehensive datasets only"""
        # Load from comprehensive datasets - no fallback to sample data
        if not self.load_data_from_csv():
            print("❌ No comprehensive datasets found. Please ensure CSV files are available in the data/ folder.")
            raise FileNotFoundError("Required CSV datasets not found. Please add electronics_balanced_10k.csv and product_reviews.csv to the data/ folder.")
    
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

if __name__ == "__main__":
    # Initialize database manager
    print("🔄 Initializing DatabaseManager...")
    db = DatabaseManager()
    
    # Load data
    print("📥 Loading data...")
    db.load_data()
    
    # Test database queries
    print("\n🔍 Testing database queries:")
    print(f"Total products: {len(db.get_all_products())}")
    
    # Test category queries
    categories = ["Smartphone", "Laptop", "Speakers"]
    for category in categories:
        products = db.get_products_by_category(category)
        print(f"\n📱 {category} products: {len(products)}")
        for product in products:
            print(f"  - {product['name']}")
            reviews = db.get_reviews_by_product(product['id'])
            print(f"    Reviews: {len(reviews)}")
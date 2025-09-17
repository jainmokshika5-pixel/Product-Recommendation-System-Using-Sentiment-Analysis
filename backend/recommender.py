"""
Product recommendation system using sentiment analysis and feature matching
Implements MLP-based ranking for product recommendations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any
from database import DatabaseManager

class MLPRecommender(nn.Module):
    """Simple MLP for product ranking"""
    
    def __init__(self, input_dim=10):
        super(MLPRecommender, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class ProductRecommender:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.model = MLPRecommender()
        
        # Feature importance weights
        self.feature_weights = {
            'battery': 0.15,
            'performance': 0.20,
            'camera': 0.15,
            'display': 0.12,
            'audio': 0.10,
            'design': 0.08,
            'price': 0.12,
            'storage': 0.05,
            'connectivity': 0.03
        }
    
    def calculate_feature_scores(self, product: Dict, reviews: List[Dict]) -> Dict[str, float]:
        """Calculate sentiment scores for each feature based on reviews"""
        feature_scores = {}
        
        for feature in self.feature_weights.keys():
            relevant_reviews = []
            
            # Find reviews mentioning this feature
            for review in reviews:
                review_text = review.get('review_text', '').lower()
                feature_keywords = self.get_feature_keywords(feature)
                
                if any(keyword in review_text for keyword in feature_keywords):
                    relevant_reviews.append(review)
            
            if relevant_reviews:
                # Calculate average sentiment score for this feature
                total_score = sum(review.get('sentiment_score', 5.0) for review in relevant_reviews)
                feature_scores[feature] = total_score / len(relevant_reviews)
            else:
                # Default neutral score if no relevant reviews
                feature_scores[feature] = 5.0
        
        return feature_scores
    
    def get_feature_keywords(self, feature: str) -> List[str]:
        """Get keywords associated with each feature"""
        keywords = {
            'battery': ['battery', 'power', 'charging', 'lasting'],
            'performance': ['fast', 'speed', 'performance', 'lag', 'smooth'],
            'camera': ['camera', 'photo', 'picture', 'quality', 'lens'],
            'display': ['screen', 'display', 'bright', 'color', 'resolution'],
            'audio': ['sound', 'audio', 'music', 'speaker', 'volume'],
            'design': ['design', 'look', 'style', 'build', 'material'],
            'price': ['price', 'cost', 'value', 'money', 'expensive', 'cheap'],
            'storage': ['storage', 'memory', 'space', 'gb', 'capacity'],
            'connectivity': ['wifi', 'bluetooth', 'network', 'connection']
        }
        return keywords.get(feature, [])
    
    def calculate_category_match_score(self, product_category: str, target_category: str) -> float:
        """Calculate how well product category matches target category"""
        if product_category.lower() == target_category.lower():
            return 1.0
        
        # Partial matches
        category_similarities = {
            'smartphone': ['phone', 'mobile'],
            'laptop': ['computer', 'pc', 'notebook'],
            'smartwatch': ['watch', 'wearable'],
            'speakers': ['headphone', 'audio', 'earphone']
        }
        
        target_lower = target_category.lower()
        product_lower = product_category.lower()
        
        for main_cat, aliases in category_similarities.items():
            if (target_lower == main_cat or target_lower in aliases) and \
               (product_lower == main_cat or product_lower in aliases):
                return 0.8
        
        return 0.1  # Very low score for mismatched categories
    
    def calculate_recommendation_score(self, product: Dict, reviews: List[Dict], 
                                     target_category: str, target_features: List[str]) -> float:
        """Calculate overall recommendation score for a product"""
        
        # 1. Category match score (40% weight)
        category_score = self.calculate_category_match_score(
            product.get('category', ''), target_category
        ) * 0.4
        
        # 2. Feature-based sentiment scores (50% weight)
        feature_scores = self.calculate_feature_scores(product, reviews)
        
        feature_score = 0.0
        for feature in target_features:
            if feature in feature_scores:
                # Normalize from 1-10 scale to 0-1 scale
                normalized_score = (feature_scores[feature] - 1) / 9
                weight = self.feature_weights.get(feature, 0.1)
                feature_score += normalized_score * weight
        
        feature_score = min(feature_score, 0.5)  # Cap at 50% contribution
        
        # 3. Overall sentiment score (10% weight)
        if reviews:
            avg_sentiment = sum(r.get('sentiment_score', 5.0) for r in reviews) / len(reviews)
            sentiment_score = ((avg_sentiment - 1) / 9) * 0.1
        else:
            sentiment_score = 0.05  # Neutral default
        
        total_score = category_score + feature_score + sentiment_score
        return min(max(total_score, 0.0), 1.0)  # Ensure score is between 0 and 1
    
    def get_recommendations(self, query: str, category: str, features: List[str], 
                          top_n: int = 6) -> List[Dict[str, Any]]:
        """Get top N product recommendations"""
        
        # Get all products and reviews from database
        products = self.db_manager.get_products_by_category(category)
        
        if not products:
            # If no products in specific category, get all products
            products = self.db_manager.get_all_products()
        
        recommendations = []
        
        for product in products:
            # Get reviews for this product
            reviews = self.db_manager.get_reviews_by_product(product['id'])
            
            # Calculate recommendation score
            score = self.calculate_recommendation_score(product, reviews, category, features)
            
            # Calculate average sentiment for display
            if reviews:
                avg_sentiment_score = sum(r.get('sentiment_score', 5.0) for r in reviews) / len(reviews)
                if avg_sentiment_score >= 6.5:
                    sentiment_label = 'positive'
                elif avg_sentiment_score <= 4.5:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
            else:
                avg_sentiment_score = 5.0
                sentiment_label = 'neutral'
            
            # Create explanation
            explanation = self.generate_explanation(product, reviews, features, avg_sentiment_score)
            
            recommendations.append({
                'id': product['id'],
                'name': product['name'],
                'category': product['category'],
                'price': product['price'],
                'description': product['description'],
                'sentiment_score': round(avg_sentiment_score, 1),
                'sentiment_label': sentiment_label,
                'explanation': explanation,
                'recommendation_score': score
            })
        
        # Sort by recommendation score and return top N
        recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        # Remove the internal recommendation_score from the response
        for rec in recommendations[:top_n]:
            rec.pop('recommendation_score', None)
        
        return recommendations[:top_n]
    
    def generate_explanation(self, product: Dict, reviews: List[Dict], 
                           target_features: List[str], avg_sentiment: float) -> str:
        """Generate explanation for why this product is recommended"""
        
        explanations = []
        
        # Feature-based explanations
        feature_scores = self.calculate_feature_scores(product, reviews)
        
        for feature in target_features[:2]:  # Focus on top 2 features
            if feature in feature_scores:
                score = feature_scores[feature]
                if score >= 7.0:
                    explanations.append(f"Highly rated for {feature} (score: {score:.1f})")
                elif score >= 5.5:
                    explanations.append(f"Good {feature} reviews (score: {score:.1f})")
        
        # Overall sentiment explanation
        if avg_sentiment >= 7.0:
            explanations.append("Excellent overall customer satisfaction")
        elif avg_sentiment >= 6.0:
            explanations.append("Good customer feedback")
        elif avg_sentiment <= 4.0:
            explanations.append("Mixed customer reviews")
        
        # Price consideration
        if product.get('price', 0) < 500:
            explanations.append("Budget-friendly option")
        elif product.get('price', 0) > 1000:
            explanations.append("Premium quality product")
        
        if not explanations:
            explanations.append("Matches your search criteria")
        
        return " • ".join(explanations[:3])  # Limit to 3 points
"""
FastAPI backend for AI-Powered Product Recommendation System
Provides endpoints for query analysis, sentiment analysis, and product recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os
import sys

# Add backend directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from nlp_utils import NLPProcessor
from sentiment_model import SentimentAnalyzer
from recommender import ProductRecommender
from database import DatabaseManager

app = FastAPI(title="AI Product Recommendation API", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
nlp_processor = NLPProcessor()
sentiment_analyzer = SentimentAnalyzer()
recommender = ProductRecommender()
db_manager = DatabaseManager()

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class ReviewRequest(BaseModel):
    review_text: str

class QueryAnalysisResponse(BaseModel):
    category: str
    features: List[str]
    confidence: float

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    score: float

class ProductRecommendation(BaseModel):
    id: int
    name: str
    category: str
    price: float
    description: str
    sentiment_score: float
    sentiment_label: str
    explanation: str

class RecommendationResponse(BaseModel):
    query_analysis: QueryAnalysisResponse
    recommendations: List[ProductRecommendation]
    total_products_analyzed: int

@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup"""
    try:
        # Load database
        db_manager.load_data()
        
        # Initialize sentiment model
        sentiment_analyzer.load_model()
        
        print("✅ Backend initialized successfully!")
        print("📊 Database loaded with sample products and reviews")
        print("🤖 ML models ready for inference")
        
    except Exception as e:
        print(f"⚠️ Warning: {str(e)}")
        print("📁 Make sure to place your CSV files in the /data folder for full functionality")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Product Recommendation API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.post("/analyze_query", response_model=QueryAnalysisResponse)
async def analyze_query(request: QueryRequest):
    """
    Analyze user query to extract category and features
    """
    try:
        analysis = nlp_processor.analyze_query(request.query)
        return QueryAnalysisResponse(**analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")

@app.post("/analyze_review", response_model=SentimentResponse)
async def analyze_review(request: ReviewRequest):
    """
    Analyze sentiment of a single review
    """
    try:
        result = sentiment_analyzer.predict_sentiment(request.review_text)
        return SentimentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.post("/recommend_products", response_model=RecommendationResponse)
async def recommend_products(request: QueryRequest):
    """
    Get product recommendations based on user query
    """
    try:
        # First analyze the query
        query_analysis = nlp_processor.analyze_query(request.query)
        
        # Get recommendations
        recommendations = recommender.get_recommendations(
            query=request.query,
            category=query_analysis["category"],
            features=query_analysis["features"]
        )
        
        return RecommendationResponse(
            query_analysis=QueryAnalysisResponse(**query_analysis),
            recommendations=[ProductRecommendation(**rec) for rec in recommendations],
            total_products_analyzed=len(db_manager.get_all_products())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get available product categories"""
    return {"categories": nlp_processor.get_categories()}

@app.get("/health")
async def health_check():
    """Detailed health check with component status"""
    return {
        "api": "healthy",
        "database": "loaded" if db_manager.is_loaded() else "not_loaded",
        "sentiment_model": "loaded" if sentiment_analyzer.is_loaded() else "not_loaded",
        "nlp_processor": "ready",
        "recommender": "ready"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
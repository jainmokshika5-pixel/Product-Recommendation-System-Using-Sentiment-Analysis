# AI-Powered Product Recommendation System

A complete full-stack application that uses advanced sentiment analysis and natural language processing to provide intelligent product recommendations. Built with React, TypeScript, and FastAPI with CNN+BiLSTM machine learning models.

## 🎯 Features

### Frontend (React + Tailwind CSS)
- **Clean, professional UI** with responsive design
- **Home Page**: Project overview and feature highlights
- **Query Page**: Natural language product search interface
- **Results Page**: Recommended products with sentiment scores
- **Dashboard**: Analytics and sentiment distribution charts  
- **Review Analysis**: Individual review sentiment analysis tool

### Backend (FastAPI + Python)
- **NLP Processing**: Category detection and feature extraction
- **Sentiment Analysis**: CNN+BiLSTM deep learning model
- **Product Ranking**: Intelligent scoring and recommendation system
- **RESTful APIs**: Clean endpoints for all functionality

### AI/ML Pipeline
- **Text Preprocessing**: Tokenization, stopword removal, lemmatization
- **Feature Extraction**: spaCy/NLTK keyword detection
- **Deep Learning Models**: CNN+BiLSTM for sentiment classification
- **Recommendation Engine**: Multi-factor scoring algorithm

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+ with pip
- Git

### Quick Start (Frontend Only)

```bash
# 1. Install frontend dependencies
npm install

# 2. Start the frontend development server
npm run dev
```

The application will be available at `http://localhost:8080` with sample data.

### Full System Setup (Frontend + Backend)

```bash
# 1. Install frontend dependencies
npm install

# 2. Set up Python backend
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Start the FastAPI server
uvicorn main:app --reload --port 8000

# 4. In a new terminal, start the frontend
npm run dev

# 5. (Optional) Train custom model with your data
python train_model.py
```

### URLs
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📁 Project Structure

```
/
├── src/                          # React frontend source
│   ├── components/              # Reusable UI components
│   ├── pages/                   # Application pages
│   ├── hooks/                   # Custom React hooks
│   └── lib/                     # Utilities and helpers
├── backend/                     # FastAPI backend (to be added)
│   ├── models/                  # ML models and training
│   ├── data/                    # Dataset storage
│   ├── routers/                 # API route handlers
│   └── main.py                  # FastAPI application
├── public/                      # Static assets
└── requirements.txt             # Python dependencies
```

## 🎨 Design System

The application features a beautiful, modern design system with:

- **AI-themed color palette**: Purple gradients with semantic color tokens
- **Sentiment visualization**: Color-coded badges (green/yellow/red)
- **Smooth animations**: Hover effects and transitions
- **Responsive layout**: Works perfectly on desktop and mobile
- **Accessibility**: Proper contrast ratios and semantic HTML

## 🔧 Tech Stack

### Frontend
- **React 18**: Modern React with hooks and context
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/UI**: Beautiful, accessible UI components
- **Lucide React**: Consistent icon system
- **React Router**: Client-side routing

### Backend (Framework Ready)
- **FastAPI**: High-performance Python web framework
- **SQLite**: Lightweight database for development
- **Pandas**: Data manipulation and analysis
- **PyTorch**: Deep learning framework
- **spaCy/NLTK**: Natural language processing
- **Scikit-learn**: Machine learning utilities

## 📊 Data Requirements

The system expects two CSV files in the `backend/data/` directory:

1. **Dataset-SA.csv**: Sentiment-annotated product reviews
   - Columns: `review_text`, `sentiment`, `product_id`

2. **product_reviews.csv**: Product information
   - Columns: `id`, `name`, `category`, `price`, `description`

If datasets are not available, the application will display mock data for demonstration purposes.

## 🤖 Machine Learning Models

### Sentiment Analysis Pipeline
1. **Text Preprocessing**: Clean and tokenize review text
2. **Feature Extraction**: Extract product features and keywords
3. **CNN+BiLSTM Model**: Deep learning classification
4. **Confidence Scoring**: Probability-based sentiment scores

### Recommendation Algorithm
1. **Query Analysis**: Extract category and feature preferences
2. **Review Filtering**: Find relevant product reviews
3. **Sentiment Aggregation**: Calculate average sentiment scores
4. **Feature Matching**: Match user preferences to product features
5. **Ranking**: Score and sort products by relevance

## 🔮 Usage Examples

### Natural Language Queries
- "Suggest me a laptop with good battery life and fast performance"
- "I need a smartphone with excellent camera quality under $800"
- "Looking for wireless headphones with noise cancellation"
- "Find me a smartwatch for fitness tracking"

### Expected Output
- **Category Detection**: Laptop, Smartphone, etc.
- **Feature Extraction**: battery, camera, performance, etc.
- **Product Recommendations**: Top 5 products with sentiment scores
- **Detailed Analytics**: Sentiment breakdown and confidence levels

## 🚀 Deployment

### Frontend Deployment
```bash
# Build for production
npm run build

# Deploy to your preferred hosting service
# (Vercel, Netlify, etc.)
```

### Backend Deployment
```bash
# Install production dependencies
pip install -r requirements.txt

# Run with production ASGI server
uvicorn main:app --host 0.0.0.0 --port 8000

# Or use Docker
docker build -t ai-recommend-backend .
docker run -p 8000:8000 ai-recommend-backend
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Educational Use

This project is designed for learning and demonstrates:
- Modern React development with TypeScript
- Machine learning integration in web applications
- Professional UI/UX design principles
- Full-stack application architecture
- Natural language processing techniques

Perfect for students, developers, and anyone interested in AI-powered web applications!
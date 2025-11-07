# 🤖 AI-Powered Product Recommendation System

An intelligent product recommendation system that combines sentiment analysis, natural language processing, and machine learning to provide personalized product recommendations based on user queries and reviews.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)

## 🌟 Features

### Core Functionality
- **🔍 Intelligent Query Analysis**: NLP-powered query understanding to extract product categories and features
- **💬 Sentiment Analysis**: Advanced CNN+BiLSTM model for accurate sentiment classification (92% accuracy)
- **🎯 Smart Recommendations**: Multi-factor recommendation engine considering sentiment, features, and user preferences
- **📊 Real-time Analytics**: Interactive dashboard with sentiment distribution and product insights
- **🔄 Multi-Model Support**: Includes RNN and 1D CNN implementations for comparison

### Machine Learning Models
1. **CNN+BiLSTM Sentiment Model** (Primary)
   - 92.25% validation accuracy
   - Trained on 12,253 product reviews
   - Handles positive, negative, and neutral sentiments

2. **1D CNN Model** (Alternative)
   - Multi-task learning (sentiment, rating, category prediction)
   - Parallel processing for faster inference
   - Early stopping to prevent overfitting

3. **RNN Model** (Experimental)
   - Bidirectional LSTM architecture
   - Multi-head attention mechanism
   - Multi-task learning capabilities

## 📁 Project Structure

```
├── backend/
│   ├── main.py                      # FastAPI server
│   ├── sentiment_model.py           # CNN+BiLSTM model
│   ├── cnn1d_product_model.py       # 1D CNN model
│   ├── rnn_product_model.py         # RNN model
│   ├── train_model.py               # Model training script
│   ├── nlp_utils.py                 # NLP processing utilities
│   ├── recommender.py               # Recommendation engine
│   ├── database.py                  # Database management
│   └── models/                      # Trained model files
├── data/
│   ├── electronics_balanced_10k.csv # Electronics dataset (11,253 reviews)
│   └── product_reviews.csv          # Product reviews dataset (1,000 reviews)
├── src/
│   ├── pages/
│   │   ├── Dashboard.tsx            # Analytics dashboard
│   │   ├── Query.tsx                # Query input interface
│   │   ├── Results.tsx              # Recommendation results
│   │   └── ReviewAnalysis.tsx       # Review sentiment analysis
│   └── components/                  # Reusable UI components
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-product-recommendation.git
cd ai-product-recommendation
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Node.js dependencies**
```bash
npm install
# or
yarn install
```

4. **Train the sentiment model** (Optional - pre-trained model included)
```bash
cd backend
python train_model.py
```

### Running the Application

#### Option 1: Run Backend and Frontend Separately

**Start the Backend Server:**
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Start the Frontend Development Server:**
```bash
npm run dev
# or
yarn dev
```

#### Option 2: Run Both Simultaneously
```bash
npm run start:full
```

The application will be available at:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📊 Dataset Information

### Electronics Balanced Dataset
- **Size**: 11,253 reviews
- **Categories**: Mobile, Watch, Laptop, TV, Cooler/AC, Headphone, Speaker, Other
- **Sentiment Distribution**: 81% Positive, 14% Negative, 5% Neutral
- **Features**: Product name, price, rating, review text, category, price range

### Product Reviews Dataset
- **Size**: 1,000 reviews
- **Categories**: Smartphones, Wearables, Smart Home, Audio, Laptops
- **Features**: Review text, sentiment, rating, feature mentions, attributes

## 🔧 API Endpoints

### Health Check
```bash
GET /
GET /health
```

### Query Analysis
```bash
POST /analyze_query
Content-Type: application/json

{
  "query": "I need a smartphone with good battery life"
}
```

### Sentiment Analysis
```bash
POST /analyze_review
Content-Type: application/json

{
  "review_text": "This product is amazing! Great quality and fast delivery."
}
```

### Product Recommendations
```bash
POST /recommend_products
Content-Type: application/json

{
  "query": "laptop for gaming with good graphics"
}
```

### Get Categories
```bash
GET /categories
```

## 🧠 Model Performance

### CNN+BiLSTM Sentiment Model
| Metric | Negative | Neutral | Positive | Overall |
|--------|----------|---------|----------|---------|
| Precision | 0.91 | 0.54 | 0.96 | 0.93 |
| Recall | 0.81 | 0.67 | 0.96 | 0.92 |
| F1-Score | 0.86 | 0.60 | 0.96 | 0.92 |
| **Accuracy** | - | - | - | **92.25%** |

### Training Details
- **Training Samples**: 9,802
- **Test Samples**: 2,451
- **Vocabulary Size**: 8,261 words
- **Epochs**: 10
- **Optimizer**: Adam (lr=0.001)
- **Architecture**: Embedding → CNN → BiLSTM → Attention → Dense

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework for model training and inference
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities
- **NLTK**: Natural language processing

### Frontend
- **React 18**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/ui**: Beautiful UI components
- **Recharts**: Data visualization

## 📈 Model Training

To train a new model from scratch:

```bash
cd backend

# Train CNN+BiLSTM model (recommended)
python train_model.py

# Train 1D CNN model
python cnn1d_product_model.py

# Train RNN model
python rnn_product_model.py
```

Training outputs:
- Model weights: `backend/models/sentiment_model.pth`
- Tokenizer: `backend/models/tokenizer.pkl`
- Training curves: `*_training_curves.png`

## 🔍 Key Features Explained

### 1. Query Analysis
- Extracts product category from natural language queries
- Identifies key features mentioned (battery life, camera, display, etc.)
- Returns confidence scores for predictions

### 2. Sentiment Analysis
- Three-class classification (Positive, Negative, Neutral)
- Handles product reviews of varying lengths
- Provides confidence scores for each prediction

### 3. Recommendation Engine
- Filters products by category
- Ranks by sentiment scores
- Considers feature matches
- Returns top-N recommendations with explanations

### 4. Multi-Model Architecture
- Primary: CNN+BiLSTM for production use
- Alternative: 1D CNN for faster inference
- Experimental: RNN with attention for research

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Dataset sources: Amazon product reviews, electronics reviews
- Inspired by modern recommendation systems
- Built with modern ML and web technologies

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. The trained models and datasets are used for demonstration only.

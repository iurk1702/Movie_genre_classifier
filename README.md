# Movie Genre Classifier

A full-stack machine learning application for multi-label movie genre classification using Natural Language Processing. This project uses the Cornell Movie Dialog Corpus to train a classifier that predicts movie genres from dialogue text.

## 🌐 Live Demo

**Frontend Application**: [https://movie-genre-classifier-frontend.vercel.app/](https://movie-genre-classifier-frontend.vercel.app/)

Try out the model by entering movie dialogue text and see real-time genre predictions with confidence scores. Explore the research tab to view comprehensive analysis including genre distributions, co-occurrence patterns, and model performance metrics.

**Backend API**: Deployed on Render (see API endpoints section below)

## 📖 About This Project

This project demonstrates a complete machine learning pipeline from data preprocessing to model deployment. The system analyzes movie dialogue text to predict multiple genres simultaneously (multi-label classification), making it useful for content categorization, recommendation systems, and film analysis.

### Key Highlights

- **Multi-label Classification**: Predicts multiple genres from a single dialogue input
- **NLP Pipeline**: Advanced text preprocessing including tokenization, lemmatization, and stopword removal
- **Interactive Web Interface**: User-friendly React frontend for testing predictions and exploring research findings
- **Comprehensive Research Dashboard**: Visualizations of genre distributions, co-occurrence matrices, word frequencies, and model performance
- **Production-Ready API**: RESTful FastAPI backend with full CORS support and comprehensive documentation

## 🎯 Features

### Backend (FastAPI)
- **Multi-label Genre Classification**: Predicts multiple genres from movie dialogue text
- **RESTful API**: Clean, well-documented FastAPI endpoints with interactive Swagger documentation
- **Research Data Endpoints**: Comprehensive analysis including genre distributions, co-occurrence matrices, and model performance metrics
- **Production Ready**: Includes deployment configuration for Render
- **CORS Enabled**: Configured for frontend integration

### Frontend (React + TypeScript)
- **Interactive Prediction Tool**: Real-time genre prediction with confidence scores
- **Research Dashboard**: Visual exploration of dataset insights and model performance
- **Modern UI**: Built with React, TypeScript, and Shadcn/ui components
- **Data Visualizations**: Interactive charts using Recharts for genre distributions, co-occurrence heatmaps, and word frequency analysis
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Machine Learning**: Scikit-learn (OneVsRestClassifier, LogisticRegression)
- **NLP**: NLTK (tokenization, lemmatization, stopwords)
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Joblib
- **API Documentation**: Swagger/OpenAPI (auto-generated)
- **Deployment**: Render

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Components**: Shadcn/ui
- **Charts**: Recharts
- **HTTP Client**: Axios
- **Deployment**: Vercel

### Machine Learning
- **Algorithm**: OneVsRestClassifier with Logistic Regression
- **Feature Extraction**: Bag of Words (CountVectorizer)
- **Evaluation Metrics**: F1-score, Precision, Recall
- **Dataset**: Cornell Movie Dialog Corpus

## 📊 Model Performance

- **F1 Score**: 0.470 (weighted)
- **Precision**: 0.570
- **Recall**: 0.419
- **Genres Supported**: 11 genres (drama, thriller, action, comedy, crime, romance, sci-fi, adventure, mystery, horror, fantasy)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Movie_genre_classifier
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

5. **Download the dataset:**
   
   The Cornell Movie Dialog Corpus is not included in this repository. You need to download it separately:
   
   - Option 1: Download from [Kaggle](https://www.kaggle.com/datasets/rajathmc/cornell-moviedialog-corpus)
   - Option 2: Use the Kaggle API (see `MovieDataset_Cornell.ipynb` for instructions)
   
   Extract the dataset to `cornell-moviedialog-corpus/` directory in the project root.

6. **Train the model:**
   ```bash
   python train_and_save_model.py
   ```
   
   This will create model files in `models/` directory:
   - `vectorizer.pkl` - CountVectorizer
   - `multilabel_binarizer.pkl` - MultiLabelBinarizer
   - `classifier.pkl` - OneVsRestClassifier
   - `metadata.json` - Model metadata

7. **Extract research data:**
   ```bash
   python scripts/extract_research_data.py
   ```
   
   This generates JSON files in `data/research/` with research insights.

8. **Run the API server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   
   The API will be available at `http://localhost:8000`
   
   Interactive API documentation: `http://localhost:8000/docs`

## 📡 API Endpoints

### Prediction

**POST** `/api/predict`

Predict genres from dialogue text.

**Request:**
```json
{
  "dialogue": "I killed him. He was coming after me. I had no choice."
}
```

**Response:**
```json
{
  "genres": ["thriller", "crime"],
  "probabilities": {
    "drama": 0.45,
    "thriller": 0.78,
    "crime": 0.65,
    ...
  },
  "top_genres": [
    {"genre": "thriller", "probability": 0.78},
    {"genre": "crime", "probability": 0.65},
    ...
  ]
}
```

### Research Data

- **GET** `/api/research/overview` - Project overview and methodology
- **GET** `/api/research/genre-distribution` - Genre frequency distribution
- **GET** `/api/research/genre-cooccurrence` - Genre co-occurrence matrix
- **GET** `/api/research/model-performance` - Model performance metrics
- **GET** `/api/research/word-frequencies/{genre}` - Word frequencies for a specific genre
- **GET** `/api/research/word-frequencies` - All word frequencies
- **GET** `/api/research/features-analysis` - Feature correlation data

## 🏗️ Project Structure

This repository contains the **backend API**. The frontend is in a separate repository.

### Backend Repository (This Repo)
```
Movie_genre_classifier/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic request/response schemas
│   ├── dependencies.py      # Model loading dependencies
│   ├── preprocessing.py     # Text preprocessing functions
│   └── routers/
│       ├── __init__.py
│       ├── predict.py       # Prediction endpoints
│       └── research.py       # Research data endpoints
├── scripts/
│   ├── __init__.py
│   └── extract_research_data.py  # Research data extraction
├── train_and_save_model.py  # Model training script
├── requirements.txt         # Python dependencies
├── render.yaml             # Render deployment configuration
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

### Frontend Repository
The frontend React application is deployed separately. It communicates with this backend API to provide:
- Interactive genre prediction interface
- Research dashboard with visualizations
- Real-time API integration

**Frontend URL**: [https://movie-genre-classifier-frontend.vercel.app/](https://movie-genre-classifier-frontend.vercel.app/)

## 🔧 Configuration

### CORS Settings

Update `app/config.py` to add your frontend URL:

```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "https://your-frontend.vercel.app",  # Add your production URL
]
```

## 🚢 Deployment

### Backend Deployment (Render)

The backend API is deployed on Render. To deploy:

1. Push your code to GitHub
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repository
4. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables:
   - `FRONTEND_URL`: Your frontend deployment URL (e.g., `https://movie-genre-classifier-frontend.vercel.app`)
6. Deploy

The `render.yaml` file contains the deployment configuration.

### Frontend Deployment (Vercel)

The frontend is deployed on Vercel at [https://movie-genre-classifier-frontend.vercel.app/](https://movie-genre-classifier-frontend.vercel.app/)

To deploy the frontend:
1. Push frontend code to GitHub
2. Import the repository on [Vercel](https://vercel.com)
3. Set environment variable:
   - `VITE_API_BASE_URL`: Your backend API URL (e.g., `https://your-backend.onrender.com`)
4. Deploy

### Important Notes for Deployment

- **Models must be trained before deployment**: The model files are gitignored. You'll need to train the model on Render or upload the model files separately.
- **Research data must be generated**: Run `scripts/extract_research_data.py` before deployment or as part of the build process.
- **Dataset access**: Ensure the dataset is accessible during training (either included in the repo or downloaded during build).
- **CORS Configuration**: Ensure the frontend URL is added to `CORS_ORIGINS` in `app/config.py` or set via `FRONTEND_URL` environment variable.

## 🧪 Development

### Running Tests

```bash
# Test the API locally
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "I love you more than anything in this world."}'
```

### Code Style

This project follows PEP 8 style guidelines. Consider using:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

## 📚 Methodology

The model uses:
- **Text Preprocessing**: Tokenization, lemmatization, stopword removal
- **Feature Engineering**: Bag of Words (CountVectorizer)
- **Model**: OneVsRestClassifier with Logistic Regression
- **Evaluation**: F1-score, precision, and recall metrics

For detailed methodology and findings, see the research endpoints or the original notebook.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Cornell Movie Dialog Corpus dataset
- Scikit-learn for machine learning tools
- FastAPI for the web framework
- NLTK for natural language processing

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

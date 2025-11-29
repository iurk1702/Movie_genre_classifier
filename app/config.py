"""
Configuration settings for the FastAPI application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODELS_DIR = BASE_DIR / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
MULTILABEL_BINARIZER_PATH = MODELS_DIR / "multilabel_binarizer.pkl"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "metadata.json"

# Research data directory
RESEARCH_DATA_DIR = BASE_DIR / "data" / "research"

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",  # Added for genre-dialogue-lab frontend
    "https://localhost:5173",
    "https://localhost:5174",
    # Add your Vercel frontend URL here after deployment
    # Example: "https://your-app.vercel.app"
]

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API settings
API_V1_PREFIX = "/api"


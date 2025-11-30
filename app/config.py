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
# Default CORS origins for local development
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",  # Added for genre-dialogue-lab frontend
    "https://localhost:5173",
    "https://localhost:5174",
]

# Get frontend URL from environment variable (for production)
FRONTEND_URL = os.getenv("FRONTEND_URL", "")

# Build CORS origins list
CORS_ORIGINS = _DEFAULT_CORS_ORIGINS.copy()
if FRONTEND_URL:
    # Add production frontend URL if provided
    CORS_ORIGINS.append(FRONTEND_URL)
    # Also add without trailing slash if present
    if FRONTEND_URL.endswith("/"):
        CORS_ORIGINS.append(FRONTEND_URL.rstrip("/"))

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# API settings
API_V1_PREFIX = "/api"


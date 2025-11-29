"""
Dependencies for FastAPI routes - model loading and initialization.
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any
from fastapi import HTTPException

from app.config import (
    VECTORIZER_PATH,
    MULTILABEL_BINARIZER_PATH,
    CLASSIFIER_PATH,
    MODEL_METADATA_PATH
)


class ModelLoader:
    """Singleton class to load and cache models."""
    
    _instance = None
    _vectorizer = None
    _multilabel_binarizer = None
    _classifier = None
    _metadata = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_models(self):
        """Load all models and metadata."""
        if self._vectorizer is None:
            try:
                self._vectorizer = joblib.load(VECTORIZER_PATH)
                self._multilabel_binarizer = joblib.load(MULTILABEL_BINARIZER_PATH)
                self._classifier = joblib.load(CLASSIFIER_PATH)
                
                if MODEL_METADATA_PATH.exists():
                    with open(MODEL_METADATA_PATH, 'r') as f:
                        self._metadata = json.load(f)
                else:
                    self._metadata = {}
                    
            except FileNotFoundError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model files not found. Please train the model first. Error: {str(e)}"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error loading models: {str(e)}"
                )
    
    @property
    def vectorizer(self):
        """Get the vectorizer model."""
        if self._vectorizer is None:
            self.load_models()
        return self._vectorizer
    
    @property
    def multilabel_binarizer(self):
        """Get the multilabel binarizer."""
        if self._multilabel_binarizer is None:
            self.load_models()
        return self._multilabel_binarizer
    
    @property
    def classifier(self):
        """Get the classifier model."""
        if self._classifier is None:
            self.load_models()
        return self._classifier
    
    @property
    def metadata(self):
        """Get model metadata."""
        if self._metadata is None:
            self.load_models()
        return self._metadata


# Global model loader instance
model_loader = ModelLoader()


def get_vectorizer():
    """Dependency to get vectorizer."""
    return model_loader.vectorizer


def get_multilabel_binarizer():
    """Dependency to get multilabel binarizer."""
    return model_loader.multilabel_binarizer


def get_classifier():
    """Dependency to get classifier."""
    return model_loader.classifier


def get_metadata():
    """Dependency to get model metadata."""
    return model_loader.metadata


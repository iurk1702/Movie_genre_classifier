"""
Prediction router for genre classification.
"""

from fastapi import APIRouter, Depends, HTTPException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

from app.models import PredictRequest, PredictResponse, GenreProbability
from app.dependencies import (
    get_vectorizer,
    get_multilabel_binarizer,
    get_classifier,
    get_metadata
)
from app.preprocessing import process_text

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("", response_model=PredictResponse)
async def predict_genre(
    request: PredictRequest,
    vectorizer: CountVectorizer = Depends(get_vectorizer),
    multilabel_binarizer: MultiLabelBinarizer = Depends(get_multilabel_binarizer),
    classifier: OneVsRestClassifier = Depends(get_classifier),
    metadata: dict = Depends(get_metadata)
):
    """
    Predict movie genres from dialogue text.
    
    Args:
        request: Contains the dialogue text
        vectorizer: Trained CountVectorizer
        multilabel_binarizer: Trained MultiLabelBinarizer
        classifier: Trained OneVsRestClassifier
        
    Returns:
        Predicted genres with confidence scores
    """
    try:
        # Preprocess the input text
        processed_text = process_text(request.dialogue)
        
        if not processed_text.strip():
            raise HTTPException(
                status_code=400,
                detail="After preprocessing, the dialogue text is empty. Please provide more substantial text."
            )
        
        # Transform text using vectorizer
        text_vector = vectorizer.transform([processed_text])
        
        # Predict probabilities
        probabilities = classifier.predict_proba(text_vector)[0]
        
        # Get predicted labels (binary predictions)
        predictions = classifier.predict(text_vector)[0]
        
        # Get genre names from multilabel binarizer
        genre_names = multilabel_binarizer.classes_
        
        # Create probability dictionary
        prob_dict = {}
        for i, genre in enumerate(genre_names):
            prob_dict[genre] = float(probabilities[i])
        
        # Get predicted genres (where prediction is 1)
        predicted_genres = [genre_names[i] for i, pred in enumerate(predictions) if pred == 1]
        
        # If no genres predicted, return top 3 by probability
        if not predicted_genres:
            top_indices = np.argsort(probabilities)[::-1][:3]
            predicted_genres = [genre_names[i] for i in top_indices]
        
        # Create top genres list sorted by probability
        genre_probs = [
            GenreProbability(genre=genre_names[i], probability=float(probabilities[i]))
            for i in range(len(genre_names))
        ]
        genre_probs.sort(key=lambda x: x.probability, reverse=True)
        top_genres = genre_probs[:5]  # Top 5 genres
        
        return PredictResponse(
            genres=predicted_genres,
            probabilities=prob_dict,
            top_genres=top_genres
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


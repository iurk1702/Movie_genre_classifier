"""
Pydantic models for API request/response schemas.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request model for genre prediction."""
    dialogue: str = Field(..., description="Movie dialogue text to classify", min_length=1)


class GenreProbability(BaseModel):
    """Genre with its probability score."""
    genre: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class PredictResponse(BaseModel):
    """Response model for genre prediction."""
    genres: List[str] = Field(..., description="Predicted genres")
    probabilities: Dict[str, float] = Field(..., description="Probability scores for each genre")
    top_genres: List[GenreProbability] = Field(..., description="Top genres sorted by probability")


class ResearchOverview(BaseModel):
    """Research overview data."""
    title: str
    objective: str
    dataset_description: str
    methodology: List[str]
    key_findings: List[str]


class GenreDistributionData(BaseModel):
    """Genre frequency distribution data."""
    genres: List[str]
    counts: List[int]
    percentages: List[float]


class GenreCooccurrenceData(BaseModel):
    """Genre co-occurrence data."""
    genres: Optional[List[str]] = None  # List of genre names in order
    matrix: Optional[List[List[int]]] = None  # NxN co-occurrence matrix
    genre_pairs: List[List[str]]  # Top pairs (backward compatibility)
    counts: List[int]
    percentages: List[float]


class ModelPerformanceData(BaseModel):
    """Model performance metrics."""
    f1_score: float
    precision: float
    recall: float
    per_genre_metrics: Dict[str, Dict[str, float]]


class WordFrequencyData(BaseModel):
    """Word frequency data for a genre."""
    genre: str
    words: List[str]
    frequencies: List[int]


class FeatureAnalysisData(BaseModel):
    """Feature correlation/importance data."""
    features: List[str]
    correlation_matrix: List[List[float]]
    feature_names: List[str]


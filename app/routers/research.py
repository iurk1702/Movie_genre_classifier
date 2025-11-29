"""
Research data router for providing research insights and visualizations.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.config import RESEARCH_DATA_DIR
from app.models import (
    ResearchOverview,
    GenreDistributionData,
    GenreCooccurrenceData,
    ModelPerformanceData,
    WordFrequencyData,
    FeatureAnalysisData
)

router = APIRouter(prefix="/research", tags=["research"])


def load_json_file(filename: str) -> Dict[str, Any]:
    """Load JSON file from research data directory."""
    file_path = RESEARCH_DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Research data file {filename} not found. Please run the data extraction script."
        )
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing JSON file {filename}: {str(e)}"
        )


@router.get("/overview", response_model=ResearchOverview)
async def get_overview():
    """Get project overview and methodology."""
    data = load_json_file("overview.json")
    return ResearchOverview(**data)


@router.get("/genre-distribution", response_model=GenreDistributionData)
async def get_genre_distribution():
    """Get genre frequency distribution data."""
    data = load_json_file("genre_distribution.json")
    return GenreDistributionData(**data)


@router.get("/genre-cooccurrence", response_model=GenreCooccurrenceData)
async def get_genre_cooccurrence():
    """Get genre co-occurrence data."""
    data = load_json_file("genre_cooccurrence.json")
    return GenreCooccurrenceData(**data)


@router.get("/model-performance", response_model=ModelPerformanceData)
async def get_model_performance():
    """Get model performance metrics."""
    data = load_json_file("model_performance.json")
    return ModelPerformanceData(**data)


@router.get("/word-frequencies/{genre}", response_model=WordFrequencyData)
async def get_word_frequencies(genre: str):
    """Get word frequency data for a specific genre."""
    try:
        data = load_json_file(f"word_frequencies_{genre.lower()}.json")
        return WordFrequencyData(**data)
    except HTTPException:
        # Try loading from all genres file
        all_data = load_json_file("word_frequencies_all.json")
        if genre.lower() in all_data:
            return WordFrequencyData(**all_data[genre.lower()])
        raise HTTPException(
            status_code=404,
            detail=f"Word frequency data for genre '{genre}' not found."
        )


@router.get("/word-frequencies")
async def get_all_word_frequencies():
    """Get word frequency data for all genres."""
    data = load_json_file("word_frequencies_all.json")
    return data


@router.get("/features-analysis", response_model=FeatureAnalysisData)
async def get_features_analysis():
    """Get feature correlation/importance analysis."""
    data = load_json_file("features_analysis.json")
    return FeatureAnalysisData(**data)


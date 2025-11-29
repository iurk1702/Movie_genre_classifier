"""
FastAPI main application for Movie Genre Classifier API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, API_V1_PREFIX
from app.routers import predict, research

# Create FastAPI app
app = FastAPI(
    title="Movie Genre Classifier API",
    description="API for predicting movie genres from dialogue text using NLP",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix=API_V1_PREFIX)
app.include_router(research.router, prefix=API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Movie Genre Classifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


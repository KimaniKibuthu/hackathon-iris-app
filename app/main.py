"""
FastAPI application for Iris flower classification.

This module provides REST API endpoints for classifying iris flowers
using a pre-trained Support Vector Classifier model.
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.exceptions import PredictionError
from app.api.routes import router
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info("Starting up Iris Classification API")
    yield
    logger.info("Shutting down Iris Classification API")


# Create FastAPI application
app = FastAPI(
    title="Iris Classification API",
    description="A FastAPI application for classifying iris flowers using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.exception_handler(PredictionError)
async def prediction_exception_handler(request, exc: PredictionError) -> JSONResponse:
    """Handle prediction errors gracefully."""
    logger.error(f"Prediction error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"error": "Prediction failed", "detail": str(exc)}
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "iris-classification-api"}


# Legacy endpoints for backward compatibility
@app.post("/predict")
async def legacy_predict(features: dict):
    """Legacy predict endpoint for backward compatibility."""
    from app.models.schemas import IrisFeatures
    from app.api.routes import classify_iris
    
    try:
        iris_features = IrisFeatures(**features)
        return await classify_iris(iris_features)
    except Exception as e:
        logger.error(f"Legacy predict error: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

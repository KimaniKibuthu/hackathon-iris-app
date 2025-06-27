"""
API routes for iris classification.
"""

import logging
import random
import time
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    IrisFeatures, 
    BatchIrisFeatures,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    RandomPredictionResponse
)
from app.models.iris_model import classifier
from app.core.exceptions import PredictionError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/classify",
    response_model=PredictionResponse,
    summary="Classify iris flower",
    description="Classify an iris flower based on sepal and petal measurements",
    response_description="Prediction result with confidence scores"
)
async def classify_iris(features: IrisFeatures) -> PredictionResponse:
    """
    Classify an iris flower species based on measurements.
    
    Args:
        features: Iris flower measurements (sepal length/width, petal length/width)
    
    Returns:
        Prediction result with confidence scores
    
    Raises:
        HTTPException: If classification fails
    """
    try:
        result = classifier.predict(features.to_list())
        return PredictionResponse(**result)
    
    except PredictionError as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/classify/batch",
    response_model=BatchPredictionResponse,
    summary="Batch classify iris flowers",
    description="Classify multiple iris flowers in a single request"
)
async def classify_batch(batch_features: BatchIrisFeatures) -> BatchPredictionResponse:
    """
    Classify multiple iris flowers in batch.
    
    Args:
        batch_features: List of iris flower measurements
    
    Returns:
        List of predictions
    
    Raises:
        HTTPException: If batch classification fails
    """
    try:
        predictions = []
        for features in batch_features.features:
            result = classifier.predict(features.to_list())
            predictions.append(result["prediction"])
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
    
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {e}")


@router.get(
    "/classify/random",
    response_model=RandomPredictionResponse,
    summary="Get random iris prediction",
    description="Generate random iris measurements and classify them"
)
async def classify_random() -> RandomPredictionResponse:
    """
    Generate random iris measurements and classify them.
    
    Returns:
        Random features and their classification
    """
    try:
        # Generate random features within typical ranges for iris flowers
        random_features = IrisFeatures(
            sepal_length=round(random.uniform(4.0, 8.0), 2),
            sepal_width=round(random.uniform(2.0, 4.5), 2),
            petal_length=round(random.uniform(1.0, 7.0), 2),
            petal_width=round(random.uniform(0.1, 2.5), 2)
        )
        
        logger.info(f"Generated random features: {random_features}")
        
        # Get prediction
        result = classifier.predict(random_features.to_list())
        
        return RandomPredictionResponse(
            random_features=random_features,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probabilities=result["probabilities"]
        )
    
    except Exception as e:
        logger.error(f"Random prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Random prediction failed: {e}")


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get model information",
    description="Get information about the loaded classification model"
)
async def get_model_info() -> ModelInfoResponse:
    """Get information about the classification model."""
    return ModelInfoResponse(
        model_type="Support Vector Classifier",
        classes=classifier.class_names,
        features=[
            "sepal_length",
            "sepal_width", 
            "petal_length",
            "petal_width"
        ],
        feature_units="centimeters"
    )


@router.get(
    "/predict/example",
    summary="Get example prediction",
    description="Get an example prediction using sample iris data"
)
async def example_prediction():
    """Get an example prediction using sample data."""
    # Example setosa measurements
    example_features = [5.1, 3.5, 1.4, 0.2]
    
    try:
        result = classifier.predict(example_features)
        return {
            "example_input": {
                "sepal_length": example_features[0],
                "sepal_width": example_features[1],
                "petal_length": example_features[2],
                "petal_width": example_features[3]
            },
            "prediction_result": result,
            "description": "This is an example prediction using typical Setosa measurements"
        }
    except Exception as e:
        logger.error(f"Error generating example prediction: {e}")
        raise HTTPException(status_code=500, detail="Could not generate example")


@router.get(
    "/test/workload",
    summary="Simulate workload",
    description="Simulate workload for testing latency and performance"
)
async def simulate_workload(seconds: Optional[int] = 1):
    """
    Simulate workload for testing purposes.
    
    Args:
        seconds: Number of seconds to simulate workload
    
    Returns:
        Success message after simulated workload
    """
    try:
        if seconds < 0 or seconds > 10:
            raise HTTPException(status_code=400, detail="Seconds must be between 0 and 10")
        
        logger.info(f"Simulating workload for {seconds} seconds")
        time.sleep(seconds)
        
        return {
            "message": f"Successfully simulated workload for {seconds} seconds",
            "status": "completed"
        }
    except Exception as e:
        logger.error(f"Workload simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Workload simulation failed: {e}")

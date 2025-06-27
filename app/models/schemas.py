"""
Pydantic models for request and response validation.
"""

from typing import Dict, List
from pydantic import BaseModel, Field, validator


class IrisFeatures(BaseModel):
    """Input features for iris classification."""
    
    sepal_length: float = Field(
        ..., 
        ge=0.0, 
        le=20.0,
        description="Sepal length in centimeters"
    )
    sepal_width: float = Field(
        ..., 
        ge=0.0, 
        le=20.0,
        description="Sepal width in centimeters"
    )
    petal_length: float = Field(
        ..., 
        ge=0.0, 
        le=20.0,
        description="Petal length in centimeters"
    )
    petal_width: float = Field(
        ..., 
        ge=0.0, 
        le=20.0,
        description="Petal width in centimeters"
    )
    
    @validator('*')
    def validate_positive(cls, v):
        """Ensure all measurements are positive."""
        if v < 0:
            raise ValueError('Measurements must be positive')
        return v
    
    def to_list(self) -> List[float]:
        """Convert to list format expected by the model."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class BatchIrisFeatures(BaseModel):
    """Batch input features for iris classification."""
    
    features: List[IrisFeatures] = Field(
        ...,
        description="List of iris flower measurements"
    )


class PredictionResponse(BaseModel):
    """Response model for iris classification."""
    
    prediction: str = Field(..., description="Predicted iris species")
    confidence: float = Field(..., description="Confidence score of the prediction")
    probabilities: Dict[str, float] = Field(..., description="Probability scores for each class")


class BatchPredictionResponse(BaseModel):
    """Response model for batch iris classification."""
    
    predictions: List[str] = Field(..., description="List of predicted iris species")
    count: int = Field(..., description="Number of predictions made")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    service: str


class ModelInfoResponse(BaseModel):
    """Model information response model."""
    
    model_type: str
    classes: List[str]
    features: List[str]
    feature_units: str


class RandomPredictionResponse(BaseModel):
    """Response model for random prediction."""
    
    random_features: IrisFeatures
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

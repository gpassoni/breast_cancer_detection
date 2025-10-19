"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel
from typing import Optional, Dict, List, Any


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    message: str
    version: Optional[str] = None
    models_loaded: Optional[Dict[str, bool]] = None


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    success: bool
    message: str
    prediction: Optional[Dict[str, Any]] = None


class SegmentationResult(BaseModel):
    """Segmentation result schema."""
    segmentation_mask: List[List[int]]
    mask_shape: List[int]
    unique_labels: List[int]


class ClassificationResult(BaseModel):
    """Classification result schema."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]

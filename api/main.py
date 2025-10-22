"""FastAPI application for breast cancer detection."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import json
from pathlib import Path
from typing import Optional
import tempfile
import os

from src.inference import BreastCancerPredictor
from src.config import config
from .schemas import PredictionResponse, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="Breast Cancer Detection API",
    description="API for thermal breast image segmentation and cancer detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor: Optional[BreastCancerPredictor] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global predictor
    
    # Load segmentation model configuration
    model_config = None
    config_path = config.get_segmentation_config_path()
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
    
    # Initialize predictor
    predictor = BreastCancerPredictor()
    
    # Load segmentation model if available
    model_path = config.get_segmentation_model_path()
    if model_path.exists():
        predictor.load_segmentation_model(str(model_path), model_config)
        print(f"Loaded segmentation model from {model_path}")
    else:
        print(f"Warning: Segmentation model not found at {model_path}")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return {
        "status": "healthy",
        "message": "Breast Cancer Detection API is running",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = {
        "segmentation": predictor.segmentation_model is not None if predictor else False,
        "classification": predictor.classification_model is not None if predictor else False
    }
    
    return {
        "status": "healthy",
        "message": "API is operational",
        "models_loaded": models_loaded
    }


@app.post("/predict/segment", response_model=PredictionResponse)
async def predict_segmentation(file: UploadFile = File(...)):
    """
    Segment breast regions from thermal image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Segmentation mask and metadata
    """
    if predictor is None or predictor.segmentation_model is None:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Run segmentation
        mask = predictor.segment_image(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Convert mask to list for JSON serialization
        mask_list = mask.tolist()
        
        return {
            "success": True,
            "message": "Segmentation completed successfully",
            "prediction": {
                "segmentation_mask": mask_list,
                "mask_shape": list(mask.shape),
                "unique_labels": [int(x) for x in np.unique(mask)]
            }
        }
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/classify")
async def predict_classification(file: UploadFile = File(...)):
    """
    Classify breast image for cancer detection.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Classification results
    """
    if predictor is None or predictor.classification_model is None:
        raise HTTPException(
            status_code=503,
            detail="Classification model not yet implemented"
        )
    
    # Classification endpoint will be fully implemented when model is ready
    return {
        "success": False,
        "message": "Classification not yet implemented"
    }


@app.post("/predict/full", response_model=PredictionResponse)
async def predict_full(file: UploadFile = File(...)):
    """
    Run full prediction pipeline (segmentation + classification).
    
    Args:
        file: Uploaded image file
        
    Returns:
        Complete prediction results
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction models not loaded"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Run full prediction
        results = predictor.predict(
            tmp_path,
            include_segmentation=True,
            include_classification=False  # Not yet implemented
        )
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Format response
        response_data = {"success": True, "message": "Prediction completed", "prediction": {}}
        
        if 'segmentation' in results:
            mask = results['segmentation']
            response_data['prediction']['segmentation_mask'] = mask.tolist()
            response_data['prediction']['mask_shape'] = list(mask.shape)
            response_data['prediction']['unique_labels'] = [int(x) for x in np.unique(mask)]
        
        if 'classification' in results:
            response_data['prediction']['classification'] = results['classification']
        
        return response_data
        
    except Exception as e:
        # Clean up on error
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)

"""Prediction pipeline for breast cancer detection."""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional

from ..models.segmentation import R2AttU_Net, init_weights
from ..preprocessing.preprocessing import preprocess_image


class BreastCancerPredictor:
    """Breast cancer detection predictor combining segmentation and classification."""
    
    def __init__(
        self,
        segmentation_model_path: Optional[str] = None,
        classification_model_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize predictor with model paths.
        
        Args:
            segmentation_model_path: Path to segmentation model weights
            classification_model_path: Path to classification model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.segmentation_model = None
        self.classification_model = None
        
        if segmentation_model_path:
            self.load_segmentation_model(segmentation_model_path)
    
    def load_segmentation_model(self, model_path: str, config: Optional[Dict] = None):
        """
        Load segmentation model from checkpoint.
        
        Args:
            model_path: Path to model weights (.pth file)
            config: Optional model configuration dict
        """
        if config is None:
            # Default configuration
            config = {
                "img_ch": 1,
                "output_ch": 3,  # multiclass: background, left breast, right breast
                "t": 4,
                "base_filters": 8
            }
        
        self.segmentation_model = R2AttU_Net(
            img_ch=config.get("img_ch", 1),
            output_ch=config.get("output_ch", 3),
            t=config.get("t", 4),
            base_filters=config.get("base_filters", 8)
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.segmentation_model.load_state_dict(state_dict)
        self.segmentation_model.to(self.device)
        self.segmentation_model.eval()
    
    def load_classification_model(self, model_path: str):
        """
        Load classification model from checkpoint.
        
        Args:
            model_path: Path to model weights (.pth file)
        """
        # Classification model loading will be implemented when needed
        raise NotImplementedError("Classification model loading not yet implemented")
    
    def segment_image(self, image: Union[str, np.ndarray], height: int = 224, width: int = 224) -> np.ndarray:
        """
        Segment breast regions from thermal image.
        
        Args:
            image: Image path or numpy array
            height: Target height for inference
            width: Target width for inference
            
        Returns:
            Segmentation mask as numpy array
        """
        if self.segmentation_model is None:
            raise RuntimeError("Segmentation model not loaded. Call load_segmentation_model first.")
        
        # Preprocess image
        if isinstance(image, str):
            img = preprocess_image(image, height, width)
        else:
            img = image
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.segmentation_model(img_tensor)
            
            # For multiclass segmentation
            if output.shape[1] > 1:
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            else:
                # For binary segmentation
                pred = torch.sigmoid(output).squeeze().cpu().numpy()
                pred = (pred > 0.5).astype(np.uint8)
        
        return pred
    
    def classify_image(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Classify breast image for cancer detection.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Dictionary with prediction scores
        """
        if self.classification_model is None:
            raise RuntimeError("Classification model not loaded. Call load_classification_model first.")
        
        # Classification inference will be implemented when needed
        raise NotImplementedError("Classification inference not yet implemented")
    
    def predict(
        self,
        image_path: str,
        include_segmentation: bool = True,
        include_classification: bool = False
    ) -> Dict:
        """
        Run full prediction pipeline.
        
        Args:
            image_path: Path to input image
            include_segmentation: Whether to include segmentation results
            include_classification: Whether to include classification results
            
        Returns:
            Dictionary containing prediction results
        """
        results = {}
        
        if include_segmentation and self.segmentation_model:
            seg_mask = self.segment_image(image_path)
            results['segmentation'] = seg_mask
        
        if include_classification and self.classification_model:
            classification = self.classify_image(image_path)
            results['classification'] = classification
        
        return results

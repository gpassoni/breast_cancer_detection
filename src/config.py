"""Configuration management for the application."""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    SRC_DIR = BASE_DIR / "src"
    MODELS_DIR = BASE_DIR / "best_model_saved"
    
    # Model paths
    SEGMENTATION_MODEL_PATH = MODELS_DIR / "best_model.pth"
    SEGMENTATION_CONFIG_PATH = MODELS_DIR / "config.json"
    
    # Model configuration
    DEFAULT_IMAGE_HEIGHT = 224
    DEFAULT_IMAGE_WIDTH = 224
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Weights & Biases configuration (for training)
    WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "breast-cancer-detection")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "")
    
    # Data paths (for training)
    ROOT_DIR = Path(os.getenv("ROOT_DIR", BASE_DIR))
    DATA_DIR = ROOT_DIR / "data"
    
    @classmethod
    def get_segmentation_model_path(cls) -> Path:
        """Get path to segmentation model weights."""
        return cls.SEGMENTATION_MODEL_PATH
    
    @classmethod
    def get_segmentation_config_path(cls) -> Path:
        """Get path to segmentation model configuration."""
        return cls.SEGMENTATION_CONFIG_PATH
    
    @classmethod
    def is_wandb_enabled(cls) -> bool:
        """Check if Weights & Biases is configured."""
        return bool(cls.WANDB_API_KEY)


# Global config instance
config = Config()

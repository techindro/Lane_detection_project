"""
Configuration file for Lane Detection System
"""

import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    """Main configuration class"""
    
    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR: str = os.path.join(PROJECT_ROOT, "models")
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    
    # Image Processing
    IMAGE_SIZE: Tuple[int, int] = (640, 360)
    ROI_VERTICES: List[Tuple[int, int]] = [
        (0, 360),      # Bottom-left
        (280, 180),    # Top-left
        (360, 180),    # Top-right
        (640, 360)     # Bottom-right
    ]
    
    # Hough Transform Parameters
    HOUGH_RHO: int = 1
    HOUGH_THETA: float = 1 * (3.141592653589793 / 180)
    HOUGH_THRESHOLD: int = 20
    HOUGH_MIN_LINE_LENGTH: int = 20
    HOUGH_MAX_LINE_GAP: int = 50
    
    # Canny Edge Detection
    CANNY_LOW_THRESHOLD: int = 50
    CANNY_HIGH_THRESHOLD: int = 150
    
    # Deep Learning
    BATCH_SIZE: int = 8
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 50
    NUM_CLASSES: int = 2  # Background and lane
    
    # Model Architecture
    MODEL_NAME: str = "unet"
    ENCODER: str = "resnet34"
    ENCODER_WEIGHTS: str = "imagenet"
    
    # Training
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.1
    TEST_SPLIT: float = 0.1
    
    # Evaluation
    METRICS: List[str] = ["accuracy", "precision", "recall", "f1", "iou"]
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Video Processing
    FPS: int = 30
    VIDEO_CODEC: str = "mp4v"
    
    # Visualization
    COLOR_LANE: Tuple[int, int, int] = (0, 255, 0)  # Green
    COLOR_CURVATURE: Tuple[int, int, int] = (255, 0, 0)  # Red
    COLOR_CENTER: Tuple[int, int, int] = (0, 0, 255)  # Blue
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            os.path.join(cls.DATA_DIR, "raw"),
            os.path.join(cls.DATA_DIR, "processed"),
            os.path.join(cls.DATA_DIR, "test_samples"),
            cls.MODEL_DIR,
            cls.OUTPUT_DIR,
            os.path.join(cls.OUTPUT_DIR, "images"),
            os.path.join(cls.OUTPUT_DIR, "videos"),
            os.path.join(cls.OUTPUT_DIR, "reports")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

# Initialize configuration
config = Config()

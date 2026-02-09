"""
Model prediction for lane detection
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .model import LaneDetectionModel
from ..config import config

class DeepLearningPredictor:
    """Deep learning based lane detection"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = "cuda"):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load trained model"""
        if model_path is None:
            # Use default model
            model = LaneDetectionModel(
                num_classes=config.NUM_CLASSES,
                encoder=config.ENCODER
            )
        else:
            # Load from checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'encoder' in checkpoint:
                model = LaneDetectionModel(
                    num_classes=checkpoint.get('num_classes', config.NUM_CLASSES),
                    encoder=checkpoint.get('encoder', config.ENCODER)
                )
            else:
                from .model import LaneNet
                model = LaneNet(num_classes=checkpoint.get('num_classes', config.NUM_CLASSES))
            
            model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model"""
        # Resize to model input size
        image = cv2.resize(image, config.IMAGE_SIZE)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.to(self.device)
    
    def postprocess(self, mask: torch.Tensor, 
                    original_shape: Tuple[int, int]) -> np.ndarray:
        """Postprocess model output"""
        # Convert to numpy
        mask = mask.squeeze().cpu().numpy()
        
        # Threshold
        mask = (mask > config.CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
        
        # Resize to original size
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
        
        return mask
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect lanes in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary containing detection results
        """
        original_shape = image.shape
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            mask = self.model.predict(input_tensor)
        
        # Postprocess
        binary_mask = self.postprocess(mask, original_shape)
        
        # Extract lane lines from mask
        lane_lines = self._extract_lane_lines(binary_mask)
        
        # Calculate curvature if lane lines found
        if len(lane_lines) >= 2:
            curvature = self._calculate_curvature(lane_lines, original_shape[0])
            offset = self._calculate_offset(lane_lines, original_shape[1])
        else:
            curvature = 0.0
            offset = 0.0
        
        # Create visualization
        visualization = self._create_visualization(image, binary_mask, lane_lines)
        
        return {
            'mask': binary_mask,
            'lane_lines': lane_lines,
            'curvature': curvature,
            'offset': offset,
            'visualization': visualization
        }
    
    def _extract_lane_lines(self, mask: np.ndarray) -> list:
        """Extract lane lines from segmentation mask"""
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lane_lines = []
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 100:
                continue
            
            # Fit line to contour
            rows, cols = mask.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate line endpoints
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            
            lane_lines.append({
                'line': (int(x), int(y), int(vx), int(vy)),
                'endpoints': ((cols - 1, righty), (0, lefty))
            })
        
        return lane_lines
    
    def _calculate_curvature(self, lane_lines: list, image_height: int) -> float:
        """Calculate average curvature from lane lines"""
        curvatures = []
        
        for lane in lane_lines:
            x, y, vx, vy = lane['line']
            
            if vy == 0:  # Avoid division by zero
                continue
            
            # Calculate curvature (simplified)
            curvature = abs(vx / vy) if vy != 0 else 0
            curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def _calculate_offset(self, lane_lines: list, image_width: int) -> float:
        """Calculate vehicle offset from center"""
        if len(lane_lines) < 2:
            return 0.0
        
        # Get x-coordinates of lane lines at bottom of image
        x_positions = []
        
        for lane in lane_lines:
            (x1, y1), (x2, y2) = lane['endpoints']
            
            # Find x at bottom of image (assuming y increases downward)
            if y1 > y2:  # y1 is bottom
                x_positions.append(x1)
            else:
                x_positions.append(x2)
        
        # Calculate lane center
        lane_center = np.mean(x_positions)
        image_center = image_width / 2
        
        # Convert to meters
        offset_pixels = lane_center - image_center
        offset_meters = offset_pixels * (3.7 / 700)  # Lane width 3.7m ~ 700 pixels
        
        return offset_meters
    
    def _create_visualization(self, image: np.ndarray, 
                             mask: np.ndarray, 
                             lane_lines: list) -> np.ndarray:
        """Create visualization of detection results"""
        # Create overlay
        overlay = image.copy()
        
        # Color mask in green
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green
        
        # Blend with original
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        # Draw lane lines
        for lane in lane_lines:
            (x1, y1), (x2, y2) = lane['endpoints']
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines
        
        return overlay

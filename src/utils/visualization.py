"""
Visualization utilities for lane detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from ..config import config

class LaneVisualizer:
    """Visualization class for lane detection results"""
    
    def __init__(self):
        self.config = config
        
    def draw_lanes(self, image: np.ndarray, 
                  left_lane: Optional[np.ndarray],
                  right_lane: Optional[np.ndarray],
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 5) -> np.ndarray:
        """
        Draw detected lanes on image
        """
        result = image.copy()
        
        if left_lane is not None:
            x1, y1, x2, y2 = left_lane
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
            
        if right_lane is not None:
            x1, y1, x2, y2 = right_lane
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)
            
        return result
    
    def draw_lane_area(self, image: np.ndarray,
                      left_lane: Optional[np.ndarray],
                      right_lane: Optional[np.ndarray],
                      color: Tuple[int, int, int] = (0, 255, 0),
                      alpha: float = 0.3) -> np.ndarray:
        """
        Draw filled lane area
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        if left_lane is not None and right_lane is not None:
            # Create polygon for lane area
            left_x1, left_y1, left_x2, left_y2 = left_lane
            right_x1, right_y1, right_x2, right_y2 = right_lane
            
            pts = np.array([
                [left_x1, left_y1],
                [left_x2, left_y2],
                [right_x2, right_y2],
                [right_x1, right_y1]
            ], np.int32)
            
            # Create overlay
            overlay = result.copy()
            cv2.fillPoly(overlay, [pts], color)
            
            # Blend with original
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
            
        return result
    
    def draw_curvature_text(self, image: np.ndarray,
                          curvature: float,
                          offset: float) -> np.ndarray:
        """
        Draw curvature and offset information
        """
        result = image.copy()
        
        # Curvature text
        curvature_text = f"Curvature: {curvature:.2f}m"
        if curvature < 1000:
            curvature_text = f"Curvature: {curvature:.2f}m"
        else:
            curvature_text = "Curvature: Straight"
            
        cv2.putText(result, curvature_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Offset text
        offset_text = f"Offset: {offset:.2f}m"
        offset_color = (0, 0, 255) if abs(offset) > 0.3 else (0, 255, 0)
        cv2.putText(result, offset_text, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, offset_color, 2)
        
        # Lane departure warning
        if abs(offset) > 0.5:
            cv2.putText(result, "LANE DEPARTURE WARNING!", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
        return result
    
    def create_comparison_plot(self, images: List[np.ndarray],
                             titles: List[str],
                             figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create comparison plot of multiple images
        """
        n_images = len(images)
        fig, axes = plt.subplots(1, n_images, figsize=figsize)
        
        if n_images == 1:
            axes = [axes]
            
        for idx, (ax, image, title) in enumerate(zip(axes, images, titles)):
            if len(image.shape) == 2:  # Grayscale
                ax.imshow(image, cmap='gray')
            else:  # Color
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            
        plt.tight_layout()
        return fig
    
    def plot_metrics_history(self, history: Dict[str, List[float]]) -> plt.Figure:
        """
        Plot training history metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = [
            ('train_loss', 'val_loss', 'Loss'),
            ('val_iou', None, 'IoU Score'),
            ('val_f1', None, 'F1 Score'),
            ('val_accuracy', None, 'Accuracy')
        ]
        
        for idx, (train_metric, val_metric, title) in enumerate(metrics):
            ax = axes[idx]
            
            if train_metric in history:
                ax.plot(history[train_metric], label='Train', linewidth=2)
                
            if val_metric and val_metric in history:
                ax.plot(history[val_metric], label='Validation', linewidth=2)
            elif not val_metric and train_metric.startswith('val_'):
                metric_name = train_metric.replace('val_', '')
                ax.plot(history[train_metric], label=metric_name, linewidth=2)
                
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig
    
    def create_animation(self, images: List[np.ndarray],
                        output_path: str,
                        fps: int = 30):
        """
        Create animation from list of images
        """
        if not images:
            return
            
        height, width = images[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for img in images:
            out.write(img)
            
        out.release()

"""
Main module for Lane Detection System
"""
import cv2
import numpy as np
from typing import Optional

from .config import config
from .traditional.hough_detector import HoughLaneDetector
from .utils.visualization import LaneVisualizer

class LaneDetectionSystem:
    """Main system class (simplified version)"""
    
    def __init__(self, method: str = "traditional"):
        self.method = method
        self.detector = HoughLaneDetector()
        self.visualizer = LaneVisualizer()
    
    def detect(self, image: np.ndarray) -> dict:
        """Detect lanes in image"""
        return self.detector.detect(image)
    
    def visualize(self, image: np.ndarray, results: dict) -> np.ndarray:
        """Visualize detection results"""
        visualized = image.copy()
        
        if 'left_lane' in results and 'right_lane' in results:
            left_lane = results['left_lane']
            right_lane = results['right_lane']
            
            if left_lane is not None and right_lane is not None:
                # Draw lane area
                visualized = self.visualizer.draw_lane_area(
                    visualized, left_lane, right_lane
                )
                
                # Draw lane lines
                visualized = self.visualizer.draw_lanes(
                    visualized, left_lane, right_lane
                )
                
                # Add info text
                curvature = results.get('curvature', 0)
                offset = results.get('offset', 0)
                visualized = self.visualizer.draw_curvature_text(
                    visualized, curvature, offset
                )
        
        return visualized
    
    def run_on_image(self, image_path: str, output_path: Optional[str] = None):
        """Run detection on single image"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        results = self.detect(image)
        visualized = self.visualize(image, results)
        
        if output_path:
            cv2.imwrite(output_path, visualized)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow("Lane Detection", visualized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

"""
Hough Transform based lane detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class HoughLaneDetector:
    """Lane detection using Hough Transform"""
    
    def __init__(self):
        # Default parameters
        self.canny_low = 50
        self.canny_high = 150
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 20
        self.min_line_length = 20
        self.max_line_gap = 50
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for lane detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        return edges
    
    def region_of_interest(self, img: np.ndarray, vertices: Optional[List] = None) -> np.ndarray:
        """Apply region of interest mask"""
        height, width = img.shape
        
        if vertices is None:
            # Default vertices for lane detection
            vertices = np.array([[
                (width * 0.1, height),           # Bottom-left
                (width * 0.45, height * 0.6),    # Top-left
                (width * 0.55, height * 0.6),    # Top-right
                (width * 0.9, height)            # Bottom-right
            ]], dtype=np.int32)
        
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(img, mask)
        
        return masked
    
    def detect_lines(self, edges: np.ndarray) -> List:
        """Detect lines using Hough Transform"""
        lines = cv2.HoughLinesP(
            edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        return lines if lines is not None else []
    
    def separate_lines(self, lines: List, image_width: int) -> Tuple[List, List]:
        """Separate lines into left and right lanes"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Avoid division by zero
            if x2 - x1 == 0:
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter horizontal lines
            if abs(slope) < 0.5:
                continue
            
            # Classify as left or right lane
            if slope < 0 and x1 < image_width // 2 and x2 < image_width // 2:
                left_lines.append(line)
            elif slope > 0 and x1 > image_width // 2 and x2 > image_width // 2:
                right_lines.append(line)
        
        return left_lines, right_lines
    
    def average_lines(self, lines: List, image_height: int) -> Optional[np.ndarray]:
        """Average multiple lines into a single line"""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        # Fit line using linear regression
        if len(x_coords) > 1:
            coeffs = np.polyfit(y_coords, x_coords, 1)
            
            # Create line from bottom to middle of image
            y1 = image_height
            y2 = int(image_height * 0.6)
            x1 = int(np.polyval(coeffs, y1))
            x2 = int(np.polyval(coeffs, y2))
            
            return np.array([x1, y1, x2, y2])
        
        return None
    
    def calculate_curvature(self, left_line: np.ndarray, 
                          right_line: np.ndarray, 
                          image_height: int) -> float:
        """Calculate road curvature"""
        if left_line is None or right_line is None:
            return 0.0
        
        # Convert pixels to meters
        ym_per_pix = 30 / image_height  # meters per pixel in y
        xm_per_pix = 3.7 / 700  # lane width in meters / pixels
        
        # Generate points for curve fitting
        ploty = np.linspace(0, image_height - 1, image_height)
        
        # Fit second order polynomials
        left_fit = np.polyfit(ploty * ym_per_pix, 
                             left_line[0] * xm_per_pix, 2)
        right_fit = np.polyfit(ploty * ym_per_pix, 
                              right_line[0] * xm_per_pix, 2)
        
        # Calculate curvature at bottom of image
        y_eval = np.max(ploty) * ym_per_pix
        
        left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit[0])
        right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit[0])
        
        return (left_curvature + right_curvature) / 2
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Main detection function"""
        height, width = image.shape[:2]
        
        # Preprocess
        edges = self.preprocess(image)
        roi_edges = self.region_of_interest(edges)
        
        # Detect lines
        lines = self.detect_lines(roi_edges)
        
        # Separate and average lines
        left_lines, right_lines = self.separate_lines(lines, width)
        left_lane = self.average_lines(left_lines, height)
        right_lane = self.average_lines(right_lines, height)
        
        # Calculate curvature
        curvature = self.calculate_curvature(left_lane, right_lane, height)
        
        # Calculate vehicle offset from center
        if left_lane is not None and right_lane is not None:
            lane_center = (left_lane[0] + right_lane[0]) / 2
            image_center = width / 2
            offset = (lane_center - image_center) * (3.7 / 700)  # Convert to meters
        else:
            offset = 0.0
        
        return {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'curvature': curvature,
            'offset': offset,
            'edges': edges,
            'roi_edges': roi_edges,
            'lines': lines
        }

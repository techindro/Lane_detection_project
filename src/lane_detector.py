import cv2
import numpy as np
from .preprocess import preprocess_frame
from .utils import draw_lines

class LaneDetector:
    def __init__(self):
        self.left_lines = []
        self.right_lines = []
        
    def hough_lines(self, img):
        lines = cv2.HoughLinesP(img, 2, np.pi/180, 100, 
                               np.array([]), minLineLength=40, maxLineGap=5)
        return lines
    
    def separate_lines(self, lines, img_shape):
        left = []
        right = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            
            # Filter horizontal lines
            if abs(slope) < 0.5: continue
                
            if slope < 0:
                left.append(line[0])
            else:
                right.append(line[0])
                
        return left, right
    
    def average_line(self, lines):
        if not lines: return np.array([])
        
        x1s = [line[0] for line in lines]
        y1s = [line[1] for line in lines]
        x2s = [line[2] for line in lines]
        y2s = [line[3] for line in lines]
        
        return np.array([np.mean(x1s), np.mean(y1s), 
                        np.mean(x2s), np.mean(y2s)], dtype=np.float32)
    
    def process_frame(self, frame):
        roi = preprocess_frame(frame)
        lines = self.hough_lines(roi)
        
        if lines is not None:
            left, right = self.separate_lines(lines, frame.shape)
            
            # Average lines
            left_line = self.average_line(left)
            right_line = self.average_line(right)
            
            # Store for averaging
            self.left_lines.append(left_line)
            self.right_lines.append(right_line)
            
            # Keep recent 10 lines for smoothing
            if len(self.left_lines) > 10:
                self.left_lines.pop(0)
                self.right_lines.pop(0)
            
            # Average recent lines
            final_left = self.average_line(self.left_lines)
            final_right = self.average_line(self.right_lines)
            
            line_img = draw_lines(frame, [final_left, final_right])
        else:
            line_img = np.zeros_like(frame)
            
        return cv2.addWeighted(frame, 0.8, line_img, 1, 1)

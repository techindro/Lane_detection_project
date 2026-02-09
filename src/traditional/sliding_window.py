"""
Sliding window based lane detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

class SlidingWindowDetector:
    """Lane detection using sliding window approach"""
    
    def __init__(self):
        # Parameters
        self.n_windows = 9
        self.margin = 100
        self.min_pixels = 50
        self.smoothing_factor = 15
        
    def perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transform to bird's eye view"""
        height, width = image.shape[:2]
        
        # Source points (trapezoid)
        src = np.float32([
            [width * 0.15, height * 0.9],
            [width * 0.45, height * 0.65],
            [width * 0.55, height * 0.65],
            [width * 0.85, height * 0.9]
        ])
        
        # Destination points (rectangle)
        dst = np.float32([
            [width * 0.2, height],
            [width * 0.2, 0],
            [width * 0.8, 0],
            [width * 0.8, height]
        ])
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        # Warp image
        warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
        
        return warped, M, Minv
    
    def create_binary_image(self, image: np.ndarray) -> np.ndarray:
        """Create binary image for lane detection"""
        # Convert to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        # Extract L and S channels
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        # Apply Sobel operator to L channel
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Threshold Sobel
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        
        # Threshold S channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1
        
        # Combine thresholds
        binary = np.zeros_like(sobel_binary)
        binary[(sobel_binary == 1) | (s_binary == 1)] = 1
        
        return binary
    
    def find_lane_pixels(self, binary_warped: np.ndarray) -> Tuple:
        """Find lane pixels using sliding window"""
        # Take histogram of bottom half
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        # Find peaks for left and right lanes
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Set window height
        window_height = np.int(binary_warped.shape[0] // self.n_windows)
        
        # Identify x and y positions of all nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Lists to receive lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through windows
        for window in range(self.n_windows):
            # Window boundaries
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            
            # Identify nonzero pixels in window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter next window if enough pixels found
            if len(good_left_inds) > self.min_pixels:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.min_pixels:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate arrays
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        return leftx, lefty, rightx, righty
    
    def fit_polynomial(self, binary_warped: np.ndarray, 
                      leftx: np.ndarray, lefty: np.ndarray,
                      rightx: np.ndarray, righty: np.ndarray) -> Tuple:
        """Fit polynomial to lane pixels"""
        # Fit second order polynomial
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
        
        return left_fit, right_fit, left_fitx, right_fitx, ploty
    
    def measure_curvature(self, ploty: np.ndarray, 
                         left_fit: np.ndarray, 
                         right_fit: np.ndarray) -> Tuple:
        """Measure curvature of lanes"""
        # Define conversions in x and y from pixels to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fit * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fit * xm_per_pix, 2)
        
        # Calculate curvature at bottom of image
        y_eval = np.max(ploty)
        
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + 
                              left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + 
                               right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
        
        return left_curverad, right_curverad
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """Main detection function"""
        # Apply perspective transform
        warped, M, Minv = self.perspective_transform(image)
        
        # Create binary image
        binary_warped = self.create_binary_image(warped)
        
        # Find lane pixels
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)
        
        # Fit polynomial
        left_fit, right_fit, left_fitx, right_fitx, ploty = \
            self.fit_polynomial(binary_warped, leftx, lefty, rightx, righty)
        
        # Measure curvature
        left_curverad, right_curverad = self.measure_curvature(ploty, left_fit, right_fit)
        curvature = (left_curverad + right_curverad) / 2
        
        # Calculate vehicle offset
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        image_center = binary_warped.shape[1] / 2
        offset = (lane_center - image_center) * (3.7 / 700)
        
        # Create visualization
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # Recast x and y points for cv2.fillPoly
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Warp back to original perspective
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        
        return {
            'left_fit': left_fit,
            'right_fit': right_fit,
            'curvature': curvature,
            'offset': offset,
            'binary_warped': binary_warped,
            'warped': warped,
            'newwarp': newwarp
        }

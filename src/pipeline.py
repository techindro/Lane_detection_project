"""
Main pipeline for lane detection
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from .config import config
from .traditional.hough_detector import HoughLaneDetector
from .traditional.sliding_window import SlidingWindowDetector
from .deep_learning.predictor import DeepLearningPredictor
from .utils.visualization import LaneVisualizer

class LaneDetectionPipeline:
    """Main pipeline class"""
    
    def __init__(self, method: str = "traditional"):
        """
        Initialize pipeline
        
        Args:
            method: Detection method
        """
        self.method = method
        self.visualizer = LaneVisualizer()
        
        # Initialize detector based on method
        if method == "traditional":
            self.detector = HoughLaneDetector()
        elif method == "sliding_window":
            self.detector = SlidingWindowDetector()
        elif method == "deep_learning":
            self.detector = DeepLearningPredictor()
        elif method == "hybrid":
            self.detectors = {
                'hough': HoughLaneDetector(),
                'sliding_window': SlidingWindowDetector(),
                'deep_learning': DeepLearningPredictor()
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            
        Returns:
            Dictionary with detection results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Process based on method
        if self.method == "hybrid":
            results = self._process_hybrid(image)
        else:
            results = self.detector.detect(image)
        
        # Visualize results
        visualized = self._visualize_results(image, results)
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, visualized)
            print(f"Result saved to: {output_path}")
        else:
            cv2.imshow("Lane Detection", visualized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results
    
    def _process_hybrid(self, image: np.ndarray) -> Dict[str, Any]:
        """Process image with all methods for comparison"""
        results = {}
        
        for name, detector in self.detectors.items():
            print(f"Running {name}...")
            results[name] = detector.detect(image.copy())
        
        return results
    
    def _visualize_results(self, image: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """Visualize detection results"""
        if self.method == "hybrid":
            # Create comparison visualization
            images = [image.copy()]
            titles = ["Original"]
            
            for name, result in results.items():
                visualized = self._visualize_single_result(image.copy(), result, name)
                images.append(visualized)
                titles.append(name.replace('_', ' ').title())
            
            # Create grid of images
            return self._create_image_grid(images, titles)
        else:
            # Visualize single method
            return self._visualize_single_result(image, results)
    
    def _visualize_single_result(self, image: np.ndarray, 
                                result: Dict[str, Any], 
                                method_name: str = "") -> np.ndarray:
        """Visualize results from single method"""
        visualized = image.copy()
        
        if 'left_lane' in result and 'right_lane' in result:
            # Traditional method results
            left_lane = result['left_lane']
            right_lane = result['right_lane']
            
            if left_lane is not None and right_lane is not None:
                # Draw lane area
                visualized = self.visualizer.draw_lane_area(
                    visualized, left_lane, right_lane
                )
                
                # Draw lane lines
                visualized = self.visualizer.draw_lanes(
                    visualized, left_lane, right_lane
                )
                
                # Add curvature and offset info
                curvature = result.get('curvature', 0)
                offset = result.get('offset', 0)
                visualized = self.visualizer.draw_curvature_text(
                    visualized, curvature, offset
                )
        
        elif 'mask' in result:
            # Deep learning method results
            mask = result['mask']
            if mask is not None:
                # Overlay mask on image
                colored_mask = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
                visualized = cv2.addWeighted(visualized, 0.7, colored_mask, 0.3, 0)
        
        # Add method name if provided
        if method_name:
            cv2.putText(visualized, method_name, (20, visualized.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return visualized
    
    def _create_image_grid(self, images: list, titles: list) -> np.ndarray:
        """Create grid of images for comparison"""
        n_images = len(images)
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Resize all images to same size
        target_size = (400, 300)
        resized_images = []
        for img in images:
            resized = cv2.resize(img, target_size)
            resized_images.append(resized)
        
        # Create grid
        rows = []
        for i in range(0, n_images, grid_size):
            row_images = resized_images[i:i+grid_size]
            row_titles = titles[i:i+grid_size]
            
            # Add titles to images
            titled_images = []
            for img, title in zip(row_images, row_titles):
                img_with_title = img.copy()
                cv2.putText(img_with_title, title, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                titled_images.append(img_with_title)
            
            # Concatenate horizontally
            if len(titled_images) < grid_size:
                # Add blank images to fill row
                blank = np.zeros_like(titled_images[0])
                for _ in range(grid_size - len(titled_images)):
                    titled_images.append(blank)
            
            row = np.concatenate(titled_images, axis=1)
            rows.append(row)
        
        # Concatenate vertically
        grid = np.concatenate(rows, axis=0)
        
        return grid
    
    def process_video(self, video_path: str, output_path: Optional[str] = None):
        """
        Process video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")
        
        # Initialize video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame for performance
            if frame_count % 3 == 0:  # Skip some frames for speed
                continue
            
            # Process frame
            start_time = cv2.getTickCount()
            
            if self.method == "hybrid":
                results = {}
                for name, detector in self.detectors.items():
                    results[name] = detector.detect(frame.copy())
                visualized = self._visualize_results(frame, results)
            else:
                results = self.detector.detect(frame)
                visualized = self._visualize_single_result(frame, results)
            
            # Calculate processing time
            end_time = cv2.getTickCount()
            processing_time = (end_time - start_time) / cv2.getTickFrequency()
            processing_times.append(processing_time)
            
            # Add FPS counter
            current_fps = 1 / processing_time if processing_time > 0 else 0
            cv2.putText(visualized, f"FPS: {current_fps:.1f}", 
                       (width - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write or display
            if output_path:
                out.write(visualized)
            else:
                cv2.imshow("Lane Detection", visualized)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
            print(f"Processed video saved to: {output_path}")
        cv2.destroyAllWindows()
        
        # Print statistics
        if processing_times:
            avg_processing_time = np.mean(processing_times)
            avg_fps = 1 / avg_processing_time
            print(f"\nProcessing Statistics:")
            print(f"Total frames processed: {len(processing_times)}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average processing time: {avg_processing_time*1000:.2f} ms")
    
    def process_webcam(self):
        """Real-time webcam processing"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        print("Starting webcam lane detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            results = self.detector.detect(frame)
            visualized = self._visualize_single_result(frame, results)
            
            # Display
            cv2.imshow("Lane Detection - Webcam", visualized)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

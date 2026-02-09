"""
Evaluation metrics for lane detection
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        pred_mask: Predicted binary mask
        true_mask: Ground truth binary mask
        
    Returns:
        Dictionary of metrics
    """
    # Flatten masks
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(true_flat, pred_flat)
    precision = precision_score(true_flat, pred_flat, zero_division=0)
    recall = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    iou = calculate_iou(pred_mask, true_mask)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

def compare_methods(dataset_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare different lane detection methods
    
    Args:
        dataset_path: Path to test dataset
        output_path: Path to save comparison results
        
    Returns:
        DataFrame with comparison results
    """
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.traditional.hough_detector import HoughLaneDetector
    from src.traditional.sliding_window import SlidingWindowDetector
    from src.deep_learning.predictor import DeepLearningPredictor
    
    # Initialize detectors
    detectors = {
        'Hough Transform': HoughLaneDetector(),
        'Sliding Window': SlidingWindowDetector(),
        'Deep Learning': DeepLearningPredictor()
    }
    
    # Collect test images
    test_images = list(Path(dataset_path).glob("*.jpg"))[:10]  # Test on 10 images
    print(f"Testing on {len(test_images)} images")
    
    results = []
    
    for image_path in test_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        for method_name, detector in detectors.items():
            try:
                # Detect lanes
                start_time = cv2.getTickCount()
                results_dict = detector.detect(image.copy())
                end_time = cv2.getTickCount()
                
                # Calculate processing time
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                fps = 1 / processing_time if processing_time > 0 else 0
                
                # Store results
                results.append({
                    'image': image_path.name,
                    'method': method_name,
                    'processing_time': processing_time,
                    'fps': fps,
                    'curvature': results_dict.get('curvature', 0),
                    'offset': results_dict.get('offset', 0)
                })
                
            except Exception as e:
                print(f"Error processing {image_path.name} with {method_name}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate average metrics per method
    summary = df.groupby('method').agg({
        'processing_time': 'mean',
        'fps': 'mean',
        'curvature': 'mean',
        'offset': 'mean'
    }).round(3)
    
    print("\n" + "=" * 60)
    print("METHOD COMPARISON RESULTS")
    print("=" * 60)
    print(summary)
    
    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        summary.to_csv(output_path.with_name(output_path.stem + '_summary.csv'))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        metrics_to_plot = ['fps', 'processing_time', 'curvature', 'offset']
        titles = ['FPS (Higher is better)', 'Processing Time (Lower is better)', 
                 'Curvature', 'Offset']
        
        for ax, metric, title in zip(axes, metrics_to_plot, titles):
            method_data = []
            for method in detectors.keys():
                method_values = df[df['method'] == method][metric]
                method_data.append(method_values.values)
            
            ax.boxplot(method_data, labels=detectors.keys())
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {output_path.with_suffix('.png')}")
    
    return df

def generate_report(results_df: pd.DataFrame, output_path: str):
    """
    Generate detailed evaluation report
    
    Args:
        results_df: DataFrame with results
        output_path: Path to save report
    """
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(output_path) as pdf:
        # Page 1: Summary statistics
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        summary = results_df.groupby('method').agg({
            'fps': ['mean', 'std'],
            'processing_time': ['mean', 'std'],
            'curvature': ['mean', 'std'],
            'offset': ['mean', 'std']
        }).round(3)
        
        # Create table
        table_data = []
        for method in summary.index:
            row = [method]
            for col in summary.columns:
                row.append(summary.loc[method, col])
            table_data.append(row)
        
        columns = ['Method'] + [f'{col[0]}_{col[1]}' for col in summary.columns]
        table = ax.table(cellText=table_data, colLabels=columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Lane Detection Methods Comparison', fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Bar charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['fps', 'processing_time', 'curvature', 'offset']
        titles = ['Frames per Second', 'Processing Time (seconds)', 
                 'Curvature (meters)', 'Offset from Center (meters)']
        
        for ax, metric, title in zip(axes, metrics, titles):
            avg_values = results_df.groupby('method')[metric].mean()
            std_values = results_df.groupby('method')[metric].std()
            
            x_pos = np.arange(len(avg_values))
            bars = ax.bar(x_pos, avg_values.values, yerr=std_values.values,
                         capsize=5, color=plt.cm.Set3(np.arange(len(avg_values))))
            
            ax.set_xlabel('Method', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(avg_values.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, avg_values.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        print(f"Report generated: {output_path}")

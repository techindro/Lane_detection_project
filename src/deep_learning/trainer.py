"""
Model training utilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

from .model import LaneDetectionModel, LaneDetectionTrainer
from .dataloader import create_dataloaders
from ..config import config

def train_model(data_dir: str, model_name: str = "unet", 
                epochs: int = 50, batch_size: int = 8,
                save_dir: str = "models"):
    """
    Train lane detection model
    
    Args:
        data_dir: Path to dataset
        model_name: Model architecture ('unet' or 'lanenet')
        epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save models
    """
    print("=" * 60)
    print("Training Lane Detection Model")
    print("=" * 60)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, batch_size=batch_size, image_size=config.IMAGE_SIZE
    )
    
    # Initialize model
    print(f"Initializing {model_name} model...")
    if model_name == "unet":
        model = LaneDetectionModel(num_classes=config.NUM_CLASSES, 
                                   encoder=config.ENCODER)
    elif model_name == "lanenet":
        model = LaneNet(num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize trainer
    trainer = LaneDetectionTrainer(model, device)
    
    # Setup tensorboard
    writer = SummaryWriter(f"runs/lane_detection_{model_name}_{int(time.time())}")
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    history = trainer.train(
        train_loader, val_loader, epochs=epochs, 
        lr=config.LEARNING_RATE,
        save_path=str(save_path / f"{model_name}_best.pth")
    )
    
    # Test model
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_metrics[0]:.4f}")
    print(f"Test F1: {test_metrics[1]:.4f}")
    print(f"Test Accuracy: {test_metrics[2]:.4f}")
    
    # Save final model
    final_model_path = save_path / f"{model_name}_final.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': test_loss,
        'metrics': test_metrics,
        'history': history
    }, final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = save_path / f"{model_name}_history.npy"
    np.save(history_path, history)
    
    # Log to tensorboard
    for epoch in range(epochs):
        writer.add_scalar('Loss/train', history['train_loss'][epoch], epoch)
        writer.add_scalar('Loss/val', history['val_loss'][epoch], epoch)
        writer.add_scalar('Metrics/IoU', history['val_iou'][epoch], epoch)
        writer.add_scalar('Metrics/F1', history['val_f1'][epoch], epoch)
        writer.add_scalar('Metrics/Accuracy', history['val_accuracy'][epoch], epoch)
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return model, history

def evaluate_model(model_path: str, test_loader: DataLoader, 
                   device: str = "cuda") -> Dict[str, float]:
    """
    Evaluate trained model
    
    Args:
        model_path: Path to saved model
        test_loader: Test dataloader
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    if 'encoder' in checkpoint:
        model = LaneDetectionModel(num_classes=checkpoint.get('num_classes', 2),
                                   encoder=checkpoint.get('encoder', 'resnet34'))
    else:
        model = LaneNet(num_classes=checkpoint.get('num_classes', 2))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Evaluate
    trainer = LaneDetectionTrainer(model, device)
    loss, metrics = trainer.validate(test_loader)
    
    return {
        'loss': loss,
        'iou': metrics[0],
        'f1': metrics[1],
        'accuracy': metrics[2]
    }

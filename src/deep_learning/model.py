"""
Deep Learning model for lane detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Tuple, Dict, Any, Optional

class LaneDetectionModel(nn.Module):
    """U-Net based lane detection model"""
    
    def __init__(self, num_classes: int = 2, encoder: str = "resnet34"):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary prediction"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

class LaneNet(nn.Module):
    """Custom LaneNet implementation"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(3, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            nn.MaxPool2d(2),
            self._conv_block(256, 512),
            nn.MaxPool2d(2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            self._upconv_block(512, 256),
            self._upconv_block(256, 128),
            self._upconv_block(128, 64),
            self._upconv_block(64, 32)
        )
        
        # Output
        self.output = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.encoder[0:2](x)
        enc2 = self.encoder[2:4](enc1)
        enc3 = self.encoder[4:6](enc2)
        enc4 = self.encoder[6:8](enc3)
        
        # Decoder
        dec1 = self.decoder[0](enc4)
        dec1 = torch.cat([dec1, enc3], dim=1)
        
        dec2 = self.decoder[1](dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec3 = self.decoder[2](dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        
        dec4 = self.decoder[3](dec3)
        
        # Output
        output = self.output(dec4)
        
        return output

class LaneDetectionTrainer:
    """Training wrapper for lane detection model"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss function
        self.criterion = smp.losses.DiceLoss(mode='binary')
        
        # Metrics
        self.metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(threshold=0.5),
            smp.utils.metrics.Accuracy(threshold=0.5)
        ]
        
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
                
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        metrics_values = [0] * len(self.metrics)
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                for i, metric in enumerate(self.metrics):
                    metrics_values[i] += metric(outputs, masks).item()
                    
        val_loss /= len(dataloader)
        metrics_values = [m / len(dataloader) for m in metrics_values]
        
        return val_loss, metrics_values
    
    def train(self, train_loader, val_loader, epochs: int = 50, 
             lr: float = 1e-4, save_path: str = None):
        """Complete training process"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        best_val_loss = float('inf')
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_f1': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(metrics[0])
            history['val_f1'].append(metrics[1])
            history['val_accuracy'].append(metrics[2])
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val IoU: {metrics[0]:.4f}")
            print(f"Val F1: {metrics[1]:.4f}")
            print(f"Val Accuracy: {metrics[2]:.4f}")
            
            # Save best model
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics
                }, save_path)
                print(f"Model saved to {save_path}")
            
            # Update learning rate
            scheduler.step(val_loss)
            
        return history

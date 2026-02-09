"""
Data loader for lane detection dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LaneDataset(Dataset):
    """Dataset class for lane detection"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform: Optional[A.Compose] = None,
                 image_size: Tuple[int, int] = (640, 360)):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            transform: Albumentations transform
            image_size: Target image size (width, height)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.image_size = image_size
        
        # Validate paths
        assert len(image_paths) == len(mask_paths), \
            "Number of images and masks must match"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if self.image_size:
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask
    
    @staticmethod
    def get_transforms(augment: bool = False) -> A.Compose:
        """Get data transforms"""
        if augment:
            # Training transforms with augmentation
            return A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Validation/Test transforms
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])

def create_dataloaders(data_dir: str, batch_size: int = 8, 
                       image_size: Tuple[int, int] = (640, 360)) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        image_size: Target image size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    # Collect image and mask paths
    image_paths = []
    mask_paths = []
    
    for image_file in data_dir.rglob("*.jpg"):
        mask_file = data_dir / "masks" / image_file.name.replace(".jpg", "_mask.png")
        if mask_file.exists():
            image_paths.append(str(image_file))
            mask_paths.append(str(mask_file))
    
    print(f"Found {len(image_paths)} images")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_images, test_images, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    val_images, test_images, val_masks, test_masks = train_test_split(
        test_images, test_masks, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    
    # Create datasets
    train_transform = LaneDataset.get_transforms(augment=True)
    val_transform = LaneDataset.get_transforms(augment=False)
    
    train_dataset = LaneDataset(train_images, train_masks, train_transform, image_size)
    val_dataset = LaneDataset(val_images, val_masks, val_transform, image_size)
    test_dataset = LaneDataset(test_images, test_masks, val_transform, image_size)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

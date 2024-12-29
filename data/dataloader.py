import torch
from torch.utils.data import DataLoader, random_split, Dataset
from datasets import load_dataset
from torchvision import transforms
import logging
from PIL import Image
import os
from datasets import config
import shutil
import time

logger = logging.getLogger(__name__)

class ImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # This is a PIL image
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        label = item['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ImageNetDataModule:
    def __init__(self, config):
        self.config = config
        self.train_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure RGB
            transforms.RandomResizedCrop(config['data']['image_size']),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),  # Ensure RGB
            transforms.Resize(256),
            transforms.CenterCrop(config['data']['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def setup(self):
        logger.info("Loading ImageNet dataset...")
        dataset = load_dataset("imagenet-1k", split="train")
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Split dataset
        total_size = len(dataset)
        train_size = int(total_size * self.config['data']['train_val_split'])
        val_size = total_size - train_size
        
        logger.info(f"Splitting dataset: {train_size} training, {val_size} validation")
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Wrap datasets with transforms
        self.train_dataset = ImageNetDataset(train_dataset, self.train_transform)
        self.val_dataset = ImageNetDataset(val_dataset, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=6,  # Increased from 4 to 6
            pin_memory=True,
            prefetch_factor=2,  # Add prefetch_factor
            persistent_workers=True  # Keep workers alive between epochs
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        ) 
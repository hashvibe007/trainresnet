import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import logging
import time
from pathlib import Path
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training']['momentum'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize learning rate scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config['training']['lr_scheduler']['step_size'],
            gamma=config['training']['lr_scheduler']['gamma']
        )
        
        # Initialize tensorboard writer with timestamp
        if config['logging']['tensorboard']:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            self.log_dir = Path.cwd() / config['logging']['log_dir'] / current_time
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            
            # Log hyperparameters
            self.writer.add_text('Hyperparameters', str(config))
            logger.info(f"TensorBoard logs will be saved to {self.log_dir}")
            
            # Flush immediately to ensure directory is created
            self.writer.flush()
        
        self.best_acc = 0.0
        self.start_epoch = 0
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
        logger.info("Initialized mixed precision training")
        
        # Check for best model and load it
        best_model_path = Path(config['checkpoint']['save_dir']) / "best_model.pth"
        if best_model_path.exists():
            logger.info(f"Found best model at {best_model_path}, loading it...")
            self.load_checkpoint(best_model_path)
            logger.info(f"Resumed from epoch {self.start_epoch} with best accuracy {self.best_acc:.2f}%")
        elif config['checkpoint']['resume'] and config['checkpoint']['resume_path']:
            self.load_checkpoint(config['checkpoint']['resume_path'])

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{self.config["training"]["epochs"]}')
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Use mixed precision training
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config['training']['gradient_accumulation_steps']
            
            # Scale loss and do backward pass
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Update metrics
            running_loss += loss.item() * self.config['training']['gradient_accumulation_steps']
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Handle remaining gradients
        if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.*correct/total
        
        # Log epoch results
        logger.info(
            f"\nEpoch {epoch+1}/{self.config['training']['epochs']}:\n"
            f"Training   - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%\n"
            f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # TensorBoard logging
        if self.config['logging']['tensorboard']:
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
            self.writer.add_scalar('Train/LearningRate', 
                                 self.optimizer.param_groups[0]['lr'], epoch)
            
            # Add histograms of model parameters
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(f'Parameters/{name}', 
                                        param.data.cpu().numpy(), epoch)
            
            self.writer.flush()
        
        return epoch_loss, epoch_acc

    def validate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Simple progress bar without metrics
        pbar = tqdm(val_loader, desc='Validating')
        
        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss/len(val_loader)
        epoch_acc = 100.*correct/total
        
        # Log validation results
        logger.info(
            f"Validation - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, "
            f"Best Acc: {self.best_acc:.2f}%\n"
            f"{'-'*80}"  # Add separator line
        )
        
        # TensorBoard logging
        if self.config['logging']['tensorboard']:
            self.writer.add_scalar('Val/Loss', epoch_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
            self.writer.add_scalar('Val/BestAccuracy', self.best_acc, epoch)
            self.writer.flush()
        
        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # Save scaler state
            'best_acc': self.best_acc,
            'val_acc': val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save(state, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if accuracy improved
        if val_acc > self.best_acc:
            prev_best = self.best_acc
            self.best_acc = val_acc
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save(state, best_model_path)
            logger.info(
                f"New best model saved! Accuracy improved from "
                f"{prev_best:.2f}% to {val_acc:.2f}%"
            )

    def load_checkpoint(self, checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Debug checkpoint contents
        logger.info("Checkpoint contains:")
        logger.info(f"- Keys: {list(checkpoint.keys())}")
        if 'val_acc' in checkpoint:
            logger.info(f"- Saved val_acc: {checkpoint['val_acc']:.2f}%")
        if 'best_acc' in checkpoint:
            logger.info(f"- Saved best_acc: {checkpoint['best_acc']:.2f}%")
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        
        # Modified accuracy loading
        if 'best_acc' in checkpoint:
            self.best_acc = checkpoint['best_acc']
        elif 'val_acc' in checkpoint:
            self.best_acc = checkpoint['val_acc']
        else:
            logger.warning("No accuracy found in checkpoint!")
            self.best_acc = 0.0
        
        logger.info(f"Successfully loaded checkpoint:")
        logger.info(f"- Epoch: {self.start_epoch}")
        logger.info(f"- Best Accuracy: {self.best_acc:.2f}%")
        logger.info(f"- Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

    def __del__(self):
        # Ensure writer is properly closed
        if hasattr(self, 'writer'):
            self.writer.close() 
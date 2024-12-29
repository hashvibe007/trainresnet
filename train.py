import yaml
import torch
import logging
from pathlib import Path
from models import resnet50
from data import ImageNetDataModule
from utils import Trainer
import torch.backends.cudnn as cudnn
import os
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Ensure directories exist
    for directory in ['runs', 'logs', 'checkpoints']:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        logger.info(f"Directory {directory} is ready at {dir_path.absolute()}")

    # Clear existing tensorboard runs if needed
    runs_dir = Path('runs')
    if runs_dir.exists():
        shutil.rmtree(runs_dir)
    runs_dir.mkdir(exist_ok=True)

    # Create model
    logger.info("Creating ResNet50 model...")
    model = resnet50(num_classes=config['model']['num_classes'])

    # Initialize data module
    logger.info("Initializing data module...")
    data_module = ImageNetDataModule(config)
    data_module.setup()

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(model, config)

    # Enable cuDNN auto-tuner
    cudnn.benchmark = True
    
    # Set environment variable for memory allocation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Empty cache before training
    torch.cuda.empty_cache()

    # Training loop
    logger.info("Starting training...")
    try:
        for epoch in range(trainer.start_epoch, config['training']['epochs']):
            # Train epoch
            train_loss, train_acc = trainer.train_epoch(
                data_module.train_dataloader(), epoch
            )
            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            )
            
            # Validate epoch
            val_loss, val_acc = trainer.validate(
                data_module.val_dataloader(), epoch
            )
            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            if (epoch + 1) % config['checkpoint']['save_freq'] == 0:
                trainer.save_checkpoint(epoch + 1, val_acc)
            
            # Step the learning rate scheduler
            trainer.scheduler.step()
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(epoch + 1, val_acc)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
        
    finally:
        if config['logging']['tensorboard']:
            trainer.writer.close()

if __name__ == '__main__':
    main() 
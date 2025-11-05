"""
Training script for BGT60TR13C finger classification model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import yaml
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import os

from model import create_model, count_parameters


class RangeDopplerDataset(Dataset):
    """Dataset for BGT60TR13C Range-Doppler maps"""
    
    def __init__(self, data_path, augment=False, aug_config=None):
        """
        Args:
            data_path: Path to .npz file with 'X' and 'y' keys
            augment: Whether to apply data augmentation
            aug_config: Augmentation configuration dictionary
        """
        data = np.load(data_path)
        self.X = data['X']  # Shape: (N, 128, 256)
        self.y = data['y']  # Shape: (N,)
        self.augment = augment
        self.aug_config = aug_config or {}
        
        print(f"Loaded dataset: {len(self.X)} samples")
        print(f"  Data shape: {self.X.shape}")
        print(f"  Label shape: {self.y.shape}")
        print(f"  Unique labels: {np.unique(self.y)}")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.y[idx]
        
        # Apply augmentation if enabled
        if self.augment and self.aug_config.get('enabled', False):
            x = self._augment(x)
        
        # Add channel dimension: (128, 256) -> (1, 128, 256)
        x = np.expand_dims(x, axis=0)
        
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)
    
    def _augment(self, x):
        """Apply data augmentation to Range-Doppler map"""
        
        # Add Gaussian noise
        if 'noise_std' in self.aug_config:
            noise = np.random.randn(*x.shape) * self.aug_config['noise_std']
            x = x + noise
        
        # Random intensity scaling
        if 'intensity_scale' in self.aug_config:
            scale_range = self.aug_config['intensity_scale']
            scale = np.random.uniform(scale_range[0], scale_range[1])
            x = x * scale
        
        # Random shift in range/Doppler
        if 'shift_range' in self.aug_config:
            shift_max = self.aug_config['shift_range']
            shift_doppler = np.random.randint(-shift_max, shift_max + 1)
            shift_range = np.random.randint(-shift_max, shift_max + 1)
            x = np.roll(x, shift=(shift_doppler, shift_range), axis=(0, 1))
        
        # Clip to valid range
        x = np.clip(x, 0, 1)
        
        return x


class Trainer:
    """Training manager for BGT60TR13C finger classification"""
    
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_name = config['training']['optimizer'].lower()
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Learning rate scheduler
        scheduler_name = config['training'].get('scheduler', 'none').lower()
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs']
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Tracking
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Paths
        self.model_dir = Path(config['paths']['model_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['paths']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs):
        """Full training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Learning rate schedule
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config['training']['learning_rate']
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
            else:
                self.epochs_without_improvement += 1
            
            # Save periodic checkpoint
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
            
            # Early stopping
            patience = self.config['training'].get('early_stopping_patience', 15)
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        # Save final model
        self.save_checkpoint(num_epochs, 'final_model.pth')
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.model_dir / filename
        torch.save(checkpoint, save_path)
        # print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train BGT60TR13C finger classifier')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--train-data', type=str, default='data/train_data.npz',
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, default='data/val_data.npz',
                       help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda, cpu, or mps')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Set device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device_name = config.get('device', 'cuda')
        if device_name == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        elif device_name == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print("BGT60TR13C Finger Classification Training")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = RangeDopplerDataset(
        args.train_data,
        augment=True,
        aug_config=config.get('augmentation', {})
    )
    
    val_dataset = RangeDopplerDataset(
        args.val_data,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Create model
    model_config = config['model'].copy()
    model_config['num_classes'] = config['data']['num_classes']
    model = create_model(model_config)
    
    print(f"\nModel: {config['model']['name']}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(config, model, train_loader, val_loader, device)
    
    # Train
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        config['training']['epochs']
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    history_path = Path(config['paths']['model_dir']) / 'training_history.npz'
    np.savez(history_path, **history)
    print(f"\nTraining history saved to: {history_path}")


if __name__ == "__main__":
    main()

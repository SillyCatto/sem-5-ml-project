"""
Training Script for Sign Language Recognition

Trains models combining RAFT optical flow and MediaPipe landmarks.
Optimized for M4 MacBook with MPS acceleration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

try:
    from .sign_classifier import create_model
    from .dataset import create_data_loaders
except ImportError:
    from sign_classifier import create_model
    from dataset import create_data_loaders


class Trainer:
    """Trainer for sign language classification models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: Path,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        use_mixed_precision: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            save_dir: Directory to save checkpoints and logs
            scheduler: Learning rate scheduler
            use_mixed_precision: Whether to use mixed precision training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision
        
        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.save_dir / "logs")
        
        # Mixed precision scaler
        if use_mixed_precision and device.type == "mps":
            self.scaler = torch.amp.GradScaler('mps')
        elif use_mixed_precision and device.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
            self.use_mixed_precision = False
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (landmarks, labels) in enumerate(pbar):
            # Move data to device
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(landmarks)
                    loss = self.criterion(logits, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(landmarks)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, epoch: int) -> Tuple[float, float, Dict]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for landmarks, labels in pbar:
                # Move data to device
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_mixed_precision:
                    with torch.amp.autocast(device_type=self.device.type):
                        logits = self.model(landmarks)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model(landmarks)
                    loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": conf_matrix
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, num_epochs: int):
        """Train the model for multiple epochs."""
        print("=" * 60)
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_mixed_precision}")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            
            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  F1: {val_metrics['f1']:.4f}, LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best model saved! (Acc: {val_acc:.4f})")
            
            # Save regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete in {elapsed/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")
        print("=" * 60)
        
        # Save final history
        self.save_history()
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "config": getattr(self, "config", {}),
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = self.save_dir / "best_model.pth"
        else:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.save_dir / "training_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for v in vals]
            for k, vals in self.history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)


def train_model(
    landmarks_dir: str,
    flow_dir: Optional[str] = None,
    model_type: str = "lstm",
    num_classes: int = 15,
    batch_size: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    save_dir: str = "checkpoints",
    device: Optional[str] = None
):
    """
    Main training function.
    
    Args:
        landmarks_dir: Directory with landmark files
        flow_dir: Deprecated. Kept for backward compatibility and ignored.
        model_type: Type of model ("lstm", "transformer", "hybrid")
        num_classes: Number of sign classes
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        save_dir: Directory to save checkpoints
        device: Device to train on
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    device = torch.device(device)
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        landmarks_dir=Path(landmarks_dir),
        batch_size=batch_size
    )
    
    print(f"\nCreating {model_type} model...")
    model = create_model(model_type=model_type, num_classes=num_classes, device=device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir=Path(save_dir),
        scheduler=scheduler,
        use_mixed_precision=True
    )
    
    # Store config so checkpoints include model architecture info
    trainer.config = {
        "model_type": model_type,
        "num_classes": num_classes,
        "hidden_dim": 256,
        "landmarks_dir": str(landmarks_dir),
        "flow_dir": str(flow_dir) if flow_dir else None,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }
    
    # Train
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    # Example training configuration
    config = {
        "landmarks_dir": "path/to/landmarks",
        "flow_dir": None,  # Optional
        "model_type": "lstm",
        "num_classes": 15,
        "batch_size": 16,
        "num_epochs": 50,
        "learning_rate": 0.001,
        "save_dir": "checkpoints/lstm_run1"
    }
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Uncomment to train
    # train_model(**config)

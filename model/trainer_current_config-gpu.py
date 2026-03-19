"""
Training Script for Sign Language Recognition

Landmark-only training pipeline tuned for NVIDIA RTX-class GPUs
(including RTX 2060 Super 8GB VRAM).
"""

import os
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
        use_mixed_precision: bool = True,
        grad_accum_steps: int = 1,
        max_grad_norm: float = 1.0,
        empty_cache_steps: int = 0,
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
            grad_accum_steps: Number of steps to accumulate gradients
            max_grad_norm: Gradient clipping threshold (<= 0 disables clipping)
            empty_cache_steps: If > 0 and CUDA, call empty_cache every N optimizer steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_mixed_precision = use_mixed_precision
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.max_grad_norm = float(max_grad_norm)
        self.empty_cache_steps = int(empty_cache_steps)
        self.non_blocking = device.type == "cuda"
        
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
        optimizer_steps = 0
        self.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, (landmarks, labels) in enumerate(pbar):
            # Move data to device
            landmarks = landmarks.to(self.device, non_blocking=self.non_blocking)
            labels = labels.to(self.device, non_blocking=self.non_blocking)
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                if self.device.type == "cuda":
                    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    autocast_ctx = torch.amp.autocast(device_type=self.device.type)

                with autocast_ctx:
                    logits = self.model(landmarks)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.grad_accum_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                should_step = (
                    ((batch_idx + 1) % self.grad_accum_steps == 0)
                    or (batch_idx + 1 == len(self.train_loader))
                )
                if should_step:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                    if (
                        self.device.type == "cuda"
                        and self.empty_cache_steps > 0
                        and optimizer_steps % self.empty_cache_steps == 0
                    ):
                        torch.cuda.empty_cache()
            else:
                logits = self.model(landmarks)
                loss = self.criterion(logits, labels)
                loss = loss / self.grad_accum_steps
                loss.backward()

                should_step = (
                    ((batch_idx + 1) % self.grad_accum_steps == 0)
                    or (batch_idx + 1 == len(self.train_loader))
                )
                if should_step:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            
            # Track metrics
            total_loss += loss.item() * self.grad_accum_steps
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

        if len(self.val_loader) == 0:
            metrics = {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confusion_matrix": np.array([]),
            }
            return 0.0, 0.0, metrics
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for landmarks, labels in pbar:
                # Move data to device
                landmarks = landmarks.to(self.device, non_blocking=self.non_blocking)
                labels = labels.to(self.device, non_blocking=self.non_blocking)
                
                # Forward pass
                if self.use_mixed_precision:
                    if self.device.type == "cuda":
                        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                    else:
                        autocast_ctx = torch.amp.autocast(device_type=self.device.type)

                    with autocast_ctx:
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
    batch_size: int = 8,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    save_dir: str = "checkpoints",
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    use_mixed_precision: bool = True,
    grad_accum_steps: int = 2,
    max_grad_norm: float = 1.0,
    empty_cache_steps: int = 0,
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
        num_workers: DataLoader workers (auto if None)
        use_mixed_precision: Enable AMP
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Gradient clipping threshold
        empty_cache_steps: CUDA cache clear interval in optimizer steps
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        gpu_name = torch.cuda.get_device_name(device)
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"[cuda] Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        if batch_size > 16:
            print("[warn] batch_size > 16 may OOM on 8GB VRAM; reduce or increase grad_accum_steps.")
    
    if flow_dir:
        print("[info] flow_dir is ignored in this trainer (landmark-only pipeline).")

    if num_workers is None:
        cpu_count = os.cpu_count() or 4
        num_workers = min(6, max(2, cpu_count // 2))

    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        landmarks_dir=Path(landmarks_dir),
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Rebuild loaders for CUDA to enable faster host->device copies.
    if device.type == "cuda":
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(
            train_loader.dataset,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_loader.dataset,
            shuffle=False,
            **loader_kwargs,
        )

    # Infer model input shape from dataset so trainer matches current extraction config.
    sample_batch, _ = next(iter(train_loader))
    target_sequence_length = int(sample_batch.shape[1])
    landmark_dim = int(sample_batch.shape[2])
    
    print(f"\nCreating {model_type} model...")
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        device=device,
        landmark_dim=landmark_dim,
    )
    
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
        use_mixed_precision=use_mixed_precision,
        grad_accum_steps=grad_accum_steps,
        max_grad_norm=max_grad_norm,
        empty_cache_steps=empty_cache_steps,
    )
    
    # Store config so checkpoints include model architecture info
    trainer.config = {
        "model_type": model_type,
        "num_classes": num_classes,
        "hidden_dim": 256,
        "landmark_dim": landmark_dim,
        "target_sequence_length": target_sequence_length,
        "landmarks_dir": str(landmarks_dir),
        "flow_dir": str(flow_dir) if flow_dir else None,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "use_mixed_precision": use_mixed_precision,
        "grad_accum_steps": grad_accum_steps,
        "max_grad_norm": max_grad_norm,
        "empty_cache_steps": empty_cache_steps,
    }
    
    # Train
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    # Example training configuration
    config = {
        "landmarks_dir": "outputs/landmarks",
        "flow_dir": None,  # Optional
        "model_type": "lstm",
        "num_classes": 15,
        "batch_size": 8,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "save_dir": "checkpoints/lstm_run1",
        "num_workers": 4,
        "use_mixed_precision": True,
        "grad_accum_steps": 2,
        "max_grad_norm": 1.0,
        "empty_cache_steps": 0,
    }
    
    print("Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Uncomment to train
    train_model(**config)

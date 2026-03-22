"""YOLACT Training Loop and Trainer.

Provides the Trainer class that handles:
    - SGD optimizer with momentum and weight decay
    - CosineAnnealingLR scheduler with linear warmup
    - Gradient clipping for training stability
    - Mixed precision training (CUDA only, FP32 for MPS)
    - Checkpoint saving/loading with best model tracking
    - Training/validation loop with loss logging

Device compatibility:
    - MPS (Apple Silicon): Full FP32 training, no AMP/GradScaler
    - CUDA: Optional AMP with GradScaler
    - CPU: Full FP32 training
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import YOLACTLoss
from src.utils.helpers import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """YOLACT model trainer with full training pipeline support.

    Handles the complete training workflow including optimizer setup,
    learning rate scheduling, gradient clipping, checkpointing, and
    training/validation loops with metric logging.

    Args:
        model: YOLACT model instance (already on device).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dictionary with training hyperparameters.
        device: PyTorch device to train on (mps, cuda, or cpu).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Extract training config with defaults
        train_cfg = config.get('training', {})
        loss_cfg = config.get('loss', {})

        self.num_epochs = train_cfg.get('epochs', 20)
        self.lr = train_cfg.get('lr', 0.001)
        self.momentum = train_cfg.get('momentum', 0.9)
        self.weight_decay = train_cfg.get('weight_decay', 5e-4)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 3)
        self.gradient_clip = train_cfg.get('gradient_clip', 10.0)
        self.val_interval = train_cfg.get('val_interval', 2)
        self.log_interval = train_cfg.get('log_interval', 20)

        # Determine if AMP is available and appropriate
        # MPS does NOT support torch.cuda.amp, so we skip AMP for MPS
        use_amp_config = train_cfg.get('amp', True)
        self.use_amp = use_amp_config and device.type == 'cuda'

        # Optimizer: SGD with momentum
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # Scheduler: Cosine annealing (applied after warmup)
        min_lr = train_cfg.get('min_lr', 1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.num_epochs - self.warmup_epochs, 1),
            eta_min=min_lr,
        )

        # GradScaler for mixed precision (CUDA only)
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Loss function
        num_classes = config.get('dataset', {}).get('num_classes', 1) + 1  # +1 for background
        self.criterion = YOLACTLoss(
            num_classes=num_classes,
            cls_weight=loss_cfg.get('cls_weight', 1.0),
            box_weight=loss_cfg.get('box_weight', 1.5),
            mask_weight=loss_cfg.get('mask_weight', 6.125),
            focal_alpha=loss_cfg.get('focal_alpha', 0.25),
            focal_gamma=loss_cfg.get('focal_gamma', 2.0),
            neg_pos_ratio=loss_cfg.get('neg_pos_ratio', 3),
        )

        # Checkpoint tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Training history
        self.history: List[Dict[str, Any]] = []

        # Output directories
        self.checkpoint_dir = Path('results/training/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path('results/training')
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized: device={device}, epochs={self.num_epochs}, "
            f"lr={self.lr}, batch_size={train_cfg.get('batch_size', 'unknown')}, "
            f"amp={self.use_amp}, warmup_epochs={self.warmup_epochs}"
        )

    def _warmup_lr(self, epoch: int) -> None:
        """Apply linear warmup to the learning rate.

        During the first `warmup_epochs` epochs, the learning rate linearly
        increases from a small fraction to the target learning rate.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr * warmup_factor

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run a single training epoch.

        Iterates over the training DataLoader, computes losses, and
        updates model weights with optional gradient clipping.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary with average loss values for the epoch:
                - 'total', 'cls', 'box', 'mask'
        """
        self.model.train()
        running_losses = {'total': 0.0, 'cls': 0.0, 'box': 0.0, 'mask': 0.0}
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # Move images to device
            images = images.to(self.device)

            # Targets are a list of dicts; move tensors to device
            targets_device = []
            for t in targets:
                targets_device.append({
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                })

            self.optimizer.zero_grad()

            if self.use_amp:
                # Mixed precision training (CUDA only)
                with torch.amp.autocast('cuda'):
                    predictions = self.model(images)
                    losses = self.criterion(predictions, targets_device)

                self.scaler.scale(losses['total']).backward()

                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training (MPS / CPU)
                predictions = self.model(images)
                losses = self.criterion(predictions, targets_device)

                losses['total'].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip
                )

                self.optimizer.step()

            # Accumulate losses
            for key in running_losses:
                running_losses[key] += losses[key].item()
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_total = running_losses['total'] / num_batches
                avg_cls = running_losses['cls'] / num_batches
                avg_box = running_losses['box'] / num_batches
                avg_mask = running_losses['mask'] / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - epoch_start
                logger.info(
                    f"  Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {avg_total:.4f} (cls: {avg_cls:.4f}, box: {avg_box:.4f}, "
                    f"mask: {avg_mask:.4f}) LR: {lr:.6f} Time: {elapsed:.1f}s"
                )

        # Average losses
        avg_losses = {
            key: val / max(num_batches, 1) for key, val in running_losses.items()
        }
        epoch_time = time.time() - epoch_start
        avg_losses['epoch_time'] = epoch_time

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics.

        Evaluates the model on the validation set, computing the same
        loss components as training. The model is set to training mode
        temporarily to get the raw prediction outputs needed for loss
        computation (since eval mode returns post-processed detections).

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary with average validation loss values:
                - 'total', 'cls', 'box', 'mask'
        """
        # We need training-mode outputs (raw predictions) for loss computation,
        # but we don't want dropout/etc. to affect results.
        # Since YOLACT only changes its return format between train/eval,
        # we keep it in train mode for loss computation but wrap in no_grad.
        was_training = self.model.training
        self.model.train()  # Need raw predictions for loss computation

        running_losses = {'total': 0.0, 'cls': 0.0, 'box': 0.0, 'mask': 0.0}
        num_batches = 0
        val_start = time.time()

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets_device = []
            for t in targets:
                targets_device.append({
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                })

            predictions = self.model(images)
            losses = self.criterion(predictions, targets_device)

            for key in running_losses:
                running_losses[key] += losses[key].item()
            num_batches += 1

        # Restore training mode
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        avg_losses = {
            key: val / max(num_batches, 1) for key, val in running_losses.items()
        }
        val_time = time.time() - val_start
        avg_losses['val_time'] = val_time

        return avg_losses

    def fit(self, num_epochs: Optional[int] = None) -> Dict[str, List]:
        """Run the full training loop.

        Executes training for the specified number of epochs with:
            - Linear warmup for the first warmup_epochs
            - Cosine annealing learning rate schedule
            - Periodic validation
            - Best model checkpointing
            - Training log saving

        Args:
            num_epochs: Number of epochs to train. If None, uses config value.

        Returns:
            Training history dictionary with per-epoch metrics.
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs

        logger.info(f"\nStarting training for {self.num_epochs} epochs")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches:   {len(self.val_loader)}")
        logger.info(f"  Device:        {self.device}")
        logger.info(f"  AMP:           {self.use_amp}")
        logger.info(f"  Warmup:        {self.warmup_epochs} epochs")
        print(f"\n{'='*70}")
        print(f"  Training YOLACT for {self.num_epochs} epochs on {self.device}")
        print(f"{'='*70}\n")

        training_start = time.time()

        for epoch in range(self.num_epochs):
            epoch_record: Dict[str, Any] = {'epoch': epoch + 1}

            # Learning rate warmup
            if epoch < self.warmup_epochs:
                self._warmup_lr(epoch)
            elif epoch == self.warmup_epochs:
                # Reset scheduler after warmup
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"\nEpoch {epoch+1}/{self.num_epochs} (lr={current_lr:.6f})")

            # --- Training ---
            train_losses = self.train_epoch(epoch)
            epoch_record['train'] = train_losses
            epoch_record['lr'] = current_lr

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"Train Loss: {train_losses['total']:.4f} "
                f"(cls: {train_losses['cls']:.4f}, "
                f"box: {train_losses['box']:.4f}, "
                f"mask: {train_losses['mask']:.4f}) "
                f"LR: {current_lr:.6f} "
                f"Time: {train_losses.get('epoch_time', 0):.1f}s"
            )

            # --- Validation ---
            if (epoch + 1) % self.val_interval == 0 or epoch == self.num_epochs - 1:
                val_losses = self.validate(epoch)
                epoch_record['val'] = val_losses

                print(
                    f"         Val Loss:   {val_losses['total']:.4f} "
                    f"(cls: {val_losses['cls']:.4f}, "
                    f"box: {val_losses['box']:.4f}, "
                    f"mask: {val_losses['mask']:.4f}) "
                    f"Time: {val_losses.get('val_time', 0):.1f}s"
                )

                # Check for best model
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    self.best_epoch = epoch + 1

                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        path=str(self.checkpoint_dir / 'best_model.pth'),
                        scheduler=self.scheduler,
                        best_metric=self.best_val_loss,
                    )
                    print(f"         ** New best model saved (val_loss={self.best_val_loss:.4f})")

            # Step scheduler after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    path=str(self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'),
                    scheduler=self.scheduler,
                    best_metric=self.best_val_loss,
                )

            self.history.append(epoch_record)

        # Training complete
        total_time = time.time() - training_start
        print(f"\n{'='*70}")
        print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Best val loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"{'='*70}\n")

        # Save training log
        self._save_training_log()

        # Return history as column-oriented dict for easy plotting
        return self._get_history_summary()

    def _save_training_log(self) -> None:
        """Save training history to a JSON file."""
        log_path = self.log_dir / 'training_log.json'
        log_data = {
            'config': {
                'lr': self.lr,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'warmup_epochs': self.warmup_epochs,
                'num_epochs': self.num_epochs,
                'gradient_clip': self.gradient_clip,
                'use_amp': self.use_amp,
                'device': str(self.device),
            },
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'history': self.history,
        }

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        logger.info(f"Training log saved to {log_path}")

    def _get_history_summary(self) -> Dict[str, List]:
        """Convert history list into column-oriented summary for plotting.

        Returns:
            Dictionary with lists for each metric across epochs.
        """
        summary: Dict[str, List] = {
            'epoch': [],
            'lr': [],
            'train_total': [],
            'train_cls': [],
            'train_box': [],
            'train_mask': [],
            'val_total': [],
            'val_cls': [],
            'val_box': [],
            'val_mask': [],
        }

        for record in self.history:
            summary['epoch'].append(record['epoch'])
            summary['lr'].append(record.get('lr', 0))

            train = record.get('train', {})
            summary['train_total'].append(train.get('total', 0))
            summary['train_cls'].append(train.get('cls', 0))
            summary['train_box'].append(train.get('box', 0))
            summary['train_mask'].append(train.get('mask', 0))

            val = record.get('val', {})
            summary['val_total'].append(val.get('total', 0))
            summary['val_cls'].append(val.get('cls', 0))
            summary['val_box'].append(val.get('box', 0))
            summary['val_mask'].append(val.get('mask', 0))

        return summary

    def resume(self, checkpoint_path: str) -> int:
        """Resume training from a checkpoint.

        Loads model weights, optimizer state, scheduler state, and
        training metadata from a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            The epoch number to resume from.
        """
        checkpoint = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_metric', float('inf'))
        self.best_epoch = start_epoch

        logger.info(
            f"Resumed from checkpoint: epoch={start_epoch}, "
            f"best_val_loss={self.best_val_loss:.4f}"
        )

        return start_epoch


if __name__ == '__main__':
    """Quick test of trainer initialization (no actual training)."""
    logging.basicConfig(level=logging.INFO)

    print("=== Trainer Module Test ===\n")
    print("Trainer class loaded successfully.")
    print("To train, run: python scripts/train.py --config configs/default.yaml")
    print("\n=== Test Complete ===")

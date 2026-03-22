#!/usr/bin/env python3
"""Main training script for YOLACT instance segmentation model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --resume results/training/checkpoints/best_model.pth
    python scripts/train.py --config configs/default.yaml --epochs 30 --batch-size 4

This script:
    1. Loads configuration from a YAML file
    2. Creates the SKU-110K dataset and DataLoaders
    3. Builds the YOLACT model with MobileNetV3-Large backbone
    4. Initializes the Trainer with SGD optimizer and cosine LR schedule
    5. Runs the training loop with periodic validation
    6. Saves the best model checkpoint and training logs
    7. Prints final training metrics and summary
"""

import argparse
import logging
import os
import sys
import time

# Ensure the project root is on the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

from src.models.yolact import YOLACT
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.utils.helpers import get_device, load_config, set_seed, format_params


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Train YOLACT model on SKU-110K dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration YAML file',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of training epochs',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate',
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Limit dataset size for quick experiments',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['mps', 'cuda', 'cpu'],
        help='Override device selection',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Override number of dataloader workers',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level',
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging format and level.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("  YOLACT Training - Dense Object Instance Segmentation")
    print("  MobileNetV3-Large + FPN + ProtoNet on SKU-110K")
    print("=" * 70)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Config loaded from: {args.config}")

    # Apply command-line overrides
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
        config.setdefault('dataset', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        config.setdefault('training', {})['lr'] = args.lr
    if args.max_images is not None:
        config.setdefault('dataset', {})['max_images'] = args.max_images
        config.setdefault('dataset', {})['train_subset'] = args.max_images
    if args.num_workers is not None:
        config.setdefault('dataset', {})['num_workers'] = args.num_workers

    # Device selection
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"\nDevice: {device}")
    if device.type == 'mps':
        print("  Apple Silicon MPS backend (FP32 training, no AMP)")
    elif device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  CUDA GPU: {gpu_name}")
    else:
        print("  CPU training (slow)")

    # ---------------------------------------------------------------
    # 1. Create datasets and dataloaders
    # ---------------------------------------------------------------
    print("\n--- Loading Dataset ---")
    dataset_cfg = config.get('dataset', {})

    # Build dataloader config from nested config
    dataloader_config = {
        'data_dir': dataset_cfg.get('data_dir', 'data'),
        'batch_size': config.get('training', {}).get('batch_size', dataset_cfg.get('batch_size', 8)),
        'num_workers': dataset_cfg.get('num_workers', 4),
        'max_images': dataset_cfg.get('train_subset', dataset_cfg.get('max_images', None)),
        'input_size': dataset_cfg.get('input_size', 550),
        'pin_memory': dataset_cfg.get('pin_memory', True),
    }

    train_loader, val_loader = get_dataloaders(dataloader_config)

    print(f"  Train: {len(train_loader.dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} images, {len(val_loader)} batches")
    print(f"  Batch size: {dataloader_config['batch_size']}")
    print(f"  Input size: {dataloader_config['input_size']}")

    # ---------------------------------------------------------------
    # 2. Create model
    # ---------------------------------------------------------------
    print("\n--- Building Model ---")
    model_config = {
        'num_classes': dataset_cfg.get('num_classes', 1) + 1,  # +1 for background
        'pretrained_backbone': config.get('backbone', {}).get('pretrained', True),
        'fpn_out_channels': config.get('fpn', {}).get('out_channels', 256),
        'num_prototypes': config.get('yolact', {}).get('num_prototypes', 32),
        'input_size': dataset_cfg.get('input_size', 550),
        'conf_threshold': config.get('yolact', {}).get('conf_threshold', 0.05),
        'max_detections': config.get('yolact', {}).get('max_detections', 300),
    }

    model = YOLACT(config=model_config).to(device)

    # Print parameter summary
    param_info = model.count_parameters()
    print(f"  Total parameters:     {format_params(param_info['total'])}")
    print(f"  Trainable parameters: {format_params(param_info['trainable'])}")
    print(f"  Backbone:  {format_params(param_info['backbone']['trainable'])} trainable")
    print(f"  FPN:       {format_params(param_info['fpn']['trainable'])} trainable")
    print(f"  ProtoNet:  {format_params(param_info['protonet']['trainable'])} trainable")
    print(f"  Head:      {format_params(param_info['prediction_head']['trainable'])} trainable")

    # ---------------------------------------------------------------
    # 3. Create trainer
    # ---------------------------------------------------------------
    print("\n--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        print(f"\n  Resuming from: {args.resume}")
        start_epoch = trainer.resume(args.resume)
        print(f"  Resumed at epoch {start_epoch}")

    # ---------------------------------------------------------------
    # 4. Train
    # ---------------------------------------------------------------
    training_cfg = config.get('training', {})
    num_epochs = training_cfg.get('epochs', 20)
    if args.epochs is not None:
        num_epochs = args.epochs

    history = trainer.fit(num_epochs=num_epochs)

    # ---------------------------------------------------------------
    # 5. Save final model and print summary
    # ---------------------------------------------------------------
    print("\n--- Training Summary ---")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch})")

    # Print final training losses
    if history['train_total']:
        final_train = history['train_total'][-1]
        print(f"  Final training loss:  {final_train:.4f}")

    if history['val_total'] and any(v > 0 for v in history['val_total']):
        last_val = [v for v in history['val_total'] if v > 0]
        if last_val:
            print(f"  Final validation loss: {last_val[-1]:.4f}")

    # Save final model
    final_model_path = 'results/training/checkpoints/final_model.pth'
    from src.utils.helpers import save_checkpoint as save_ckpt
    save_ckpt(
        model=model,
        optimizer=trainer.optimizer,
        epoch=num_epochs,
        path=final_model_path,
        scheduler=trainer.scheduler,
        best_metric=trainer.best_val_loss,
    )
    print(f"\n  Final model saved to: {final_model_path}")
    print(f"  Best model saved to:  results/training/checkpoints/best_model.pth")
    print(f"  Training log saved to: results/training/training_log.json")

    print(f"\n{'='*70}")
    print(f"  Training complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

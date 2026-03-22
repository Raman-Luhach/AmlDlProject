"""Miscellaneous utility functions for the YOLACT project.

Provides device management, configuration loading, checkpoint management,
reproducibility, and parameter counting utilities.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return the best available device for computation.

    Priority order: MPS (Apple Silicon) > CUDA (NVIDIA GPU) > CPU.

    Returns:
        torch.device for the best available hardware accelerator.
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary containing configuration key-value pairs.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    scheduler: Optional[Any] = None,
    best_metric: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Saves model state dict, optimizer state, epoch number, and optional
    scheduler state and best metric to a file.

    Args:
        model: PyTorch model to save.
        optimizer: Optimizer whose state to save.
        epoch: Current epoch number.
        path: File path for the checkpoint.
        scheduler: Optional learning rate scheduler.
        best_metric: Optional best validation metric achieved so far.
        extra: Optional dictionary of extra data to save.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if best_metric is not None:
        checkpoint['best_metric'] = best_metric

    if extra is not None:
        checkpoint.update(extra)

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Loads checkpoint data and optionally restores model, optimizer, and
    scheduler states.

    Args:
        path: Path to the checkpoint file.
        model: Optional model to load state dict into.
        optimizer: Optional optimizer to load state dict into.
        scheduler: Optional scheduler to load state dict into.
        device: Device to map checkpoint tensors to.

    Returns:
        Dictionary containing all checkpoint data (epoch, best_metric, etc.).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_location = device if device is not None else get_device()
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for Python's random, NumPy, and PyTorch (CPU and GPU).
    Also configures PyTorch for deterministic behavior where possible.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms where possible
    torch.use_deterministic_algorithms(False)  # Set True for strict determinism
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_params(num: int) -> str:
    """Format parameter count with appropriate suffix.

    Args:
        num: Number of parameters.

    Returns:
        Formatted string, e.g. '5.4M', '1.2K', '245'.
    """
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


if __name__ == '__main__':
    # Test device detection
    device = get_device()
    print(f"Best available device: {device}")

    # Test seed setting
    set_seed(42)
    print(f"Random seed set to 42")
    print(f"  Random int: {random.randint(0, 100)}")
    print(f"  Numpy random: {np.random.rand():.4f}")
    print(f"  Torch random: {torch.rand(1).item():.4f}")

    # Test format_params
    print(f"\nParameter formatting:")
    print(f"  {format_params(5_400_000)}")
    print(f"  {format_params(12_500)}")
    print(f"  {format_params(245)}")

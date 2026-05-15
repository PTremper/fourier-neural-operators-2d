"""General utility functions for training, evaluation, and experiment management.

This module contains reusable helper functions that are shared across
multiple parts of the project, including:

    - velocity field postprocessing
    - model checkpoint saving
    - standardized filename generation

The utilities in this module are intentionally lightweight and
independent of model architecture to keep them broadly reusable across
training and visualization workflows.
"""

from pathlib import Path

import numpy as np
import torch


def velocity_magnitude(u: np.ndarray) -> np.ndarray:
    """Compute the velocity magnitude from the velocity field components.

    Takes element [0] and [1] of the input's last axis.
    Then computes the square root of their squared sum.

    Input:
    -----
    u : np.ndarray
        Velocity field (..., 2+).
        e.g. shape (Nx, Ny, 2)

    Return:
    ------
    np.ndarray
        Velocity magnitude (...).
        e.g. shape (Nx, Ny)

    """
    return np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str | Path,
) -> None:
    """Save a full training checkpoint.

    Contents:
    --------
        - model weights
        - optimizer state
        - epoch
        - training loss

    """
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, save_path)


def generate_filename(
    file_extension: str,
    epoch: int,
    train_steps: int = 1,
    val_steps: int = 1,
    loss: float | None = None,
) -> str:
    """Generate a filename for the animation/snapshot/checkpoint."""
    epoch_part = f"epoch_{(epoch + 1):04d}"
    train_steps_part = f"_train_steps_{train_steps}" if train_steps > 1 else ""
    val_steps_part = f"_val_steps_{val_steps}" if val_steps > 1 else ""
    loss_part = f"_loss_{loss:.4f}" if loss is not None else ""
    return f"{epoch_part}{train_steps_part}{val_steps_part}{loss_part}.{file_extension}"

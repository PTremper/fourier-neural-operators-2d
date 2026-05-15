"""Utilities for multi-step animation generation.

This module provides helper functions for visualizing multi-step predictions
as animations and single-step predictions as snapshots.

    - create_prediction_animation():
        Create an animation comparing prediction and ground truth.
    - create_snapshot_figure():
        Create a figure showing a single-step prediction.
    - save_animation():
        Save the generated animation to disk.
    - save_snapshot():
        Save the generated snapshot to disk.


Prediction visualizations are useful for evaluating temporal stability,
error accumulation, and long-term dynamical behavior of the model.
"""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage
from tqdm import tqdm

from fno_utils.types import TypeHints
from fno_utils.utils import velocity_magnitude


def _create_prediction_visualization(
    preds: TypeHints.Timeseries,
    true_seq: TypeHints.Timeseries,
    title: str = "",
) -> tuple[plt.Figure, Callable, int]:
    """Visualize the prediction of a model on a given sequence.

    Generate an initial image of the first frame of the prediction sequence.
    Provides an update function that can be used in conjunction with
    `matplotlib.animation.FuncAnimation` to create an animation of the prediction.

    Input:
    -----
        preds : np.ndarray
            predictions, shape [steps, Nx, Ny, C_out]
        true_seq : np.ndarray
            true values, shape [steps, Nx, Ny, C_out]

    Return:
    ------
        fig : plt.Figure
            the figure object
        _update : Callable
            update function for the animation
        num_frames : int
            the number of frames in the animation

    """
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), constrained_layout=True)

    # Precompute magnitudes
    pred_mags: list[TypeHints.Datapoint] = [velocity_magnitude(p.transpose(1, 0, 2)) for p in preds]
    true_mags: list[TypeHints.Datapoint] = [
        velocity_magnitude(t.transpose(1, 0, 2)) for t in true_seq
    ]

    # Global scaling
    all_vals: np.ndarray = np.concatenate(
        [
            np.concatenate([p.ravel() for p in pred_mags]),
            np.concatenate([t.ravel() for t in true_mags]),
        ],
    )
    min_val: float = np.percentile(all_vals, 1)
    max_val: float = np.percentile(all_vals, 99)

    # Error scaling
    all_errors: np.ndarray = np.concatenate(
        [(t - p).ravel() for t, p in zip(true_mags, pred_mags, strict=True)],
    )
    err_lim: float = np.percentile(np.abs(all_errors), 99)

    # Initial frame to define frame layout
    fig_title = fig.suptitle(title)

    im0 = axs[0].imshow(
        true_mags[0],
        vmin=min_val,
        vmax=max_val,
        cmap="viridis",
        origin="lower",
    )
    axs[0].set_title("Ground Truth")

    im1 = axs[1].imshow(
        pred_mags[0],
        vmin=min_val,
        vmax=max_val,
        cmap="viridis",
        origin="lower",
    )
    axs[1].set_title("Prediction")

    im2 = axs[2].imshow(
        true_mags[0] - pred_mags[0],
        vmin=-err_lim,
        vmax=err_lim,
        cmap="seismic",
        origin="lower",
    )
    axs[2].set_title("Error")

    for ax in axs:
        ax.axis("off")

    cbar_mag = fig.colorbar(im0, ax=axs[:2], fraction=0.025, pad=0.02)
    cbar_mag.set_label("Velocity Magnitude")

    cbar_err = fig.colorbar(im2, ax=axs[2], fraction=0.025, pad=0.02)
    cbar_err.set_label("Velocity Magnitude Error")

    def _update(i: int) -> tuple[AxesImage, AxesImage, AxesImage]:

        # update frame data
        im0.set_data(true_mags[i])
        im1.set_data(pred_mags[i])
        im2.set_data(true_mags[i] - pred_mags[i])

        fig_title.set_text(f"Step {i}")

        return im0, im1, im2

    return fig, _update, len(preds)


def create_prediction_animation(
    preds: TypeHints.Timeseries,
    true_seq: TypeHints.Timeseries,
) -> FuncAnimation:
    """Animate the prediction of a model on a given sequence.

    Input:
    -----
        preds : np.ndarray
            predictions, shape [steps, Nx, Ny, C_out]
        true_seq : np.ndarray
            true values, shape [steps, Nx, Ny, C_out]

    """
    fig, _update, num_frames = _create_prediction_visualization(preds, true_seq)

    return animation.FuncAnimation(
        fig,
        _update,
        frames=num_frames,
        blit=False,
    )


def create_snapshot_figure(
    pred: TypeHints.Datapoint,
    true: TypeHints.Datapoint,
    title: str = "",
) -> plt.Figure:
    """Create a plot to visualize predictions vs ground truth.

    Input:
    -----
    pred : np.ndarray
        Predicted velocity field [Nx, Ny, C_out].
    true : np.ndarray
        Ground truth velocity field [Nx, Ny, C_out].
    title : str, optional
        Title for the plot.

    Return:
    ------
    fig : plt.Figure
        The figure object.

    """
    preds = pred[None, ...]
    true_seq = true[None, ...]
    fig, _, _ = _create_prediction_visualization(preds, true_seq, title=title)
    return fig


def save_animation(
    ani: animation.FuncAnimation,
    save_path: str | Path,
    frames: int,
) -> None:
    """Save the animation to a file."""
    pbar = tqdm(total=frames, desc="Generating video", leave=False)
    ani.save(
        save_path,
        writer="ffmpeg",
        fps=10,
        progress_callback=lambda _current_frame, _total_frames: pbar.update(1),
    )
    pbar.close()


def save_snapshot(fig: plt.Figure, save_path: str | Path) -> None:
    """Save the figure to a file."""
    fig.savefig(save_path)

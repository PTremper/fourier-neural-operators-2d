"""Utilities for multi-step prediction evaluation.

This module provides helper functions for generating multi-step model predictions.

Provides the functions:
    - run_multi_step_prediction()
        Generate predictions over multiple future steps, recursively feeding
        the model's output back into itself to predict the next step.

    - run_single_step_prediction()
        Run a single-step prediction using the given model and input/output datapoints.
        Uses `run_multi_step_prediction()` internally and handles input/output types
        to accept datapoints instead of timeseries.

"""

from collections.abc import Callable

import numpy as np
import torch

from fno_utils.types import TypeHints


def run_multi_step_prediction(  # noqa: PLR0913  - allow more than 5 arguments
    model: torch.nn.Module,
    x: TypeHints.Datapoint,
    y_timeseries: TypeHints.Timeseries,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[TypeHints.Timeseries, list[float]]:
    """Compute a multi-step prediction from a given input.

    Input:
    ------
        model : torch.nn.Module
            The model to use for prediction.
        x : np.ndarray
            The input datapoint to predict on.
            shape: [Nx, Ny, C_in]
        y_timeseries : np.ndarray
            The target / ground truth timeseries.
            shape: [steps, Nx, Ny, C_out]
        loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The function to use for computing the loss.
        device : torch.device | str | None, optional
            The device to use for computation. If None, the device is automatically determined.
        dtype : torch.dtype | None, optional
            The data type to use for computation. If None, the default dtype is used.

    Return:
    ------
        y_preds_timeseries : np.ndarray
            The predicted timeseries as numpy array.
            shape: [steps, Nx, Ny, C_out]
        pred_losses : list[float]
            The loss at each step of the prediction.

    """
    steps = len(y_timeseries)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    # add batch dimensions to x and y_timeseries as expected by the model
    # x (input datapoint):
    #     [Nx, Ny, C_in] -> [1, Nx, Ny, C_in]
    # y_timeseries (target timeseries):
    #     [val_steps, Nx, Ny, C_out] -> [val_steps, 1, Nx, Ny, C_out]
    x_torch = torch.from_numpy(x).to(dtype).unsqueeze(0).to(device, non_blocking=True)
    y_timeseries_torch = (
        torch.from_numpy(y_timeseries).to(dtype).unsqueeze(1).to(device, non_blocking=True)
    )

    y_preds_list: list[TypeHints.Datapoint] = []
    pred_losses: list[float] = []

    model.eval()
    for step in range(steps):
        with torch.no_grad():
            y_pred_torch = model(x_torch)
            pred_loss = loss_function(y_pred_torch, y_timeseries_torch[step])

        y_preds_list.append(y_pred_torch.detach().cpu().numpy()[0])
        pred_losses.append(pred_loss.item())

        # feed prediction back in
        x_torch[..., :2] = y_pred_torch  # overwrite velocity channels

    model.train()

    # [steps, Nx, Ny, C_out]
    y_preds_timeseries: TypeHints.Timeseries = np.stack(y_preds_list, axis=0)
    return y_preds_timeseries, pred_losses


def run_single_step_prediction(  # noqa: PLR0913  - allow more than 5 arguments
    model: torch.nn.Module,
    x: TypeHints.Datapoint,
    y: TypeHints.Datapoint,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[TypeHints.Datapoint, float]:
    """Run a single-step prediction using the given model and input/output datapoints.

    A wrapper around `run_multi_step_prediction()` that handles single-step predictions.
    Handles input/output types to accept datapoints instead of timeseries.

    Input:
    -----
        model : torch.nn.Module
            The model to use for prediction.
        x : np.ndarray
            The input datapoint to predict on.
            shape: [Nx, Ny, C_in]
        y : np.ndarray
            The target / ground truth datapoint.
            shape: [Nx, Ny, C_out]
        loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            The function to use for computing the loss.
        device : torch.device | str | None, optional
            The device to use for computation. If None, the device is automatically determined.
        dtype : torch.dtype | None, optional
            The data type to use for computation. If None, the default dtype is used.

    Return:
    ------
        y_pred : np.ndarray
            The predicted datapoint as numpy array.
            shape: [Nx, Ny, C_out]
        pred_loss : float
            The loss of the prediction.

    """
    # Convert y to a timeseries with a single step
    y_timeseries = y[None, ...]
    y_preds_timeseries, pred_losses = run_multi_step_prediction(
        model,
        x,
        y_timeseries,
        loss_function=loss_function,
        device=device,
        dtype=dtype,
    )
    # Return the single-step prediction and loss
    return y_preds_timeseries.squeeze(0), pred_losses[0]

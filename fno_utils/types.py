"""Centralized type aliases and type hint definitions used throughout the project.

This module provides structured NumPy shape annotations for
commonly used data representations, including:

    - velocity fields
    - temporal trajectories
    - geometry masks
    - model inputs and outputs

The goal is to improve code readability, editor support, and consistency
across dataset, training, and visualization modules.
"""

from typing import Literal

import numpy as np


class TypeHints:
    """Collection of shared type aliases for simulation data structures.

    Type Hints:
    ----------
        TypeHints.Datapoint : Single datapoint of shape [Nx, Ny, C]

        TypeHints.Timeseries : Timeseries of shape [T, Nx, Ny, C]

        TypeHints.Batched_Data : Batched data of shape [B, Nx, Ny, C]

        TypeHints.Mask2D : geometry mask of shape [Nx, Ny], boolean entries.

        TypeHints.FeatureSize : Feature size, either 2, 3, 4, or 5
            len(C), corresponding to the number of velocity components (ux, uy)
            and optional spatial grid (x, y) and optional mask (mask).
            - 2 (ux, uy)
            - 3 (ux, uy, mask)
            - 4 (ux, uy, x, y)
            - 5 (ux, uy, x, y, mask)

    These aliases are primarily intended for improving readability and
    static analysis when working with high-dimensional NumPy arrays and
    tensors throughout the FNO training pipeline.

    """

    FeatureSize = Literal[2, 3, 4, 5]
    Datapoint = np.ndarray[tuple[int, int, FeatureSize], np.dtype[np.floating]]
    Timeseries = np.ndarray[tuple[int, int, int, FeatureSize], np.dtype[np.floating]]
    Batched_Data = np.ndarray[tuple[int, int, int, FeatureSize], np.dtype[np.floating]]
    Mask2D = np.ndarray[tuple[int, int], np.dtype[np.bool_]]

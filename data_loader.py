"""Dataset utilities for loading and preparing LBM simulation data for FNO training.

This module provides the `LBMDataset` class, which converts raw fluid
simulation trajectories into supervised learning samples suitable for
autoregressive operator learning.

The dataset supports:
    - single-step prediction training
    - multi-step autoregressive training
    - configurable temporal stride (`time_stride`)
    - optional spatial coordinate encoding
    - optional geometry mask features

Each dataset sample consists of:

    x : [Nx, Ny, C_in]
        Input features at time t, including velocity components and
        optional auxiliary channels such as spatial coordinates and
        obstacle masks.

    y : [train_steps, Nx, Ny, C_out]
        Sequence of future target states beginning at time
        t + time_stride.

The dataset preserves temporal ordering between consecutive indices,
allowing it to be used for multi-step prediction evaluation and sequence sampling.
"""

from typing import Literal

import numpy as np
import torch

from fno_utils.types import TypeHints


def load_data(
    train_data_path: str,
    val_data_path: str | None = None,
    geometry_mask_path: str | None = None,
) -> tuple[TypeHints.Timeseries, TypeHints.Timeseries, TypeHints.Mask2D | None]:
    """Load the dataset from numpy files.

    Input:
    -----
        train_data_path:
            Path to the training data file.
            Expected shape: (T, Nx, Ny, 2) with features (ux, uy).
        val_data_path:
            Path to the validation data file (optional).
            Expected shape: (T, Nx, Ny, 2) with features (ux, uy).
        geometry_mask_path:
            Path to the geometry mask file (optional).
            Expected shape: (Nx, Ny) with boolean values.

    Return:
    ------
        A tuple of (train_data, val_data, mask).

    Notes:
    -----
        - if no validation data path is provided, the training data is split into train/val sets.
        - if no geometry mask is provided, the returned mask will be None.

    """
    # Expects data as numpy arrays of shape (T, Nx, Ny, 2) with features (ux, uy)
    if val_data_path is not None:
        print(f"Loading train data from {train_data_path}")
        train_data: TypeHints.Timeseries = np.load(train_data_path)
        print(f"Loading val data from {val_data_path}")
        val_data: TypeHints.Timeseries = np.load(val_data_path)
    else:
        print(
            "No validation data path provided, "
            f"splitting {train_data_path} 80/20 into train/val sets",
        )
        data: TypeHints.Timeseries = np.load(train_data_path)

        train_val_split_idx = int(0.8 * len(data))

        train_data: TypeHints.Timeseries = data[:train_val_split_idx]
        val_data: TypeHints.Timeseries = data[train_val_split_idx:]

    if geometry_mask_path is not None:
        print(f"Loading geometry mask from {geometry_mask_path}")
        mask: TypeHints.Mask2D = np.load(geometry_mask_path)  # (Nx, Ny)
    else:
        print("No geometry mask path provided, mask set to None")
        mask = None

    return train_data, val_data, mask


class LBMDataset(torch.utils.data.Dataset):
    """Prepare the LBM dataset for multi-step FNO training.

    Input: Takes time series data as numpy array of shape [T, Nx, Ny, 2]

    Shifts data to create pairs of (X, Y) where each y(t) is x(t+1)
    Augments and normalizes the pairs, which are provided as tuple via __getitem__
    (X, Y), where
        X is of shape [Nx, Ny, 5] and
        Y is of shape [train_steps, Nx, Ny, 2]

    Defaults to train_steps = 1 to produce data for single step training.

    Output: Pairs of (X, Y)
        where X is of shape [Nx, Ny, 5] and Y is of shape [train_steps, Nx, Ny, 2]
        These pairs can be used with the PyTorch DataLoader for batching/shuffling before training.

    """

    def __init__(
        self,
        data: TypeHints.Timeseries,
        mask: TypeHints.Mask2D | None = None,
        train_steps: int = 1,
        time_stride: int = 1,
        *,
        add_coordinates: bool = False,
    ) -> None:
        """Prepare the LBM dataset for multi-step FNO training.

        Input:
        ------
        data : np.ndarray
            Input data of shape (T, Nx, Ny, 2).
        add_coordinates: bool, optional
            Whether to add grid coordinates to the input. Default is False.
        mask : np.ndarray, optional
            Mask data of shape (Nx, Ny), boolean values.
        train_steps: int, optional
            The number of steps to predict. Default is 1.
        time_stride: int, optional
            The time step size. Default is 1.

        """
        X, Y = self._create_pairs(data)
        print(f"X shape: {X.shape}, Y shape: {Y.shape}")

        if add_coordinates:
            X = self._add_grid(X)
            print(f"Added grid coordinates: X shape: {X.shape}")

        if mask is not None:
            X = self._add_mask(X, mask)
            print(f"Added mask: X shape: {X.shape}")

        X, Y, self.mean, self.std = self._normalize(X, Y)

        print(f"Normalized: X shape: {X.shape}")

        self.X: TypeHints.Timeseries = X
        self.Y: TypeHints.Timeseries = Y

        self.train_steps: int = train_steps
        self.time_stride: int = time_stride
        self.T: int = len(data)

    def _create_pairs(
        self,
        data: TypeHints.Timeseries,
    ) -> tuple[TypeHints.Timeseries, TypeHints.Timeseries]:
        """Generate pairs of consecutive frames from the data.

        Input:
        ------
        data: (T, Nx, Ny, 2)

        Return:
        ------
            X: (T-1, Nx, Ny, 2)
            Y: (T-1, Nx, Ny, 2)

        """
        X = data[:-1]
        Y = data[1:]
        return X, Y

    def _create_grid(
        self,
        Nx: int,
        Ny: int,
    ) -> np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.floating]]:
        """Create a grid of points in the unit square.

        Input:
        ------
        Nx: int
            Number of points in the x direction.
        Ny: int
            Number of points in the y direction.

        Return:
        ------
            grid: (Nx, Ny, 2)

        """
        x = np.linspace(0, 1, Nx)
        y = np.linspace(0, 1, Ny)

        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")

        return np.stack([grid_x, grid_y], axis=-1)  # (Nx, Ny, 2)

    def _add_grid(self, X: TypeHints.Timeseries) -> TypeHints.Timeseries:
        """Add a grid of points to the input data.

        A single datapoint becomes: (ux, uy) --> (ux, uy, x, y)

        Input:
        ----------
        X : np.ndarray
            Input data of shape (T, Nx, Ny, 2).

        Return:
        ------
        np.ndarray
            Augmented data of shape (T, Nx, Ny, 4).

        """
        T, Nx, Ny, _ = X.shape

        grid = self._create_grid(Nx, Ny)  # (Nx, Ny, 2)
        grid = np.repeat(grid[None, ...], T, axis=0)  # (T, Nx, Ny, 2)

        return np.concatenate([X, grid], axis=-1)

    def _add_mask(self, X: TypeHints.Timeseries, mask: TypeHints.Mask2D) -> TypeHints.Timeseries:
        """Add a mask to the input data.

        A single datapoint becomes: (ux, uy, x, y) --> (ux, uy, x, y, mask)

        Input:
        ----------
        X : np.ndarray
            Input data of shape (T, Nx, Ny, 4).
        mask : np.ndarray
            Mask of shape (Nx, Ny), boolean values.

        Return:
        ------
        np.ndarray
            Augmented data of shape (T, Nx, Ny, 5).

        """
        T = X.shape[0]
        mask = mask[None, ..., None]  # (1, Nx, Ny, 1)
        mask = np.repeat(mask, T, axis=0)

        return np.concatenate([X, mask], axis=-1)

    def _normalize(
        self,
        X: TypeHints.Timeseries,
        Y: TypeHints.Timeseries,
    ) -> tuple[TypeHints.Timeseries, TypeHints.Timeseries, np.ndarray, np.ndarray]:
        """Normalize the input data.

        Input:
        ----------
        X : np.ndarray
            Input data of shape (T, Nx, Ny, n).
        Y : np.ndarray
            Target data of shape (T, Nx, Ny, 2).

        Return:
        ------
        X_norm: Timeseries
            Normalized input data. (T, Nx, Ny, n)
        Y_norm : Timeseries
            Normalized target data. (T, Nx, Ny, 2)
        mean : np.ndarray
            Mean values used for normalization. (1, 1, 1, n)
        std : np.ndarray
            Standard deviation values used for normalization. (1, 1, 1, n)

        """
        mean = X.mean(axis=(0, 1, 2), keepdims=True)
        std = X.std(axis=(0, 1, 2), keepdims=True) + 1e-6

        X_norm = (X - mean) / std
        Y_norm = (Y - mean[..., : Y.shape[-1]]) / std[..., : Y.shape[-1]]

        return X_norm, Y_norm, mean, std

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        This function explicitly subtracts `train_steps * time_stride`
            from the total number of samples.
        """
        return self.T - self.train_steps * self.time_stride

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[TypeHints.Datapoint, TypeHints.Timeseries]:
        """Return a single item from the dataset.

        Input:
        -----
            index: int
                The index of the item to return.

        Return:
        ------
            x: (Nx, Ny, 5)
            y: (train_steps, Nx, Ny, 2)

        """
        X: TypeHints.Datapoint = self.X[index]

        if isinstance(index, slice):
            msg = "Slicing is not supported."
            raise NotImplementedError(msg)

        # count from index... index + (train_steps-1)
        # since create_pairs already shifted the index of Y so Y[t] is the next step after X[t]
        Y_seq: TypeHints.Timeseries = np.stack(
            [self.Y[index + i * self.time_stride] for i in range(self.train_steps)],
            axis=0,
        )

        return X, Y_seq


if __name__ == "__main__":
    # unit test with mock data

    rng = np.random.default_rng()
    data = rng.random((100, 64, 64, 2))
    mask = rng.random((64, 64)) < 0.5  # noqa: PLR2004 allow explicit value in comparison

    print(f"Data shape: {data.shape}")
    print(f"Mask shape: {mask.shape}")

    dataset = LBMDataset(data, mask, add_coordinates=True)
    print(f"Dataset size: {len(dataset)}")

    X0, Y0 = dataset[10]
    print(f"LBMDataset - X1 shape: {X0.shape}, Y1 shape: {Y0.shape}")

    from torch.utils.data import DataLoader

    BATCH_SIZE = 4
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # optional
        pin_memory=True,  # good for GPU
    )

    for X_batch, Y_batch in dataloader:
        # this would raise an error if batching required slicing the dataset class
        foo, bar = X_batch.shape, Y_batch.shape
        break

    print("Slicing of data not required to obtain batches through PyTorch DataLoader")
    print(f"X_batch shape: {X_batch.shape}, Y_batch shape: {Y_batch.shape}")

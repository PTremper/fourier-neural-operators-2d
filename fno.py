"""Fourier Neural Operator implementation for 2D fluid dynamics.

This module contains the core neural network components used to model the
time evolution of 2D fluid flow fields.

Implemented components:
- SpectralConv2d:
    Fourier-space convolution layer operating on low-frequency modes.
- FNOBlock:
    Combination of spectral and pointwise convolutions.
- FNO2d:
    Full Fourier Neural Operator architecture for predicting future fluid
    states.

The model is designed for supervised learning on spatial simulation data
such as:
- velocity fields
- geometry-aware flow domains
- time-dependent CFD simulations

The implementation follows the general architecture introduced in:

    Li et al. (2020)
    "Fourier Neural Operator for Parametric Partial Differential Equations"

Tensor conventions:
-------------------
The repository primarily uses channel-last tensor layout for dataset and
training code:

    [B, Nx, Ny, C]

where:
    B:
        batch size
    Nx, Ny:
        spatial grid dimensions
    C:
        feature channels

Internally, Fourier layers use PyTorch's channel-first convention:

    [B, C, Nx, Ny]

The code is intentionally structured for readability and learning, with:
- explicit tensor shape annotations
- semantic variable naming
- descriptive comments
- lightweight type aliases
"""

import torch
from torch import nn

# Data Shape is [Batch, Channels_in, Nx, Ny]
Data_bcxy = torch.Tensor

# Permuted Data Shape is [Batch, Nx, Ny, Channels_in]
Data_bxyc = torch.Tensor

# Data Shape after lift is [Batch, Nx, Ny, width]
Data_bxyw = torch.Tensor

# Permuted Data Shape after lift is [Batch, width, Nx, Ny]
Data_bwxy = torch.Tensor

# Output Data Shape [Batch, Channels_out, Nx, Ny]
Data_out = torch.Tensor


class SpectralConv2d(nn.Module):
    """2D spectral convolution layer used in the Fourier Neural Operator.

    This layer performs convolution in Fourier space instead of physical
    space. The input is transformed using a 2D Fast Fourier Transform (FFT),
    multiplied with learnable complex-valued weights for a limited number of
    low-frequency modes, and transformed back using the inverse FFT.

    Restricting the operation to low-frequency modes allows the layer to
    efficiently learn large-scale spatial structures and long-range
    interactions in fluid flows.

    The layer expects channel-first tensor layout:

    Input:
    -----
        [B, C_in, Nx, Ny]

    Output:
    ------
        [B, C_out, Nx, Ny]

    where:
        B:
            batch size
        C_in:
            number of input feature channels
        C_out:
            number of output feature channels
        Nx, Ny:
            spatial grid dimensions

    Reference:
    ---------
        Li et al. (2020)
        "Fourier Neural Operator for Parametric Partial Differential Equations"

    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        """Initialize the spectral convolution layer.

        Args:
        ----
            in_channels : int
                Number of input feature channels.

            out_channels : int
                Number of output feature channels.

            modes_x : int
                Number of Fourier modes retained along the x-axis.

            modes_y : int
                Number of Fourier modes retained along the y-axis.

        Notes:
        -----
            Only the lower-frequency Fourier modes are learned. Higher
            frequencies are discarded, acting as an implicit low-pass filter.

            The learnable weights are complex-valued because Fourier
            coefficients contain both amplitude and phase information.

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Learnable complex weights
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat),
        )

    def forward(self, x: Data_bwxy) -> Data_bwxy:
        """Apply the spectral convolution to the input tensor.

        Args:
        ----
            x : torch.tensor
                [B, width, Nx, Ny]

        Returns:
        -------
            torch.tensor
                [B, width, Nx, Ny]

        """
        B, _w, Nx, Ny = x.shape

        # Check that the number of modes is valid
        if self.modes_x > Nx:
            msg = f"modes_x={self.modes_x} is greater than Nx={Nx}"
            raise ValueError(msg)
        if self.modes_y > Ny // 2 + 1:
            msg = f"modes_y={self.modes_y} is greater than Ny // 2 + 1={Ny // 2 + 1}"
            raise ValueError(msg)

        # 1. FFT
        x_ft = torch.fft.rfft2(x)

        # 2. Allocate output in Fourier space
        out_ft = torch.zeros(
            B,
            self.out_channels,
            Nx,
            Ny // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        # 3. Multiply only low-frequency modes
        out_ft[:, :, : self.modes_x, : self.modes_y] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes_x, : self.modes_y],
            self.weights,
        )

        # 4. Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(Nx, Ny))

        return x  # noqa: RET504 - allow trivial assignment before return


class FNOBlock(nn.Module):
    """Single Fourier Neural Operator block.

    An FNO block combines:
    - a spectral convolution for learning global spatial interactions
    - a pointwise 1x1 convolution for local feature mixing

    The spectral path captures long-range structures in Fourier space,
    while the pointwise path acts similarly to a residual/local correction
    term in physical space.

    The outputs of both paths are added together and passed through a
    non-linear activation function.

    Input:
    ------
        [B, width, Nx, Ny]

    Output:
    -------
        [B, width, Nx, Ny]

    where:
        B:
            batch size
        width:
            latent feature dimension
        Nx, Ny:
            spatial grid dimensions
    """

    def __init__(self, width: int, modes_x: int, modes_y: int) -> None:
        """Initialize the Fourier Neural Operator block.

        Args:
        ----
            width : int
                Number of latent feature channels used throughout the block.

            modes_x : int
                Number of Fourier modes retained along the x-axis in the
                spectral convolution.

            modes_y : int
                Number of Fourier modes retained along the y-axis in the
                spectral convolution.

        Notes:
        -----
            The spectral convolution learns large-scale spatial interactions,
            while the pointwise convolution provides local feature mixing and
            improves representational flexibility.

        """
        super().__init__()

        self.spectral = SpectralConv2d(
            in_channels=width,
            out_channels=width,
            modes_x=modes_x,
            modes_y=modes_y,
        )
        self.pointwise = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=1)

        self.activation = nn.GELU()

    def forward(self, x: Data_bwxy) -> Data_bwxy:
        """Apply the FNO block to the input tensor.

        Args:
        ----
            x : torch.tensor
                [B, width, Nx, Ny]

        Returns:
        -------
            torch.tensor
                [B, width, Nx, Ny]

        """
        # Global interaction
        x1: Data_bwxy = self.spectral(x)

        # Local interaction (skip path)
        x2: Data_bwxy = self.pointwise(x)

        return self.activation(x1 + x2)


class FNO2d(nn.Module):
    """2D Fourier Neural Operator for fluid flow prediction.

    This model learns the time evolution of 2D spatial fields such as
    fluid velocity fields.

    The architecture consists of three stages:

    1. Input lifting
       The input features are projected into a higher-dimensional latent
       representation.

    2. Fourier processing
       A sequence of FNO blocks applies spectral convolutions in Fourier
       space to learn global spatial interactions and flow dynamics.

    3. Output projection
       The latent representation is projected back to the desired output
       channels.

    The model expects channel-last input tensors:

    Input:
    ------
        [B, Nx, Ny, C_in]

    Output:
    -------
        [B, Nx, Ny, C_out]

    where:
        B:
            batch size
        Nx, Ny:
            spatial grid dimensions
        C_in:
            number of input features per grid cell
        C_out:
            number of predicted output features per grid cell

    Typical input features include:
    - velocity components
    - spatial coordinates
    - geometry masks

    Typical outputs:
    - predicted future velocity field
    """

    def __init__(  # noqa: PLR0913  - allow more than 5 arguments
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int = 16,
        modes_y: int = 16,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        """Initialize the 2D Fourier Neural Operator model.

        Args:
        ----
            in_channels : int
                Number of input feature channels.

            out_channels : int
                Number of output feature channels.

            modes_x : int, default=16
                Number of retained Fourier modes along the x-axis.

            modes_y : int, default=16
                Number of retained Fourier modes along the y-axis.

            width : int, default=64
                Width of the latent feature representation used throughout
                the Fourier blocks.

            depth : int, default=4
                Number of stacked Fourier Neural Operator blocks.

        Notes:
        -----
            Larger values for `width` and `depth` increase model capacity
            but also increase memory usage and computational cost.

            The number of Fourier modes controls how much frequency
            information is retained during spectral convolutions.

        """
        super().__init__()

        self.width = width

        # Lift input to higher dimension
        self.fc0 = nn.Linear(in_channels, width)

        # Fourier blocks
        self.blocks = nn.ModuleList([FNOBlock(width, modes_x, modes_y) for _ in range(depth)])

        # Projection back to output
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: Data_bxyc) -> Data_out:
        """Pass input data through the blocks of FNO layers.

        Args:
        ----
            x : torch.tensor
                [B, Nx, Ny, C_in]

        Returns:
        -------
            torch.tensor
                [B, Nx, Ny, C_out]

        """
        # 1. Lift
        x: Data_bxyw = self.fc0(x)  # [B, Nx, Ny, C_in] -> [B, Nx, Ny, width]

        # 2. Move channels to PyTorch format
        x: Data_bwxy = x.permute(0, 3, 1, 2)  # [B, Nx, Ny, width] -> [B, width, Nx, Ny]

        # 3. Fourier layers - blocks output same shape as input
        for block in self.blocks:
            x: Data_bwxy = block(x)  # [...] -> [...]

        # 4. Reverse channel permutation
        x: Data_bxyw = x.permute(0, 2, 3, 1)  # [B, width, Nx, Ny] -> [B, Nx, Ny, width]

        # 5. Projection
        x = self.fc1(x)  # [B, Nx, Ny, width] -> [B, Nx, Ny, 128]
        x = torch.relu(x)
        x: Data_out = self.fc2(x)  # [B, Nx, Ny, 128] -> [B, Nx, Ny, C_out]

        return x


def test_spectral_conv2d() -> None:
    """Test the SpectralConv2d layer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running test on device: {device}")

    B, C_in, C_out = 2, 3, 4
    Nx, Ny = 32, 32

    layer = SpectralConv2d(in_channels=C_in, out_channels=C_out, modes_x=8, modes_y=8).to(device)

    x = torch.randn(B, C_in, Nx, Ny, device=device, requires_grad=True)

    # Forward pass
    y = layer(x)

    # Check shape
    if y.shape != (B, C_out, Nx, Ny):
        msg = f"Output shape mismatch: Expected {(B, C_out, Nx, Ny)}, got {y.shape}"
        raise ValueError(msg)

    # Check finite values
    if not torch.isfinite(y).all():
        msg = "Output contains NaNs or infs"
        raise ValueError(msg)

    # Backward pass
    loss = y.mean()
    loss.backward()

    # Check gradients exist
    if x.grad is None:
        msg = "No gradient for input"
        raise ValueError(msg)
    if layer.weights.grad is None:
        msg = "No gradient for weights"
        raise ValueError(msg)

    print("✅ SpectralConv2d test passed")


if __name__ == "__main__":
    test_spectral_conv2d()

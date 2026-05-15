"""Run model training for the Fourier Neural Operator."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from dateutil import tz
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import LBMDataset, load_data
from fno import FNO2d
from fno_utils.predictions import (
    run_multi_step_prediction,
    run_single_step_prediction,
)
from fno_utils.utils import (
    generate_filename,
    save_checkpoint,
)
from fno_utils.visualization import (
    create_prediction_animation,
    create_snapshot_figure,
    save_animation,
    save_snapshot,
)

if TYPE_CHECKING:
    from fno_utils.types import TypeHints

BASE_FOLDER = "outputs"
VIDEO_FOLDER = "animations"
IMAGE_FOLDER = "snapshots"
CHECKPOINT_FOLDER = "checkpoints"

RANDOM_SEED = 42


# fmt: off
def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Data command line arguments
    parser.add_argument("--train_data", type=str,
        help="Path to the training data. Required Input.")
    parser.add_argument("--val_data", type=str, default=None,
        help="Path to the validation data. If None will split from the training data."
        " Default is None.")
    parser.add_argument("--geometry_mask", type=str, default=None,
        help="Path to the geometry mask data. Default is None.")

    # Validation / Visualization command line arguments
    parser.add_argument("--snapshot_interval", type=int, default=5,
        help="Frequency of snapshot saving every n epochs. 0 means no saving. Default is 5.")
    parser.add_argument("--val_interval", type=int, default=10,
        help="Frequency of validation prediction visualization every n epochs. 0 means no saving."
        " Default is 10.")
    parser.add_argument("--val_steps", type=int, default=20,
        help="Number of steps to visualize in the validation prediction. Default is 20.")
    parser.add_argument("--interactive_val", action="store_true",
        help="Show validation prediction interactively. Blocks computation until closed!"
        " Default is False.")

    # Training command line arguments
    parser.add_argument("--epochs", type=int, default=100,
        help="Number of epochs for training. Default is 100.")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Batch size for training. Default is 8.")
    parser.add_argument("--train_steps", type=int, default=1,
        help="Amount of steps for multi step training (1 is single step training). Default is 1.")
    parser.add_argument("--time_stride", type=int, default=1,
        help="stride for multi step training (1 is consequtive steps). Default is 1.")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate. Default is 1e-3.")
    parser.add_argument("--checkpoint_interval", type=int, default=0,
        help="Save checkpoint after every n epochs. 0 means no peridic saving."
        " (Final checkpoint is always saved). Default is 0.")

    return parser.parse_args()
# fmt: on


def main() -> None:  # noqa: PLR0915  - allow large function
    """Run a Lattice Boltzmann simulation and save the results."""
    args = _parse_args()

    # create the folders for this run in a timestamped folder in the base folder
    timestamp = datetime.now(tz=tz.tzlocal()).strftime("%Y-%m-%d_%H-%M-%S")

    base_path = Path(BASE_FOLDER) / timestamp
    snapshot_path = base_path / IMAGE_FOLDER
    animation_path = base_path / VIDEO_FOLDER
    checkpoint_path = base_path / CHECKPOINT_FOLDER

    for path in (snapshot_path, animation_path, checkpoint_path):
        path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- start actual script -----
    # Load data
    # Expected shapes:
    #   train_data: [T, Nx, Ny, C_in=2]
    #   val_data:   [T, Nx, Ny, C_in=2]
    #   mask:       [Nx, Ny]
    train_data, val_data, mask = load_data(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        geometry_mask_path=args.geometry_mask,
    )

    if len(val_data) < args.val_steps:
        msg = (
            f"validation data ({len(val_data)}) is less than"
            f" val_steps ({args.val_steps})."
            "Increase the dataset size or reduce val_steps."
        )
        raise ValueError(msg)

    # dataset returns (x, y) pairs, where
    # x is [T, Nx, Ny, C_in=5] with features: (ux, uy, x, y, mask)
    # y is [T, train_steps, Nx, Ny, C_out=2] with targets: (ux, uy)
    train_dataset = LBMDataset(
        data=train_data,
        add_coordinates=True,
        mask=mask,
        train_steps=args.train_steps,
        time_stride=args.time_stride,
    )

    val_dataset = LBMDataset(
        data=val_data,
        add_coordinates=True,
        mask=mask,
        train_steps=args.val_steps,
        time_stride=1,
    )

    # set the point in time of the validation dataset
    rng = np.random.default_rng(RANDOM_SEED)
    val_time: int = int(rng.integers(low=0, high=len(val_dataset)))

    # dataloader returns batches of (x, y) pairs
    # where x is [B, Nx, Ny, C_in] and y is [B, train_steps, Nx, Ny, C_out]
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # optional
        pin_memory=True,  # good for GPU
    )

    in_channels = train_dataset[0][0].shape[-1]  # 5: (ux, uy, x, y, mask)
    out_channels = train_dataset[0][1].shape[-1]  # 2: (ux, uy)
    train_steps = train_dataset.train_steps
    print(f"in_channels={in_channels}, out_channels={out_channels}, train_steps={train_steps}")

    model = FNO2d(in_channels=in_channels, out_channels=out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # key: epoch index, value: average loss over (multi-step) training / validation prediction
    loss_history: dict[int, float] = {}
    val_loss_history: dict[int, float] = {}

    # ----- start training -----
    # Multi-step training:
    #
    # predicting single time step evolution (train_steps = 1):
    # x = u(t) -> pred = u(t+1)
    # or predicting multi time step evolution (train_steps > 1):
    # x = u(t) -> pred = u(t+1) -> pred = u(t+2) -> ... -> pred = u(t+train_steps)
    #
    # Predictions are recursively fed back into the model input.

    for epoch in (pbar := tqdm(range(args.epochs), desc="")):
        epoch_losses: list[float] = []

        for x_, y_ in tqdm(train_dataloader, desc="Batches", leave=False):
            # y_ is of shape [B, train_steps, Nx, Ny, C_out]

            x = x_.to(torch.float32).to(device, non_blocking=True)  # [B, Nx, Ny, C_in]
            y = y_.to(torch.float32).to(device, non_blocking=True)  # [B, Nx, Ny, C_out]

            pred = model(x)
            loss = loss_fn(pred, y[:, 0])

            # for train_steps==1, the loop has zero iterations
            for k in range(1, train_steps):
                # expand predicted velocity with grid and mask to use it as input again
                # detach to avoid iterative backpropagation through all steps
                x_in = torch.cat([pred.detach(), x[..., 2:]], dim=-1)
                pred = model(x_in)
                loss += loss_fn(pred, y[:, k])

            loss /= train_steps
            epoch_losses.append(loss.item())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # pytorch automatically frees memory after backward and reuses it

            pbar.set_description(f"Loss: {loss.item():.4f}")

        # ----- Snapshot hook -----
        if args.snapshot_interval > 0 and (
            (epoch % args.snapshot_interval == 0) or (epoch + 1 == args.epochs)
        ):
            # ----- define block input/output variables -----
            val_x: TypeHints.Datapoint
            val_y_timeseries: TypeHints.Timeseries
            # val_x:
            #     [Nx, Ny, C_in]
            # val_y_timeseries:
            #     [val_steps, Nx, Ny, C_out]

            val_y: TypeHints.Datapoint
            pred: TypeHints.Datapoint
            # val_y:
            #     [Nx, Ny, C_out]
            # pred:
            #     [Nx, Ny, C_out]
            # -----------------------------------------------

            val_x, val_y_timeseries = val_dataset[val_time]
            val_y = val_y_timeseries[0]

            pred, val_loss = run_single_step_prediction(
                model=model,
                x=val_x,
                y=val_y,
                loss_function=loss_fn,
                device=device,
                dtype=torch.float32,
            )

            fig = create_snapshot_figure(
                pred,
                val_y,
                title=f"Epoch {epoch + 1}, Loss: {val_loss:.4f}",
            )

            snapshot_filename = generate_filename(
                epoch=epoch,
                train_steps=args.train_steps,
                loss=loss.item(),
                file_extension="png",
            )
            save_snapshot(fig=fig, save_path=snapshot_path / snapshot_filename)
            plt.close(fig)

        # ----- Validation hook -----
        if args.val_interval > 0 and (
            (epoch % args.val_interval == 0) or (epoch + 1 == args.epochs)
        ):
            # ----- define block input/output variables -----
            val_x: TypeHints.Datapoint
            val_y_timeseries: TypeHints.Timeseries
            # val_x:
            #     [Nx, Ny, C_in]
            # val_y_timeseries:
            #     [val_steps, Nx, Ny, C_out]

            val_preds_timeseries: TypeHints.Timeseries
            val_losses: list[float]
            # val_preds:
            #     [val_steps, Nx, Ny, C_out]
            # val_losses:
            #     [val_steps]
            # -----------------------------------------------

            val_x, val_y_timeseries = val_dataset[val_time]

            val_preds_timeseries, val_losses = run_multi_step_prediction(
                model=model,
                x=val_x,
                y_timeseries=val_y_timeseries,
                loss_function=loss_fn,
                device=device,
                dtype=torch.float32,
            )

            val_loss_history[epoch] = sum(val_losses) / len(val_losses)

            # -------------------------

            val_animation = create_prediction_animation(
                preds=val_preds_timeseries,
                true_seq=val_y_timeseries,
            )

            animation_filename = generate_filename(
                epoch=epoch,
                val_steps=args.val_steps,
                train_steps=args.train_steps,
                loss=loss.item(),
                file_extension="mp4",
            )
            save_animation(
                ani=val_animation,
                save_path=animation_path / animation_filename,
                frames=args.val_steps,
            )

            if args.interactive_val:
                plt.show()

            # close all figures to avoid memory leaks
            plt.close("all")

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            checkpoint_filename = "checkpoint_" + generate_filename(
                epoch=epoch,
                file_extension="pt",
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss.item(),
                save_path=checkpoint_path / checkpoint_filename,
            )

        # calculate total epoch loss by averaging over the epoch
        loss_history[epoch] = sum(epoch_losses) / len(epoch_losses)

        # save metrics after each epoch in case training is stopped early
        # train data shape, val data shape
        run_metrics = {
            "timestamp": timestamp,
            "train_data_shape": train_data.shape,
            "val_data_shape": val_data.shape,
            "hyperparameters": {
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "train_steps": args.train_steps,
                "time_stride": args.time_stride,
                "epochs": args.epochs,
            },
            "metrics": {
                "train_loss": loss_history,  # dict of per-epoch loss
                "val_loss": val_loss_history,  # dict of per-epoch loss
            },
        }

        metrics_file = base_path / "metrics.json"

        with Path.open(metrics_file, "w") as f:
            json.dump(run_metrics, f, indent=4)

    # always save checkpoint at the end
    checkpoint_filename = "final_checkpoint_" + generate_filename(epoch=epoch, file_extension="pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=loss.item(),
        save_path=checkpoint_path / checkpoint_filename,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.KAN import KAN
from models.common import (
    autocast_if_available,
    ensure_directory,
    normalize_device,
    select_embedding_slice,
    should_use_amp,
)
from utils.utils_files import write_gene_list

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Flexible regression loss supporting MSE, RMSE, and MAE."""

    def __init__(self, mode: str = "RMSE") -> None:
        super().__init__()
        mode = mode.upper()
        if mode not in {"MSE", "RMSE", "MAE"}:
            raise ValueError(f"Unsupported loss mode: {mode}")
        self.mode = mode

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        if self.mode == "MSE":
            return diff.pow(2).mean()
        if self.mode == "RMSE":
            return diff.pow(2).mean().sqrt()
        return diff.abs().mean()


class SpotDataset(Dataset):
    """Mini-batch friendly wrapper for cell embeddings grouped by spot."""

    def __init__(self, cells: torch.Tensor, targets: torch.Tensor, group_size: int) -> None:
        if cells.ndim != 2:
            raise ValueError(f"Expected 2D tensor for cells, got shape {cells.shape}")
        if cells.shape[0] % group_size != 0:
            raise ValueError(
                f"Cell embeddings ({cells.shape[0]}) must be divisible by group_size ({group_size})"
            )
        if targets.ndim != 2:
            raise ValueError(f"Expected 2D tensor for targets, got shape {targets.shape}")

        num_spots = cells.shape[0] // group_size
        if num_spots != targets.shape[0]:
            raise ValueError(
                f"Mismatch between inferred spots ({num_spots}) and targets ({targets.shape[0]})"
            )

        self.cells = cells.view(num_spots, group_size, cells.shape[1]).contiguous()
        self.targets = targets
        self.group_size = group_size

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cells[index], self.targets[index]


@dataclass
class TrainerConfig:
    lr: float
    min_lr: float
    num_warmup_steps: int
    total_epochs: int
    batch_size: int
    loss_mode: str
    device: torch.device
    use_amp: bool
    use_scheduler: bool
    embedding_part: str

    @classmethod
    def from_args(cls, args: object) -> "TrainerConfig":
        device = normalize_device(getattr(args, "device", "cuda:0"))
        return cls(
            lr=float(getattr(args, "lr", 5e-4)),
            min_lr=float(getattr(args, "min_lr", 1e-5)),
            num_warmup_steps=int(getattr(args, "num_warmup_steps", 1000)),
            total_epochs=int(getattr(args, "total_epoches", getattr(args, "total_epochs", 1))),
            batch_size=int(getattr(args, "batch_size", 64)),
            loss_mode=str(getattr(args, "loss_mode", "MSE")),
            device=device,
            use_amp=bool(getattr(args, "use_amp", False)),
            use_scheduler=bool(getattr(args, "use_scheduler", False)),
            embedding_part=str(getattr(args, "embedding_part", "all")),
        )


def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, steps_per_epoch: int, config: TrainerConfig
) -> LambdaLR:
    total_steps = config.total_epochs * max(1, steps_per_epoch)
    min_lr_scale = config.min_lr / config.lr if config.lr else 0.0

    def lr_lambda(step: int) -> float:
        if total_steps <= config.num_warmup_steps:
            return 1.0
        if step < config.num_warmup_steps:
            return float(step) / float(max(1, config.num_warmup_steps))
        progress = (step - config.num_warmup_steps) / float(total_steps - config.num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * max(0.0, cosine)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _prepare_dataset(folder: Path, config: TrainerConfig) -> tuple[SpotDataset, int, int]:
    train_data_path = folder / "save_embedding" / "Embed_training.pt"
    train_blob = torch.load(train_data_path, map_location="cpu")

    cells = train_blob["x"].float()
    targets = train_blob["y"].float()

    if targets.shape[0] == 0:
        raise ValueError("Training targets are empty.")
    group_size, remainder = divmod(cells.shape[0], targets.shape[0])
    if remainder != 0:
        raise ValueError("Cell embeddings cannot be evenly divided into spot groups.")

    dataset = SpotDataset(cells, targets, group_size=group_size)
    input_dim = cells.shape[1]

    cnts_path = folder / "cnts.tsv"
    cnts = pd.read_csv(cnts_path, sep="\t")
    num_genes = len(cnts.columns) - 1

    return dataset, input_dim, num_genes


def _build_dataloader(dataset: SpotDataset, config: TrainerConfig) -> DataLoader:
    batch_size = max(1, min(config.batch_size, len(dataset)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=config.device.type == "cuda",
    )


def _run_training_loop(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    save_dir: Path,
    config: TrainerConfig,
    group_size: int,
) -> None:
    device = config.device
    model.to(device)

    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = (
        _cosine_schedule_with_warmup(optimizer, len(dataloader), config)
        if config.use_scheduler
        else None
    )

    use_amp = should_use_amp(config.use_amp, device)
    scaler = GradScaler(enabled=use_amp)

    best_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config.total_epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{config.total_epochs}",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_cells, batch_targets in progress:
            optimizer.zero_grad(set_to_none=True)

            batch_cells = select_embedding_slice(
                batch_cells.to(device, non_blocking=True),
                config.embedding_part,
            ).contiguous()
            batch_targets = batch_targets.to(device, non_blocking=True)

            spot_count = batch_cells.shape[0]
            batch_cells = batch_cells.view(spot_count * group_size, -1)

            with autocast_if_available(use_amp):
                cell_predictions = model(batch_cells)
                spot_predictions = cell_predictions.view(spot_count, group_size, -1).sum(dim=1)
                loss = loss_fn(spot_predictions, batch_targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            running_loss += loss.item() * spot_count
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        epoch_loss = running_loss / len(dataloader.dataset)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            ensure_directory(save_dir)
            torch.save(model, save_dir / "predictor.pth")

        tqdm.write(f"Epoch {epoch}: loss={epoch_loss:.4f} (best epoch {best_epoch})")

    tqdm.write("Training complete.")


def train_predictor(folder: str, args: object | None = None) -> None:
    """Train the predictor network for a given slide folder."""
    folder_path = Path(folder)
    config = TrainerConfig.from_args(args or object())

    dataset, input_dim, num_genes = _prepare_dataset(folder_path, config)
    logger.info("Training predictor for %s (%d spots)", folder_path.name, len(dataset))

    if config.embedding_part.lower() == "sub":
        input_dim = 1536
    else:
        input_dim = input_dim

    model = KAN(
        layers_hidden=[input_dim, 256, 256, 256, 256, num_genes],
        grid_range=[-1, 1],
        base_output_only=False,
    )

    dataloader = _build_dataloader(dataset, config)
    if len(dataloader) == 0:
        raise RuntimeError("Training dataset is empty.")

    loss_fn = CombinedLoss(mode=config.loss_mode)

    write_gene_list(folder)
    predictor_dir = folder_path / "save_predictor"

    _run_training_loop(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        save_dir=predictor_dir,
        config=config,
        group_size=dataset.group_size,
    )

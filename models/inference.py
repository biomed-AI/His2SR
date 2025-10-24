from __future__ import annotations

import logging
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.common import autocast_if_available, normalize_device, select_embedding_slice, should_use_amp
from utils.utils_files import save_pickle

logger = logging.getLogger(__name__)


def _load_inference_embeddings(folder: Path) -> torch.Tensor:
    inference_path = folder / "save_embedding" / "Embed_inference.pt"
    tensor = torch.load(inference_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Inference file did not contain a tensor: {inference_path}")
    return tensor.float()


def _load_trained_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = torch.load(model_path, map_location=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected a torch.nn.Module at {model_path}, found {type(model).__name__}")
    return model.to(device).eval()


def _build_dataloader(data: torch.Tensor, batch_size: int, device: torch.device) -> DataLoader:
    dataset = TensorDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )


def _reshape_predictions(cells: torch.Tensor, folder: Path, tile_size: int) -> torch.Tensor:
    image_path = folder / f"HE_scaled_pad{tile_size}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Scaled image not found at {image_path}")
    height, width, _ = image.shape
    prediction_size = (height // 16, width // 16)
    return cells.view(prediction_size[0], prediction_size[1], -1).cpu().numpy()


def run_inference(folder: str, tile_size: int = 224, args: object | None = None) -> None:
    folder_path = Path(folder)
    args = args or object()
    device = normalize_device(getattr(args, "device", "cuda:0"))
    embedding_part = getattr(args, "embedding_part", "all")
    batch_size = int(getattr(args, "inference_batch_size", 256))
    use_amp = should_use_amp(getattr(args, "use_amp", False), device)

    model_path = folder_path / "save_predictor" / "predictor.pth"

    logger.info("Running inference for %s", folder_path.name)
    embeddings = _load_inference_embeddings(folder_path)
    model = _load_trained_model(model_path, device)
    dataloader = _build_dataloader(embeddings, batch_size=batch_size, device=device)

    predictions: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for (batch_cells,) in tqdm(dataloader, desc="Predicting", unit="batch", dynamic_ncols=True):
            batch_cells = select_embedding_slice(
                batch_cells.to(device, non_blocking=True),
                embedding_part,
            ).contiguous()
            with autocast_if_available(use_amp):
                cell_predictions = model(batch_cells)
            predictions.append(cell_predictions.detach().cpu())

    merged = torch.cat(predictions, dim=0)
    reshaped = _reshape_predictions(merged, folder_path, tile_size)

    result_dir = folder_path / "result" / "His2SR"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(reshaped, result_dir / "Hist2SRpred.pickle")

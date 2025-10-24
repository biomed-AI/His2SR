from pathlib import Path
from typing import Optional

import torch
from argparse import Namespace

from feature_extractor.run_embedding_shift import run_whole_pipline
from models.inference import run_inference
from models.train import train_predictor
from utils.utils_files import check_required_files
from utils.utils_image import rescale_and_padding
from utils.utils_spot import spot_coord
from visualization.plot_heatmap import run_prediction_visual


def _load_encoder(weight_path: Path, device_index: int) -> torch.nn.Module:
    map_location = torch.device(f"cuda:{device_index}")
    model = torch.load(weight_path, map_location=map_location)
    return model.eval()


def run_HistToSR(
    folder,
    tile_encoder_path,
    slide_encoder_path,
    epochs,
    lr=5e-4,
    min_lr=1e-5,
    use_amp=False,
    use_scheduler=False,
    tile_size=224,
    stride=56,
    device=0,
    tile_encoder: Optional[torch.nn.Module] = None,
    slide_encoder: Optional[torch.nn.Module] = None,
    loss_mode: str = "MSE",
):
    """
    Before running, please make sure the following files exist in the folder:
        HE_raw.jpg
        pixel-size-raw.txt
        radius-raw.txt
        locs-raw.tsv
        cnts.tsv
    """

    folder_path = Path(folder)
    embedding_dir = folder_path / "save_embedding"
    embed_inference = embedding_dir / "Embed_inference.pt"
    embed_training = embedding_dir / "Embed_training.pt"

    embeddings_ready = embed_inference.exists() and embed_training.exists()

    if embeddings_ready:
        print("\033[33mEmbedding files exist. No need to extract features.\033[0m")
    else:
        check_required_files(str(folder_path))
        scaled_image = folder_path / f"HE_scaled_pad{tile_size}.jpg"
        if not scaled_image.exists():
            rescale_and_padding(str(folder_path), target_pixel_size=0.5, tile_size=tile_size)
        spot_coord(str(folder_path))

        encoder_tile = tile_encoder or _load_encoder(Path(tile_encoder_path), device)
        encoder_slide = slide_encoder or _load_encoder(Path(slide_encoder_path), device)

        run_whole_pipline(
            str(folder_path),
            encoder_tile,
            encoder_slide,
            device,
            tile_size=tile_size,
            stride=stride,
        )

    args = Namespace(
        lr=lr,
        num_warmup_steps=1000,
        min_lr=min_lr,
        batch_size=64,
        total_epoches=epochs,
        device=device,
        lr_anneal_steps=200,
        loss_mode=loss_mode,
        use_amp=use_amp,
        use_scheduler=use_scheduler,
        embedding_part="all",  # all or sub
    )

    train_predictor(str(folder_path), args=args)
    run_inference(str(folder_path), tile_size=tile_size, args=args)
    run_prediction_visual(str(folder_path))

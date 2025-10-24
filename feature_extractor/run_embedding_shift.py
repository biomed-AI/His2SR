import os
import time
from einops import rearrange, reduce, repeat
import numpy as np
import torch
from feature_extractor.embedding import get_embedding
import cv2
import timm.models.vision_transformer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.utils_files import save_pickle, get_locs, read_string, load_pickle, get_gene_counts
from utils.utils_spot import get_disk_mask, get_patches_flat, get_coord
import random
import shutil
from skimage.measure import block_reduce
from tqdm import tqdm
import torch.nn.functional as F

def seed_torch(device, seed=7):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_embeddings_shift(
        folder, tile_encoder, slide_encoder_model, img, margin=224, stride=56, device=3):
    factor = 16
    H, W, _ = img.shape
    shape_emb = (H // factor, W // factor)


    agg_cls = torch.zeros((shape_emb[0], shape_emb[1], 768), dtype=torch.float32)
    agg_sub = torch.zeros((shape_emb[0], shape_emb[1], 1536), dtype=torch.float32)

    
    dir_to_save_tiles = os.path.join(folder, "temp_tiles")

    if os.path.exists(dir_to_save_tiles):
        for filename in os.listdir(dir_to_save_tiles):
            file_path = os.path.join(dir_to_save_tiles, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    os.makedirs(dir_to_save_tiles, exist_ok=True)
    

    if stride<=margin:
        count_map = torch.zeros(shape_emb, dtype=torch.float32)
        

        start_list = list(range(0, margin, stride))
        if start_list[-1] != margin:
            start_list.append(margin)

        total_shifts = len(start_list) ** 2
        shift_idx = 0

        for y_shift in start_list:
            for x_shift in start_list:
                shift_idx += 1
                y0 = y_shift
                y1 = min(H - (margin - y_shift), H)
                x0 = x_shift
                x1 = min(W - (margin - x_shift), W)
                im = img[y0:y1, x0:x1]
                cls, sub = get_embedding(
                        im, tile_encoder=tile_encoder, slide_encoder_model=slide_encoder_model, device=device, slide_dir=dir_to_save_tiles)
                del im
                sta0 = y0 // factor
                sta1 = x0 // factor
                sto0 = sta0 + sub.shape[0]
                sto1 = sta1 + sub.shape[1]
                if agg_cls is not None:
                    agg_cls[sta0:sto0, sta1:sto1] += cls
                    del cls
                agg_sub[sta0:sto0, sta1:sto1] += sub
                del sub
                count_map[sta0:sto0, sta1:sto1] += 1
                print(f"[Shift {shift_idx}/{total_shifts}] start=({y_shift},{x_shift})")

        count_map[count_map == 0] = 1.0
        if agg_cls is not None:
            agg_cls /= count_map[..., None]

        agg_sub /= count_map[..., None]

        if os.path.exists(dir_to_save_tiles):
            print(f"Deleting temp tiles dir...")
            shutil.rmtree(dir_to_save_tiles)
        return agg_cls, agg_sub
    
    else:
        print("\033[31mInvalid shift stride. Shift embedding is not available\033[0m")
        cls, sub = get_embedding(
                img, tile_encoder=tile_encoder, slide_encoder_model=slide_encoder_model, device=device, slide_dir=dir_to_save_tiles)
        del img
        return cls, sub




def handle_embeddings(folder, embs):
    """
    embs['cls']: None or torch.Tensor, (H, W, C1)
    embs['sub']: torch.Tensor, (H, W, C2)
    embs['rgb']: torch.Tensor, (H, W, 3)
    """
    print('\033[32mProcessing embeddings...\033[0m')
    sub = embs['sub'].to(torch.float32).cpu()  # (H, W, C2)
    rgb = embs['rgb'].to(torch.float32).cpu()  # (H, W, 3)

    if embs['cls'] is not None:
        cls = embs['cls'].to(torch.float32).cpu()  # (H, W, C1)
        emb = torch.cat([cls, sub, rgb], dim=-1)   # (H, W, C1+C2+3)
        del cls
    else:
        emb = torch.cat([sub, rgb], dim=-1)        # (H, W, C2+3)

    del sub, rgb

    H, W, D = emb.shape
    cell_embeddings = emb.view(-1, D)  # Flatten to (H*W, D)
    del emb 

    locs = get_locs(folder)
    coord_path = os.path.join(folder, "embeddings-spot-coord.pickle")
    spot_to_cells, coord = get_coord(coord_path, (H, W), spot_num=len(locs))

    spot_embeds = []
    for cells_in_spot in spot_to_cells.values():
        spot_embeds.append(cell_embeddings[cells_in_spot])  # Tensor slice

    train_x = torch.cat(spot_embeds, dim=0)  # (total_cells_in_spots, D)
    del spot_embeds
    gene_expressions = get_gene_counts(folder)
    train_y = torch.tensor(gene_expressions.values, dtype=torch.float32)

    train_data_no_graph = {"x": train_x, "y": train_y}

    embedding_save_path = os.path.join(folder, f"save_embedding")
    os.makedirs(embedding_save_path, exist_ok=True)
    torch.save(train_data_no_graph, os.path.join(embedding_save_path, "Embed_training.pt"))
    del train_data_no_graph, train_x
    torch.save(cell_embeddings, os.path.join(embedding_save_path, "Embed_inference.pt"))
    del cell_embeddings

    print(f'\033[32mFinished! PT files saved to {embedding_save_path}\033[0m')



def run_whole_pipline(folder, tile_encoder, slide_encoder_model, device, tile_size=224, stride=56):
    image_path = os.path.join(folder, f"HE_scaled_pad{tile_size}.jpg")
    wsi = cv2.imread(image_path)

    emb_cls, emb_sub = get_embeddings_shift(folder, tile_encoder, slide_encoder_model, wsi, stride=stride,device=device)
    embs = dict(cls=emb_cls, sub=emb_sub)
    embs['rgb'] = np.stack([
            reduce(
                wsi[..., i].astype(np.float16) / 255.0,
                '(h1 h) (w1 w) -> h1 w1', 'mean',
                h=16, w=16).astype(np.float32)
            for i in range(3)])

    embs['rgb'] = torch.from_numpy(block_reduce(wsi, block_size=(16, 16, 1), func=np.mean).astype(np.float32) / 255.0)


    handle_embeddings(folder, embs)
    

def downsample_tensor(wsi_tensor, block_size=16):
    if wsi_tensor.ndim == 3:
        wsi_tensor = wsi_tensor.unsqueeze(0)
    if wsi_tensor.max() > 1.0:
        wsi_tensor = wsi_tensor.float() / 255.0
    pooled = F.avg_pool2d(wsi_tensor, kernel_size=block_size)
    return pooled.squeeze(0)  # (3, H//block, W//block)
import os
from gigapath.pipeline import TileEncodingDataset
from gigapath.pipeline import load_tile_encoder_transforms
from feature_extractor.use_gigapath import run_inference_with_tile_encoderv2, run_inference_with_slide_encoderv2
from einops import rearrange
from utils.utils_files import save_pickle
import torch
import numpy as np
import cv2
import pandas as pd
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def sort_key(file_path): 
    filename = os.path.basename(file_path)  
    x, y = filename.split('.png')[0].split('_')
    x, y = int(x.replace('x', '')), int(y.replace('y', ''))
    return (y, x)  



def get_embedding(img, device, tile_encoder, slide_encoder_model, slide_dir):

    split_image_into_tiles(img,slide_dir)

    # Get the dimensions of the image
    height, width, _ = img.shape

    # Calculate the number of tiles
    num_tiles_y = height // 224
    num_tiles_x = width // 224


    image_paths = [os.path.join(slide_dir, im) for im in os.listdir(slide_dir) if im.endswith('.png')]

    # image_paths sort
    image_paths = sorted(image_paths, key=sort_key)


    # tile embedding
    tile_encoder_outputs = run_inference_with_tile_encoderv2(image_paths, tile_encoder, device=device)
    # (Num_patch, 196, 1536)
    sub_embeds = tile_encoder_outputs['tile_embeds_sub'].reshape(num_tiles_y,num_tiles_x,196,1536)
    sub_embeds = sub_embeds.reshape(num_tiles_y,num_tiles_x,14,14,1536)
    sub_embeds = rearrange(sub_embeds, "p1 p2 h w c-> (p1 h) (p2 w) c") 

    
    #slide embedding
    slide_embeds = run_inference_with_slide_encoderv2(slide_encoder_model=slide_encoder_model, tile_embeds_cls=tile_encoder_outputs['tile_embeds_cls'], coords=tile_encoder_outputs['coords'], device=device)
    cls = slide_embeds['last_layer_embed'].permute(1,0,2).repeat(1, 196, 1)

    cls_embeds = cls.reshape(num_tiles_y,num_tiles_x,196,768)
    cls_embeds = cls_embeds.reshape(num_tiles_y,num_tiles_x,14,14,768)
    cls_embeds = rearrange(cls_embeds, "p1 p2 h w c-> (p1 h) (p2 w) c") 

    return cls_embeds, sub_embeds


def split_image_into_tiles(img, slide_dir, tile_size=224):

    height, width, _ = img.shape
    num_tiles_y = height // tile_size
    num_tiles_x = width // tile_size

    data = []
    os.makedirs(slide_dir, exist_ok=True)

    # Split the image and save the tiles
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Calculate the coordinates
            y_start = i * tile_size
            y_end = y_start + tile_size
            x_start = j * tile_size
            x_end = x_start + tile_size
            
            # Extract the tile
            tile = img[y_start:y_end, x_start:x_end]
            
            # Calculate the center coordinates
            center_x = x_start + tile_size // 2
            center_y = y_start + tile_size // 2
            
            # Name the tile based on the center coordinates
            tile_name = f"{center_x}x_{center_y}y.png"
            
            # Save the tile as a PNG file
            tile_path = os.path.join(slide_dir, tile_name)
            cv2.imwrite(tile_path, tile)
            
            # Save the center coordinates and name of the tile
            data.append({'tile_name': tile_name, 'center_x': center_x, 'center_y': center_y})

    # Save the data to a CSV file
    df = pd.DataFrame(data)
    # df.to_csv(os.path.join(slide_dir, 'tiles_centers.csv'), index=False)

    # print(f"Tiles have been saved to the directory: {slide_dir}")


@torch.no_grad()
def run_inference_with_resnet(image_paths: List[str], resnet: torch.nn.Module, batch_size: int=256, device:int = 0):
    print(f'\033[32mrun resnet on device {device}\033[0m')
    resnet = resnet.cuda(device)
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    # run inference
    resnet.eval()
    collated_outputs = {'tile_embeds_sub': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with resnet'):
            x = resnet(batch['img'].cuda(device)).detach().cpu()
            collated_outputs['tile_embeds_sub'].append(x)
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}



@torch.no_grad()
def run_inference_with_densenet(image_paths: List[str], densenet: torch.nn.Module, batch_size: int=256, device:int = 0):
    print(f'\033[32mrun resnet on device {device}\033[0m')
    densenet = densenet.cuda(device)
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    # run inference
    densenet.eval()
    collated_outputs = {'tile_embeds_sub': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with densenet'):
            x = densenet(batch['img'].cuda(device)).detach().cpu()
            collated_outputs['tile_embeds_sub'].append(x)
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}
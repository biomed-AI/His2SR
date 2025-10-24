from gigapath.pipeline import TileEncodingDataset
from gigapath.pipeline import load_tile_encoder_transforms
from gigapath.slide_encoder import LongNetViT
import torch
from typing import Optional, List
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x

def pool_all(self, x:torch.Tensor, pool_type: Optional[str]=None) -> torch.Tensor:
    if self.attn_pool is not None:
        x = self.attn_pool(x)
        return
    x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
    return x

def forward_head_all(self, x:torch.Tensor, pre_logits:bool=False) -> torch.Tensor:
    x = self.pool_all(x)
    x = self.fc_norm(x)
    x = self.head_drop(x)
    return x if pre_logits else self.head(x)

def forward_all(self, x:torch.Tensor) -> torch.Tensor:
    x = self.forward_features(x)
    x = self.forward_head_all(x)
    return x


VisionTransformer.pool_all = pool_all
VisionTransformer.forward_head_all = forward_head_all
VisionTransformer.forward_all = forward_all

def LongNetViT_forward_all(self, x, coords, all_layer_embed=False):
        """
        The forward pass of the model

        Arguments:
        ----------
        x: torch.Tensor
            The input tile embeddings, of shape [N, L, D]
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        all_layer_embed: bool
            Whether to return embeddings from all layers or not
        """
        # embed patches
        x = self.patch_embed(x)

        # get pos indices
        pos = self.coords_to_pos(coords, self.tile_size) # [N, L]

        x = x + self.pos_embed[:, pos, :].squeeze(0)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        if all_layer_embed:
            x_list = self.encoder(src_tokens=None, token_embeddings=x, return_all_hiddens=all_layer_embed)["encoder_states"]
        else:
            x_list = [self.encoder(src_tokens=None, token_embeddings=x)["encoder_out"]]

        outcomes = []
        for x in x_list:
            if self.global_pool:
                #x = x[:, 1:, :].mean(dim=1)  # global average pooling
                x = x[:, 1:, :]
                outcome = self.norm(x)
            else:
                x = self.norm(x)
                outcome = x[:, 0]
            outcomes.append(outcome)

        return outcomes


LongNetViT.forward_all = LongNetViT_forward_all


# https://github.com/prov-gigapath/prov-gigapath/blob/main/gigapath/pipeline.py
@torch.no_grad()
def run_inference_with_slide_encoderv2(tile_embeds_cls: torch.Tensor, coords: torch.Tensor, slide_encoder_model: torch.nn.Module, device:int = 3) -> torch.Tensor:
    """
    Run inference with the slide encoder, variant of Gigapath's run_inference_with_slide_encoder

    Arguments:
    ----------
    tile_embeds : torch.Tensor
        Tile embeddings
    coords : torch.Tensor
        Coordinates of the tiles
    slide_encoder_model : torch.nn.Module
        Slide encoder model
    """
    print(f'\033[32mrun slide encoder on device {device}\033[0m')
    if len(tile_embeds_cls.shape) == 2:
        tile_embeds_cls = tile_embeds_cls.unsqueeze(0)
        coords = coords.unsqueeze(0)

    slide_encoder_model = slide_encoder_model.cuda( device)
    slide_encoder_model.eval()
    # run inference
    with torch.cuda.amp.autocast(dtype=torch.float16):
        slide_embeds = slide_encoder_model.forward_all(tile_embeds_cls.cuda( device), coords.cuda( device), all_layer_embed=True)

    outputs = {"layer_{}_embed".format(i): slide_embeds[i].cpu() for i in range(len(slide_embeds))}
    outputs["last_layer_embed"] = slide_embeds[-1].cpu()
    return outputs


# https://github.com/prov-gigapath/prov-gigapath/blob/main/gigapath/pipeline.py
@torch.no_grad()
def run_inference_with_tile_encoderv2(image_paths: List[str], tile_encoder: torch.nn.Module, batch_size: int=256, device:int = 3) -> dict:
    """
    Run inference with the tile encoder, variant of Gigapath's run_inference_with_tile_encoder

    Arguments:
    ----------
    image_paths : List[str]
    List of image paths, each image is named with its coordinates
    tile_encoder : torch.nn.Module
        Tile encoder model
    """
    print(f'\033[32mrun tile encoder on device {device}\033[0m')
    tile_encoder = tile_encoder.cuda(device)
    # make the tile dataloader
    tile_dl = DataLoader(TileEncodingDataset(image_paths, transform=load_tile_encoder_transforms()), batch_size=batch_size, shuffle=False)
    # run inference
    tile_encoder.eval()
    collated_outputs = {'tile_embeds_sub': [],'tile_embeds_cls': [], 'coords': []}
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for batch in tqdm(tile_dl, desc='Running inference with tile encoder'):
        # for batch in tile_dl:
            x = tile_encoder.forward_all(batch['img'].cuda(device)).detach().cpu()
            collated_outputs['tile_embeds_sub'].append(x[:,1:,:])
            collated_outputs['tile_embeds_cls'].append(x[:,0,:])
            collated_outputs['coords'].append(batch['coords'])
    return {k: torch.cat(v) for k, v in collated_outputs.items()}




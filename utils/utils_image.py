from time import time
from skimage.transform import rescale
from utils.utils_files import load_image, save_image, write_string, read_string
import os
import numpy as np



def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img



def crop_image(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img



def adjust_margins(image_path, pad, pad_value=255):
    img_name, img_format = os.path.splitext(os.path.basename(image_path))
    img = load_image(image_path)
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    save_image(img, os.path.dirname(image_path)+"/"+ img_name + f"_pad{pad}.jpg")



def rescale_and_padding(folder, target_pixel_size=0.5, tile_size=224):

    print(f'\033[33mRecaling HE...\033[0m')
    image_path = os.path.join(folder, "HE_raw.jpg")
    raw_pxsize_path = os.path.join(folder, "pixel-size-raw.txt")
    scaled_image_path = os.path.join(folder, "HE_scaled.jpg")

    # Rescaling
    raw_pixel_size = float(read_string(raw_pxsize_path))
    target_pixel_size = float(target_pixel_size)
    scale = raw_pixel_size / target_pixel_size
    img = load_image(image_path)
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    t0 = time()
    img = rescale_image(img, scale)
    print(int(time() - t0), 'sec')
    img = img.astype(np.uint8)
    save_image(img, scaled_image_path)

    # Padding
    adjust_margins(scaled_image_path, tile_size, pad_value=255)
    print(f'\033[32mSuccessfully rescaled\033[0m')
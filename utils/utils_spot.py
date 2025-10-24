from utils.utils_files import get_locs, read_string, save_pickle, write_string, load_pickle
import numpy as np
import pandas as pd
import cv2
import os

def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin



def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list



def spot_coord(folder):
    print(f'\033[33mCalculating sopt coordinates...\033[0m')
    locs_raw_path = os.path.join(folder, "locs-raw.tsv")
    loc_path = os.path.join(folder, "locs.tsv")
    padded_image_path = os.path.join(folder, "HE_scaled_pad224.jpg")
    radius_raw_path = os.path.join(folder, "radius-raw.txt")
    radius_save_path = os.path.join(folder, "radius.txt")
    embeddings_spot_coord_save_path = os.path.join(folder, "embeddings-spot-coord.pickle")

    coords = pd.read_csv(locs_raw_path,index_col=0,sep='\t')
    target_pixel_size = float(0.5)
    locs = coords / target_pixel_size
    locs = locs[['x','y']].round().astype(int)
    locs.to_csv(loc_path,index=True,sep='\t')

    locs = get_locs(folder)
    img = cv2.imread(padded_image_path)

    height, width = int(img.shape[0]/16),int(img.shape[1]/16)
    y_indices, x_indices = np.indices((height, width))
    coordinates = np.stack((y_indices, x_indices), axis=-1)
    radius_raw = float(read_string(radius_raw_path))
    radius = radius_raw/0.5

    write_string(int(radius), radius_save_path)

    radius = radius / 16 
    mask = get_disk_mask(radius)
    x = get_patches_flat(coordinates, locs, mask)
    save_pickle(x,embeddings_spot_coord_save_path)
    
    print(f'\033[32mFinished\033[0m')



def get_coord(coord_path, embedding_shape, spot_num):
    coord = load_pickle(coord_path)
    coord = coord.reshape(-1, 2)
    def calculate_new_index(row, col, num_cols, order='C'):
       if order == 'C':
           return row * num_cols + col
       elif order == 'F':
           return col * embedding_shape[0] + row
       else:
           raise ValueError("Order must be 'C' or 'F'.")

    num_rows = embedding_shape[0]
    num_cols = embedding_shape[1]

    new_indices_row_major = []
    for point in coord:
       row, col = point
       new_index_row_major = calculate_new_index(row, col, num_cols, order='C')
       new_indices_row_major.append(new_index_row_major)

    flattened_row_major = np.array(new_indices_row_major)
    spot_labels = np.array([i // 145 for i in range(spot_num*145)])
    cell_spot_labels = np.column_stack((flattened_row_major, spot_labels))

    spot_dict = {}
    for cell_coord, spot in cell_spot_labels:
        if spot not in spot_dict:
            spot_dict[spot] = []
        spot_dict[spot].append(cell_coord)
    return spot_dict, coord
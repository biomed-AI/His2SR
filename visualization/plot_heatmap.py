import numpy as np
import matplotlib.pyplot as plt
from utils.utils_files import save_image, read_gene_list, load_pickle
import os

def save_heatmap(x, save_path):
    x = x.copy()
    mask_x = np.isfinite(x)
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min) / (x_max - x_min + 1e-12)
    cmap = plt.get_cmap('turbo')
    img1 = cmap(x)[..., :3]
    img1[~mask_x] = 1.0
    img1 = (img1 * 255).astype(np.uint8)
    save_image(img1, save_path)

def generate_heatmap(x):
    x = x.copy()
    mask_x = np.isfinite(x)
    x_max = np.max(x)
    x_min = np.min(x)
    x = (x - x_min) / (x_max - x_min + 1e-12)
    cmap = plt.get_cmap('turbo')
    img = cmap(x)[..., :3]
    img[~mask_x] = 1.0
    img = (img * 255).astype(np.uint8)
    return img



def run_prediction_visual(folder):
    save_dir = os.path.join(folder,"visualization")
    os.makedirs(save_dir,exist_ok=True)
    gene_list = read_gene_list(folder)
    prediton = load_pickle(os.path.join(folder, "result", f"His2SR", "Hist2SRpred.pickle"))
    gene_num = len(gene_list)
    for i in range(gene_num):
        save_path = os.path.join(save_dir, f"{gene_list[i]}.png")
        save_heatmap(prediton[:,:,i], save_path)
    
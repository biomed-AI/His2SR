import torch
from utils.utils_files import load_pickle, load_tsv, read_gene_list, save_pickle, get_gene_counts
from scipy.stats import wasserstein_distance
from visualization.plot_heatmap import generate_heatmap
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
from utils.utils_files import load_tsv,read_gene_list
import os
import ot
import numpy as np
from tqdm import tqdm
from geomloss import SamplesLoss
from scipy.special import rel_entr
from scipy.stats import spearmanr
from utils.utils_files import read_gene_list
import pandas as pd
from torchmetrics.functional import structural_similarity_index_measure as ssim_gpu

def performance_summary(pred_path, ground_truth_path, gene_list_dir):
    """
    Compare predicted and ground truth spatial transcriptome data by computing 
    RMSE, SSIM, PCC, and Spearman correlation for each gene.
    
    Both inputs must be pickle files with matching gene order.

    Saves a CSV with the results in the same folder as pred_path.
    """
    pred = load_pickle(pred_path)
    gt = load_pickle(ground_truth_path)
    gene_list = read_gene_list(gene_list_dir)

    if pred.shape[-1] != gt.shape[-1]:
        raise FileNotFoundError(
            f"Inconsistency in the number of genes between the predicted spatial transcriptome ({pred.shape[-1]}) and the ground truth ({gt.shape[-1]})."
        )

    pred = pred[:gt.shape[0], :gt.shape[1], :]

    rmse_list = []
    ssim_list = []
    pcc_list = []

    for i in tqdm(range(pred.shape[-1]), desc="Calculating metrics..."):
        rmse_list.append(cal_rmse(pred[:, :, i], gt[:, :, i]))
        ssim_list.append(cal_ssim(generate_heatmap(pred[:, :, i]), generate_heatmap(gt[:, :, i]), ws=21))
        pcc_list.append(cal_pcc(pred[:, :, i], gt[:, :, i]))

    metrics_df = pd.DataFrame({
        "Gene": gene_list,
        "RMSE": rmse_list,
        "SSIM": ssim_list,
        'PCC': pcc_list
    })

    save_dir = os.path.dirname(pred_path)
    csv_path = os.path.join(save_dir, "performance_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    return rmse_list, ssim_list, pcc_list



def cal_rmse(prediction, groundtruth):
    x = prediction.copy().astype(float)
    y = groundtruth.copy().astype(float)
    x = np.clip(x, 0, None)
    y = np.clip(y, 0, None)
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    x = (x - min_x)/(max_x - min_x)
    y = (y - min_y)/(max_y - min_y)
    rmse = np.sqrt(((x - y) ** 2).mean())
    return  rmse



def cal_ssim(img1, img2, ws=7):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(img1, img2, full=True, win_size=ws)
    return score


def cal_pcc(matrix1, matrix2):
    x = matrix1.copy().astype(float)
    y = matrix2.copy().astype(float)
    x = np.clip(x, 0, None)
    y = np.clip(y, 0, None)
    max_x = np.max(x)
    min_x = np.min(x)
    max_y = np.max(y)
    min_y = np.min(y)
    x = (x - min_x)/(max_x - min_x)
    y = (y - min_y)/(max_y - min_y)

    vector_a = x.flatten()
    vector_b = y.flatten()
    mean_a = np.mean(vector_a)
    mean_b = np.mean(vector_b)
    covariance = np.sum((vector_a - mean_a) * (vector_b - mean_b)) / (x.size - 1)
    std_dev_a = np.std(vector_a, ddof=1)
    std_dev_b = np.std(vector_b, ddof=1)
    pcc = covariance / (std_dev_a * std_dev_b)  

    return  pcc


def cal_emd(matrix1, matrix2):
    """
    Calculate Earth Mover's Distance (EMD) between two spatial gene expression maps.

    Args:
        matrix1 (np.ndarray): 2D predicted spatial expression map.
        matrix2 (np.ndarray): 2D ground-truth spatial expression map.

    Returns:
        float: EMD score.
    """
    x = matrix1.copy().astype(float).flatten()
    y = matrix2.copy().astype(float).flatten()


    x = np.clip(x, 0, None)
    y = np.clip(y, 0, None)


    x = x / (np.sum(x) + 1e-8)
    y = y / (np.sum(y) + 1e-8)


    bins = np.arange(len(x))

    emd = wasserstein_distance(bins, bins, x, y)
    return emd

def cal_spearmanr(pred, gt):
    score, _ = spearmanr(pred.flatten(), gt.flatten())
    return score

def build_ground_truth_from_tsv(folder, ground_turth_dir):
    cnts_ordered = get_gene_counts(folder)
    gene_list = cnts_ordered.columns.to_list()
    result = []
    for i in tqdm(range(len(gene_list)),desc="Reading ground truth..."):
        matrix = load_tsv(os.path.join(ground_turth_dir, f"{gene_list[i]}.tsv"),index=False)
        result.append(matrix.values)
    result = np.stack(result,axis=-1)
    os.makedirs(os.path.join(folder, "ground_truth"),exist_ok=True)
    save_pickle(result, os.path.join(folder, "ground_truth","ground_truth.pickle"))
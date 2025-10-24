import itertools
from PIL import Image
import pickle
import os

import numpy as np
import pandas as pd
import yaml


Image.MAX_IMAGE_PIXELS = None


def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def load_image(filename):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img


def load_mask(filename):
    mask = load_image(filename)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)



def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def read_string(filename):
    return read_lines(filename)[0]


def write_lines(strings, filename):
    mkdir(filename)
    with open(filename, 'w') as file:
        for s in strings:
            file.write(f'{s}\n')



def write_string(string, filename):
    return write_lines([string], filename)


def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)


def join(x):
    return list(itertools.chain.from_iterable(x))


def get_most_frequent(x):
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
            labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]


def get_locs(prefix):
    locs_path = os.path.join(prefix, "locs.tsv")
    locs = load_tsv(locs_path)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs = locs.astype(float)
    locs /= 16
    locs = locs.round().astype(int)
    return locs


def get_locs_from_csv(coord_path):
    locs = pd.read_csv(coord_path, index_col=0)
    locs = np.stack([locs['imagecol'], locs['imagerow']], -1)
    locs = locs.astype(float)
    locs /= 16
    locs = locs.round().astype(int)
    return locs

def get_gene_counts(prefix, reorder_genes=True):

    cnts_path = os.path.join(prefix, 'cnts.tsv')
    cnts = load_tsv(cnts_path)
    if reorder_genes:
        order = cnts.var().to_numpy().argsort()[::-1]
        cnts = cnts.iloc[:, order]
    return cnts

def write_gene_list(prefix):
    cnts_ordered = get_gene_counts(prefix)
    gene_list = cnts_ordered.columns.to_list()
    save_path = os.path.join(prefix,"gene_list.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        for item in gene_list:
            f.write(f"{item}\n")

def read_gene_list(prefix):
    file_path = os.path.join(prefix,"gene_list.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        gene_list = [line.strip() for line in f]
    return gene_list


def check_required_files(folder):
    required_files = {
        "HE_raw.jpg": "Raw H&E image file.",
        "pixel-size-raw.txt": "Pixel size: the physical length (in microns) represented by one pixel in HE_raw.jpg.",
        "locs-raw.tsv": "Spot coordinates (in microns).",
        "radius-raw.txt": "Radius of a spot (in microns).",
        "cnts.tsv": "Gene expression profile for each spot."
    }
    missing_files = []
    for filename in required_files:
        full_path = os.path.join(folder, filename)
        if not os.path.isfile(full_path):
            missing_files.append((full_path, required_files[filename]))

    if missing_files:
        missing_info = "\n".join([f"{path}  <-- {desc}" for path, desc in missing_files])
        raise FileNotFoundError(
            f"The following required files are missing in folder '{folder}':\n\n"
            f"{missing_info}\n\n"
            "Please ensure all required files are present before proceeding.\n"
            "For specific file details and examples, please refer to the github page of this project."
        )
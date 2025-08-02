import sys

import os
import anndata
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_pickle, save_image, read_lines, load_image, build_her2st_data


def plot_super(
        x, outfile, underground=None, truncate=None):

    x = x.copy()
    mask = np.isfinite(x)

    if truncate is not None:
        x -= np.nanmean(x)
        x /= np.nanstd(x) + 1e-12
        x = np.clip(x, truncate[0], truncate[1])
    x -= np.nanmean(x, (0, 1))
    x /= np.nanstd(x, (0, 1)) + 1e-12

    x -= np.nanmin(x)
    x /= np.nanmax(x) + 1e-12

    cmap = plt.get_cmap('turbo')
    # cmap = cmap_turbo_truncated
    if underground is not None:
        under = underground.mean(-1, keepdims=True)
        under -= under.min()
        under /= under.max() + 1e-12

    img = cmap(x)[..., :3]
    if underground is not None:
        img = img * 0.5 + under * 0.5
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    save_image(img, outfile)


def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'

    adata = sc.read_visium(path=prefix,count_file='filtered_feature_bc_matrix.h5')
    # adata = build_her2st_data(prefix, 'H1')
    adata.var_names_make_unique()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=1000)
    gene_names = adata.var[adata.var['highly_variable']].index

    # gene_names = read_lines(f'{prefix}gene-names.txt')
    # mask = load_image(f'{prefix}mask-small.png') > 0

    for i, gn in tqdm(enumerate(gene_names), total=len(gene_names)):
        if gn in ['HSPA1A', 'DNER', 'C11orf96']:
            cnts = load_pickle(f'{prefix}cnts-super/{gn}.pickle', verbose=False)
            # cnts[~mask] = np.nan
            plot_super(cnts, f'{prefix}cnts-super-plots/{gn}.png')


if __name__ == '__main__':
    main()

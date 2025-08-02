import sys

import numpy as np
import scanpy as sc
import pandas as pd
from einops import reduce
from tqdm import tqdm

from utils import load_image, load_tsv, read_lines, read_string, setToArray, build_her2st_data
from visual import plot_spots


def plot_spots_multi(
        cnts, locs, gene_names, radius, img, prefix,
        disk_mask=True):
    for i, gname in tqdm(enumerate(gene_names), total=len(gene_names)):
        if gname in ['HSPA1A', 'DNER', 'C11orf96']:
            ct = cnts.iloc[:, i]
            outfile = f'{prefix}{gname}.png'
            plot_spots(
                    img=img, cnts=ct, locs=locs, radius=radius,
                    cmap='turbo', weight=1.0,
                    disk_mask=disk_mask,
                    outfile=outfile)


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    factor = 16

    # adata = sc.read_visium(path='data/151673',count_file='151673_filtered_feature_bc_matrix.h5')
    # adata = sc.read_visium(path=prefix,count_file='V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5')
    # adata = build_her2st_data(prefix, 'G1')
    adata = sc.read_visium(path=prefix,count_file='filtered_feature_bc_matrix.h5')

    adata.var_names_make_unique()
    # adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cnts = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
#     gene_names = read_lines(f'{prefix}gene-names.txt')
    gene_names = adata.var[adata.var['highly_variable']].index
    cnts = cnts[gene_names]
    locs = adata.obsm['spatial']
    locs = np.stack([locs[:,1], locs[:,0]], -1)
    locs *= 2

    sample_index = np.random.choice(range(cnts.shape[0]), size=round(1 * cnts.shape[0]),replace=False)
    sample_index = setToArray(set(sample_index))
    cnts = cnts.iloc[sample_index]
    locs = locs[sample_index]

    factor = 16
    radius = 80
    spot_radius = np.round(radius / factor).astype(int)
    locs = (locs / factor).round().astype(int)

    img = load_image(f'{prefix}image.jpg')
    img = reduce(
            img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)

    # plot spot-level gene expression measurements
    plot_spots_multi(
            cnts=cnts,
            locs=locs, gene_names=gene_names,
            radius=spot_radius, disk_mask=True,
            img=img, prefix=prefix+'spots/')


if __name__ == '__main__':
    main()

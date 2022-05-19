from scvi.data import brainlarge_dataset
import scanpy as sc
import numpy as np

adata = brainlarge_dataset(n_genes_to_keep=720)
x_norm = sc.pp.normalize_total(adata, target_sum=1, inplace=False)["X"]
dense_x = x_norm.todense()
np.save("brainlarge_normalized.npy", dense_x)

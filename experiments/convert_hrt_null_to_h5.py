import numpy as np
import torch
import h5py
from flowselect.util import write_h5

# Load file
hrt_null_samples = torch.load(
    "./output/experiments/gaussian_ar_mixture_linear_d100_n100000_rho0.98_/run001/nojoint/hrt/null_samples.npy"
)
dataset = torch.load(
    "./output/experiments/gaussian_ar_mixture_linear_d100_n100000_rho0.98_/run001/dataset"
)
X_true = dataset[2].X[:2000]

X_true.shape

X_hrt_12 = np.stack((X_true[:, 0], hrt_null_samples[1][0]), axis=1)

write_h5(
    "./hrt_null_12.h5",
    {"X_hrt_12": X_hrt_12},
)

from pathlib import Path
from numpy.core.numeric import roll

import torch
import numpy as np
import pandas as pd
import h5py
import matplotlib
import tqdm
from matplotlib import pyplot as plt
from torch.nn.functional import threshold
from flowselect.crt import confusion_matrix, benjamini_hochberg
from flowselect.util import write_h5

DATASETS = [
    "gaussian_ar_mixture_linear_d100_n100000_rho0.98_",
    "gaussian_ar_mixture_sin_cos5_d100_n100000_rho0.98_",
    "rnasq_cor_linear_a40_n10_nv100_nvc5_nt100000_",
    "rnasq_cor_sin_cos5_a40_n10_nv100_nvc5_nt100000_",
]

FEATURE_STATS = [
    "fast_lasso_10000",
    "fast_rf_10000",
    "fast_lasso_10000",
    "fast_rf_10000",
]


ROOT_PATH = Path("output/experiments/")

DGP_PATH = Path(
    "output/experiments/gaussian_ar_mixture_sin_cos5_d100_n100000_rho0.98_/"
)


FEATURE_STATS_PATH = Path("gaussmaf_5/mcmc_n10000_s1000/fast_rf_10000/feature_stats.h5")


def get_fdr_and_power(p_values, beta, nominal_fdrs):
    is_relevant = ~np.isclose(beta, 0.0)
    res = []
    thresholds = []
    for fdr in nominal_fdrs:
        threshold, accept = benjamini_hochberg(p_values, fdr)
        thresholds.append(threshold)
        _, fp, _, tp = confusion_matrix(
            y_true=is_relevant, y_pred=accept.astype(int)
        ).ravel()
        fdp = fp / max(fp + tp, 1.0)
        power = tp / max(1.0, is_relevant.sum())
        res.append((fdp, power))
    threshold_df = pd.DataFrame(
        list(zip(nominal_fdrs, thresholds)), columns=["n_fdrs", "threshold"]
    )
    pvalue_df = pd.DataFrame(
        list(zip(p_values, is_relevant, beta)),
        columns=["p_values", "is_relevant", "beta"],
    )
    fdrs, powers = list(zip(*res))
    power_df = pd.DataFrame(
        list(zip(nominal_fdrs, fdrs, powers)),
        columns=["n_fdrs", "fdr", "power"],
    )
    return threshold_df, pvalue_df, power_df


def get_fdr_and_power_for_run(runid, dataset_idx):
    dgp_path = ROOT_PATH / DATASETS[dataset_idx]
    run_path = dgp_path / f"run{runid:03d}"
    feature_stats_path = (
        Path("gaussmaf_5/mcmc_n10000_s1000")
        / FEATURE_STATS[dataset_idx]
        / "feature_stats.h5"
    )
    fname = run_path / feature_stats_path
    dataset_fname = run_path / "dataset"
    with h5py.File(fname, "r") as f:
        feature_stats = f["feature_stats"][()]

    feature_stats_less_than_true = feature_stats[:, 0:1] <= feature_stats[:, 1:]

    rolling_sum = feature_stats_less_than_true.cumsum(axis=1)
    rolling_divisors = np.array(range(2, rolling_sum.shape[1] + 2))
    rolling_pvalues = (rolling_sum + 1) / rolling_divisors.reshape(1, -1)

    train, _, _ = torch.load(dataset_fname)

    beta = train.beta["beta"].flatten()
    nominal_fdrs = np.array([0.05, 0.1, 0.25])

    power_dfs = []

    for i in range(rolling_pvalues.shape[1]):
        _, _, power_df = get_fdr_and_power(rolling_pvalues[:, i], beta, nominal_fdrs)
        power_dfs.append(power_df)

    fdr_010 = [df["fdr"].to_numpy() for df in power_dfs]
    power = [df["power"].to_numpy() for df in power_dfs]
    return rolling_pvalues, np.stack(fdr_010), np.stack(power)


for (i, dataset) in enumerate(DATASETS):
    rolling_pvalues_runs = []
    fdrs = []
    powers = []
    for run in tqdm.tqdm(range(1, 21)):
        rolling_pvalues, fdr, power = get_fdr_and_power_for_run(run, i)
        rolling_pvalues_runs.append(rolling_pvalues)
        fdrs.append(fdr)
        powers.append(power)
    rolling_pvalues_runs = np.stack(rolling_pvalues_runs)
    fdrs = np.stack(fdrs)
    powers = np.stack(powers)
    write_h5(
        dataset + "_asymptotic_fdr_and_power.h5",
        {
            "rolling_pvalues_runs": rolling_pvalues_runs,
            "fdrs": fdrs,
            "powers": powers,
        },
    )

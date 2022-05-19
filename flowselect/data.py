import os
import torch

import numpy as np
import pandas as pd
import torch.functional as F

from hydra.utils import to_absolute_path
from torch.utils.data import Dataset
from typing import Sequence, Optional

import flowselect.response as response
from flowselect.util import save_and_symlink
from flowselect.ddlk.ddlk.data import GaussianAR1, GaussianMixtureAR1
from pathlib import Path


def get_data(cfg, log):
    getters = {
        "gaussian_ar": GaussianAR1DataGenerator,
        "gaussian_ar_mixture": GaussianAR1MixtureDataGenerator,
        "rnasq": RNAsqDataGenerator,
        "soybean": SoybeanDataGenerator,
    }
    if cfg.data.type == "dgp":
        data_getter = getters[cfg.dgp_covariate.name](cfg, log)
    else:
        data_getter = getters[cfg.data.label](cfg, log)
    train, valid, test = data_getter()
    return train, valid, test


class DataGenerator:
    def __init__(self, cfg, log):
        self.cfg = cfg
        self.data = cfg.data
        self.covariate = cfg.dgp_covariate
        self.resp = cfg.dgp_response
        self.log = log

    @property
    def id(self):
        return (
            f"{self.covariate.name}_{self.resp.name}_"
            + f"d{self.covariate.d}_n{self.covariate.n_obs}"
            + f"/run{self.data.run_id:03d}"
        )

    @property
    def save_path(self):
        return (
            Path(to_absolute_path(self.cfg.output))
            .joinpath(self.id)
            .joinpath("dataset")
        )

    def __call__(self):
        self.cfg.data["data_id"] = self.id
        self.log.info(f"data_id: {self.id}")
        if self.save_path.exists():
            self.log.info(f"Using existing datafile at {self.save_path.resolve()}")
            train, valid, test = torch.load(self.save_path.resolve())
        else:
            train, valid, test = self.generate_data()
        save_and_symlink((train, valid, test), self.save_path, "dataset")
        return train, valid, test


class GaussianAR1DataGenerator(DataGenerator):
    def generate_data(self):
        data_sampler = GaussianAR1(p=self.covariate.d, rho=self.covariate.rho)
        X = data_sampler.sample(n=self.covariate.n_obs, use_torch=False)

        response_model = response.make_response(self.resp)
        _, Y, beta = response_model(X)
        X = X.astype("float32")
        Y = Y.astype("float32")

        dataset = JointDataset(X, Y, beta=beta, dgp=data_sampler)
        train_nobs = int(np.ceil(self.covariate.n_obs * self.data.train_size))
        valid_test_nobs = self.covariate.n_obs - train_nobs
        test_nobs = int(np.ceil(2 * self.data.test_size * valid_test_nobs))
        valid_nobs = valid_test_nobs - test_nobs
        train, valid_test = dataset.random_split([train_nobs, valid_test_nobs])
        valid, test = valid_test.random_split([valid_nobs, test_nobs])

        return train, valid, test


class GaussianAR1MixtureDataGenerator(DataGenerator):
    @property
    def id(self):
        return (
            f"{self.covariate.name}_{self.resp.name}_"
            + f"d{self.covariate.d}_n{self.covariate.n_obs}_"
            + f"rho{self.covariate.rho_base}_"
            + f"/run{self.data.run_id:03d}"
        )

    def generate_data(self):
        prop = (
            2 + (0.5 + np.arange(self.covariate.k) - self.covariate.k / 2) ** 2
        ) ** 0.9
        prop = prop / prop.sum()
        rho_list = [
            self.covariate.rho_base ** (i * self.covariate.rho_decay + 0.9)
            for i in range(self.covariate.k)
        ]

        data_sampler = GaussianMixtureAR1(
            p=self.covariate.d,
            rho_list=rho_list,
            mu_list=[self.covariate.mu_shift * i for i in range(self.covariate.k)],
            proportions=prop,
        )
        X = data_sampler.sample(n=self.covariate.n_obs, use_torch=False)

        response_model = response.make_response(self.resp)
        _, Y, beta = response_model(X)
        X = X.astype("float32")
        Y = Y.astype("float32")

        dataset = JointDataset(X, Y, beta=beta, dgp=data_sampler)
        train_nobs = int(np.ceil(self.covariate.n_obs * self.data.train_size))
        valid_test_nobs = self.covariate.n_obs - train_nobs
        test_nobs = int(np.ceil(2 * self.data.test_size * valid_test_nobs))
        valid_nobs = valid_test_nobs - test_nobs
        train, valid_test = dataset.random_split([train_nobs, valid_test_nobs])
        valid, test = valid_test.random_split([valid_nobs, test_nobs])

        return train, valid, test


class RNAsqDataGenerator(DataGenerator):
    @property
    def id(self):
        return (
            f"rnasq_cor_{self.resp.name}_"
            + f"a{int(self.resp.signal_a)}_"
            + f"n{int(self.resp.signal_n)}_"
            + f"nv{int(self.data.n_vars)}_"
            + f"nvc{int(self.data.n_vars_corr)}_"
            + f"nt{int(self.data.n_threshold)}_"
            + f"/run{self.data.run_id:03d}"
        )

    def generate_data(self):
        rnasq = np.load(to_absolute_path(self.data.location))
        if self.data.n_threshold:
            rnasq = rnasq[
                np.random.choice(rnasq.shape[0], self.data.n_threshold, replace=False)
            ]
        corrs = np.corrcoef(rnasq, rowvar=False)
        np.fill_diagonal(corrs, -2.0)
        corr_max = corrs.max(axis=1)
        highest_correlated = np.argsort(-corr_max)
        features = highest_correlated[: self.data.n_vars]

        rnasq = rnasq[:, features]
        X1 = (rnasq - rnasq.mean()) / rnasq.std()
        X = X1 + (np.random.standard_normal(X1.shape) * self.data.noise) * (
            X1 == X1.min()
        )
        n_obs, d = X.shape

        response_model = response.make_response(self.resp)
        _, Y, beta = response_model(
            X, beta_nonzero_in=range(0, self.data.n_vars_corr * 2, 2)
        )
        X = X.astype("float32")
        Y = Y.astype("float32")

        dataset = JointDataset(X, Y, beta=beta, dgp=None)
        train_nobs = int(np.ceil(n_obs * self.data.train_size))
        valid_test_nobs = n_obs - train_nobs
        test_nobs = int(np.ceil(2 * self.data.test_size * valid_test_nobs))
        valid_nobs = valid_test_nobs - test_nobs
        train, valid_test = dataset.random_split([train_nobs, valid_test_nobs])
        valid, test = valid_test.random_split([valid_nobs, test_nobs])

        return train, valid, test


def get_soybeans_flows(trait="_height.txt", imputed=False, path="./data/soybeans/"):
    path = Path(path)
    start = "IMP" if imputed else "QA"
    geno_pheno = pd.read_csv(path.joinpath(start + trait), sep="\t")
    genotypes = geno_pheno.iloc[:, 4:]  # X's start at column 4
    genotypes = torch.as_tensor(genotypes.values, dtype=torch.int8)
    # genotypes = genotypes.long()  # convert to long for 1-hot encoding
    # genotypes = F.one_hot(genotypes + 1)  # 1-hot encoding
    # genotypes = genotypes.char()  # make uint8 to use less memory
    traits = geno_pheno.iloc[:, 1]  # extract the normalized trait
    return genotypes, traits


class SoybeanDataGenerator(DataGenerator):
    @property
    def id(self):
        return f"soybean_snps{self.data.n_snps}" + f"/trait_{self.data.trait}"

    def generate_data(self):
        genotypes, traits = get_soybeans_flows(
            f"_{self.data.trait}.txt", path=to_absolute_path(self.data.path)
        )
        X = genotypes.numpy() + 1
        print(f"Min class: {X.min()}")
        print(f"Max class: {X.max()}")
        if self.data.n_snps != "All":
            X = X[:, : self.data.n_snps]
        Y = traits.to_numpy()
        n_obs = X.shape[0]

        dataset = JointDataset(X, Y, beta=None, dgp=None)
        train_nobs = int(np.ceil(n_obs * self.data.train_size))
        valid_test_nobs = n_obs - train_nobs
        test_nobs = int(np.ceil(2 * self.data.test_size * valid_test_nobs))
        valid_nobs = valid_test_nobs - test_nobs
        train, valid_test = dataset.random_split([train_nobs, valid_test_nobs])
        valid, test = valid_test.random_split([valid_nobs, test_nobs])

        return train, valid, test


class JointDataset(Dataset):
    def __init__(self, X, Y=None, beta=None, d=None, mode="generation", dgp=None):
        self.X = X
        self.Y = Y
        self.beta = beta
        self.set_d(d)  # initial stage
        self.mode = mode
        self.dgp = dgp

        # generation returns only xs
        # prediction returns xs and ys
        assert self.mode in ["generation", "prediction"]

    def __getitem__(self, index):
        if self.mode == "generation":
            x = self.X[index]
            if self.d is None:
                return tuple([x])
            else:
                if self.d == 0:
                    return tuple([torch.tensor([1.0]), x[0]])
                else:
                    return tuple([x[: self.d], x[self.d]])
        elif self.mode == "prediction":
            assert self.Y is not None, "Y cannot be None..."
            x = self.X[index]
            y = self.Y[index]
            return tuple([x, y])
        else:
            raise NotImplementedError(
                f"Data loader mode [{self.mode}] is not implemented..."
            )

    def set_mode(self, mode="generation"):
        """Outputs label data in addition to training data
        x data is all columns of X, y data is Y
        """
        self.mode = mode

    def set_d(self, d=None):
        """Chooses dimension to train on:
        d = 0 -> x data is just 1s, y data is first column of X
        d = 1 -> x data is 1st column of X, y data is second column
        d = 2 -> x data is 1st 2 columns of X, y data is third column

        d must be in the range [0, X.shape[-1] - 1]
        """
        self.d = d

    def reset(self):
        """Resets JointDataset to original state"""
        self.set_mode()
        self.set_d()

    def __len__(self):
        return len(self.X)

    def random_split(
        self,
        lengths: Sequence[int],
        generator: Optional[torch.Generator] = torch.default_generator,
    ):
        r"""
        Copied from Torch; made to return subsets of the same class.
        Randomly split a dataset into non-overlapping new datasets of given lengths.
        Optionally fix the generator for reproducible results, e.g.:

        >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

        Arguments:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths of splits to be produced
            generator (Generator): Generator used for the random permutation.
        """
        # Cannot verify that dataset is Sized
        if sum(lengths) != len(self):  # type: ignore
            raise ValueError(
                "Sum of input lengths does not equal the length of the input dataset!"
            )

        indices = torch.randperm(sum(lengths), generator=generator).tolist()
        out = []
        for offset, length in zip(torch._utils._accumulate(lengths), lengths):
            idxs = indices[offset - length : offset]
            out.append(
                self.__class__(
                    self.X[idxs],
                    self.Y[idxs],
                    beta=self.beta,
                    d=self.d,
                    mode=self.mode,
                    dgp=self.dgp,
                )
            )
        return out

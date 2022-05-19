from abc import abstractmethod
import argparse
from itertools import cycle
from flowselect.joint import DiscreteFlowModel
from os import makedirs
from time import time_ns

from torch.distributions.categorical import Categorical
from flowselect.selection_methods import (
    feature_statistic_factory,
    HRTFeatureStatistic,
    FastNNFeatureStatistic,
)
from flowselect.util import save_and_symlink, write_h5
from joblib.parallel import Parallel, delayed
import pandas as pd
import torch
import numpy as np

from pathlib import Path
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
from torch.nn import Module
from einops import rearrange, repeat

from flowselect.ddlk.utils import get_two_moments, extract_data


class CRTFactory:
    def __init__(self):
        self.crt_fitters = {
            "mcmc": MCMCCRTFitter,
            "flow_crt": FlowCRTFitter,
        }

    def get_crt_fitter(self, cfg, name):
        fitter_class = self.crt_fitters[name]
        fitter = fitter_class(cfg)
        return fitter


class CRTFitter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = cfg.data
        self.joint = cfg.joint
        self.nominal_fdrs = np.arange(1, 51) / 100
        self.device = self.cfg.crt.device
        self.batch_size = self.cfg.crt.batch_size
        self.n_obs = self.cfg.crt.n_obs

    @abstractmethod
    def create_and_fit_crt_sampler(self, train, valid, q_joint):
        pass

    def __call__(self, train, valid, test, q_joint):
        t = time_ns()
        q_crt = self.create_and_fit_crt_sampler(train, valid, q_joint)
        trainloader, valloader, testloader = self.create_loaders(train, valid, test)
        trainloader.dataset.set_mode("prediction")
        valloader.dataset.set_mode("prediction")
        testloader.dataset.set_mode("prediction")
        # extract training and validation data
        xTr, yTr = extract_data(trainloader)
        xVal, yVal = extract_data(valloader)
        ## concatenate xTr and xVal to use in HRT
        xTr = torch.cat([xTr, xVal], axis=0)
        yTr = torch.cat([yTr, yVal], axis=0)
        xTr = xTr.float()  ## Data generating process
        xTr = xTr[: min(self.n_obs, xTr.size(0))]
        yTr = yTr[: min(self.n_obs, xTr.size(0))]
        # extract test data
        xTe, yTe = extract_data(testloader)
        xTe = xTe.float()
        xTe = xTe[: min(self.n_obs, xTe.size(0))]
        yTe = yTe[: min(self.n_obs, xTe.size(0))]

        q_crt.to(self.device)
        xTr = xTr.to(self.device)
        xTe = xTe.to(self.device)
        d = xTr.size(1)
        mcmc_sample_file = "crt_mcmc_samples.pt"
        gbl_mcmc_sample_file = self.save_folder.joinpath(mcmc_sample_file)

        xTr_tilde, xTe_tilde = None, None
        if gbl_mcmc_sample_file.exists():
            xTr_tilde, xTe_tilde = torch.load(gbl_mcmc_sample_file)
        else:
            if "fast_" in self.cfg.crt.feature_statistic:
                if self.cfg.crt.feature_statistic != "fast_nn":
                    xTe_tilde = self._sample_null_features(xTe, q_crt)
            else:
                xTr_tilde = self._sample_null_features(xTr, q_crt)
                xTe_tilde = self._sample_null_features(xTe, q_crt)

        save_and_symlink((xTr_tilde, xTe_tilde), gbl_mcmc_sample_file, mcmc_sample_file)

        xTr_np = xTr.cpu().numpy()
        xTe_np = xTe.cpu().numpy()
        if xTr_tilde is None:
            xTr_tilde_np = None
        else:
            xTr_tilde_np = xTr_tilde.numpy()
        if xTe_tilde is None:
            xTe_tilde_np = None
        else:
            xTe_tilde_np = xTe_tilde.numpy()

        crt_feature_statistic = get_crt_feature_statistic(self.cfg, xTr, yTr, q_crt)
        feature_stats = crt_feature_statistic(
            xTr_np, xTr_tilde_np, yTr, xTe_np, xTe_tilde_np, yTe
        )
        feature_stat_file = "feature_stats.h5"
        gbl_feature_stat_file = self.varsel_save_folder.joinpath(feature_stat_file)
        if gbl_feature_stat_file.is_symlink():
            gbl_feature_stat_file.unlink()
        save_and_symlink(
            None,
            gbl_feature_stat_file,
            feature_stat_file,
            savefun=lambda: write_h5(
                feature_stat_file, {"feature_stats": feature_stats}
            ),
        )
        p_values = (
            (feature_stats[:, 0:1] <= feature_stats[:, 1:]).sum(axis=1) + 1
        ) / feature_stats.shape[1]
        crt_time = time_ns() - t
        self.save_timings_to_disk(crt_time)

        threshold_df, pvalue_df, power_df = self._get_fdr_and_power(
            p_values, trainloader
        )
        if power_df is not None:
            fnames = ["thresholds.csv", "pvalues.csv", "power.csv"]
            dfs = [threshold_df, pvalue_df, power_df]
        else:
            fnames = ["thresholds.csv", "pvalues.csv"]
            dfs = [threshold_df, pvalue_df]
        for fname, df in zip(fnames, dfs):
            gbl_file = self.varsel_save_folder.joinpath(fname)
            if gbl_file.is_symlink():
                gbl_file.unlink()
            save_and_symlink(
                None,
                gbl_file,
                fname,
                savefun=lambda: df.to_csv(fname),
            )
        return pvalue_df, power_df

    def _sample_null_features(self, x, q_crt):
        x_samples = []
        n_batches = int(np.ceil(x.size(0) / self.batch_size))
        for i in tqdm(range(n_batches)):
            xTr_tilde_i = q_crt.sample(
                x[(i * self.batch_size) : min((i + 1) * self.batch_size, x.size(0))]
            )
            x_samples.append(xTr_tilde_i)
        x_tilde = torch.cat(x_samples, dim=1)
        return x_tilde

    def _get_fdr_and_power(self, p_values, trainloader):
        known_response = not (trainloader.dataset.beta is None)
        if known_response:
            beta = trainloader.dataset.beta["beta"].flatten()
            is_relevant = ~np.isclose(beta, 0.0)
            res = []
        # beta = trainloader.dataset.beta["beta"].flatten()
        # pvalue_df = pd.DataFrame(
        #     list(zip(p_values, is_relevant, beta)),
        #     columns=["p_values", "is_relevant", "beta"],
        # )
        thresholds = []
        for fdr in self.nominal_fdrs:
            threshold, accept = benjamini_hochberg(p_values, fdr)
            thresholds.append(threshold)
            if known_response:
                _, fp, _, tp = confusion_matrix(
                    y_true=is_relevant, y_pred=accept.astype(int)
                ).ravel()
                fdp = fp / max(fp + tp, 1.0)
                power = tp / max(1.0, is_relevant.sum())
                res.append((fdp, power))
        threshold_df = pd.DataFrame(
            list(zip(self.nominal_fdrs, thresholds)), columns=["n_fdrs", "threshold"]
        )
        if known_response:
            pvalue_df = pd.DataFrame(
                list(zip(p_values, is_relevant, beta)),
                columns=["p_values", "is_relevant", "beta"],
            )
            fdrs, powers = list(zip(*res))
            power_df = pd.DataFrame(
                list(zip(self.nominal_fdrs, fdrs, powers)),
                columns=["n_fdrs", "fdr", "power"],
            )
        else:
            pvalue_df = pd.DataFrame(
                list(zip(p_values)),
                columns=["p_values"],
            )
            power_df = None
        return threshold_df, pvalue_df, power_df

    def create_loaders(self, train, valid, test=None):
        train_cfg = argparse.Namespace(
            batch_size_train=64,
            batch_size_valid=1000,
            batch_size_test=1000,
            drop_last=False,
        )
        trainloader = DataLoader(
            train,
            batch_size=train_cfg.batch_size_train,
            shuffle=True,
            drop_last=train_cfg.drop_last,
        )
        valloader = DataLoader(
            valid,
            batch_size=train_cfg.batch_size_valid,
            shuffle=False,
            drop_last=train_cfg.drop_last,
        )
        if test is not None:
            testloader = DataLoader(
                test,
                batch_size=train_cfg.batch_size_test,
                shuffle=False,
                drop_last=train_cfg.drop_last,
            )
            return trainloader, valloader, testloader
        else:
            return trainloader, valloader

    @property
    def id(self):
        return self.__class__.__name__

    @property
    def varsel_id(self):
        return self.cfg.crt.feature_statistic + "_" + str(self.cfg.crt.n_obs)

    @property
    def save_folder(self):
        path = (
            Path(to_absolute_path(self.cfg.output))
            .joinpath(self.data.data_id)
            .joinpath(self.joint.joint_id)
            .joinpath(self.id)
        )
        return path

    @property
    def varsel_save_folder(self):
        return self.save_folder.joinpath(self.varsel_id)

    def save_path(self, savename):
        path = self.save_folder.joinpath(savename)
        return path

    def varsel_save_path(self, savename):
        path = self.varsel_save_folder.joinpath(savename)
        return path

    def save_timings_to_disk(self, time):
        if not self.varsel_save_folder.is_dir():
            makedirs(self.varsel_save_folder)
        path = self.varsel_save_folder.joinpath("time.txt")
        with path.open("w") as fp:
            fp.writelines([f"{time}"])


class MCMCCRTFitter(CRTFitter):
    @property
    def id(self):
        return f"mcmc_n{self.cfg.crt.n_obs}_s{self.cfg.crt.mcmc_steps}"

    def save_state(self, q_knockoff):
        pass

    def create_and_fit_crt_sampler(self, train, valid, q_joint):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = get_two_moments(trainloader)
        X_sigma = torch.from_numpy(np.cov(train.X.transpose()))
        if isinstance(q_joint, DiscreteFlowModel):
            q_crt = DiscreteFlowMCMC(self.cfg, q_joint)
        else:
            q_crt = MCMCCRT(self.cfg, q_joint, X_mu, X_sigma)
        return q_crt


class MCMCCRT(Module):
    def __init__(self, cfg, q_joint, X_mu, X_sigma):
        super().__init__()
        self.cfg = cfg
        self.q_joint = q_joint
        self.X_mu = X_mu
        self.X_sigma = X_sigma
        self.mcmc_steps = self.cfg.crt.mcmc_steps

        ## Calculate conditional variance for each covariate
        cond_var = torch.stack(
            [self.calc_cond_var(self.X_sigma, j) for j in range(self.X_sigma.size(0))]
        )
        self.register_buffer("cond_var", cond_var)

    def forward(self, X):
        return self.sample(X)

    def sample(self, X):
        self.cond_var = self.cond_var.type_as(X)
        # X is N by D
        # We want it to be an N x D x D-1 array where
        # X[i, j] is a vector of all covariates except j
        # X_long = torch.stack(
        #     [self.delete_at_index(X, j) for j in range(X.size(1))], dim=1
        # )
        X_long = X.unsqueeze(-2).repeat([1, X.size(1), 1])
        X_current = X
        samples = [X_current.cpu()]
        tbar = tqdm(range(self.mcmc_steps))
        for _ in tbar:
            X_new, accept = self.mh_step(X_current, X_long)
            samples.append(X_new.cpu())
            tbar.set_description(f"Acceptance rate : {accept.float().mean():.3f}")
            X_current = X_new
        X_out = torch.stack(samples)
        return X_out

    @staticmethod
    def delete_at_index(X, j):
        return torch.cat((X[:, 0:j], X[:, (j + 1) :]), dim=1)

    @staticmethod
    def calc_cond_var(Sigma, j):
        no_j_idx = [i for i in range(Sigma.size(0)) if i != j]
        Sigma_without_j = Sigma[no_j_idx][:, no_j_idx]
        B, _ = torch.solve(Sigma[no_j_idx, j : (j + 1)], Sigma_without_j)
        return Sigma[j, j] - Sigma[j : (j + 1), no_j_idx].matmul(B)[0, 0]

    def mh_step(self, X_current, X_long):
        X_tilde = X_current + (self.cond_var.sqrt() * torch.randn_like(X_current))
        q_tilde = self._calc_acceptance_prob(X_tilde, X_long)
        q_orig = self._calc_acceptance_prob(X_current, X_long)
        # Since it is a random walk, the proposal densities are equal
        log_p_accept = q_tilde - q_orig
        accept = (
            torch.rand(
                [X_current.size(0), X_current.size(1)], device=log_p_accept.device
            ).log()
            < log_p_accept
        )
        X_out = X_current.clone()
        X_out[accept] = X_tilde[accept]
        return X_out, accept

    def _calc_acceptance_prob(self, X_sample, X_long):
        # First set the diagonal of X_long to the current sample
        torch.diagonal(X_long, dim1=1, dim2=2)[:, :] = X_sample
        # Now make X_long a (ND) x D matrix for use with joint density
        X_long = X_long.reshape(-1, X_long.size(-1))
        # Calculate the joint density
        p = self.q_joint.log_prob(X_long)
        # Reshape back the joint density so it is N x D (same as X_sample)
        p = p.reshape(X_sample.shape)
        return p


import torch.nn.functional as F
from torch.distributions import OneHotCategorical


class DiscreteFlowMCMC(Module):
    def __init__(self, cfg, q_joint):
        super().__init__()
        self.cfg = cfg
        self.q_joint = q_joint
        self.mcmc_steps = self.cfg.crt.mcmc_steps

    def forward(self, X):
        return self.sample(X)

    def fit(self, X):
        with torch.no_grad():
            # X is N by D
            # We want it to be an N x D x D-1 array where
            # X[i, j] is a vector of all covariates except j
            # X_long = torch.stack(
            #     [self.delete_at_index(X, j) for j in range(X.size(1))], dim=1
            # )
            N, D = X.shape
            X_oh = X.long()  # convert to long for 1-hot encoding
            X_oh = F.one_hot(X_oh)  # 1-hot encoding
            X_oh = X_oh.char().float()  # make uint8 to use less memory
            N, D, K = X_oh.shape
            # X_long = X.unsqueeze(-2).repeat([1, X.size(1), 1]) # (N x D x D x K)
            self.log_prob = torch.zeros(
                size=(N, D, K), dtype=torch.float32, device=X.device
            )
            print("Calculating probabilities...")
            oh = F.one_hot(torch.arange(K, device=X_oh.device), num_classes=K).float()
            tbar = tqdm(range(D))
            for j in tbar:
                xj = X_oh[:, j, :].clone()
                for k in range(K):
                    X_oh[:, j, :] = oh[k]
                    p = self.q_joint(X_oh)
                    self.log_prob[:, j, k] = p
                X_oh[:, j, :] = xj

    def sample_from_fit(self, n_samples=None):
        print("Sampling ...")
        sampler = Categorical(logits=self.log_prob)
        samples = []
        # X_current = X
        # samples = [X_current.cpu()]
        if n_samples is None:
            n_samples = self.mcmc_steps
        tbar = tqdm(range(n_samples))
        for _ in tbar:
            X_new = sampler.sample()
            samples.append(X_new.cpu())
            X_current = X_new
        X_out = torch.stack(samples)
        return X_out

    # @staticmethod
    # def delete_at_index(X, j):
    #     return torch.cat((X[:, 0:j], X[:, (j + 1) :]), dim=1)

    # @staticmethod
    # def calc_cond_var(Sigma, j):
    #     no_j_idx = [i for i in range(Sigma.size(0)) if i != j]
    #     Sigma_without_j = Sigma[no_j_idx][:, no_j_idx]
    #     B, _ = torch.solve(Sigma[no_j_idx, j : (j + 1)], Sigma_without_j)
    #     return Sigma[j, j] - Sigma[j : (j + 1), no_j_idx].matmul(B)[0, 0]

    def sample(self, X):
        self.fit(X)
        return self.sample_from_fit()

    def mh_step(self, X_current, X_long):
        X_tilde = X_current + (self.cond_var.sqrt() * torch.randn_like(X_current))
        q_tilde = self._calc_acceptance_prob(X_tilde, X_long)
        q_orig = self._calc_acceptance_prob(X_current, X_long)
        # Since it is a random walk, the proposal densities are equal
        log_p_accept = q_tilde - q_orig
        accept = (
            torch.rand(
                [X_current.size(0), X_current.size(1)], device=log_p_accept.device
            ).log()
            < log_p_accept
        )
        X_out = X_current.clone()
        X_out[accept] = X_tilde[accept]
        return X_out, accept

    def _calc_acceptance_prob(self, X_sample, X_long):
        # First set the diagonal of X_long to the current sample
        torch.diagonal(X_long, dim1=1, dim2=2)[:, :] = X_sample
        # Now make X_long a (ND) x D matrix for use with joint density
        X_long = X_long.reshape(-1, X_long.size(-1))
        # Calculate the joint density
        p = self.q_joint.log_prob(X_long)
        # Reshape back the joint density so it is N x D (same as X_sample)
        p = p.reshape(X_sample.shape)
        return p


class FlowCRTFitter(CRTFitter):
    @property
    def id(self):
        return "flow_crt"

    def save_state(self, q_knockoff):
        pass

    def create_and_fit_crt_sampler(self, train, valid, q_conditional):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = get_two_moments(trainloader)
        X_sigma = torch.from_numpy(np.cov(train.X.transpose()))
        q_crt = FlowCRT(self.cfg, q_conditional)
        return q_crt


class FlowCRT(Module):
    def __init__(self, cfg, q_conditional):
        super().__init__()
        self.cfg = cfg
        assert isinstance(q_conditional, GenericConditionalFlow)
        self.q_conditional = q_conditional
        self.n_samples = self.cfg.crt.mcmc_steps

    def forward(self, X):
        return self.sample(X)

    def sample(self, X):
        self.q_conditional = self.q_conditional.to(X.device)
        samples = []
        for _ in range(self.n_samples):
            sample = self.q_conditional.sample([X.size(0)], cond_inputs=X)
            samples.append(sample)
        X_out = torch.stack(samples)
        return X_out


def get_crt_feature_statistic(cfg, x_in=None, y_in=None, q_crt=None):
    if cfg.crt.feature_statistic == "fast_nn":
        return FastNNFeatureStatistic(cfg, q_crt)
    elif "fast_" in cfg.crt.feature_statistic:
        return HRTFeatureStatistic(
            cfg.crt.feature_statistic, cfg.crt.n_jobs, cfg.crt.one_hot
        )
    else:
        return CRTFeatureStatistic(cfg, x_in=x_in, y_in=y_in)


class CRTFeatureStatistic:
    def __init__(self, cfg, x_in=None, y_in=None):
        self.cfg = cfg
        self.feature_stat = feature_statistic_factory(
            self.cfg.crt.feature_statistic,
            self.cfg.variable_selection,
            x_in=x_in,
            y_in=y_in,
        )
        self.n_jobs = self.cfg.crt.n_jobs

    def __call__(self, xTr, xTr_tilde, yTr, xTe, xTe_tilde, yTe):
        out = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(self._call_feature_stat)(xTr, xTr_tilde, yTr)
            for (xTr, xTr_tilde, yTr) in zip(cycle([xTr]), xTr_tilde, cycle([yTr]))
        )
        feature_stats = np.stack(out, axis=1)
        return feature_stats

    def _call_feature_stat(self, xTr, xTr_tilde, yTr):
        xTr = xTr.copy()
        d = xTr.shape[1]
        feature_stats = np.zeros(d)
        for j in range(d):
            features_orig = xTr[:, j].copy()
            xTr[:, j] = xTr_tilde[:, j]
            feature_stats[j] = self.feature_stat.fit(xTr, yTr)[j]
            xTr[:, j] = features_orig
        return feature_stats


def benjamini_hochberg(pvalues, alpha, yekutieli=False):
    """
    Bejamini-Hochberg procedure.
    Extracted from the implementation in sklearn.
    """
    n_features = len(pvalues)
    sv = np.sort(pvalues)
    criteria = float(alpha) / n_features * np.arange(1, n_features + 1)
    if yekutieli:
        c_m = np.log(n_features) + np.euler_gamma + 1 / (2 * n_features)
        criteria /= c_m
    selected = sv[sv <= criteria]
    if selected.size == 0:
        return 0.0, np.zeros_like(pvalues, dtype=bool)
    return selected.max(), pvalues <= selected.max()

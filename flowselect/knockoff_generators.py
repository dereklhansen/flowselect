from time import time_ns
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from flowselect.selection_methods import get_selection_method
from flowselect.ddlk.testing.hrt import get_fdp_power
import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np

from pathlib import Path
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torch.nn import Identity
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

import flowselect.flows as flows
from flowselect.alternatives.KnockoffGAN import KnockoffGAN
from flowselect.util import save_and_symlink, write_h5
from flowselect.ddlk.utils import get_two_moments, extract_data
from flowselect.ddlk.ddlk.ddlk import DDLK
from flowselect.ddlk.ddlk.data import GaussianAR1, GaussianMixtureAR1
from flowselect.modelx import ModelXKnockoff, ModelXKnockoffMixture

import pytorch_lightning as pl


path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)

from flowselect.ddlk.ddlk.swap import RandomSwapper, GumbelSwapper


def fit_knockoff(cfg, train, valid, test, q_joint):
    fitter_dict = {
        "ddlk": DDLKFitter,
        "modelx": ModelXKnockoffFitter,
        "gan": KnockoffGANFitter,
        "deepknockoff": DeepKnockoffFitter,
        "mass": MASSKnockoffFitter,
    }
    fitter = fitter_dict[cfg.knockoff.model_name](cfg)
    results, power = fitter(train, valid, test, q_joint)
    return results, power


class KnockoffFitter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = cfg.data
        self.joint = cfg.joint
        self.knockoff = cfg.knockoff

    def create_loaders(self, train, valid, test=None):
        trainloader = DataLoader(
            train,
            batch_size=self.knockoff.batch_size_train,
            shuffle=True,
            drop_last=self.knockoff.drop_last,
        )
        valloader = DataLoader(
            valid,
            batch_size=self.knockoff.batch_size_valid,
            shuffle=False,
            drop_last=self.knockoff.drop_last,
        )
        if test is not None:
            testloader = DataLoader(
                test,
                batch_size=self.knockoff.batch_size_test,
                shuffle=False,
                drop_last=self.knockoff.drop_last,
            )
            return trainloader, valloader, testloader
        else:
            return trainloader, valloader

    @property
    def id(self):
        return self.__class__.__name__

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
        return self.save_folder.joinpath(
            self.cfg.variable_selection.variable_selection_id
        )

    def save_path(self, savename):
        path = self.save_folder.joinpath(savename)
        return path

    @property
    def state_save_path(self):
        return self.save_path("knockoff_state")

    def save_state(self, q_knockoff):
        save_and_symlink(
            q_knockoff.state_dict(), self.state_save_path, "knockoff_state"
        )

    def __call__(self, train, valid, test, q_joint):
        t = time_ns()
        q_knockoff = self.create_and_fit_knockoff(train, valid, q_joint)
        self.knockoff["knockoff_id"] = self.id
        self.save_state(q_knockoff)
        results, power = self.variable_selection(train, valid, test, q_knockoff)
        knockoff_time = time_ns() - t
        self.save_timings_to_disk(knockoff_time)
        return results, power

    def variable_selection(self, train, valid, test, q_knockoff):
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
        # extract test data
        xTe, yTe = extract_data(testloader)
        xTe = xTe.float()

        xTr = xTr[: min(xTr.size(0), self.cfg.variable_selection.train_n_obs)]
        xTe = xTe[: min(xTe.size(0), self.cfg.variable_selection.test_n_obs)]
        yTr = yTr[: xTr.size(0)]
        yTe = yTe[: xTe.size(0)]
        knockoff_sample_location = self.save_path("knockoff_sample")
        knockoff_sample_h5_location = self.save_path("knockoff_sample.h5")
        density_plot_location = self.save_path("density_plot.png")

        if not knockoff_sample_location.exists():
            knockoff_sample_gpu = (
                None
                if self.cfg.variable_selection.knockoff_sample_gpu == "none"
                else self.cfg.variable_selection.knockoff_sample_gpu
            )
            if knockoff_sample_gpu is not None:
                device = torch.device(
                    "cuda:" + str(self.cfg.variable_selection.knockoff_sample_gpu)
                )
            else:
                device = torch.device("cpu")

            # with torch.no_grad():
            q_knockoff = q_knockoff.to(device)
            xTr_tilde = q_knockoff.sample(xTr.to(device)).cpu().numpy()
            xTe_tilde = q_knockoff.sample(xTe.to(device)).cpu().numpy()
            knockoff_sample = {"xTr_tilde": xTr_tilde, "xTe_tilde": xTe_tilde}
        else:
            knockoff_sample = torch.load(knockoff_sample_location)
            xTr_tilde = knockoff_sample["xTr_tilde"]
            xTe_tilde = knockoff_sample["xTe_tilde"]

        save_and_symlink(knockoff_sample, knockoff_sample_location, "knockoff_sample")
        save_and_symlink(
            None,
            knockoff_sample_h5_location,
            "knockoff_sample.h5",
            lambda: write_h5("knockoff_sample.h5", knockoff_sample),
        )

        # assert self.cfg.variable_selection.train_n_obs <= xTr_tilde.shape[0]
        max_obs_train = min(self.cfg.variable_selection.train_n_obs, xTr_tilde.shape[0])
        xTr_tilde = xTr_tilde[:max_obs_train]
        # assert self.cfg.variable_selection.test_n_obs <= xTe_tilde.shape[0]
        max_obs_test = min(self.cfg.variable_selection.test_n_obs, xTe_tilde.shape[0])
        xTe_tilde = xTe_tilde[:max_obs_test]

        selection_method = get_selection_method(self.cfg.variable_selection)
        variable_selection_id = (
            self.cfg.variable_selection.method_name
            + "_"
            + str(self.cfg.variable_selection.train_n_obs)
        )
        if self.cfg.variable_selection.method_name == "khrt":
            variable_selection_id = (
                variable_selection_id
                + "_"
                + self.cfg.variable_selection.feature_statistic
            )
        self.cfg.variable_selection["variable_selection_id"] = variable_selection_id
        knockoff_statistics = selection_method(
            self.cfg.variable_selection, xTr, yTr, xTr_tilde, xTe, yTe, xTe_tilde
        )

        knockoff_statistics = pd.Series(knockoff_statistics)
        results = pd.DataFrame(knockoff_statistics, columns=["statistic"]).join(
            pd.DataFrame(
                trainloader.dataset.beta["beta"].flatten(),
                index=np.arange(trainloader.dataset.beta["beta"].flatten().shape[0]),
                columns=["beta"],
            )
        )
        results.index.name = "feature"

        lcl_results_file = "results.csv"
        gbl_results_file = self.varsel_save_folder.joinpath(lcl_results_file)

        if gbl_results_file.is_symlink():
            gbl_results_file.unlink()
        save_and_symlink(
            None,
            gbl_results_file,
            lcl_results_file,
            savefun=lambda: results.to_csv(lcl_results_file),
        )

        power = self.export_power_pdf(results)
        lcl_power_file = "power.csv"
        gbl_power_file = self.varsel_save_folder.joinpath(lcl_power_file)
        if gbl_power_file.is_symlink():
            gbl_power_file.unlink()
        save_and_symlink(
            None,
            gbl_power_file,
            lcl_power_file,
            savefun=lambda: power.to_csv(lcl_power_file),
        )

        return results, power

    @staticmethod
    def export_power_pdf(results):
        """
        results: output from run_knockoffs
        """
        nominal_fdrs = (
            np.arange(1, 51) / 100
        )  # to do: make this an argument in the function

        W = np.array(results["statistic"])
        beta = np.array(1 * (results["beta"] != 0))

        fdrs = []
        powers = []
        for fdr in nominal_fdrs:
            f, p = get_fdp_power(W, beta, offset=0, nominal_fdr=fdr)
            fdrs.append(f)
            powers.append(p)

        df = pd.DataFrame(
            list(zip(nominal_fdrs, fdrs, powers)), columns=["n_fdrs", "fdr", "power"]
        )

        return df

    def save_timings_to_disk(self, time):
        if not self.varsel_save_folder.is_dir():
            os.makedirs(self.varsel_save_folder)
        path = self.varsel_save_folder.joinpath("time.txt")
        with path.open("w") as fp:
            fp.writelines([f"{time}"])


class DDLKFitter(KnockoffFitter):
    @property
    def id(self):
        return "ddlk"

    def create_and_fit_knockoff(self, train, valid, q_joint):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = get_two_moments(trainloader)
        hparams = argparse.Namespace(
            X_mu=X_mu,
            X_sigma=X_sigma,
            lr=self.knockoff.lr,
            init_type=self.knockoff.init_type,
            hidden_layers=self.knockoff.hidden_layers,
        )
        tb_logdir = self.state_save_path.parent.parent.joinpath("tb_logs_knockoff")
        logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.knockoff.model_name)
        q_knockoff = DDLK(hparams, q_joint=q_joint)
        if self.knockoff.early_stopping:
            ddlk_callbacks = [EarlyStopping(monitor="val_loss")]
        else:
            ddlk_callbacks = []
        trainer = pl.Trainer(
            max_epochs=self.knockoff.max_epochs,
            num_sanity_val_steps=self.knockoff.num_sanity_val_steps,
            deterministic=self.knockoff.deterministic,
            gradient_clip_val=self.knockoff.gradient_clip_val,
            weights_summary=self.knockoff.weights_summary,
            logger=logger,
            gpus=None if self.knockoff.gpu == "none" else self.knockoff.gpu,
            callbacks=ddlk_callbacks,
        )

        if not self.state_save_path.exists():
            trainer.fit(
                q_knockoff, train_dataloader=trainloader, val_dataloaders=valloader
            )
        else:
            q_knockoff.load_state_dict(torch.load(self.state_save_path))

        return q_knockoff


class ModelXKnockoffFitter(KnockoffFitter):
    @property
    def id(self):
        return "modelx"

    def save_state(self, q_knockoff):
        pass

    def create_and_fit_knockoff(self, train, valid, q_joint):
        dgp = train.dgp
        if isinstance(dgp, GaussianAR1):
            q_knockoff = ModelXKnockoff(dgp.mu, dgp.Sigma)
        elif isinstance(dgp, GaussianMixtureAR1):
            mus = [d.mu for d in dgp.normals]
            Sigmas = [d.Sigma for d in dgp.normals]
            q_knockoff = ModelXKnockoffMixture(mus, Sigmas, dgp.proportions)
        else:
            raise NotImplementedError("Model-X not available for this data type")
        return q_knockoff


class MASSKnockoffFitter(KnockoffFitter):
    @property
    def id(self):
        return "mass"

    def save_state(self, q_knockoff):
        pass

    def create_and_fit_knockoff(self, train, valid, q_joint):
        trainloader, valloader = self.create_loaders(train, valid)
        (X,) = extract_data(trainloader)
        # components_to_try=[1, 2, 3, 5, 10, 20]
        components_to_try = self.knockoff.components_to_try
        best_mixture = None
        best_aic = None
        best_n_components = None
        aic = None
        print(X.shape)
        with tqdm(
            components_to_try, desc="Finding best mixture for MASS knockoffs"
        ) as tbar:
            for n_components in tbar:
                mixture, aic = self._fit_gmm(X, n_components)
                if (best_aic is None) or (aic < best_aic):
                    best_aic = aic
                    best_mixture = mixture
                    best_n_components = n_components
                tbar.set_postfix(
                    {"aic": aic, "best_aic": best_aic, "best_k": best_n_components}
                )

        mus = [best_mixture.means_[k] for k in range(best_n_components)]
        Sigmas = [best_mixture.covariances_[k] for k in range(best_n_components)]
        proportions = best_mixture.weights_
        q_knockoff = ModelXKnockoffMixture(mus, Sigmas, proportions)
        return q_knockoff

    @staticmethod
    def _fit_gmm(X, n_components):
        mixture = GaussianMixture(n_components=n_components, n_init=1).fit(X)
        aic = mixture.aic(X)
        return mixture, aic


class KnockoffGANFitter(KnockoffFitter):
    @property
    def id(self):
        return "gan"

    def save_state(self, q_knockoff):
        pass

    def create_and_fit_knockoff(self, train, valid, q_joint):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = get_two_moments(trainloader)
        (X,) = extract_data(trainloader)
        return KnockoffGANSampler(self.cfg, X)


class KnockoffGANSampler:
    def __init__(self, cfg, X) -> None:
        self.cfg = cfg

        ## Sample distribution (Normal, Uniform)
        self.x_name = self.cfg.knockoff.x_name
        ##
        self.lamda = self.cfg.knockoff.lamda
        ##
        self.mu = self.cfg.knockoff.mu
        ##
        self.mb_size = self.cfg.knockoff.mb_size
        ##
        self.niter = self.cfg.knockoff.niter

        X_np = X.cpu().numpy()
        self.sampler = KnockoffGAN(
            X_np, self.x_name, self.lamda, self.mu, self.mb_size, self.niter
        )

    def sample(self, X):
        X_np = X.cpu().numpy()
        X_tilde_np = self.sampler(X_np)
        X_tilde = torch.from_numpy(X_tilde_np)
        return X_tilde

    def to(self, device):
        return self


class DeepKnockoffFitter(KnockoffFitter):
    @property
    def id(self):
        return "deepknockoff"

    def save_state(self, q_knockoff):
        pass

    def create_and_fit_knockoff(self, train, valid, q_joint):
        trainloader, valloader = self.create_loaders(train, valid)
        return DeepKnockoffSampler(self.cfg, trainloader)


from flowselect.alternatives.DeepKnockoffs import GaussianKnockoffs, KnockoffMachine
import flowselect.alternatives.DeepKnockoffs.parameters as deep_knockoff_parameters
from rpy2.robjects.packages import importr


class DeepKnockoffSampler:
    def __init__(self, cfg, trainloader):
        self.cfg = cfg
        self.knockoff = importr("knockoff")

        ((X_mu,), (X_sigma,)) = get_two_moments(trainloader)
        self.X_mu = X_mu
        self.X_sigma = X_sigma

        (X,) = extract_data(trainloader)
        X = X[: self.cfg.knockoff.train_obs]

        X_std = (X - self.X_mu) / self.X_sigma
        X_train = X_std.numpy()
        n, p = X_train.shape
        # Compute the empirical covariance matrix of the training data
        SigmaHat = np.cov(X_train, rowvar=False)

        # Initialize generator of second-order knockoffs
        if self.cfg.knockoff.second_order_method == "asdp":
            Ds = np.array(self.knockoff.create_solve_asdp(SigmaHat))
        elif self.cfg.knockoff.second_order_method == "sdp":
            second_order = GaussianKnockoffs(
                SigmaHat, mu=np.mean(X_train, 0), method="sdp"
            )
            Ds = np.diag(second_order.Ds)
        else:
            raise NotImplementedError("Invalid second_order_method selected")

        corr_g = (np.diag(SigmaHat) - Ds) / np.diag(SigmaHat)
        model = "gaussian"
        training_params = deep_knockoff_parameters.GetTrainingHyperParams(model)
        # Set the parameters for training deep knockoffs
        pars = dict()
        # Number of epochs
        pars["epochs"] = self.cfg.knockoff.epochs
        # Number of iterations over the full data per epoch
        pars["epoch_length"] = self.cfg.knockoff.epoch_length
        # Data type, either "continuous" or "binary"
        pars["family"] = "continuous"
        # Dimensions of the data
        pars["p"] = p
        # Size of the test set
        pars["test_size"] = int(0.1 * n)
        # Batch size
        pars["batch_size"] = int(0.45 * n)
        # Learning rate
        pars["lr"] = 0.01
        # When to decrease learning rate (unused when equal to number of epochs)
        pars["lr_milestones"] = [pars["epochs"]]
        # Width of the network (number of layers is fixed to 6)
        pars["dim_h"] = int(10 * p)
        # Penalty for the MMD distance
        pars["GAMMA"] = training_params["GAMMA"]
        # Penalty encouraging second-order knockoffs
        pars["LAMBDA"] = training_params["LAMBDA"]
        # Decorrelation penalty hyperparameter
        pars["DELTA"] = training_params["DELTA"]
        # Target pairwise correlations between variables and knockoffs
        pars["target_corr"] = corr_g
        # Kernel widths for the MMD measure (uniform weights)
        pars["alphas"] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

        # Initialize the machine
        self.machine = KnockoffMachine(pars)

        # Train the machine
        # print("Fitting the knockoff machine...")
        with torch.cuda.device(self.cfg.knockoff.device):
            self.machine.train(X_train)

        # Generate deep knockoffs

    def sample(self, X):
        X_std = (X - self.X_mu) / self.X_sigma
        X_train = X_std.numpy()
        with torch.cuda.device(self.cfg.knockoff.device):
            Xk_train_m = self.machine.generate(X_train)
        # print("Size of the deep knockoff dataset: %d x %d." %(Xk_train_m.shape))

        X_tilde_std = torch.from_numpy(Xk_train_m)
        X_tilde = X_tilde_std * self.X_sigma + self.X_mu
        return X_tilde

    def to(self, device):
        return self

from abc import abstractmethod
from itertools import cycle
import os
import sys
from pytorch_lightning import LightningModule, Trainer
import torch

import numpy as np

from joblib import Parallel, delayed

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)

import flowselect.ddlk as ddlk
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from einops import rearrange, repeat


class FeatureStatistic:
    def __init__(self, cfg=None, x_in=None, y_in=None):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass


class HRTFeatureStatistic:
    def __init__(self, feature_statistic, n_jobs, one_hot=[]):
        self.feature_stat_class = get_fast_feature(feature_statistic)
        self.n_jobs = n_jobs
        self.one_hot = one_hot

    def __call__(self, xTr, xTr_tilde, yTr, xTe, xTe_tilde, yTe):
        feature_stat = self.feature_stat_class(
            xTr, yTr, n_jobs=self.n_jobs, one_hot=self.one_hot
        )

        out = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(self._call_feature_stat)(xTe, xTe_tilde, yTe, feature_stat)
            for (xTe, xTe_tilde, yTe, feature_stat) in zip(
                cycle([xTe]), xTe_tilde, cycle([yTe]), cycle([feature_stat])
            )
        )
        feature_stats = np.stack(out, axis=1)

        return feature_stats

    @staticmethod
    def _call_feature_stat(xTe, xTe_tilde, yTe, feature_stat):
        xTe = xTe.copy()
        d = xTe.shape[1]
        feature_stats = np.zeros(d)
        for j in range(d):
            features_orig = xTe[:, j].copy()
            xTe[:, j] = xTe_tilde[:, j]
            feature_stats[j] = feature_stat.importance(xTe, yTe)
            xTe[:, j] = features_orig
        return feature_stats


def get_fast_feature(name):
    d = {"fast_ridge": FastRidge, "fast_rf": FastRF, "fast_lasso": FastLasso}
    return d[name]


class FastRidge:
    def __init__(self, x_train, y_train, n_jobs=None, one_hot=[]):
        self.mu = x_train.mean(0)
        self.sigma = x_train.std(0)
        x_train_std = (x_train - self.mu) / self.sigma
        self.model = RidgeCV(cv=None).fit(x_train_std, y_train)

    def importance(self, x_test, y_test):
        x_test_std = (x_test - self.mu) / self.sigma
        return self.model.score(x_test_std, y_test)


class FastLasso:
    def __init__(self, x_train, y_train, n_jobs=2, one_hot=[]):
        self.mu = x_train.mean(0)
        self.sigma = x_train.std(0)
        x_train_std = (x_train - self.mu) / self.sigma
        self.model = LassoCV(
            cv=5,
            random_state=2345,
            max_iter=10_000_000,
            n_jobs=n_jobs,
        ).fit(x_train_std, y_train)

    def importance(self, x_test, y_test):
        x_test_std = (x_test - self.mu) / self.sigma
        return self.model.score(x_test_std, y_test)


class FastRF:
    def __init__(self, x_train, y_train, n_jobs=None, one_hot=[]):
        if one_hot == "All":
            self.one_hot = [j for j in range(x_train.shape[-1])]
            self.not_one_hot = []
        else:
            self.one_hot = one_hot
            self.not_one_hot = [
                j for j in range(x_train.shape[-1]) if j not in self.one_hot
            ]
        if len(self.one_hot) > 0:
            self.lb = LabelBinarizer().fit(x_train[:, self.one_hot].reshape(-1))
            x_train = self._to_one_hot(x_train)
        self.model = RandomForestRegressor().fit(x_train, y_train)

    def importance(self, x_test, y_test):
        if len(self.one_hot) > 0:
            x_test = self._to_one_hot(x_test)
        return self.model.score(x_test, y_test)

    def _to_one_hot(self, x):
        x_in = rearrange(x[:, self.one_hot], "N D -> (N D)")
        x_oh = self.lb.transform(x_in)
        x_oh2 = rearrange(x_oh, "(N D) K -> N (D K)", N=x.shape[0])
        x_out = np.concatenate((x[:, self.not_one_hot], x_oh2), axis=1)
        return x_out


def feature_statistic_factory(name, cfg, x_in=None, y_in=None):
    feat_dict = {
        "ridge": RidgeFeatureStatistic,
        "rf": RandomForestFeatureStatistic,
        "lasso": LassoFeatureStatistic,
    }
    return feat_dict[name](cfg, x_in, y_in)


def get_selection_method(cfg):
    f = eval(f"{cfg.method_name}_fit")
    return f


def khrt_fit(cfg, xTr, yTr, xTr_tilde, xTe, yTe, xTe_tilde):
    hrt_model = HRTFeatureStatistic(feature_statistic=cfg.feature_statistic, n_jobs=1)
    xTe_tilde_in = np.stack((xTe, xTe_tilde), axis=0)
    feature_stats = hrt_model(
        xTr.numpy(), xTr_tilde, yTr.numpy(), xTe.numpy(), xTe_tilde_in, yTe.numpy()
    )
    knockoff_statistics = feature_stats[:, 0] - feature_stats[:, 1]
    return knockoff_statistics


class RandomForestFeatureStatistic(FeatureStatistic):
    def fit(self, x, y):
        rf = RandomForestRegressor()  # can try to tune this with CV
        rf.fit(x, y)  # fit random forest
        feature_statistics = rf.feature_importances_  # extract feature importanc
        return feature_statistics


def rf_fit(cfg, xTr, yTr, xTr_tilde, xTe, yTe, xTe_tilde):
    d = xTr.shape[1]
    xTr_joint = np.hstack((xTr, xTr_tilde))  # form joint matrix
    xTe_joint = np.hstack((xTe, xTe_tilde))

    feature_statistics = RandomForestFeatureStatistic(cfg).fit(xTr_joint, yTr)

    # possibly check for negative values in feature_statistics and convert to 0?
    knockoff_statistics = [
        abs(feature_statistics[i]) - abs(feature_statistics[i + d]) for i in range(d)
    ]
    return knockoff_statistics


class LassoFeatureStatistic(FeatureStatistic):
    # def __init__(self, n_folds, random_state, max_iter, n_jobs):
    def __init__(self, cfg, x_in=None, y_in=None):
        super().__init__()
        self.n_folds = cfg.n_folds
        self.random_state = cfg.random_state
        self.max_iter = cfg.max_iter
        self.n_jobs = cfg.n_jobs

        if x_in is not None:
            if isinstance(x_in, torch.Tensor):
                x_in = x_in.cpu().numpy()
            if isinstance(y_in, torch.Tensor):
                y_in = y_in.cpu().numpy()
            # Fit model
            self.mu = x_in.mean(0)
            self.sigma = x_in.std(0)
            x_std = (x_in - self.mu) / self.sigma
            lasso_model = ElasticNetCV(
                l1_ratio=0.99,
                cv=self.n_folds,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
            ).fit(x_std, y_in)
            # Get the best regularization
            self.alpha = lasso_model.alpha_
        else:
            self.alpha = None

    def fit(self, x, y):

        if self.alpha is not None:
            xTr_joint_std = (x - self.mu) / self.sigma
            lasso_final = ElasticNet(
                l1_ratio=0.99, alpha=self.alpha, max_iter=self.max_iter
            ).fit(xTr_joint_std, y)
        else:
            mu = x.mean(0)
            sigma = x.std(0)
            xTr_joint_std = (x - mu) / sigma
            lasso_final = ElasticNetCV(
                l1_ratio=0.99,
                cv=self.n_folds,
                random_state=self.random_state,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
            ).fit(xTr_joint_std, y)
        feature_statistics = np.abs(lasso_final.coef_)
        return feature_statistics


# Implemented following this link:
# https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html
def lasso_fit(
    cfg,
    xTr,
    yTr,
    xTr_tilde,
    xTe,
    yTe,
    xTe_tilde,
):
    d = xTr.shape[1]
    xTr_joint = np.hstack((xTr, xTr_tilde))  # form joint matrix
    xTe_joint = np.hstack((xTe, xTe_tilde))
    feature_statistics = LassoFeatureStatistic(cfg).fit(xTr_joint, yTr)
    knockoff_statistics = [
        feature_statistics[i] - feature_statistics[i + d] for i in range(d)
    ]
    return knockoff_statistics


# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html
class RidgeFeatureStatistic(FeatureStatistic):
    def fit(self, x, y):
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        ridge_final = RidgeCV(cv=None).fit(x, y)
        feature_statistics = np.abs(ridge_final.coef_)
        return feature_statistics


def ridge_fit(
    cfg,
    xTr,
    yTr,
    xTr_tilde,
    xTe,
    yTe,
    xTe_tilde,
):
    d = xTr.shape[1]
    xTr_joint = np.hstack((xTr, xTr_tilde))  # form joint matrix
    xTe_joint = np.hstack((xTe, xTe_tilde))
    feature_statistics = RidgeFeatureStatistic(cfg).fit(xTr_joint, yTr)
    knockoff_statistics = [
        feature_statistics[i] - feature_statistics[i + d] for i in range(d)
    ]
    return knockoff_statistics


import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    """
    A Multi-layer perceptron of dense layers with non-linear activation layers
    """

    def __init__(
        self, in_features, hs, out_features, act=nn.ReLU, final=None, p_dropout=0.0
    ):
        self.in_features = in_features
        self.out_features = out_features
        layers = []
        for i, h in enumerate(hs):
            layers.append(nn.Linear(in_features if i == 0 else hs[i - 1], h))
            layers.append(act())
            layers.append(nn.Dropout(p_dropout))
        layers.append(nn.Linear(hs[-1], out_features))
        if final is not None:
            layers.append(final())
        super().__init__(*layers)


class FeedForwardRegressor(LightningModule):
    def __init__(
        self, n_inputs, input_dropout_rate=0.2, dropout_rate=0.2, lr=1e-5, l2_reg=1e-10
    ):
        super().__init__()
        self.mlp = MLP(n_inputs, [128, 256, 128], 1, p_dropout=dropout_rate)
        # self.fc1 = nn.Linear(n_inputs, 128)
        # self.fc2 = nn.Linear(128, 256)
        # self.fc3 = nn.Linear(256, 128)
        # self.fc4 = nn.Linear(128, 1)

        # self.dropout1 = nn.Dropout(dropout_rate)
        # self.dropout2 = nn.Dropout(dropout_rate)
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout1 = nn.Dropout(input_dropout_rate)
        self.lr = lr
        self.l2_reg = l2_reg

    def forward(self, x):
        x = self.dropout1(x)
        y = self.mlp(x)
        return y

    def training_step(self, batch, batch_idx, valid=False):
        X, y = batch
        y_hat = self(X)
        loss = F.mse_loss(y.view(-1), y_hat.view(-1))
        if not valid:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, valid=True)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


from tqdm import tqdm

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class FastNNFeatureStatistic:
    def __init__(self, cfg, q_crt):
        self.cfg = cfg
        self.q_crt = q_crt

    def __call__(self, xTr, _xTr_tilde, yTr, xTe, _xTe_tilde, yTe):
        xTr = torch.from_numpy(xTr).float()
        yTr = yTr.float()
        xVd = xTr[-500:]
        xTr = xTr[:-500]
        yVd = yTr[-500:]
        yTr = yTr[:-500]
        traindata = list(zip(xTr, yTr))
        valdata = list(zip(xVd, yVd))
        train_loader = DataLoader(traindata, batch_size=32)
        val_loader = DataLoader(valdata, batch_size=500)
        self.model = FeedForwardRegressor(xTr.shape[1])
        # self.model.train()
        trainer = Trainer(
            max_epochs=100,
            logger=None,
            gpus=self.cfg.gpu,
            callbacks=[EarlyStopping(monitor="val_loss")],
        )
        trainer.fit(self.model, train_loader, val_loader)

        xTe = torch.from_numpy(xTe)
        # xTe_tilde = torch.from_numpy(xTe_tilde)
        # yTe = torch.from_numpy(yTe)
        if self.cfg.gpu is not None:
            self.model = self.model.to(f"cuda:{self.cfg.gpu}")
            xVd = xVd.to(f"cuda:{self.cfg.gpu}")
            yVd = yVd.to(f"cuda:{self.cfg.gpu}")
            xTe = xTe.to(f"cuda:{self.cfg.gpu}")
            # xTe_tilde = xTe_tilde.to(f"cuda:{self.cfg.gpu}")
            yTe = yTe.to(f"cuda:{self.cfg.gpu}")
        with torch.no_grad():
            self.model.eval()
            val_loss = F.mse_loss(yVd.view(-1), self.model(xVd).view(-1))
            print(f"val_loss : {val_loss.item():.3f}")

            # batch_size = 100
            batch_size = self.cfg.fastnn.batch_size
            n_samples = self.cfg.crt.mcmc_steps
            n_batches = int(np.ceil(n_samples / batch_size))
            feature_stats = np.ndarray((n_samples + 1, xTr.shape[1]), dtype=np.float)
            self.q_crt.fit(xTe)
            for i in tqdm(range(n_batches)):
                start = i * batch_size
                stop = min((i + 1) * batch_size, n_samples)
                if i == 0:
                    xTe_tilde = self.q_crt.sample_from_fit(n_samples=(stop - start) - 1)
                    xTe_tilde = torch.cat((xTe.cpu().unsqueeze(0), xTe_tilde), dim=0)
                else:
                    xTe_tilde = self.q_crt.sample_from_fit(n_samples=(stop - start))
                print("Calculating feature statistics...")
                for j in tqdm(range(xTr.shape[1])):
                    # xTe_large = repeat(xTe, "N D -> K N D", K=stop - start)
                    xTe_large = xTe.unsqueeze(0).repeat(stop - start, 1, 1)
                    xTe_large[:, :, j] = xTe_tilde[:, :, j]
                    y_hat = self.model(xTe_large).squeeze(-1)
                    feature_stat = -(yTe.unsqueeze(0) - y_hat).pow(2).mean(dim=1)
                    feature_stats[start:stop, j] = feature_stat.cpu().numpy()

        return feature_stats.transpose()

    @staticmethod
    def _call_feature_stat(xTe, xTe_tilde, yTe, feature_stat):
        xTe = xTe.copy()
        d = xTe.shape[1]
        feature_stats = np.zeros(d)
        for j in range(d):
            features_orig = xTe[:, j].copy()
            xTe[:, j] = xTe_tilde[:, j]
            feature_stats[j] = feature_stat.importance(xTe, yTe)
            xTe[:, j] = features_orig
        return feature_stats

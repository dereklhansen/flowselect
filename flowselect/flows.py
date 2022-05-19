import os
import sys
import math
import pytorch_lightning as pl
import numpy as np
import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from collections import OrderedDict

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)

import flowselect.bisection_grad as bisect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

## Implementation


def log_sigmoid_with_derivative(Y):
    spY = F.softplus(Y)
    s = Y - spY
    ds = s - spY
    return s, ds


## This is probably a stupid function to implement.
## If we have a large tensor of square matrices,
## it might be more efficient to multiply them batch-wise
def reduce2_bmm(Xs):
    if Xs.size(0) % 2 == 1:
        Xs = torch.cat([torch.bmm(Xs[0:1], Xs[1:2]), Xs[2:]], dim=0)
    Ys = torch.bmm(Xs[0::2], Xs[1::2])
    return Ys


def reduce_bmm(Xs, keepdim=False):
    while Xs.size(0) > 1:
        Xs = reduce2_bmm(Xs)
    if not keepdim:
        Xs = Xs[0]
    return Xs


class GenericFlow(pl.LightningModule):
    def __init__(self, d, dist=None, lr=1e-3):
        super().__init__()
        if dist is None:
            dist = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.d = d
        self.dist = dist
        self.lr = lr

    def forward(self, inputs, cond_inputs=None, mode="direct"):
        if mode == "direct":
            U, log_dets = self.direct(inputs, cond_inputs)
            return U, log_dets
        else:
            X, log_dets = self.inverse(inputs, cond_inputs)
            return X, log_dets

    def direct(self, X, cond_inputs=None):
        raise NotImplementedError("direct not defined for GenericFlow")

    # From U to X
    def inverse(self, U, cond_inputs=None):
        raise NotImplementedError("inverse not defined for GenericFlow")

    def log_prob(self, X, cond_inputs=None):
        U, log_dets = self.direct(X, cond_inputs=cond_inputs)
        assert U.isinf().sum() < 1
        assert U.isnan().sum() < 1
        assert log_dets.isnan().sum() < 1
        assert log_dets.isinf().sum() < 1
        prob_U = self.dist.log_prob(U).sum(dim=1)
        prob_X = prob_U + log_dets
        assert prob_X.isnan().sum() < 1
        assert prob_X.isinf().sum() < 1
        return prob_X

    def sample(self, sample_shape=torch.Size([1, 1]), cond_inputs=None):
        if self.dist.loc.device != self.device:
            self.dist.loc = self.dist.loc.to(self.device)
            self.dist.scale = self.dist.scale.to(self.device)
        noise = self.dist.rsample([sample_shape[0], self.d])
        X = self.inverse(noise, cond_inputs=cond_inputs)[0]
        return X

    def training_step(self, batch, batch_idx, val_step=False):
        (X,) = batch
        elbo = self.log_prob(X).mean()
        loss = -elbo
        if val_step:
            prefix = "val_"
        else:
            prefix = ""
        self.log(f"{prefix}loss", loss)
        self.log(f"{prefix}elbo", elbo)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            return self.training_step(batch, batch_idx, True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        out = dict(val_loss=avg_loss)
        out["log"] = tensorboard_logs
        out["progress_bar"] = tensorboard_logs
        return out

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class IdentityFlow(GenericFlow):
    def direct(self, X, cond_inputs=None):
        log_dets = torch.zeros(X.size(0), device=X.device)
        return X, log_dets

    def inverse(self, U, cond_inputs=None):
        log_dets = torch.zeros(U.size(0), device=U.device)
        return U, log_dets


class SequentialFlow(GenericFlow):
    def __init__(self, d, flows: nn.Sequential, dist=None, lr=1e-3):
        super().__init__(d=d, dist=dist, lr=lr)
        self.flows = flows

    def direct(self, X, cond_inputs=None):
        U = X
        log_dets = torch.zeros(X.size(0), device=X.device)
        for module in self.flows._modules.values():
            X = U
            U, log_dets_step = module.direct(X, cond_inputs=cond_inputs)
            log_dets += log_dets_step
        return U, log_dets

    def inverse(self, U, cond_inputs=None):
        X = U
        log_dets = torch.zeros(U.size(0), device=U.device)
        for module in reversed(self.flows._modules.values()):
            U = X
            X, log_dets_step = module.inverse(U, cond_inputs=cond_inputs)
            log_dets += log_dets_step
        return X, log_dets


class StandardizationFlow(GenericFlow):
    def __init__(self, mu, sigma):
        self.d = mu.size(0)
        super().__init__(d=self.d)
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def direct(self, X, cond_inputs=None):
        U = (X - self.mu) / self.sigma
        log_det = -self.sigma.log().sum()
        return U, log_det

    def inverse(self, U, cond_inputs=None):
        X = U * self.sigma + self.mu
        log_det = self.sigma.log().sum()
        return X, log_det


class GFKernelF(GenericFlow):
    def __init__(self, d, k, dist=None, min_h=0.1, pclamp=1e-7):
        super().__init__(d=d, dist=dist)
        self.min_h = min_h
        self.mus = nn.Parameter(torch.randn([1, d, k]))
        self.log_hs = nn.Parameter(torch.zeros([1, d, k]))
        self.BI = bisect.bi(
            lambda X: self.direct(X)[0], lower=-1e6, upper=1e6, max_iter=1000
        )
        self.k = self.mus.size(2)
        self.pclamp_l = pclamp
        self.pclamp_u = 1 - pclamp

    @property
    def hs(self):
        return self.log_hs.exp()

    def direct(self, X, cond_inputs=None):
        Y = (X.unsqueeze(-1) - self.mus) / self.hs
        s = torch.sigmoid(Y).clamp(self.pclamp_l, self.pclamp_u)
        U = s.mean(dim=2)
        log_ds = s.log() - F.softplus(Y)
        log_dU = log_ds - self.log_hs
        log_det = (
            torch.logsumexp(log_dU, dim=(2)).sum(dim=1) - math.log(self.k) * self.d
        )
        assert U.isnan().sum() < 1
        assert U.isinf().sum() < 1
        assert log_det.isnan().sum() < 1
        assert log_det.isinf().sum() < 1
        return U, log_det

    # From U to X
    def inverse(self, U, cond_inputs=None):
        U = U.clamp(self.pclamp_l, self.pclamp_u)
        X = self.BI.apply(U)
        if U.requires_grad:
            self.BI.calc_back_grad()
        _, direct_log_det = self.direct(X)
        log_det = -direct_log_det
        return X, log_det


class GFKernelInvG(GenericFlow):
    def __init__(self, d, pclamp=1e-7):
        super().__init__(d=d)
        self.dist = torch.distributions.Normal(0.0, 1.0)
        self.pclamp_l = pclamp
        self.pclamp_u = 1 - pclamp

    def direct(self, X, cond_inputs=None):
        X = X.clamp(self.pclamp_l, self.pclamp_u)
        U = self.dist.icdf(X)
        log_dets = -self.dist.log_prob(U)
        log_det = log_dets.sum(dim=1)
        assert U.isinf().sum() < 1
        assert U.isnan().sum() < 1
        assert log_det.isnan().sum() < 1
        assert log_det.isinf().sum() < 1
        return U, log_det

    def inverse(self, U, cond_inputs=None):
        X = self.dist.cdf(U)
        log_dets = self.dist.log_prob(U)
        log_det = log_dets.sum(dim=1)
        return X, log_det


def householder(v):
    d = v.size(0)
    return torch.eye(d) - ((2 * v * v.transpose(1, 0)) / v.pow(2).sum())


class GFOrtho(GenericFlow):
    def __init__(self, n_iter, d):
        super().__init__(d=d)
        self.V = nn.Parameter(torch.ones(n_iter, d, 1))
        self.register_buffer("eye", torch.eye(d, d).unsqueeze(0))
        self.n_iter = self.V.size(0)

    def rotation_matrix(self):
        VVT = torch.bmm(self.V, self.V.permute(0, 2, 1))
        vnorm2 = self.V.pow(2).sum(dim=(1)).unsqueeze(2)
        Hs = self.eye - (2 * VVT / vnorm2)
        H = reduce_bmm(Hs, keepdim=False)
        return H

    def direct(self, X, cond_inputs=None):
        H = self.rotation_matrix()
        U = torch.matmul(X, H)
        return U, torch.zeros(1, device=U.device)

    def inverse(self, U, cond_inputs=None):
        H = self.rotation_matrix()
        X = torch.matmul(U, H.transpose(1, 0))
        return X, torch.zeros(1, device=X.device)


class CouplingFlow(GenericFlow):
    """"""

    def __init__(
        self, d, n_hidden, mask, num_cond_inputs=None, s_act="tanh", t_act="relu"
    ):
        super().__init__(d=d)

        self.n_hidden = n_hidden
        self.register_buffer("mask", mask.float())

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = d + num_cond_inputs
        else:
            total_inputs = d

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, n_hidden),
            s_act_func(),
            nn.Linear(n_hidden, n_hidden),
            s_act_func(),
            nn.Linear(n_hidden, d),
        )
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, n_hidden),
            t_act_func(),
            nn.Linear(n_hidden, n_hidden),
            t_act_func(),
            nn.Linear(n_hidden, d),
        )

    def direct(self, X, cond_inputs=None):
        masked_X = X * self.mask
        if cond_inputs is not None:
            masked_X = torch.cat([masked_X, cond_inputs], -1)
        log_s = self.scale_net(masked_X) * (1 - self.mask)
        t = self.translate_net(masked_X) * (1 - self.mask)
        s = torch.exp(log_s)
        return X * s + t, log_s.sum(-1)

    def inverse(self, U, cond_inputs=None):
        masked_U = U * self.mask
        if cond_inputs is not None:
            masked_U = torch.cat([masked_U, cond_inputs], -1)
        log_s = self.scale_net(masked_U) * (1 - self.mask)
        t = self.translate_net(masked_U) * (1 - self.mask)
        s_reciprocal = torch.exp(-log_s)
        return (U - t) * s_reciprocal, -log_s.sum(-1)


class BatchNormFlow(GenericFlow):
    """"""

    def __init__(self, d, momentum=0.0, eps=1e-5):
        super().__init__(d=d)

        self.log_gamma = nn.Parameter(torch.zeros(d))
        self.beta = nn.Parameter(torch.zeros(d))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))

    def direct(self, X, cond_inputs=None):
        if self.training:
            self.batch_mean = X.mean(0)
            self.batch_var = (X - self.batch_mean).pow(2).mean(0) + self.eps
            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.add_(self.batch_var.data * (1 - self.momentum))
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (X - mean) / var.sqrt()
        y = torch.exp(self.log_gamma) * x_hat + self.beta
        return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1, keepdim=True)

    def inverse(self, U, cond_inputs=None):
        if self.training and hasattr(self, "batch_mean"):
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (U - self.beta) / torch.exp(self.log_gamma)
        y = x_hat * var.sqrt() + mean
        return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(-1, keepdim=True)


class BatchNormFlowReversed(BatchNormFlow):
    def __init__(self, d, momentum=0.0, eps=1e-5):
        super().__init__(d=d, momentum=momentum, eps=1e-5)

    def direct(self, X, cond_inputs=None):
        return super().inverse(X, cond_inputs=cond_inputs)

    def inverse(self, U, cond_inputs=None):
        return super().direct(U, cond_inputs=cond_inputs)


## Interface
class GaussianizationFlow(SequentialFlow):
    """
    Implementation of a Gaussianization flow.
    Inherits methods .direct(), .inverse(), and .log_prob()
    """

    def __init__(self, d, k, n_iter, n_layers, mu=None, sigma=None, lr=1e-3):
        """
        d : Dimension of input and output
        k : Number of mixture components
        n_iter: Number of householder rotations
        n_layers: Number of layers
        mu: Mean for standardization flow
        sigma: Scale for standardization flow
        lr: Learning rate for optimization
        """
        flows = []
        if (mu is not None) and (sigma is not None):
            flows.append(StandardizationFlow(mu, sigma))
        for _ in range(n_layers):
            flows.append(GFKernelF(d, k))
            flows.append(GFKernelInvG(d))
            flows.append(GFOrtho(n_iter, d))
        super().__init__(d, nn.Sequential(*flows), lr=lr)
        self.d = d
        self.k = k
        self.n_iter = n_iter
        self.n_layers = n_layers


class RealNVP(SequentialFlow):
    """"""

    def __init__(
        self,
        d,
        n_layers,
        n_hidden,
        mu=None,
        sigma=None,
        num_cond_inputs=None,
        reverse_batch_norm=False,
        lr=1e-3,
    ):
        mask = torch.arange(0, d) % 2
        flows = []
        if (mu is not None) and (sigma is not None):
            flows.append(StandardizationFlow(mu, sigma))
        if reverse_batch_norm:
            BNF = BatchNormFlowReversed
        else:
            BNF = BatchNormFlow
        for _ in range(n_layers):
            flows.append(
                CouplingFlow(
                    d, n_hidden, mask, num_cond_inputs, s_act="tanh", t_act="relu"
                )
            )
            flows.append(BNF(d))
            mask = 1 - mask
        super().__init__(d, nn.Sequential(*flows), lr=lr)
        self.d = d
        self.n_layers = n_layers
        self.n_hidden = n_hidden


from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform, Transform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm, BatchNorm
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)


class StandardizationTransform(Transform):
    def __init__(self, mu, sigma):
        self.d = mu.size(0)
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

    def forward(self, X, context=None):
        U = (X - self.mu) / self.sigma
        log_det = -self.sigma.log().sum()
        return U, log_det

    def inverse(self, U, context=None):
        X = U * self.sigma + self.mu
        log_det = self.sigma.log().sum()
        return X, log_det


class NeuralSplineFlow(GenericFlow):
    def __init__(
        self,
        d,
        n_layers,
        mu=None,
        sigma=None,
        lr=1e-3,
        hidden_features=64,
        tail_bound=3.0,
        # k=6
    ):
        super().__init__(d=d, lr=lr)
        base_dist = StandardNormal(shape=[d])
        transforms = []
        transforms.append(StandardizationTransform(mu, sigma))
        # transforms.append(GFKernelFTransform(d, k))
        for _ in range(n_layers):
            transforms.append(LULinear(d))
            transforms.append(ActNorm(d))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    d,
                    hidden_features=hidden_features,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )
        transform = CompositeTransform(transforms)
        self.flow = Flow(transform, base_dist)

    def log_prob(self, X, cond_inputs=None):
        return self.flow.log_prob(inputs=X, context=cond_inputs)

    def sample(self, sample_shape=torch.Size([1, 1]), cond_inputs=None):
        return self.flow.sample(sample_shape[0], context=cond_inputs)


from nflows.flows import MaskedAutoregressiveFlow


class MAF(GenericFlow):
    def __init__(
        self,
        d,
        n_layers,
        mu=None,
        sigma=None,
        lr=1e-3,
        hidden_features=64,
        tail_bound=3.0,
    ):
        super().__init__(d=d, lr=lr)
        self.flow = MaskedAutoregressiveFlow(
            d,
            hidden_features,
            n_layers,
            num_blocks_per_layer=2,
            batch_norm_between_layers=True,
        )

    def log_prob(self, X, cond_inputs=None):
        return self.flow.log_prob(inputs=X, context=cond_inputs)

    def sample(self, sample_shape=torch.Size([1, 1]), cond_inputs=None):
        return self.flow.sample(sample_shape[0], context=cond_inputs)


class GaussMAF(GenericFlow):
    """An autoregressive flow that uses affine transforms with masking.

    Reference:
    > G. Papamakarios et al., Masked Autoregressive Flow for Density Estimation,
    > Advances in Neural Information Processing Systems, 2017.
    """

    def __init__(
        self,
        d,
        hidden_features,
        num_layers,
        mu,
        sigma,
        k=6,
        X=None,
        num_blocks_per_layer=2,
        use_residual_blocks=True,
        use_random_masks=False,
        use_random_permutations=False,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        batch_norm_between_layers=True,
        lr=1e-4,
    ):
        super().__init__(d, lr=lr)
        features = d
        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        layers.append(StandardizationTransform(mu, sigma))
        layers.append(GFKernelFTransform(d, k, X=X))
        layers.append(GFKernelInvGTransform(d))
        self.gaussflow = Flow(
            transform=CompositeTransform(layers[0:3]),
            distribution=StandardNormal([features]),
        )
        self.use_gauss_flow = False

        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        self.flow = Flow(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features]),
        )

    def log_prob(self, X, cond_inputs=None):
        if self.use_gauss_flow:
            return self.gaussflow.log_prob(inputs=X, context=cond_inputs)
        else:
            return self.flow.log_prob(inputs=X, context=cond_inputs)

    def sample(self, sample_shape=torch.Size([1, 1]), cond_inputs=None):
        return self.flow.sample(sample_shape[0], context=cond_inputs)

    def set_use_gauss_flow(self, use_gauss_flow):
        self.use_gauss_flow = use_gauss_flow


from sklearn.cluster import KMeans, kmeans_plusplus


class GFKernelFTransform(Transform):
    def __init__(self, d, k, dist=None, min_h=0.1, pclamp=1e-7, X=None):
        super().__init__()
        self.d = d
        self.min_h = min_h
        if X is None:
            self.mus = nn.Parameter(torch.randn([1, d, k]))
        else:
            self.mus = nn.Parameter(self._kmeans_init(X, k))
        self.log_hs = nn.Parameter(torch.zeros([1, d, k]))
        self.BI = bisect.bi(
            lambda X: self.forward(X)[0], lower=-1e6, upper=1e6, max_iter=1000
        )
        self.k = self.mus.size(2)
        self.pclamp_l = pclamp
        self.pclamp_u = 1 - pclamp

    @property
    def hs(self):
        return self.log_hs.exp()

    def forward(self, X, context=None):
        Y = (X.unsqueeze(-1) - self.mus) / self.hs
        s = torch.sigmoid(Y).clamp(self.pclamp_l, self.pclamp_u)
        U = s.mean(dim=2)
        log_ds = s.log() - F.softplus(Y)
        log_dU = log_ds - self.log_hs
        log_det = (
            torch.logsumexp(log_dU, dim=(2)).sum(dim=1) - math.log(self.k) * self.d
        )
        assert U.isnan().sum() < 1
        assert U.isinf().sum() < 1
        assert log_det.isnan().sum() < 1
        assert log_det.isinf().sum() < 1
        return U, log_det

    # From U to X
    def inverse(self, U, context=None):
        U = U.clamp(self.pclamp_l, self.pclamp_u)
        X = self.BI.apply(U)
        if U.requires_grad:
            self.BI.calc_back_grad()
        _, direct_log_det = self.forward(X)
        log_det = -direct_log_det
        return X, log_det

    @staticmethod
    def _kmeans_init(X, k):
        centers, _ = kmeans_plusplus(X, k)
        means = torch.from_numpy(centers)
        return means.transpose(1, 0).unsqueeze(0)


class GFKernelInvGTransform(Transform):
    def __init__(self, d, pclamp=1e-7):
        super().__init__()
        self.dist = torch.distributions.Normal(0.0, 1.0)
        self.pclamp_l = pclamp
        self.pclamp_u = 1 - pclamp

    def forward(self, X, context=None):
        X = (X + self.pclamp_l) / (1.0 + self.pclamp_l) * self.pclamp_u
        U = self.dist.icdf(X)
        log_dets = (
            -np.log(1.0 + self.pclamp_l)
            + np.log(self.pclamp_u)
            + -self.dist.log_prob(U)
        )
        log_det = log_dets.sum(dim=1)
        assert U.isinf().sum() < 1
        assert U.isnan().sum() < 1
        assert log_det.isnan().sum() < 1
        assert log_det.isinf().sum() < 1
        return U, log_det

    def inverse(self, U, context=None):
        X = self.dist.cdf(U)
        X = X / self.pclamp_u * (1.0 + self.pclamp_l) - self.pclamp_l
        log_dets = (
            np.log(1.0 + self.pclamp_l) - np.log(self.pclamp_u) + self.dist.log_prob(U)
        )
        log_det = log_dets.sum(dim=1)
        return X, log_det

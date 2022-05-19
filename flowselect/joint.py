import os
import sys
import subprocess as sp
from time import time_ns
import torch
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import argparse
import pandas as pd

import torch.nn as nn

from torch.optim import Adam
from hydra.utils import to_absolute_path
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

path = os.path.abspath("../..")
if path not in sys.path:
    sys.path.insert(0, path)

import flowselect.ddlk as ddlk
from flowselect.flows import (
    GaussMAF,
    MAF,
    RealNVP,
    GaussianizationFlow,
    SequentialFlow,
)
from flowselect.util import save_and_symlink, write_h5
from flowselect.discrete_flows.made import MADE
from flowselect.discrete_flows.disc_models import (
    DiscreteAutoFlowModel,
    DiscreteAutoregressiveFlow,
    DiscreteBipartiteFlow,
)


def fit_joint(cfg, train, valid):
    fitter_dict = {
        "ddlk": DDLKJointFitter,
        "realnvp": RealNVPJointFitter,
        "gaussflow": GaussFlowJointFitter,
        "truth": TrueDGPJointFitter,
        "maf": MAFFitter,
        "gaussmaf": GaussMAFFitter,
        "discrete_flow": DiscreteFlowFitter,
    }

    fitter = fitter_dict[cfg.joint.model_name](cfg)
    q_joint = fitter(train, valid)
    return q_joint


class JointFitter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = cfg.data
        self.joint = cfg.joint

    def __call__(self, train, valid):
        t = time_ns()
        q_joint = self.create_and_fit_joint(train, valid)
        joint_time = time_ns() - t
        self.joint["joint_id"] = self.id
        self.save_joint_state_to_disk(q_joint)
        self.save_joint_info_to_disk(q_joint, train, valid)
        self.save_joint_timings_to_disk(joint_time)
        return q_joint

    @property
    def id(self):
        return self.__class__.__name__

    @property
    def save_path(self):
        path = (
            Path(to_absolute_path(self.cfg.output))
            .joinpath(self.data.data_id)
            .joinpath(self.id)
            .joinpath("joint_state")
        )
        return path

    def create_and_fit_joint(self, trainloader, valloader):
        raise NotImplementedError(
            "JointFitter is not meant to be used directly; implement a subclass"
        )

    def create_loaders(self, train, valid):
        train_cfg = argparse.Namespace(
            batch_size_train=self.joint.batch_size_train,
            batch_size_valid=self.joint.batch_size_valid,
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
        return trainloader, valloader

    def save_joint_state_to_disk(self, q_joint):
        save_and_symlink(q_joint.state_dict(), self.save_path, "joint_state")

    def save_joint_info_to_disk(self, q_joint, train, valid):
        trainloader, valloader = self.create_loaders(train, valid)
        lcl_sample_name = "joint_sample.h5"

        def export_sample():
            with torch.no_grad():
                q_joint.train(False)
                xTr = ddlk.utils.extract_data(trainloader)[0]
                xTr = xTr[: self.joint.plot_nobs]
                xTr_tilde = q_joint.sample(
                    sample_shape=torch.Size([self.joint.plot_nobs, 1])
                )
                sample_data = {"xTr": xTr.numpy(), "xTr_tilde": xTr_tilde.numpy()}
                write_h5(lcl_sample_name, sample_data)

        save_and_symlink(
            None,
            self.save_path.parent.joinpath(lcl_sample_name),
            lcl_sample_name,
            export_sample,
        )

        # Save mapped data
        lcl_flow_space_name = "data_in_flow_space.h5"
        if self.joint.model_name in ["gaussflow", "gaussnvp", "realnvp"]:

            def export_mapped_sample():
                with torch.no_grad():
                    q_joint.train(False)
                    xTr = ddlk.utils.extract_data(trainloader)[0]
                    uTr, _ = q_joint.direct(xTr)
                    write_h5(
                        lcl_flow_space_name,
                        {"xTr": uTr.numpy(), "xTr_tilde": uTr.numpy()},
                    )

            save_and_symlink(
                None,
                self.save_path.parent.joinpath(lcl_flow_space_name),
                lcl_flow_space_name,
                export_mapped_sample,
            )

        # Save heldout log-likelihood
        lcl_valid_file = "valid_loglik.csv"

        def export_heldout_loglik():
            with torch.no_grad():
                q_joint.train(False)
                xValid = ddlk.utils.extract_data(valloader)[0]
                valid_logliks = q_joint.log_prob(xValid)
                valid_ll_mean = valid_logliks.mean(0, keepdim=True)
                valid_ll_std = valid_logliks.std(0, keepdim=True)
                valid_df = pd.DataFrame(
                    {
                        "n": valid_logliks.size(0),
                        "loglik_mean": valid_ll_mean,
                        "loglik_std": valid_ll_std,
                    }
                )
                valid_df.to_csv(lcl_valid_file)

        save_and_symlink(
            None,
            self.save_path.parent.joinpath(lcl_valid_file),
            lcl_valid_file,
            export_heldout_loglik,
        )

    def save_joint_density_plot(
        self, out, q_joint, xTr, xTr_label="data", xTr_tilde_label="ddlk"
    ):
        with torch.no_grad():
            q_joint.train(False)
            xTr = xTr[: self.joint.plot_nobs]
            xTr_tilde = q_joint.sample(
                sample_shape=torch.Size([self.joint.plot_nobs, 1])
            )
        # select 2 coordinates at random
        j1, j2 = np.random.permutation(xTr.shape[1])[:2]

        fig, axarr = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        sns.kdeplot(
            xTr[:, j1],
            xTr[:, j2],
            label=xTr_label,
            shade=False,
            levels=15,
            color="#577590",
            ax=axarr[0],
        )
        ylim = axarr[0].get_ylim()
        xlim = axarr[0].get_xlim()
        sns.kdeplot(
            xTr_tilde[:, j1],
            xTr_tilde[:, j2],
            label=xTr_tilde_label,
            shade=False,
            levels=15,
            color="#f94144",
            ax=axarr[1],
        )
        axarr[1].set_xlim(*xlim)
        axarr[1].set_ylim(*ylim)
        axarr[0].legend()
        axarr[1].legend()
        plt.savefig(out)

    def save_joint_timings_to_disk(self, time):
        if not self.save_path.parent.is_dir():
            os.makedirs(self.save_path.parent)
        path = self.save_path.parent.joinpath("time.txt")
        if not path.exists():
            with path.open("w") as fp:
                fp.writelines([f"{time}"])


class DDLKJointFitter(JointFitter):
    @property
    def id(self):
        return "ddlk"

    def create_and_fit_joint(self, train, valid):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = ddlk.utils.get_two_moments(trainloader)
        hparams = argparse.Namespace(
            X_mu=X_mu,
            X_sigma=X_sigma,
            init_type=self.joint.init_type,
            hidden_layers=self.joint.hidden_layers,
        )
        q_joint = ddlk.mdn.MDNJoint(hparams)
        tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_joint")
        logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
        if self.joint.early_stopping:
            ddlk_callbacks = [EarlyStopping(monitor="val_loss")]
        else:
            ddlk_callbacks = []
        trainer = pl.Trainer(
            max_epochs=self.joint.max_epochs,
            num_sanity_val_steps=self.joint.num_sanity_val_steps,
            weights_summary=self.joint.weights_summary,
            deterministic=self.joint.deterministic,
            gpus=self.joint.gpu,
            logger=logger,
            callbacks=ddlk_callbacks,
        )
        if not self.save_path.exists():
            trainer.fit(
                q_joint, train_dataloader=trainloader, val_dataloaders=valloader
            )
        else:
            q_joint.load_state_dict(torch.load(self.save_path))
        return q_joint


class GaussFlowJointFitter(JointFitter):
    @property
    def id(self):
        return (
            f"gaussflow_k{self.joint.k}_ni{self.joint.n_iter}_nl{self.joint.n_layers}"
        )

    def create_and_fit_joint(self, train, valid):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = ddlk.utils.get_two_moments(trainloader)
        q_joint = GaussianizationFlow(
            X_mu.size(0),
            self.joint.k,
            self.joint.n_iter,
            self.joint.n_layers,
            X_mu,
            X_sigma,
            lr=self.joint.lr,
        )
        if not self.save_path.exists():
            if self.joint.train_marg_separate:
                marginal_flow = SequentialFlow(
                    q_joint.d,
                    nn.Sequential(q_joint.flows[0], q_joint.flows[1], q_joint.flows[2]),
                    lr=self.joint.marg_lr,
                )
                tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_marginal")
                logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
                trainer = pl.Trainer(
                    max_epochs=self.joint.marg_max_epochs,
                    num_sanity_val_steps=self.joint.num_sanity_val_steps,
                    weights_summary=self.joint.weights_summary,
                    deterministic=self.joint.deterministic,
                    gpus=self.joint.gpu,
                    logger=logger,
                )
                trainer.fit(
                    marginal_flow,
                    train_dataloader=trainloader,
                    val_dataloaders=valloader,
                )
                marginal_flow.freeze()

            tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_joint")
            logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
            trainer = pl.Trainer(
                max_epochs=self.joint.max_epochs,
                num_sanity_val_steps=self.joint.num_sanity_val_steps,
                weights_summary=self.joint.weights_summary,
                deterministic=self.joint.deterministic,
                gpus=self.joint.gpu,
                logger=logger,
            )
            # if not joint_location.exists():
            trainer.fit(
                q_joint, train_dataloader=trainloader, val_dataloaders=valloader
            )
        else:
            q_joint.load_state_dict(torch.load(self.save_path))

        return q_joint


class RealNVPJointFitter(JointFitter):
    @property
    def id(self):
        return f"realnvp_nl{self.joint.n_layers}"

    def create_and_fit_joint(self, train, valid):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = ddlk.utils.get_two_moments(trainloader)
        q_joint = RealNVP(
            X_mu.size(0),
            self.joint.n_layers,
            self.joint.n_hidden,
            mu=X_mu,
            sigma=X_sigma,
            lr=self.joint.lr,
        )
        tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_self.")
        logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
        trainer = pl.Trainer(
            max_epochs=self.joint.max_epochs,
            num_sanity_val_steps=self.joint.num_sanity_val_steps,
            weights_summary=self.joint.weights_summary,
            deterministic=self.joint.deterministic,
            gpus=self.joint.gpu,
            logger=logger,
        )
        if not self.save_path.exists():
            trainer.fit(
                q_joint, train_dataloader=trainloader, val_dataloaders=valloader
            )
        else:
            q_joint.load_state_dict(torch.load(self.save_path))
        return q_joint


class TrueDGPJointFitter(JointFitter):
    @property
    def id(self):
        return f"true_dgp"

    def create_and_fit_joint(self, train, valid):
        q_joint = train.dgp
        return q_joint

    def save_joint_state_to_disk(self, q_joint):
        pass


class MAFFitter(JointFitter):
    @property
    def id(self):
        return f"maf_{self.joint.n_layers}"

    def create_and_fit_joint(self, train, valid):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = ddlk.utils.get_two_moments(trainloader)
        q_joint = MAF(
            X_mu.size(0),
            self.joint.n_layers,
            mu=X_mu,
            sigma=X_sigma,
            lr=self.joint.lr,
            hidden_features=self.joint.hidden_features,
        )
        tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_self.")
        logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
        trainer = pl.Trainer(
            max_epochs=self.joint.max_epochs,
            num_sanity_val_steps=self.joint.num_sanity_val_steps,
            weights_summary=self.joint.weights_summary,
            deterministic=self.joint.deterministic,
            gpus=self.joint.gpu,
            logger=logger,
            gradient_clip_val=self.joint.gradient_clip_val,
        )
        if not self.save_path.exists():
            trainer.fit(
                q_joint, train_dataloader=trainloader, val_dataloaders=valloader
            )
        else:
            q_joint.load_state_dict(torch.load(self.save_path))
        return q_joint


class GaussMAFFitter(JointFitter):
    @property
    def id(self):
        return f"gaussmaf_{self.joint.n_layers}"

    def create_and_fit_joint(self, train, valid):
        trainloader, valloader = self.create_loaders(train, valid)
        ((X_mu,), (X_sigma,)) = ddlk.utils.get_two_moments(trainloader)
        q_joint = GaussMAF(
            X_mu.size(0),
            self.joint.hidden_features,
            self.joint.n_layers,
            mu=X_mu,
            sigma=X_sigma,
            lr=self.joint.lr,
            X=None,
        )
        if self.joint.train_marg_separate and (not self.save_path.exists()):
            q_joint.set_use_gauss_flow(True)
            tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_marginal")
            logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
            trainer = pl.Trainer(
                max_epochs=self.joint.marg_max_epochs,
                num_sanity_val_steps=self.joint.num_sanity_val_steps,
                weights_summary=self.joint.weights_summary,
                deterministic=self.joint.deterministic,
                gpus=self.joint.gpu,
                logger=logger,
            )
            trainer.fit(
                q_joint,
                train_dataloader=trainloader,
                val_dataloaders=valloader,
            )
            q_joint.set_use_gauss_flow(False)
            # if self.joint.marg_freeze:
            #     q_joint.freeze()
        tb_logdir = self.save_path.parent.parent.joinpath("tb_logs_joint")
        logger = pl.loggers.TensorBoardLogger(tb_logdir, name=self.id)
        trainer = pl.Trainer(
            max_epochs=self.joint.max_epochs,
            num_sanity_val_steps=self.joint.num_sanity_val_steps,
            weights_summary=self.joint.weights_summary,
            deterministic=self.joint.deterministic,
            gpus=self.joint.gpu,
            logger=logger,
            gradient_clip_val=self.joint.gradient_clip_val,
        )
        if not self.save_path.exists():
            trainer.fit(
                q_joint, train_dataloader=trainloader, val_dataloaders=valloader
            )
        else:
            q_joint.load_state_dict(torch.load(self.save_path))
        return q_joint


class DiscreteFlowModel(nn.Module):
    def __init__(
        self, flow, disc_layer_type, base_log_probs, batch_size, sequence_length
    ):
        super().__init__()
        self.flow = flow
        self.disc_layer_type = disc_layer_type
        self.base_log_probs = nn.Parameter(base_log_probs)
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def forward(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        zs = self.flow(x)

        if self.disc_layer_type == "bipartite":
            zs = zs.view(
                self.batch_size, self.sequence_length, -1
            )  # adding back in sequence dimension
        base_log_probs_sm = torch.nn.functional.log_softmax(self.base_log_probs, dim=-1)
        # print(zs.shape, base_log_probs_sm.shape)
        logprob = (
            (zs * base_log_probs_sm).sum(-1).sum(-1)
        )  # zs are onehot so zero out all other logprobs.
        return logprob


import torch.nn.functional as F


class DiscreteFlowFitter(JointFitter):
    @property
    def id(self):
        return f"discreteflow"

    def create_and_fit_joint(self, train, valid):
        # import pandas as pd
        # import torch

        # from disc_models import *
        # from made import *

        genotypes = torch.from_numpy(train.X)
        genotypes = genotypes.long()  # convert to long for 1-hot encoding
        genotypes = F.one_hot(genotypes)  # 1-hot encoding
        genotypes = genotypes.char()  # make uint8 to use less memory
        soybeans_data = genotypes.float()
        n, sequence_length, vocab_size = soybeans_data.shape
        vector_length = sequence_length * vocab_size

        num_flows = 1  # number of flow steps. This is different to the number of layers used inside each flow
        temperature = 0.1  # used for the straight-through gradient estimator. Value taken from the paper
        disc_layer_type = "autoreg"  #'autoreg' #'bipartite'

        # This setting was previously used for the MLP and MADE networks.
        nh = sequence_length + 1  # number of hidden units per layer
        batch_size = 256

        flows = []
        for i in range(num_flows):
            if disc_layer_type == "autoreg":
                # layer = EmbeddingLayer([batch_size, sequence_length, vocab_size], output_size=vocab_size)
                # MADE network is much more powerful.
                layer = MADE(
                    [batch_size, sequence_length, vocab_size], vocab_size, [nh, nh]
                )

                disc_layer = DiscreteAutoregressiveFlow(layer, temperature, vocab_size)

            # elif disc_layer_type == 'bipartite':
            if disc_layer_type == "bipartite":
                # MLP will learn the factorized distribution and not perform well.
                # layer = MLP(vector_length//2, vector_length//2, nh)

                layer = torch.nn.Embedding(vector_length // 2, vector_length // 2)

                disc_layer = DiscreteBipartiteFlow(
                    layer, i % 2, temperature, vocab_size, vector_length, embedding=True
                )
                # i%2 flips the parity of the masking. It splits the vector in half and alternates
                # each flow between changing the first half or the second.

            flows.append(disc_layer)

        model = DiscreteAutoFlowModel(flows)

        print(model)

        base_log_probs = torch.tensor(
            torch.randn(sequence_length, vocab_size), requires_grad=True
        )
        base = torch.distributions.OneHotCategorical(logits=base_log_probs)
        q_joint = DiscreteFlowModel(
            model, disc_layer_type, base_log_probs, batch_size, sequence_length
        )
        if not self.save_path.exists():
            if self.joint.gpu is not None:
                q_joint = q_joint.to(f"cuda:{self.joint.gpu}")
                soybeans_data = soybeans_data.to(f"cuda:{self.joint.gpu}")

            # epochs = 1200
            epochs = self.joint.epochs
            learning_rate = 0.001
            print_loss_every = max(epochs // 10, 1)

            losses = []
            optimizer = Adam(q_joint.parameters(), lr=learning_rate)
            q_joint.train()
            for e in range(epochs):
                batch = np.random.choice(range(n), batch_size, False)
                x = soybeans_data[batch]

                if disc_layer_type == "bipartite":
                    x = x.view(x.shape[0], -1)  # flattening vector

                optimizer.zero_grad()
                # zs = model.forward(x)

                # if disc_layer_type == 'bipartite':
                #     zs = zs.view(batch_size, sequence_length, -1) # adding back in sequence dimension

                # base_log_probs_sm = torch.nn.functional.log_softmax(base_log_probs, dim=-1)
                # #print(zs.shape, base_log_probs_sm.shape)
                # logprob = zs*base_log_probs_sm # zs are onehot so zero out all other logprobs.
                logprob = q_joint.log_prob(x)
                loss = -torch.sum(logprob) / batch_size

                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if e % print_loss_every == 0:
                    print("epoch:", e, "loss:", loss.item())

            genotypes = torch.from_numpy(valid.X)
            genotypes = genotypes.long()  # convert to long for 1-hot encoding
            genotypes = F.one_hot(genotypes)  # 1-hot encoding
            genotypes = genotypes.char()  # make uint8 to use less memory
            x = genotypes.float()
            if self.joint.gpu is not None:
                x = x.to(f"cuda:{self.joint.gpu}")
            batch_size = x.shape[0]
            logprob = q_joint.log_prob(x)
            loss = -torch.sum(logprob) / batch_size
            print("held out loss: ", loss)
        else:
            q_joint.load_state_dict(torch.load(self.save_path))

        return q_joint

    def save_joint_info_to_disk(self, q_joint, train, valid):
        pass

from itertools import cycle
from os import makedirs
from pathlib import Path
from time import time_ns

from hydra.utils import to_absolute_path
from joblib.parallel import Parallel, delayed
from flowselect.util import save_and_symlink
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from .pyhrt.hrt import hrt
from .selection_methods import get_fast_feature
from .crt import benjamini_hochberg


class HRTFitter:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.nominal_fdrs = np.arange(1, 51) / 100
        self.n_obs = self.cfg.hrt.n_obs
        self.n_jobs = self.cfg.hrt.n_jobs

    def __call__(self, train, valid, test):
        t = time_ns()
        # n = 1000
        # p = 30
        # X = np.random.normal(size=(n,1)) * 0.5 + np.random.normal(size=(n,p)) * 0.5
        # beta = np.random.normal(size=4)
        # Y = np.random.normal(X[:,1:1+len(beta)].dot(beta))

        # Create train/test splits
        # X_train, Y_train = X[:900], Y[:900]
        # X_test, Y_test = X[900:], Y[900:]
        tr_obs = min(self.n_obs, train.X.shape[0])
        te_obs = min(self.n_obs, test.X.shape[0])
        X_train, Y_train = train.X[:tr_obs], train.Y[:tr_obs]
        X_test, Y_test = test.X[:te_obs], test.Y[:te_obs]

        # Simple OLS regression
        # from sklearn.linear_model import LinearRegression
        # model = LinearRegression().fit(X_train, Y_train)
        SelectionModelClass = get_fast_feature(self.cfg.hrt.feature_statistic)
        model = SelectionModelClass(X_train, Y_train)

        # Use mean squared error as the empirical risk metric
        # tstat_fn = lambda X_eval: ((Y_test - model.predict(X_eval))**2).mean()
        tstat_fn = lambda X_eval: -model.importance(X_eval, Y_test)

        # Run the HRT
        p_values = []
        results = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(hrt)(
                feature,
                tstat_fn,
                X_test,
                nbootstraps=nbootstraps,
                nperms=nperms,
                nepochs=nepochs,
                save_nulls=True,
            )
            for (feature, tstat_fn, X_test, nbootstraps, nperms, nepochs) in zip(
                range(X_train.shape[1]),
                cycle([tstat_fn]),
                cycle([X_test]),
                cycle([self.cfg.hrt.nbootstraps]),
                cycle([self.cfg.hrt.nperms]),
                cycle([self.cfg.hrt.nepochs]),
            )
        )
        p_values = [r["p_value"] for r in results]
        nulls = [r["samples_null"] for r in results]
        save_and_symlink(
            nulls, self.save_folder.joinpath("null_samples.npy"), "null_samples.npy"
        )
        pvalue_df, power_df = self._get_fdr_and_power(p_values, train)

        hrt_time = time_ns() - t
        self.save_timings_to_disk(hrt_time)

        for fname, df in zip(["pvalues.csv", "power.csv"], [pvalue_df, power_df]):
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

    def _get_fdr_and_power(self, p_values, train):
        res = []
        beta = train.beta["beta"].flatten()
        is_relevant = ~np.isclose(beta, 0.0)
        pvalue_df = pd.DataFrame(
            list(zip(p_values, is_relevant, beta)),
            columns=["p_values", "is_relevant", "beta"],
        )
        for fdr in self.nominal_fdrs:

            threshold, accept = benjamini_hochberg(p_values, fdr)
            _, fp, _, tp = confusion_matrix(
                y_true=is_relevant, y_pred=accept.astype(int)
            ).ravel()

            fdp = fp / max(fp + tp, 1.0)
            power = tp / max(1.0, is_relevant.sum())
            res.append((fdp, power))
        fdrs, powers = list(zip(*res))
        power_df = pd.DataFrame(
            list(zip(self.nominal_fdrs, fdrs, powers)),
            columns=["n_fdrs", "fdr", "power"],
        )
        return pvalue_df, power_df

    @property
    def id(self):
        return "hrt"

    @property
    def varsel_id(self):
        return self.cfg.hrt.feature_statistic + "_" + str(self.cfg.hrt.n_obs)

    @property
    def save_folder(self):
        path = (
            Path(to_absolute_path(self.cfg.output))
            .joinpath(self.cfg.data.data_id)
            .joinpath(self.cfg.joint.joint_id)
            .joinpath(self.id)
        )
        return path

    @property
    def varsel_save_folder(self):
        return self.save_folder.joinpath(self.varsel_id)

    def save_timings_to_disk(self, time):
        if not self.varsel_save_folder.is_dir():
            makedirs(self.varsel_save_folder)
        path = self.varsel_save_folder.joinpath("time.txt")
        with path.open("w") as fp:
            fp.writelines([f"{time}"])

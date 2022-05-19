import pytest
import flowselect.joint as joint
from hydra.experimental import initialize, compose
from os import chdir
from os.path import abspath
from pathlib import Path
from shutil import rmtree


@pytest.fixture(scope="session")
def output_path():
    p = Path(abspath("./output/pytest"))
    yield p
    rmtree(p)


@pytest.fixture(scope="session")
def cfg_ddlk(output_path):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="global",
            overrides=[
                "joint.max_epochs=1",
                "knockoff.max_epochs=1",
                f"output={output_path.as_posix()}",
                "joint=ddlk",
                "knockoff=ddlk",
                "variable_selection=khrt",
                "variable_selection.feature_statistic=fast_lasso",
            ],
        )
    return cfg


@pytest.fixture(scope="session")
def cfg_modelx(output_path):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="global",
            overrides=[
                "dgp_covariate=gaussian_ar",
                "dgp_response=linear",
                "joint.max_epochs=1",
                "knockoff.max_epochs=1",
                f"output={output_path.as_posix()}",
                "joint=truth",
                "knockoff=modelx",
                "variable_selection=lasso",
                "variable_selection.max_iter=1",
                "variable_selection.n_jobs=1",
                "variable_selection.n_folds=2",
            ],
        )
    return cfg


@pytest.fixture(scope="session")
def cfg_crt(output_path):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="global",
            overrides=[
                "dgp_response=linear",
                "joint.max_epochs=1",
                "knockoff.max_epochs=1",
                f"output={output_path.as_posix()}",
                "joint=gaussmaf",
                "joint.n_layers=1",
                "joint.hidden_features=1",
                "joint.marg_max_epochs=1",
                "varsel_method=crt",
                "crt.model_name=mcmc",
                "crt.feature_statistic=fast_lasso",
                "crt.n_jobs=1",
                "crt.mcmc_steps=2",
                "crt.batch_size=1000",
            ],
        )
    return cfg


@pytest.fixture(scope="session")
def cfg_kgan(output_path):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="global",
            overrides=[
                "dgp_response=linear",
                "joint.max_epochs=1",
                "knockoff.max_epochs=1",
                f"output={output_path.as_posix()}",
                "joint=truth",
                "knockoff=gan",
                "knockoff.niter=1",
            ],
        )
    return cfg


@pytest.fixture(scope="session")
def cfg_deepknockoff(output_path):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="global",
            overrides=[
                "dgp_response=linear",
                "joint.max_epochs=1",
                "knockoff.max_epochs=1",
                f"output={output_path.as_posix()}",
                "joint=truth",
                "knockoff=deepknockoff",
                "knockoff.epochs=1",
                "knockoff.epoch_length=1",
            ],
        )
    return cfg


@pytest.fixture(scope="session")
def cfg_hrt(output_path):
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="global",
            overrides=[
                "dgp_response=linear",
                "joint.max_epochs=1",
                "knockoff.max_epochs=1",
                f"output={output_path.as_posix()}",
                "joint=truth",
                "varsel_method=hrt",
                "hrt.feature_statistic=fast_lasso",
                "hrt.nbootstraps=1",
                "hrt.nperms=2",
                "hrt.nepochs=1",
            ],
        )
    return cfg


@pytest.fixture(scope="session")
def wd():
    p = Path(abspath("pytest_wd"))
    p.mkdir()
    yield p
    chdir(p.parent)
    rmtree(p)

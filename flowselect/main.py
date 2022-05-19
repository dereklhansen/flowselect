from hydra.utils import to_absolute_path
from flowselect.crt import CRTFactory
import os
import sys
import torch
import pytorch_lightning as pl
import hydra
import logging
from time import time_ns
from omegaconf import DictConfig

path = os.path.abspath("..")
if path not in sys.path:
    sys.path.insert(0, path)

from flowselect.data import get_data
from flowselect.joint import fit_joint
from flowselect.knockoff_generators import fit_knockoff
from flowselect.hrt import HRTFitter

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="../conf/global")
def main(cfg: DictConfig) -> None:
    ## Seed for reproduceability (corresponds to deterministic=true in training loop)
    pl.trainer.seed_everything(cfg.seed + cfg.data.run_id)
    torch.set_num_threads(cfg.torch_threads)  # Limit number of threads on server
    train, valid, test = get_data(cfg, log)
    if (
        (cfg.varsel_method == "crt")
        or (cfg.varsel_method == "knockoff")
        and (cfg.knockoff.model_name in ["ddlk", "realnvp"])
    ):
        q_joint = fit_joint(cfg, train, valid)
    else:
        q_joint = None
        cfg.joint["joint_id"] = "nojoint"

    if cfg.varsel_method == "knockoff":
        results, power = fit_knockoff(cfg, train, valid, test, q_joint)
    elif cfg.varsel_method == "crt":
        crt_fitter = CRTFactory().get_crt_fitter(cfg, cfg.crt.model_name)
        results, power = crt_fitter(train, valid, test, q_joint)
    elif cfg.varsel_method == "hrt":
        hrt_fitter = HRTFitter(cfg)
        results, power = hrt_fitter(train, valid, test)
    else:
        raise NotImplementedError("Invalid varsel_method selected")
    print(results)
    if power is not None:
        print(power)


if __name__ == "__main__":
    main()

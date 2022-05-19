import os
import pytest
import torch
from flowselect.main import main
from pathlib import Path
from shutil import rmtree
from sklearn.exceptions import ConvergenceWarning
from warnings import catch_warnings, filterwarnings


class TestSmoke:
    @staticmethod
    def _run_smoke_test(wd, cfg, name):
        print(cfg)
        wd = wd.joinpath(name)
        wd.mkdir()
        os.chdir(wd)
        main(cfg)

    def test_ddlk(self, wd, cfg_ddlk):
        self._run_smoke_test(wd, cfg_ddlk, "ddlk")

    def test_modelx(self, wd, cfg_modelx):
        self._run_smoke_test(wd, cfg_modelx, "modelx")

    def test_crt(self, wd, cfg_crt):
        self._run_smoke_test(wd, cfg_crt, "crt")

    def test_kgan(self, wd, cfg_kgan):
        self._run_smoke_test(wd, cfg_kgan, "kgan")

    def test_deepknockoff(self, wd, cfg_deepknockoff):
        self._run_smoke_test(wd, cfg_deepknockoff, "deepknockoff")

    def test_hrt(self, wd, cfg_hrt):
        self._run_smoke_test(wd, cfg_hrt, "hrt")

import os
from shutil import which
import subprocess as sp
from warnings import warn
import torch
import h5py

from pathlib import Path


def save_and_symlink(save_data, gbl_data_path, lcl_data_name, savefun=None):
    """

    Saves "save_data" to "lcl_data_name" in the current working directory,
    or symlinks gbl_data_path if that is available.
    Then, if applicable, we then create a symlink in a different directory

    save_data: the data to be saved
    gbl_data_path: A path. Points to the global path for the file
    lcl_data_name: A string where to save the data in the local cwd.
    """
    gbl_data_path = Path(gbl_data_path)
    lcl_data_name = Path(lcl_data_name)
    if gbl_data_path.is_symlink():
        if not gbl_data_path.exists():
            msg = f"\
            The symlink for the file {gbl_data_path.as_posix()} is broken.\
            Likely something went wrong in writing it out."
            warn(msg, RuntimeWarning)
        os.symlink(os.path.relpath(gbl_data_path.resolve()), lcl_data_name)
    else:
        if savefun is None:
            torch.save(save_data, lcl_data_name)
        else:
            savefun()
        gbl_data_fldr = gbl_data_path.parent
        if not gbl_data_fldr.is_dir():
            os.makedirs(gbl_data_fldr)
        if not gbl_data_path.exists():
            # log.info(f"Symlinking {gbl_data_path} to generated data")
            link_source = os.path.relpath(
                Path.cwd().joinpath(lcl_data_name), start=gbl_data_fldr
            )
            gbl_data_path.symlink_to(link_source)


def write_h5(path, data_dict):
    with h5py.File(path, "w") as f:
        for key in data_dict:
            f[key] = data_dict[key]

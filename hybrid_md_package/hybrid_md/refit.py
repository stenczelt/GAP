#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021.
"""
Refitting of model on the fly

This is a generic refitting function, specific ones and tweaks of
this one with the same interface are to be implemented here.
"""

import importlib
import os.path
import shutil
import subprocess
from time import time

import ase.io
import numpy as np
from hybrid_md.state_objects import HybridMD

DEFAULT_FIT_NUM_THREADS = "32"


def refit(state: HybridMD):
    """Refit a GAP model, with in-place update

    This is a generic one, which can import the function

    Parameters
    ----------
    state: HybridMD

    """
    if state.settings.refit_function_name is None:
        return refit_generic(
            state,
            descriptor_strs=state.settings.refit_descriptor_str,
            default_sigma=state.settings.refit_default_sigma,
            extra_gap_parameters=state.settings.refit_extra_gap_opts,
        )
    else:
        refit_function_import = state.settings.refit_function_name

        # separate import path
        module_name = ".".join(refit_function_import.split(".")[:-1])
        function_name = refit_function_import.split(".")[-1]

        # import the module of the refit function
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            raise RuntimeError(f"Refit function's module not found: {module_name}")

        # class of the calculator
        if hasattr(module, function_name):
            refit_function = getattr(module, function_name)
            assert callable(refit_function)
        else:
            raise RuntimeError(
                f"Refit function ({function_name}) not found in module {module_name}"
            )

        # YAY, all great now
        return refit_function(state)


def refit_generic(
    state: HybridMD,
    descriptor_strs: str = None,
    default_sigma: str = None,
    extra_gap_parameters: str = None,
):
    """Refit a GAP model, with in-place update

    This is a generic very simple solution, that should work as a
    first try starting from scratch. Change this function to your
    own system and fitting settings as needed.

    Parameters
    ----------
    state: HybridMD
    descriptor_strs : str
        descriptor strings, ':' separated, no brackets around them
    default_sigma : str
        default sigma, four numbers separated by ':'
    extra_gap_parameters : str
        extra GAP parameters to be appended to the

    """
    if default_sigma is None:
        default_sigma = "0.005 0.050 0.1 1.0"

    # 2B + SOAP model
    gp_name = "GAP.xml"
    frames_train = ase.io.read(state.xyz_filename, ":") + state.get_previous_data()

    if descriptor_strs is None:
        # generic 2B+SOAP, need the frames for delta
        delta = np.std([at.info["QM_energy"] / len(at) for at in frames_train]) / 4
        desc_str_2b = (
            f"distance_Nb order=2 n_sparse=20 cutoff=4.5 cutoff_transition_width=1.0 "
            f"compact_clusters covariance_type=ard_se theta_uniform=1.0 sparse_method=uniform "
            f"f0=0.0 add_species=T delta={delta}"
        )
        desc_str_soap = (
            f"soap n_sparse=200 n_max=8 l_max=4 cutoff=4.0 cutoff_transition_width=1.0 "
            f"atom_sigma=0.5 add_species=True "
            f"delta={delta} covariance_type=dot_product zeta=4 sparse_method=cur_points"
        )
        descriptor_strs = desc_str_2b + " : " + desc_str_soap

    if extra_gap_parameters is None:
        extra_gap_parameters = " sparse_jitter=1.0e-8 "

    # save the previous model
    if os.path.isfile(gp_name):
        shutil.move(gp_name, f"save__{time()}__{gp_name}")

    # training structures & delta
    ase.io.write("train.xyz", frames_train)
    if os.path.isfile("train.xyz.idx"):
        os.remove("train.xyz.idx")

    # assemble the fitting string
    if state.settings.e0 in ["average", "isolated"]:
        e0_method = f"e0_method={state.settings.e0}"
    else:
        e0_method = f"e0={state.settings.e0}"

    fit_str = (
        f"gap_fit at_file=train.xyz gp_file={gp_name}"
        f" energy_parameter_name=QM_energy"
        f" force_parameter_name=QM_forces"
        f" virial_parameter_name=QM_virial_NOPE "
        f" do_copy_at_file=F sparse_separate_file=T "
        f" default_sigma={{ {default_sigma} }} {e0_method} "
        f" gap={{ {descriptor_strs} }} "
        f" {extra_gap_parameters}"
    )

    with open("debug_output.txt", "w") as file:
        file.write(fit_str)

    # fit the model
    if state.settings.refit_num_threads is None:
        num_threads = DEFAULT_FIT_NUM_THREADS
    else:
        num_threads = str(state.settings.refit_num_threads)
    os.environ["OMP_NUM_THREADS"] = num_threads
    proc = subprocess.run(
        fit_str, shell=True, capture_output=True, text=True, check=True
    )
    os.environ["OMP_NUM_THREADS"] = "1"

    # print the outputs to file
    with open(f"stdout_{gp_name}_at_{time()}__.txt", "w") as file:
        file.write(proc.stdout)
    with open(f"stderr_{gp_name}_at_{time()}__.txt", "w") as file:
        file.write(proc.stderr)

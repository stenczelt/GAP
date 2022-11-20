#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021-2022.
"""
Settings of the calculation
"""
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Type

import marshmallow_dataclass
import yaml
from marshmallow import Schema


@dataclass
class Tolerances:
    """Represents the tolerances we are setting for the checking steps in the calculation"""

    ediff: Optional[float]  # in eV
    fmax: Optional[float]  # in eV/A
    frmse: Optional[float]  # in eV/A
    vmax: Optional[float]  # in eV

    def get(self, key: str):
        if not isinstance(key, str):
            raise KeyError("Non-string keys are not supported")
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"No attribute {key} found in the {self.__class__.__name__}")


@dataclass
class Refit:
    # custom function
    function_name: Optional[str]

    # list of str
    previous_data: Optional[List[str]] = field(default_factory=list)

    # GAP parameters
    gp_name: str = "GAP.xml"
    default_sigma: str = "0.005 0.050 0.1 1.0"
    descriptor_str: Optional[str] = None
    extra_gap_opts: str = "sparse_jitter=1.0e-8"
    e0: Optional[str] = None
    e0_method: Optional[str] = "average"

    # for switching to OMP-parallel mode on the fly
    num_threads: Optional[int] = None

    @property
    def use_omp(self):
        """Whether we want to use OMP-parallel fitting"""
        return bool(self.num_threads) and self.num_threads > 1


@dataclass
class AdaptiveMethodSettings:
    """Settings for the adaptive method"""

    n_min: Optional[int]
    n_max: Optional[int]
    factor: Optional[int]


@marshmallow_dataclass.dataclass
class MainSettings:
    # nested objects
    tolerances: Tolerances
    adaptive_method_parameters: Optional[AdaptiveMethodSettings]
    refit: Refit = field(default_factory=Refit)

    # if model can be updated, if False then we can only measure performance
    can_update: bool = False

    # intervals
    check_interval: int = 1
    num_initial_steps: int = 0

    # for mypy & code editors (this is filled in by marshmallow_dataclass)
    Schema: ClassVar[Type[Schema]] = Schema

    @classmethod
    def read_input(cls, filename) -> "MainSettings":
        # reads input settings of calculation
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        return cls.Schema().load(data)


# class Settings:
#     # tolerance
#     tolerances = dict(
#         ediff=None,  # in eV
#         fmax=None,  # in eV/A
#         frmse=None,  # in eV/A
#         vmax=None,  # in eV -- not used this time
#     )
#     # if model can be updated, if False then we can only measure performance
#     can_update = False
#
#     # intervals
#     check_interval = 1
#     num_initial_steps = 0
#
#     # for the adaptive interval method
#     adaptive_method_parameters = dict()
#
#     # refitting-related
#     previous_data = None
#     refit_function_name = None
#     refit_default_sigma = None
#     refit_descriptor_str = None
#     refit_extra_gap_opts = None
#     refit_num_threads = None
#     e0 = None
#
#     def read_input(self, filename):
#         # reads input settings of calculation
#         with open(filename, "r") as file:
#             data = yaml.safe_load(file)
#
#         # unpack
#         self.tolerances = data.get("tolerances", dict())
#         self.can_update = data.get("can_update", False)
#         self.check_interval = data.get("check_interval", 1)
#         self.num_initial_steps = data.get("num_initial_steps", 0)
#         self.previous_data = data.get("previous_data", None)
#         self.refit_function_name = data.get("refit_function_name", None)
#         self.refit_descriptor_str = data.get("refit_descriptor_str", None)
#         self.refit_default_sigma = data.get("refit_default_sigma", None)
#         self.refit_extra_gap_opts = data.get("refit_extra_gap_opts", None)
#         self.refit_num_threads = data.get("refit_num_threads", None)
#         self.e0 = data.get("e0", "average")
#         self.adaptive_method_parameters = data.get("adaptive_method_parameters", dict())
#
#     def validate(self):
#         # any validation of the settings
#         if self.num_initial_steps > 0 and not self.can_update:
#             raise ValueError("Requesting initial DFT steps but cannot update model!")


if __name__ == "__main__":
    serializer = MainSettings.Schema()

    with open(
        "/Users/tks32/work/castep-restart/GAP/hybrid_md_package/examples/SiC_gap_initial_steps/sic_md.hybrid-md-input.yaml",
        "r",
    ) as ff:
        dd = yaml.safe_load(ff)

    settings = serializer.load(dd)
    print(settings)

#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021-2022.
"""
Settings of the calculation
"""
import yaml


class Settings:
    # tolerance
    tolerance_met = True
    tolerances = dict(
        ediff=None,  # in eV
        fmax=None,  # in eV/A
        frmse=None,  # in eV/A
        vmax=None,  # in eV -- not used this time
    )
    # if model can be updated, if False then we can only measure performance
    can_update = False

    # intervals
    check_interval = 1
    num_initial_steps = 0

    # for the adaptive interval method
    adaptive_method_parameters = dict()

    # refitting-related
    previous_data = None
    refit_function_name = None
    refit_default_sigma = None
    refit_descriptor_str = None
    refit_extra_gap_opts = None
    refit_num_threads = None
    e0 = None

    def read_input(self, filename):
        # reads input settings of calculation
        with open(filename, "r") as file:
            data = yaml.safe_load(file)

        # unpack
        self.tolerances = data.get("tolerances", dict())
        self.can_update = data.get("can_update", False)
        self.check_interval = data.get("check_interval", 1)
        self.num_initial_steps = data.get("num_initial_steps", 0)
        self.previous_data = data.get("previous_data", None)
        self.refit_function_name = data.get("refit_function_name", None)
        self.refit_descriptor_str = data.get("refit_descriptor_str", None)
        self.refit_default_sigma = data.get("refit_default_sigma", None)
        self.refit_extra_gap_opts = data.get("refit_extra_gap_opts", None)
        self.refit_num_threads = data.get("refit_num_threads", None)
        self.e0 = data.get("e0", "average")
        self.adaptive_method_parameters = data.get("adaptive_method_parameters", dict())

    def validate(self):
        # any validation of the settings
        if self.num_initial_steps > 0 and not self.can_update:
            raise ValueError("Requesting initial DFT steps but cannot update model!")

#  Hybrid MD decision making package
#
#  Copyright (c) Tamas K. Stenczel 2021-2022.
"""
Objects representing the state of a calculation
"""
from abc import ABC
from enum import Enum, auto, unique

import ase.io
import numpy as np
import yaml

from hybrid_md.settings import Settings


@unique
class StepKinds(Enum):
    """The kinds of steps we can have while the MD"""

    INITIAL = auto()
    LAST_INITIAL = auto()
    CHECK = auto()
    GENERIC = auto()


class HybridMD:
    def __init__(self, seed: str, md_iteration: int = None):
        self.seed = seed

        # associated sub-state objects
        self.xyz_this_run = SubStateXYZ(seed)
        self.settings = Settings()
        self.carry = CarriedState(seed)

        self.log_filename = f"{self.seed}.hybrid-md-temporary-log"
        self.input_filename = f"{self.seed}.hybrid-md-input.yaml"
        self.xyz_filename = f"{self.seed}.hybrid-md.xyz"

        # read input -> tolerances, etc.
        self.read_input()

        # dummy arrays for results
        self.md_iteration = md_iteration

        # validation
        self.settings.validate()

    # -----------------------------------------------------------------------------------
    # IO for carried info

    def read_input(self):
        self.settings.read_input(self.input_filename)

    # -----------------------------------------------------------------------------------
    # step's IO

    def io_initial_step_banner(self):
        lines = ["\n", "Using Hybrid MD\n", "! input file" + "-" * 66 + "\n"]

        with open(self.input_filename, "r") as file:
            lines.extend(file.readlines())

        lines.append("! end of input file" + "-" * 61)
        lines.append("")

        self.write_to_tmp_log(lines, append=False)

    def write_to_tmp_log(self, lines, append):
        # write log to temporary file

        if append:
            mode = "a"
        else:
            mode = "w"

        with open(self.log_filename, mode) as file:
            file.writelines(lines)

    # -----------------------------------------------------------------------------------
    # Error table for IO

    def _tolerance_line(self, name: str, value: float, tolerance, unit: str):
        if tolerance is not None:
            # checks tolerance as well
            self.tolerance_met = self.settings.tolerances and tolerance > value

            # line to be printed
            yes_no = self._bool_to_str(tolerance > value)

            return f" {name:>12} | {value:16.8f} | {tolerance:16.8f} | {unit:10} | {yes_no:3} | <-- Hybrid-MD\n"

        return f" {name:>12} | {value:16.8f} |              Off | {unit:10} |     | <-- Hybrid-MD\n"

    @staticmethod
    def _tolerance_line_cumulative(name: str, value: float, unit: str):

        return f" {name:>12}{value:30.8f}               | {unit:10} | <-- Hybrid-MD-Cumul\n"

    def check_tolerances(self):
        # todo: implement species specific check as well, and perhaps X out of Y to pass rather than all()

        # pass if none of them are given
        checked_tolerances = [True]

        for val, tol in [
            (self.xyz_this_run.get_ediff(), self.get_tolerance("ediff")),
            (self.xyz_this_run.get_fmax(), self.get_tolerance("fmax")),
            (self.xyz_this_run.get_frmse(), self.get_tolerance("frmse")),
            (self.xyz_this_run.get_vmax(), self.get_tolerance("vmax")),
        ]:
            if tol is None:
                continue

            # perform the check
            checked_tolerances.append(val < tol)

        return all(checked_tolerances)

    def error_table(self):
        # line formatting
        separator = " -------------+------------------+------------------+------------+-----+ <-- Hybrid-MD\n"
        refit_str = "Refitting!" if self.carry.do_update_model else "No Refit"

        # container for table -> write to file later
        lines = [
            f"\n      Hybrid-MD: MD iteration{self.md_iteration:8}                                    <-- Hybrid-MD\n"
            if self.md_iteration is not None
            else "\n",
            separator,
            "    Parameter |      value       |     tolerance    |    units   | OK? | <-- Hybrid-MD\n",
            separator,
            self._tolerance_line(
                "|Ediff|",
                self.xyz_this_run.get_ediff(),
                self.get_tolerance("ediff"),
                "eV/at",
            ),
            self._tolerance_line(
                "max |Fdiff|",
                self.xyz_this_run.get_fmax(),
                self.get_tolerance("fmax"),
                "eV/Å",
            ),
            self._tolerance_line(
                "force RMSE",
                self.xyz_this_run.get_frmse(),
                self.get_tolerance("frmse"),
                "eV/Å",
            ),
            self._tolerance_line(
                "max |Vdiff|",
                self.xyz_this_run.get_vmax(),
                self.get_tolerance("vmax"),
                "eV",
            ),
            separator,
            f"{refit_str:>72} <-- Hybrid-MD\n",
        ]

        self.write_to_tmp_log(lines, append=True)

    def cumulative_error_table(self):

        postfix = " <-- Hybrid-MD-Cumul"

        separator = f" -------------+------------------+------------------+------------+-----+{postfix}\n"

        lines = [
            separator,
            f"  Cumulative RMSE       value            count: {self.xyz_this_run.get_count():>8}"
            f"  |    units   |{postfix}\n",
            separator,
            self._tolerance_line_cumulative(
                "Energy", self.xyz_this_run.get_cumulative_energy_rmse(), "eV/atom"
            ),
            self._tolerance_line_cumulative(
                "Forces", self.xyz_this_run.get_cumulative_force_rmse(), "eV/Å"
            ),
            self._tolerance_line_cumulative(
                "Virial", self.xyz_this_run.get_cumulative_virial_rmse(), "eV"
            ),
            separator,
        ]

        self.write_to_tmp_log(lines, append=True)

    def get_tolerance(self, key: str):
        if not self.xyz_this_run.use_virial and key in ["vmax"]:
            return None
        return self.settings.tolerances.get(key, None)

    # -----------------------------------------------------------------------------------
    # XYZ IO
    def read_xyz(self):
        self.xyz_this_run.read_xyz()

    def get_previous_data(self):
        # read the previous data from files given
        if self.settings.previous_data is None:
            return []
        else:
            frames = []
            for fn in self.settings.previous_data:
                frames.extend(ase.io.read(fn, ":"))
            return frames

    @staticmethod
    def _bool_to_str(value):
        if value:
            return "Yes"
        return " No"


class SeedAwareState(ABC):
    def __init__(self, seed: str):
        self.seed = seed


class CarriedState(SeedAwareState):
    do_comparison = False  # print the error table
    do_update_model = False  # REFIT
    next_ab_initio = False
    next_is_pre_step = True  # do them in order
    last_check_step = -1
    current_check_interval = -1

    def __init__(self, seed: str):
        super().__init__(seed)
        self.state_filename = f"{self.seed}.hybrid-md-state.yaml"

    def dump(self):
        # save state to file
        with open(self.state_filename, "w") as file:
            yaml.dump(self.carry_dict(), file)

    def load(self):
        # load state from file
        with open(self.state_filename, "r") as file:
            values = yaml.safe_load(file)
        self.unpack_dump(values)

    def carry_dict(self):
        return dict(
            do_comparison=self.do_comparison,
            do_update_model=self.do_update_model,
            next_is_pre_step=self.next_is_pre_step,
            next_ab_initio=self.next_ab_initio,
            last_check_step=self.last_check_step,
            current_check_interval=self.current_check_interval,
        )

    def unpack_dump(self, values: dict):
        self.do_comparison = values.get("do_comparison")
        self.do_update_model = values.get("do_update_model")
        self.next_is_pre_step = values.get("next_is_pre_step")
        self.next_ab_initio = values.get("next_ab_initio")
        self.last_check_step = values.get("last_check_step")
        self.current_check_interval = values.get("current_check_interval")

    def reset(self):
        # reset the info
        self.do_comparison = False
        self.do_update_model = False
        self.next_is_pre_step = True


class SubStateXYZ(SeedAwareState):
    """Represents XYZ structural data gathered during the accelerated MD"""

    def __init__(self, seed: str):
        super().__init__(seed)
        self.xyz_filename = f"{self.seed}.hybrid-md.xyz"

        self.use_virial = False

        # arrays: filled with data from structures
        self.len_atoms = 1
        self.atomic_numbers = np.zeros(1)

        self.energy_ff = np.zeros(1)
        self.forces_pp = np.zeros(1)
        self.virial_pp = np.zeros(1)

        self.energy_qm = np.zeros(1)
        self.forces_pw = np.zeros(1)
        self.virial_pw = np.zeros(1)

    # -----------------------------------------------------------------------------------
    # XYZ IO
    def read_xyz(self):
        frames = ase.io.read(self.xyz_filename, ":")

        # global atoms information
        self.len_atoms = len(frames[0])
        self.atomic_numbers = frames[0].get_atomic_numbers()

        # energy : (n_frames)
        self.energy_qm = self._unpack_info(frames, "QM_energy")
        self.energy_ff = self._unpack_info(frames, "FF_energy")

        # forces : (n_frames, n_atoms, 3)
        self.forces_pw = self._unpack_arrays(frames, "QM_forces")
        self.forces_pp = self._unpack_arrays(frames, "FF_forces")

        # virial : (n_frames, 6)
        if "QM virial" in frames[0].info.keys():
            self.use_virial = True
            self.virial_pw = self._unpack_info(frames, "QM_virial")
            self.virial_pp = self._unpack_info(frames, "FF_virial")

    # -----------------------------------------------------------------------------------
    # last step's error measures
    def get_ediff(self):
        # energy difference per atom
        return np.abs(self.energy_ff[-1] - self.energy_qm[-1]) / self.len_atoms

    def get_fmax(self, atomic_number: int = None):
        if atomic_number is None:
            # species agnostic maximum force component difference
            return np.max(np.abs(self.forces_pp[-1] - self.forces_pw[-1]))

        return self._max_abs(
            self.forces_pp[-1, self._z_mask(atomic_number)]
            - self.forces_pw[-1, self._z_mask(atomic_number)]
        )

    def get_frmse(self, atomic_number: int = None):
        if atomic_number is None:
            # species agnostic force component RMSE
            return self._rmse(self.forces_pp[-1] - self.forces_pw[-1])

        return self._rmse(
            self.forces_pp[-1, self._z_mask(atomic_number)]
            - self.forces_pw[-1, self._z_mask(atomic_number)]
        )

    def get_vmax(self):
        # maximum virial component difference
        return np.max(np.abs(self.virial_pp[-1] - self.virial_pw[-1]))

    # -----------------------------------------------------------------------------------
    # cumulative error measures

    def get_count(self):
        return len(self.energy_qm)

    def get_cumulative_energy_rmse(self):
        # cumulative energy RMSE per atom
        return self._rmse((self.energy_ff - self.energy_qm) / self.len_atoms)

    def get_cumulative_force_rmse(self, atomic_number: int = None):
        if atomic_number is None:
            # species agnostic force component RMSE
            return self._rmse(self.forces_pp - self.forces_pw)

        return self._rmse(
            self.forces_pp[:, self._z_mask(atomic_number)]
            - self.forces_pw[:, self._z_mask(atomic_number)]
        )

    def get_cumulative_virial_rmse(self):
        return self._rmse(self.virial_pw - self.virial_pp)

    # -----------------------------------------------------------------------------------
    # helper functions

    @staticmethod
    def _rmse(array):
        return np.sqrt(np.mean(np.square(array)))

    @staticmethod
    def _max_abs(array):
        return np.max(np.abs(array))

    def _z_mask(self, atomic_number: int):
        return self.atomic_numbers == atomic_number

    @staticmethod
    def _unpack_info(frames, key: str):
        return np.array([at.info[key] for at in frames])

    @staticmethod
    def _unpack_arrays(frames, key: str):
        return np.array([at.arrays[key] for at in frames])

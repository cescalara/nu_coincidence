import h5py
import numpy as np
from abc import abstractmethod, ABCMeta


class Simulation(object, metaclass=ABCMeta):
    """
    Generic simulation base class.
    """

    def __init__(self, file_name="output/test_sim", group_base_name="survey", N=1):

        self._N = N

        self._file_name = file_name

        self._group_base_name = group_base_name

        self._param_servers = []

        self._setup_param_servers()

        self._initialise_output_file()

    @abstractmethod
    def _setup_param_servers(self):

        raise NotImplementedError()

    def _initialise_output_file(self):

        with h5py.File(self._file_name, "w") as f:

            f.attrs["file_name"] = np.string_(self._file_name)
            f.attrs["group_base_name"] = np.string_(self._group_base_name)
            f.attrs["N"] = self._N

    @abstractmethod
    def run(self):

        raise NotImplementedError()

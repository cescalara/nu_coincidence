import os
import h5py
import numpy as np
from popsynth.utils.configuration import popsynth_config

from cosmic_coincidence.blazars.fermi_interface import (
    FermiPopParams,
    VariableFermiPopParams,
)
from cosmic_coincidence.blazars.bllac import (
    BLLacPopWrapper,
    VariableBLLacPopWrapper,
)


class Simulation(object):
    """
    Set up and run simulations.
    """

    def __init__(self, file_name="output/test_sim.h5", group_base_name="survey", N=1):

        self._N = N

        self._file_name = file_name

        self._group_base_name = group_base_name

        self._param_servers = []

        popsynth_config["show_progress"] = False

        self._setup_param_servers()

    def _setup_param_servers(self):

        for i in range(self._N):

            param_server = VariableFermiPopParams(
                A=3.39e4,
                gamma1=0.27,
                Lstar=0.28e48,
                gamma2=1.86,
                zcstar=1.34,
                p1star=2.24,
                tau=4.92,
                p2=-7.37,
                alpha=4.53e-2,
                mustar=2.1,
                beta=6.46e-2,
                sigma=0.26,
                boundary=4e-12,
                hard_cut=True,
                variability_weight=0.05,
                flare_rate_min=1 / 7.5,
                flare_rate_max=15,
                flare_rate_index=1.5,
                obs_time=10,
            )

            param_server.seed = i

            param_server.file_path = self._file_name
            param_server.group_name = self._group_base_name + "_%i" % i

            self._param_servers.append(param_server)

    def _pop_wrapper(self, param_server):

        return VariableBLLacPopWrapper(param_server)

    def run(self, client=None):

        # Parallel
        if client is not None:

            futures = client.map(self._pop_wrapper, self._param_servers)

            results = client.gather(futures)

            for res in results:

                res._survey.addto(
                    res._parameter_server.file_path, res._parameter_server.group_name
                )

            del results
            del futures

        else:

            # Serial
            results = [
                self._pop_wrapper(param_server) for param_server in self._param_servers
            ]

    def save(self):

        pass

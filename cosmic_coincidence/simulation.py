import os

# from popsynth.utils.configuration import popsynth_config

from cosmic_coincidence.blazars.fermi_interface import FermiPopParams
from cosmic_coincidence.blazars.bllac import BLLacPopWrapper


class Simulation(object):
    """
    Set up and run simulations.
    """

    def __init__(self, survey_base_name="test_survey", save_path="output", N=1):

        self._N = N

        self._survey_base_name = survey_base_name

        self._save_path = save_path

        self._param_servers = []

        # popsynth_config["show_progress"] = False

        self._setup_param_servers()

    def _setup_param_servers(self):

        for i in range(self._N):

            param_server = FermiPopParams(
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
            )

            param_server.seed = i

            param_server.file_path = os.path.join(
                self._save_path,
                self._survey_base_name + "_%i.h5" % i,
            )

            self._param_servers.append(param_server)

    def _pop_wrapper(self, param_server):

        return BLLacPopWrapper(param_server)

    def run(self, client=None):

        if client is not None:

            futures = client.map(self._pop_wrapper, self._param_servers)

            results = client.gather(futures)

            del results
            del futures

        else:

            results = [
                self._pop_wrapper(param_server) for param_server in self._param_servers
            ]

    def save(self):

        pass

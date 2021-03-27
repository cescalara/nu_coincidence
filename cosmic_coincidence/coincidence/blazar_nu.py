import time
from dask.distributed import as_completed
from popsynth.utils.configuration import popsynth_config

from cosmic_coincidence.blazars.fermi_interface import VariableFermiPopParams
from cosmic_coincidence.blazars.bllac import VariableBLLacPopWrapper
from cosmic_coincidence.blazars.fsrq import VariableFSRQPopWrapper
from cosmic_coincidence.neutrinos.icecube import (
    IceCubeObsParams,
    IceCubeObsWrapper,
)
from cosmic_coincidence.simulation import Simulation


class BlazarNuCoincidence(object):
    """
    Check for coincidences of interest.
    """

    def __init__(self, bllac_pop, fsrq_pop, nu_obs):

        self._bllac_pop = bllac_pop

        self._fsrq_pop = fsrq_pop

        self._nu_obs = nu_obs

        self._run()

    def _run(self):

        time.sleep(10)

        return 0


class BlazarNuSimulation(Simulation):
    """
    Set up and run simulations.
    """

    def __init__(
        self,
        file_name="output/test_sim.h5",
        group_base_name="survey",
        N=1,
    ):

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
        )

        popsynth_config["show_progress"] = False

    def _setup_param_servers(self):

        self._bllac_param_servers = []
        self._fsrq_param_servers = []
        self._nu_param_servers = []

        for i in range(self._N):

            # BL Lacs
            bllac_param_server = VariableFermiPopParams(
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

            bllac_param_server.seed = i
            bllac_param_server.file_name = self._file_name
            bllac_param_server.group_name = self._group_base_name + "_%i" % i

            self._bllac_param_servers.append(bllac_param_server)

            # FSRQs
            fsrq_param_server = VariableFermiPopParams(
                A=3.06e4,
                gamma1=0.21,
                Lstar=0.84e48,
                gamma2=1.58,
                zcstar=1.47,
                p1star=7.35,
                tau=0,
                p2=-6.51,
                alpha=0.21,
                mustar=2.44,
                beta=0,
                sigma=0.18,
                boundary=4e-12,
                hard_cut=True,
                variability_weight=0.4,
                flare_rate_min=1 / 7.5,
                flare_rate_max=15,
                flare_rate_index=1.5,
                obs_time=10,
            )

            fsrq_param_server.seed = i
            fsrq_param_server.file_name = self._file_name
            fsrq_param_server.group_name = self._group_base_name + "_%i" % i

            self._fsrq_param_servers.append(fsrq_param_server)

            # Neutrinos
            nu_param_server = IceCubeObsParams(
                Emin=1e5,
                Emax=1e8,
                Enorm=1e5,
                Emin_det=2e5,
                atmo_flux_norm=2.5e-18,
                atmo_index=3.7,
                diff_flux_norm=1e-18,
                diff_index=2.19,
                obs_time=10,
                max_cosz=0.1,
            )

            nu_param_server.seed = i
            nu_param_server.file_name = self._file_name
            nu_param_server.group_name = self._group_base_name + "_%i" % i

            self._nu_param_servers.append(nu_param_server)

    def _bllac_pop_wrapper(self, param_server):

        return VariableBLLacPopWrapper(param_server)

    def _fsrq_pop_wrapper(self, param_server):

        return VariableFSRQPopWrapper(param_server)

    def _nu_obs_wrapper(self, param_server):

        return IceCubeObsWrapper(param_server)

    def _pop_wrapper(self, param_servers):

        bllac_server, fsrq_server = param_servers

        bllac = VariableBLLacPopWrapper(bllac_server)

        fsrq = VariableFSRQPopWrapper(fsrq_server)

        return bllac, fsrq

    def _coincidence_check(self, bllac_pop, fsrq_pop, nu_obs):

        return BlazarNuCoincidence(bllac_pop, fsrq_pop, nu_obs)

    def run(self, client=None):

        # Parallel
        if client is not None:

            # New ideas
            bllac_pop = client.map(
                self._bllac_pop_wrapper,
                self._bllac_param_servers,
            )

            fsrq_pop = client.map(
                self._fsrq_pop_wrapper,
                self._fsrq_param_servers,
            )

            nu_obs = client.map(
                self._nu_obs_wrapper,
                self._nu_param_servers,
            )

            coincidence = client.map(
                self._coincidence_check,
                bllac_pop,
                fsrq_pop,
                nu_obs,
            )

            for future, result in as_completed(coincidence, with_results=True):

                result._bllac_pop.write()
                result._fsrq_pop.write()
                result._nu_obs.write()

                del future, result

            del coincidence

        # Serial
        else:

            bllac_pop = [
                self._bllac_pop_wrapper(param_server)
                for param_server in self._bllac_param_servers
            ]

            fsrq_pop = [
                self._fsrq_pop_wrapper(param_server)
                for param_server in self._fsrq_param_servers
            ]

            nu_obs = [
                self._nu_obs_wrapper(param_server)
                for param_server in self._nu_param_servers
            ]

            for bllac, fsrq, nu in zip(bllac_pop, fsrq_pop, nu_obs):

                self._coincidence_check(bllac, fsrq, nu)

                bllac.write()
                fsrq.write()
                nu.write()

            del bllac_pop, fsrq_pop, nu_obs

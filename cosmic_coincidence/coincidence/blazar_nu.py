from dask.distributed import as_completed
from popsynth.utils.configuration import popsynth_config

from cosmic_coincidence.blazars.fermi_interface import VariableFermiPopParams
from cosmic_coincidence.blazars.bllac import VariableBLLacPopWrapper
from cosmic_coincidence.blazars.fsrq import VariableFSRQPopWrapper
from cosmic_coincidence.simulation import Simulation


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

    def _bllac_pop_wrapper(self, param_server):

        return VariableBLLacPopWrapper(param_server)

    def _fsrq_pop_wrapper(self, param_server):

        return VariableFSRQPopWrapper(param_server)

    def _pop_wrapper(self, param_servers):

        bllac_server, fsrq_server = param_servers

        bllac = VariableBLLacPopWrapper(bllac_server)

        fsrq = VariableFSRQPopWrapper(fsrq_server)

        return bllac, fsrq

    def run(self, client=None):

        # Parallel
        if client is not None:

            param_servers = []
            for bllac_server, fsrq_server in zip(
                self._bllac_param_servers, self._fsrq_param_servers
            ):
                param_servers.append((bllac_server, fsrq_server))

            futures = client.map(self._pop_wrapper, param_servers)

            for future, result in as_completed(futures, with_results=True):

                bllac, fsrq = result

                bllac.write()
                fsrq.write()

                del bllac, fsrq
                del future, result

            del futures

            # bllac_futures = client.map(
            #     self._bllac_pop_wrapper, self._fsrq_param_servers
            # )

            # for future, result in as_completed(bllac_futures, with_results=True):

            #     result.write()

            #     del result
            #     del future

            # del bllac_futures

            # fsrq_futures = client.map(self._fsrq_pop_wrapper, self._fsrq_param_servers)

            # for future, result in as_completed(fsrq_futures, with_results=True):

            #     result.write()

            #     del result
            #     del future

            # del fsrq_futures

        # Serial
        else:

            bllac_results = [
                self._bllac_pop_wrapper(param_server)
                for param_server in self._bllac_param_servers
            ]

            for result in bllac_results:

                result.write()

            del bllac_results

            fsrq_results = [
                self._fsrq_pop_wrapper(param_server)
                for param_server in self._fsrq_param_servers
            ]

            for result in fsrq_results:

                result.write()

            del fsrq_results

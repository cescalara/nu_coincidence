from abc import ABCMeta, abstractmethod
from joblib import (
    parallel_backend,
    register_parallel_backend,
    Parallel,
    delayed,
)

from cosmic_coincidence.populations.popsynth_wrapper import (
    PopsynthParams,
    PopsynthWrapper,
)
from cosmic_coincidence.simulation import Simulation
from cosmic_coincidence.utils.package_data import get_path_to_config
from cosmic_coincidence.neutrinos.icecube import (
    IceCubeObsParams,
    IceCubeObsWrapper,
    IceCubeTracksWrapper,
    IceCubeAlertsParams,
    IceCubeAlertsWrapper,
)
from cosmic_coincidence.utils.parallel import FileWritingBackend

register_parallel_backend("file_write", FileWritingBackend)


class BlazarNuSim(Simulation, metaclass=ABCMeta):
    """
    Abstract base class for blazar
    neutrino simulations.
    """

    def __init__(
        self,
        file_name="output/test_sim.h5",
        group_base_name="survey",
        N=1,
        bllac_config: str = None,
        fsrq_config: str = None,
        nu_config: str = None,
        nu_hese_config: str = None,
        nu_ehe_config: str = None,
        seed=1000,
    ):

        self._bllac_config = bllac_config
        self._fsrq_config = fsrq_config
        self._nu_config = nu_config
        self._nu_hese_config = nu_hese_config
        self._nu_ehe_config = nu_ehe_config
        self._seed = seed

        self._bllac_param_servers = []
        self._fsrq_param_servers = []
        self._nu_param_servers = []

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
        )

    def _setup_param_servers(self):

        self._bllac_param_servers = []
        self._fsrq_param_servers = []
        self._nu_param_servers = []

        for i in range(self._N):

            seed = self._seed + i

            # BL Lacs
            bllac_spec = get_path_to_config(self._bllac_config)
            bllac_param_server = PopsynthParams(bllac_spec)
            bllac_param_server.seed = seed
            bllac_param_server.file_name = self._file_name
            bllac_param_server.group_name = self._group_base_name + "_%i" % i

            self._bllac_param_servers.append(bllac_param_server)

            # FSRQs
            fsrq_spec = get_path_to_config(self._fsrq_config)
            fsrq_param_server = PopsynthParams(fsrq_spec)
            fsrq_param_server.seed = seed
            fsrq_param_server.file_name = self._file_name
            fsrq_param_server.group_name = self._group_base_name + "_%i" % i

            self._fsrq_param_servers.append(fsrq_param_server)

            # Neutrinos
            if self._nu_config is not None:

                nu_spec = get_path_to_config(self._nu_config)
                nu_param_server = IceCubeObsParams(nu_spec)

            else:

                nu_hese_spec = get_path_to_config(self._nu_hese_config)
                nu_ehe_spec = get_path_to_config(self._nu_ehe_config)
                nu_param_server = IceCubeAlertsParams(nu_hese_spec, nu_ehe_spec)

            nu_param_server.seed = seed
            nu_param_server.file_name = self._file_name
            nu_param_server.group_name = self._group_base_name + "_%i" % i

            self._nu_param_servers.append(nu_param_server)

    def _bllac_pop_wrapper(self, param_server):

        return PopsynthWrapper(param_server)

    def _fsrq_pop_wrapper(self, param_server):

        return PopsynthWrapper(param_server)

    def _nu_obs_wrapper(self, param_server):

        if self._nu_config is not None:

            return IceCubeTracksWrapper(param_server)

        else:

            return IceCubeAlertsWrapper(param_server)

    @abstractmethod
    def _blazar_nu_wrapper(self, bllac_pop, fsrq_pop, nu_obs):

        raise NotImplementedError()

    def _sim_wrapper(self, bllac_param_server, fsrq_param_server, nu_param_server):

        bllac_pop = self._bllac_pop_wrapper(bllac_param_server)

        fsrq_pop = self._fsrq_pop_wrapper(fsrq_param_server)

        nu_obs = self._nu_obs_wrapper(nu_param_server)

        result = self._blazar_nu_wrapper(bllac_pop, fsrq_pop, nu_obs)

        del bllac_pop, fsrq_pop, nu_obs

        return result

    def run(self, parallel=True, n_jobs=4):

        # Parallel
        if parallel:

            # Writes to file upon completion
            with parallel_backend("file_write"):

                Parallel(n_jobs=n_jobs)(
                    delayed(self._sim_wrapper)(bllac_ps, fsrq_ps, nu_ps)
                    for bllac_ps, fsrq_ps, nu_ps in zip(
                        self._bllac_param_servers,
                        self._fsrq_param_servers,
                        self._nu_param_servers,
                    )
                )

        # Serial
        else:

            for bllac_ps, fsrq_ps, nu_ps in zip(
                self._bllac_param_servers,
                self._fsrq_param_servers,
                self._nu_param_servers,
            ):

                result = self._sim_wrapper(
                    bllac_ps,
                    fsrq_ps,
                    nu_ps,
                )

                result.write()

                del result


class BlazarNuAction(object, metaclass=ABCMeta):
    """
    Abstract base class for different actions
    that can be applied to blazar and neutrino
    observations e.g. coincidence checks or
    connected simulations.
    """

    def __init__(
        self,
        bllac_pop: PopsynthWrapper,
        fsrq_pop: PopsynthWrapper,
        nu_obs: IceCubeObsWrapper,
        name="blazar_nu_action",
    ):

        self._name = name

        self._bllac_pop = bllac_pop

        self._fsrq_pop = fsrq_pop

        self._nu_obs = nu_obs

        self._file_name = nu_obs._parameter_server.file_name

        self._group_name = nu_obs._parameter_server.group_name

        self._run()

    @abstractmethod
    def _run(self):

        raise NotImplementedError()

    @abstractmethod
    def write(self):

        raise NotImplementedError()

    @property
    def name(self):

        return self._name

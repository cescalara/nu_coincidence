from abc import ABCMeta, abstractmethod
from joblib import (
    parallel_backend,
    register_parallel_backend,
    Parallel,
    delayed,
)

from nu_coincidence.populations.popsynth_wrapper import (
    PopsynthParams,
    PopsynthWrapper,
)
from nu_coincidence.simulation import Simulation
from nu_coincidence.utils.package_data import get_path_to_config
from nu_coincidence.neutrinos.icecube import (
    IceCubeObsParams,
    IceCubeObsWrapper,
    IceCubeTracksWrapper,
    IceCubeAlertsParams,
    IceCubeAlertsWrapper,
)
from nu_coincidence.utils.parallel import FileWritingBackend

register_parallel_backend("file_write", FileWritingBackend)


class PopNuSim(Simulation, metaclass=ABCMeta):
    """
    Abstract base class for popsynth
    neutrino simulations.
    """

    def __init__(
        self,
        file_name="output/test_sim.h5",
        group_base_name="survey",
        N=1,
        pop_config: str = None,
        nu_config: str = None,
        nu_hese_config: str = None,
        nu_ehe_config: str = None,
        seed=1000,
    ):

        self._pop_config = pop_config
        self._nu_config = nu_config
        self._nu_hese_config = nu_hese_config
        self._nu_ehe_config = nu_ehe_config
        self._seed = seed

        self._pop_param_servers = []
        self._nu_param_servers = []

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
        )

    def _setup_param_servers(self):

        self._pop_param_servers = []
        self._nu_param_servers = []

        for i in range(self._N):

            seed = self._seed + i

            pop_spec = get_path_to_config(self._pop_config)
            pop_param_server = PopsynthParams(pop_spec)
            pop_param_server.seed = seed
            pop_param_server.file_name = self._file_name
            pop_param_server.group_name = self._group_base_name + "_%i" % i

            self._pop_param_servers.append(pop_param_server)

            # Neutrinos
            if self._nu_config is not None:

                nu_spec = get_path_to_config(self._nu_config)
                nu_param_server = IceCubeObsParams.from_file(nu_spec)

            else:

                nu_hese_spec = get_path_to_config(self._nu_hese_config)
                nu_ehe_spec = get_path_to_config(self._nu_ehe_config)
                nu_param_server = IceCubeAlertsParams(nu_hese_spec, nu_ehe_spec)

            nu_param_server.seed = seed
            nu_param_server.file_name = self._file_name
            nu_param_server.group_name = self._group_base_name + "_%i" % i

            self._nu_param_servers.append(nu_param_server)

    def _pop_wrapper(self, param_server):

        return PopsynthWrapper(param_server)

    def _nu_obs_wrapper(self, param_server):

        if self._nu_config is not None:

            return IceCubeTracksWrapper(param_server)

        else:

            return IceCubeAlertsWrapper(param_server)

    @abstractmethod
    def _pop_nu_wrapper(self, pop, nu_obs):

        raise NotImplementedError()

    def _sim_wrapper(self, pop_param_server, nu_param_server):

        pop = self._pop_wrapper(pop_param_server)

        nu_obs = self._nu_obs_wrapper(nu_param_server)

        result = self._pop_nu_wrapper(pop, nu_obs)

        del pop, nu_obs

        return result

    def run(self, parallel=True, n_jobs=4):

        # Parallel
        if parallel:

            # Writes to file upon completion
            with parallel_backend("file_write"):

                Parallel(n_jobs=n_jobs)(
                    delayed(self._sim_wrapper)(pop_ps, nu_ps)
                    for pop_ps, nu_ps in zip(
                        self._pop_param_servers,
                        self._nu_param_servers,
                    )
                )

        # Serial
        else:

            for pop_ps, nu_ps in zip(
                self._pop_param_servers,
                self._nu_param_servers,
            ):

                result = self._sim_wrapper(
                    pop_ps,
                    nu_ps,
                )

                result.write()

                del result


class PopNuAction(object, metaclass=ABCMeta):
    """
    Abstract base class for different actions
    that can be applied to popsynth and neutrino
    observations e.g. coincidence checks or
    connected simulations.
    """

    def __init__(
        self,
        pop: PopsynthWrapper,
        nu_obs: IceCubeObsWrapper,
        name="pop_action",
    ):

        self._name = name

        self._pop = pop

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

import h5py
import numpy as np
from joblib import (
    register_parallel_backend,
    parallel_backend,
    Parallel,
    delayed,
)
from collections import OrderedDict

from cosmic_coincidence.coincidence.coincidence import (
    check_spatial_coincidence,
    check_temporal_coincidence,
)
from cosmic_coincidence.popsynth_wrapper import (
    PopsynthParams,
    PopsynthWrapper,
)
from cosmic_coincidence.populations.aux_samplers import (
    VariabilityAuxSampler,
    FlareRateAuxSampler,
    FlareTimeAuxSampler,
    FlareDurationAuxSampler,
)
from cosmic_coincidence.populations.selection import GalacticPlaneSelection
from cosmic_coincidence.neutrinos.icecube import (
    IceCubeAlertsParams,
    IceCubeAlertsWrapper,
)
from cosmic_coincidence.simulation import Simulation
from cosmic_coincidence.utils.package_data import get_path_to_data
from cosmic_coincidence.utils.parallel import FileWritingBackend

register_parallel_backend("file_write", FileWritingBackend)


class BlazarNuCoincidence(object):
    """
    Check for coincidences of interest.
    """

    def __init__(
        self,
        bllac_pop,
        fsrq_pop,
        nu_obs,
        name="blazar_nu_coincidence",
    ):

        self._name = name

        self._bllac_pop = bllac_pop

        self._fsrq_pop = fsrq_pop

        self._nu_obs = nu_obs

        self._file_name = nu_obs._parameter_server.file_name

        self._group_name = nu_obs._parameter_server.group_name

        self._bllac_coincidence = OrderedDict()

        self._fsrq_coincidence = OrderedDict()

        self._run()

    def _run(self):

        self._check_spatial()

        self._check_temporal()

    @property
    def name(self):

        return self._name

    @property
    def bllac_coincidence(self):

        return self._bllac_coincidence

    @property
    def fsrq_coincidence(self):

        return self._fsrq_coincidence

    def write(self):

        with h5py.File(self._file_name, "r+") as f:

            if self._group_name not in f.keys():

                group = f.create_group(self._group_name)

            else:

                group = f[self._group_name]

            subgroup = group.create_group(self.name)

            bllac_group = subgroup.create_group("bllac")

            for key, value in self.bllac_coincidence.items():

                if key != "spatial_match_inds":

                    bllac_group.create_dataset(key, data=value)

            fsrq_group = subgroup.create_group("fsrq")

            for key, value in self.fsrq_coincidence.items():

                if key != "spatial_match_inds":

                    fsrq_group.create_dataset(key, data=value)

    def _check_spatial(self):
        """
        Check for spatial coincidences between
        the *detected* blazar populations and
        neutrinos
        """

        observation = self._nu_obs.observation

        # BL Lacs
        survey = self._bllac_pop.survey

        n_match_spatial, spatial_match_inds = check_spatial_coincidence(
            np.deg2rad(observation.ra),
            np.deg2rad(observation.dec),
            np.deg2rad(observation.ang_err),
            np.deg2rad(survey.ra[survey.selection]),
            np.deg2rad(survey.dec[survey.selection]),
        )

        self.bllac_coincidence["n_spatial"] = n_match_spatial
        self.bllac_coincidence["spatial_match_inds"] = spatial_match_inds

        # FSRQs
        survey = self._fsrq_pop.survey

        n_match_spatial, spatial_match_inds = check_spatial_coincidence(
            np.deg2rad(observation.ra),
            np.deg2rad(observation.dec),
            np.deg2rad(observation.ang_err),
            np.deg2rad(survey.ra[survey.selection]),
            np.deg2rad(survey.dec[survey.selection]),
        )

        self.fsrq_coincidence["n_spatial"] = n_match_spatial
        self.fsrq_coincidence["spatial_match_inds"] = spatial_match_inds

    def _check_temporal(self):
        """
        Check for temporal coincidences between
        the *detected* blazar populations and
        neutrinos, which are also spatial
        coincidences.
        """

        observation = self._nu_obs.observation

        # BL Lacs
        survey = self._bllac_pop.survey

        n_match_variable, n_match_flaring = check_temporal_coincidence(
            observation.times,
            self.bllac_coincidence["spatial_match_inds"],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
        )

        self.bllac_coincidence["n_variable"] = n_match_variable
        self.bllac_coincidence["n_flaring"] = n_match_flaring

        # FSRQs
        survey = self._fsrq_pop.survey

        n_match_variable, n_match_flaring = check_temporal_coincidence(
            observation.times,
            self.fsrq_coincidence["spatial_match_inds"],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
        )

        self.fsrq_coincidence["n_variable"] = n_match_variable
        self.fsrq_coincidence["n_flaring"] = n_match_flaring


class BlazarNuCoincidenceSim(Simulation):
    """
    Set up and run simulations for blazar-neutrino
    coincidences. Assumes blazars and neutrinos have
    no underlying connection.
    """

    def __init__(
        self,
        file_name="output/test_coincidence_sim.h5",
        group_base_name="survey",
        N=1,
    ):

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

            seed = i * 100

            # BL Lacs
            bllac_spec = get_path_to_data("bllac.yml")
            bllac_param_server = PopsynthParams(bllac_spec)
            bllac_param_server.seed = seed
            bllac_param_server.file_name = self._file_name
            bllac_param_server.group_name = self._group_base_name + "_%i" % i

            self._bllac_param_servers.append(bllac_param_server)

            # FSRQs
            fsrq_spec = get_path_to_data("fsrq.yml")
            fsrq_param_server = PopsynthParams(fsrq_spec)
            fsrq_param_server.seed = seed
            fsrq_param_server.file_name = self._file_name
            fsrq_param_server.group_name = self._group_base_name + "_%i" % i

            self._fsrq_param_servers.append(fsrq_param_server)

            # Neutrinos
            nu_hese_spec = get_path_to_data("diffuse_hese_nu.yml")
            nu_ehe_spec = get_path_to_data("diffuse_ehe_nu.yml")
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

        return IceCubeAlertsWrapper(param_server)

    def _sim_wrapper(self, bllac_param_server, fsrq_param_server, nu_param_server):

        bllac_pop = self._bllac_pop_wrapper(bllac_param_server)

        fsrq_pop = self._fsrq_pop_wrapper(fsrq_param_server)

        nu_obs = self._nu_obs_wrapper(nu_param_server)

        result = self._coincidence_check(bllac_pop, fsrq_pop, nu_obs)

        del bllac_pop, fsrq_pop, nu_obs

        return result

    def _coincidence_check(self, bllac_pop, fsrq_pop, nu_obs):

        return BlazarNuCoincidence(bllac_pop, fsrq_pop, nu_obs)

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


def BlazarNuConnectedSim(Simulation):
    """
    Set up and run simulations of neutrinos produced
    by blazars.
    """

    def __init__(
        self,
        file_name="output/test_connected_sim.h5",
        group_base_name="survey",
        N=1,
    ):

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

            seed = i * 100

            # BL Lacs
            bllac_spec = get_path_to_data("bllac.yml")
            bllac_param_server = PopsynthParams(bllac_spec)
            bllac_param_server.seed = seed
            bllac_param_server.file_name = self._file_name
            bllac_param_server.group_name = self._group_base_name + "_%i" % i

            self._bllac_param_servers.append(bllac_param_server)

            # FSRQs
            fsrq_spec = get_path_to_data("fsrq.yml")
            fsrq_param_server = PopsynthParams(fsrq_spec)
            fsrq_param_server.seed = seed
            fsrq_param_server.file_name = self._file_name
            fsrq_param_server.group_name = self._group_base_name + "_%i" % i

            self._fsrq_param_servers.append(fsrq_param_server)

            # Neutrinos
            nu_hese_spec = get_path_to_data("connected_hese_nu.yml")
            nu_ehe_spec = get_path_to_data("connected_ehe_nu.yml")
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

        return IceCubeAlertsWrapper(param_server)

    def _connected_nu_wrapper(self, nu_obs, bllac_pop, fsrq_pop):

        return BlazarNuConnection(nu_obs, bllac_pop, fsrq_pop)

import h5py
import numpy as np
from abc import ABCMeta, abstractmethod
from joblib import (
    register_parallel_backend,
    parallel_backend,
    Parallel,
    delayed,
)
from collections import OrderedDict

from icecube_tools.neutrino_calculator import NeutrinoCalculator

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
    IceCubeObsParams,
    IceCubeObsWrapper,
    IceCubeTracksWrapper,
    IceCubeAlertsParams,
    IceCubeAlertsWrapper,
    _get_point_source,
    _run_sim_for,
)
from cosmic_coincidence.simulation import Simulation
from cosmic_coincidence.utils.package_data import get_path_to_data
from cosmic_coincidence.utils.parallel import FileWritingBackend

register_parallel_backend("file_write", FileWritingBackend)

erg_to_GeV = 624.151


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
    ):

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
        )

        self._bllac_config = bllac_config
        self._fsrq_config = fsrq_config
        self._nu_config = nu_config
        self._nu_hese_config = nu_hese_config
        self._nu_ehe_config = nu_ehe_config

        self._bllac_param_servers = []
        self._fsrq_param_servers = []
        self._nu_param_servers = []

    def _setup_param_servers(self):

        self._bllac_param_servers = []
        self._fsrq_param_servers = []
        self._nu_param_servers = []

        for i in range(self._N):

            seed = i * 100

            # BL Lacs
            bllac_spec = get_path_to_data(self._bllac_config)
            bllac_param_server = PopsynthParams(bllac_spec)
            bllac_param_server.seed = seed
            bllac_param_server.file_name = self._file_name
            bllac_param_server.group_name = self._group_base_name + "_%i" % i

            self._bllac_param_servers.append(bllac_param_server)

            # FSRQs
            fsrq_spec = get_path_to_data(self._fsrq_config)
            fsrq_param_server = PopsynthParams(fsrq_spec)
            fsrq_param_server.seed = seed
            fsrq_param_server.file_name = self._file_name
            fsrq_param_server.group_name = self._group_base_name + "_%i" % i

            self._fsrq_param_servers.append(fsrq_param_server)

            # Neutrinos
            if self._nu_config is not None:

                nu_spec = get_path_to_data(self._nu_config)
                nu_param_server = IceCubeObsParams(nu_spec)

            else:

                nu_hese_spec = get_path_to_data(self._nu_hese_config)
                nu_ehe_spec = get_path_to_data(self._nu_ehe_config)
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


class BlazarNuCoincidenceSim(BlazarNuSim):
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
        bllac_config: str = None,
        fsrq_config: str = None,
        nu_config: str = None,
        nu_hese_config: str = None,
        nu_ehe_config: str = None,
    ):

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
            bllac_config=bllac_config,
            fsrq_config=fsrq_config,
            nu_config=nu_config,
            nu_hese_config=nu_hese_config,
            nu_ehe_config=nu_ehe_config,
        )

    def _blazar_nu_wrapper(self, bllac_pop, fsrq_pop, nu_obs):

        return BlazarNuCoincidence(bllac_pop, fsrq_pop, nu_obs)


class BlazarNuConnectedSim(BlazarNuSim):
    """
    Set up and run simulations of neutrinos produced
    by blazars.
    """

    def __init__(
        self,
        file_name="output/test_connected_sim.h5",
        group_base_name="survey",
        N=1,
        bllac_config: str = None,
        fsrq_config: str = None,
        nu_config: str = None,
        nu_hese_config: str = None,
        nu_ehe_config: str = None,
    ):

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
            bllac_config=bllac_config,
            fsrq_config=fsrq_config,
            nu_config=nu_config,
            nu_hese_config=nu_hese_config,
            nu_ehe_config=nu_ehe_config,
        )

    def _blazar_nu_wrapper(self, nu_obs, bllac_pop, fsrq_pop):

        return BlazarNuConnection(nu_obs, bllac_pop, fsrq_pop)


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


class BlazarNuCoincidence(BlazarNuAction):
    """
    Check for coincidences of interest.
    """

    def __init__(
        self,
        bllac_pop: PopsynthWrapper,
        fsrq_pop: PopsynthWrapper,
        nu_obs: IceCubeObsWrapper,
        name="blazar_nu_coincidence",
    ):

        self._bllac_coincidence = OrderedDict()

        self._fsrq_coincidence = OrderedDict()

        super().__init__(
            bllac_pop=bllac_pop,
            fsrq_pop=fsrq_pop,
            nu_obs=nu_obs,
            name=name,
        )

    def _run(self):

        self._check_spatial()

        self._check_temporal()

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


class BlazarNuConnection(BlazarNuAction):
    """
    Handle connected blazar and neutrino
    observations.
    """

    def __init__(
        self,
        bllac_pop: PopsynthWrapper,
        fsrq_pop: PopsynthWrapper,
        nu_obs: IceCubeObsWrapper,
        name="blazar_nu_connection",
    ):

        self._bllac_connection = OrderedDict()

        self._fsrq_connection = OrderedDict()

        super().__init__(
            bllac_pop=bllac_pop,
            fsrq_pop=fsrq_pop,
            nu_obs=nu_obs,
            name=name,
        )

    @property
    def bllac_connection(self):

        return self._bllac_connection

    @property
    def fsrq_connection(self):

        return self._fsrq_connection

    def _run(self):

        # Need to implement alerts case
        if isinstance(self._nu_obs, IceCubeAlertsWrapper):

            raise NotImplementedError()

        # Neutrino info
        nu_params = self._nu_obs._parameter_server
        Emin = nu_params.connection["lower_energy"]
        Emax = nu_params.connection["upper_energy"]
        Enorm = nu_params.connection["normalisation_energy"]
        flux_factor = nu_params.connection["flux_factor"]
        effective_area = self._nu_obs.detector.effective_area
        seed = nu_params.seed

        # Loop over BL Lacs
        survey = self._bllac_pop.survey
        N = len(survey.distances)

        self.bllac_connection["Nnu"] = np.zeros(N)
        self.bllac_connection["nu_Erecos"] = []
        self.bllac_connection["nu_ras"] = []
        self.bllac_connection["nu_decs"] = []
        self.bllac_connection["nu_ang_errs"] = []
        self.bllac_connection["nu_times"] = []
        self.bllac_connection["src_detected"] = []
        self.bllac_connection["src_flare"] = []

        for i in range(N):

            ra = np.deg2rad(survey.ra[i])
            dec = np.deg2rad(survey.dec[i])
            z = survey.distances[i]
            spectral_index = survey.spectral_index[i]

            # Calculate steady emission
            # Coming soon!

            # Calculate flared emission
            if survey.variability[i] and survey.flare_times[i].size > 0:

                # Loop over flares
                for time, duration, amp in zip(
                    survey.flare_times[i],
                    survey.flare_durations[i],
                    survey.flare_amplitudes[i],
                ):

                    L_flare = survey.luminosities_latent[i] * amp  # erg s^-1
                    L_flare = L_flare * erg_to_GeV  # GeV s^-1
                    L_flare_nu = L_flare * flux_factor  # Neutrinos

                    source = _get_point_source(
                        L_flare_nu,
                        spectral_index,
                        z,
                        ra,
                        dec,
                        Emin,
                        Emax,
                        Enorm,
                    )

                    # Calulate expected neutrino number per source
                    nu_calc = NeutrinoCalculator([source], effective_area)
                    Nnu_ex_flare = nu_calc(
                        time=duration, min_energy=Emin, max_energy=Emax
                    )[0]

                    # Sample actual number of neutrinos per flare
                    Nnu_flare = np.random.poisson(Nnu_ex_flare)
                    self.bllac_connection["Nnu"][i] += Nnu_flare

                    # Sample times of nu
                    if Nnu_flare > 0:
                        self.bllac_connection["nu_times"].extend(
                            np.random.uniform(time, time + duration, Nnu_flare)
                        )

            # Simulate neutrino observations
            if self.bllac_connection["Nnu"][i] > 0:

                sim = _run_sim_for(
                    self.bllac_connection["Nnu"][i],
                    spectral_index,
                    ra,
                    dec,
                    Emin,
                    Emax,
                    Enorm,
                    self._nu_obs.detector,
                    seed,
                )

                self.bllac_connection["nu_Erecos"].extend(sim.reco_energy)
                self.bllac_connection["nu_ras"].extend(sim.ra)
                self.bllac_connection["nu_decs"].extend(sim.dec)
                self.bllac_connection["nu_ang_errs"].extend(sim.ang_err)
                self.bllac_connection["src_detected"].extend(
                    np.repeat(survey.selection[i], self.bllac_connection["Nnu"][i])
                )
                self.bllac_connection["src_flare"].extend(
                    np.repeat(True, self.bllac_connection["Nnu"][i])
                )

        # Loop over FSRQs

        # Calculate steady emission

        # Calculate flared emission

        # Simulate neutrino observations

    def write(self):

        pass

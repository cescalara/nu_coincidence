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
from cosmic_coincidence.utils.package_data import get_path_to_config
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

        self._bllac_config = bllac_config
        self._fsrq_config = fsrq_config
        self._nu_config = nu_config
        self._nu_hese_config = nu_hese_config
        self._nu_ehe_config = nu_ehe_config

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

            seed = i * 100

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

        (
            n_match_variable,
            n_match_flaring,
            matched_flare_amplitudes,
        ) = check_temporal_coincidence(
            observation.times,
            self.bllac_coincidence["spatial_match_inds"],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
            survey.flare_amplitudes[survey.selection],
        )

        self.bllac_coincidence["n_variable"] = n_match_variable
        self.bllac_coincidence["n_flaring"] = n_match_flaring
        self.bllac_coincidence["matched_flare_amplitudes"] = matched_flare_amplitudes

        # FSRQs
        survey = self._fsrq_pop.survey

        (
            n_match_variable,
            n_match_flaring,
            matched_flare_amplitudes,
        ) = check_temporal_coincidence(
            observation.times,
            self.fsrq_coincidence["spatial_match_inds"],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
            survey.flare_amplitudes[survey.selection],
        )

        self.fsrq_coincidence["n_variable"] = n_match_variable
        self.fsrq_coincidence["n_flaring"] = n_match_flaring
        self.fsrq_coincidence["matched_flare_amplitudes"] = matched_flare_amplitudes


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
        flare_only=False,
    ):

        self._bllac_connection = OrderedDict()

        self._fsrq_connection = OrderedDict()

        self._flare_only = flare_only

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

        self._initialise(self._bllac_connection)
        self._initialise(self._fsrq_connection)

        if isinstance(self._nu_obs, IceCubeAlertsWrapper):

            hese_nu_params = self._nu_obs._parameter_server.hese
            ehe_nu_params = self._nu_obs._parameter_server.ehe

            hese_nu_detector = self._nu_obs.hese_detector
            ehe_nu_detector = self._nu_obs.ehe_detector

            # BL Lacs
            self._connected_sim(
                self._bllac_pop,
                hese_nu_params,
                hese_nu_detector,
                self._bllac_connection,
            )
            self._connected_sim(
                self._bllac_pop,
                ehe_nu_params,
                ehe_nu_detector,
                self._bllac_connection,
            )

            # FSRQs
            self._connected_sim(
                self._fsrq_pop,
                hese_nu_params,
                hese_nu_detector,
                self._fsrq_connection,
            )
            self._connected_sim(
                self._fsrq_pop,
                ehe_nu_params,
                ehe_nu_detector,
                self._fsrq_connection,
            )

        else:

            nu_params = self._nu_obs._parameter_server
            nu_detector = self._nu_obs.detector

            # BL Lacs
            self._connected_sim(
                self._bllac_pop, nu_params, nu_detector, self._bllac_connection
            )

            # FSRQs
            self._connected_sim(
                self._fsrq_pop, nu_params, nu_detector, self._fsrq_connection
            )

    def _initialise(self, connection):

        connection["nu_Erecos"] = np.array([])
        connection["nu_ras"] = np.array([])
        connection["nu_decs"] = np.array([])
        connection["nu_ang_errs"] = np.array([])
        connection["nu_times"] = np.array([])
        connection["src_detected"] = np.array([])
        connection["src_flare"] = np.array([])
        connection["src_id"] = np.array([], dtype=np.int64)

    def _connected_sim(self, pop, nu_params, nu_detector, connection):
        """
        Run a connected sim for pop and store
        the results in connection.
        """

        # Neutrino info
        Emin = nu_params.connection["lower_energy"]
        Emax = nu_params.connection["upper_energy"]
        Emin_sim = nu_params.connection["lower_energy_sim"]
        Emin_det = nu_params.detector["Emin_det"]
        Enorm = nu_params.connection["normalisation_energy"]
        flux_factor = nu_params.connection["flux_factor"]
        flavour_factor = nu_params.detector["flavour_factor"]
        effective_area = nu_detector.effective_area
        seed = nu_params.seed

        np.random.seed(seed)

        survey = pop.survey
        N = len(survey.distances)

        connection["Nnu_steady"] = np.zeros(N)
        connection["Nnu_ex_steady"] = np.zeros(N)
        connection["Nnu_flare"] = np.zeros(N)
        connection["Nnu_ex_flare"] = np.zeros(N)

        for i in range(N):

            ra = np.deg2rad(survey.ra[i])
            dec = np.deg2rad(survey.dec[i])
            z = survey.distances[i]
            spectral_index = survey.spectral_index[i]

            # Calculate steady emission
            L_steady = survey.luminosities_latent[i]  # erg s^-1 [0.1 - 100 GeV]
            # L_steady = _convert_energy_range(
            #    L_steady, spectral_index, 0.1, 100, 1, 100
            # )  # erg s^-1 [1 - 100 GeV]
            L_steady = L_steady * erg_to_GeV  # GeV s^-1
            L_steady = L_steady * flux_factor  # Neutrinos
            L_steady = L_steady * flavour_factor  # Detected flavours

            source = _get_point_source(
                L_steady,
                spectral_index,
                z,
                ra,
                dec,
                Emin,
                Emax,
                Enorm,
            )

            # Time spent not flaring
            total_duration = nu_params.detector["obs_time"]
            steady_duration = total_duration - sum(survey.flare_durations[i])

            nu_calc = NeutrinoCalculator([source], effective_area)
            Nnu_ex_steady = nu_calc(
                time=steady_duration,
                min_energy=Emin_sim,
                max_energy=Emax,
            )[0]
            connection["Nnu_ex_steady"][i] += Nnu_ex_steady
            Nnu_steady = np.random.poisson(Nnu_ex_steady)
            connection["Nnu_steady"][i] += Nnu_steady

            if Nnu_steady > 0 and not self._flare_only:

                # TODO: remove flare periods
                if steady_duration < total_duration:

                    connection["nu_times"] = np.append(
                        connection["nu_times"],
                        np.random.uniform(0, total_duration, Nnu_steady),
                    )

                else:

                    connection["nu_times"] = np.append(
                        connection["nu_times"],
                        np.random.uniform(0, total_duration, Nnu_steady),
                    )

                sim = _run_sim_for(
                    connection["Nnu_steady"][i],
                    spectral_index,
                    z,
                    ra,
                    dec,
                    Emin_sim,
                    Emax,
                    Enorm,
                    nu_detector,
                    seed,
                )

                connection["nu_Erecos"] = np.append(
                    connection["nu_Erecos"], sim.reco_energy
                )
                connection["nu_ras"] = np.append(connection["nu_ras"], sim.ra)
                connection["nu_decs"] = np.append(connection["nu_decs"], sim.dec)
                connection["nu_ang_errs"] = np.append(
                    connection["nu_ang_errs"], sim.ang_err
                )
                connection["src_detected"] = np.append(
                    connection["src_detected"],
                    np.repeat(survey.selection[i], connection["Nnu_steady"][i]),
                )
                connection["src_flare"] = np.append(
                    connection["src_flare"],
                    np.repeat(False, connection["Nnu_steady"][i]),
                )
                connection["src_id"] = np.append(
                    connection["src_id"], np.repeat(i, connection["Nnu_steady"][i])
                )

            # Calculate flared emission
            if survey.variability[i] and survey.flare_times[i].size > 0:

                # Loop over flares
                for time, duration, amp in zip(
                    survey.flare_times[i],
                    survey.flare_durations[i],
                    survey.flare_amplitudes[i],
                ):

                    L_flare = (
                        survey.luminosities_latent[i] * amp
                    )  # erg s^-1 [0.1 - 100 GeV]
                    # L_flare = _convert_energy_range(
                    #   L_flare, spectral_index, 0.1, 100, 1, 100
                    # )  # erg s^-1 [1 - 100 GeV]
                    L_flare = L_flare * erg_to_GeV  # GeV s^-1
                    L_flare_nu = L_flare * flux_factor  # Neutrinos
                    L_flare_nu = L_flare_nu * flavour_factor  # Detected flavours

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
                        time=duration, min_energy=Emin_sim, max_energy=Emax
                    )[0]

                    # Sample actual number of neutrinos per flare
                    Nnu_flare = np.random.poisson(Nnu_ex_flare)
                    connection["Nnu_ex_flare"][i] += Nnu_ex_flare
                    connection["Nnu_flare"][i] += Nnu_flare

                    # Sample times of nu
                    if Nnu_flare > 0:
                        connection["nu_times"] = np.append(
                            connection["nu_times"],
                            np.random.uniform(time, time + duration, Nnu_flare),
                        )

            # Simulate neutrino observations
            if connection["Nnu_flare"][i] > 0:

                sim = _run_sim_for(
                    connection["Nnu_flare"][i],
                    spectral_index,
                    z,
                    ra,
                    dec,
                    Emin_sim,
                    Emax,
                    Enorm,
                    nu_detector,
                    seed,
                )

                connection["nu_Erecos"] = np.append(
                    connection["nu_Erecos"], sim.reco_energy
                )
                connection["nu_ras"] = np.append(connection["nu_ras"], sim.ra)
                connection["nu_decs"] = np.append(connection["nu_decs"], sim.dec)
                connection["nu_ang_errs"] = np.append(
                    connection["nu_ang_errs"], sim.ang_err
                )
                connection["src_detected"] = np.append(
                    connection["src_detected"],
                    np.repeat(survey.selection[i], connection["Nnu_flare"][i]),
                )
                connection["src_flare"] = np.append(
                    connection["src_flare"],
                    np.repeat(True, connection["Nnu_flare"][i]),
                )
                connection["src_id"] = np.append(
                    connection["src_id"], np.repeat(i, connection["Nnu_flare"][i])
                )

            # Select above Emin_det
            selection = connection["nu_Erecos"] > Emin_det

            connection["nu_Erecos"] = connection["nu_Erecos"][selection]
            connection["nu_ras"] = connection["nu_ras"][selection]
            connection["nu_decs"] = connection["nu_decs"][selection]
            connection["nu_ang_errs"] = connection["nu_ang_errs"][selection]
            connection["nu_times"] = connection["nu_times"][selection]
            connection["src_detected"] = connection["src_detected"][selection]
            connection["src_flare"] = connection["src_flare"][selection]
            connection["src_id"] = connection["src_id"][selection]

    def write(self):

        with h5py.File(self._file_name, "r+") as f:

            if self._group_name not in f.keys():

                group = f.create_group(self._group_name)

            else:

                group = f[self._group_name]

            subgroup = group.create_group(self.name)

            bllac_group = subgroup.create_group("bllac")

            for key, value in self.bllac_connection.items():

                bllac_group.create_dataset(key, data=value)

            fsrq_group = subgroup.create_group("fsrq")

            for key, value in self.fsrq_connection.items():

                fsrq_group.create_dataset(key, data=value)


def _convert_energy_range(luminosity, spectral_index, Emin, Emax, new_Emin, new_Emax):
    """
    Convert value of luminosity to be defined
    over a different energy range. The units of all
    energy quantities must be consient. Assumes a
    power-law spectrum.

    :param luminosity: L in erg s^-1
    :param Emin: Current Emin
    :param Emax: Current Emax
    :param new_Emin: New Emin
    :param new_Emax: New Emax
    """

    if spectral_index == 2:

        numerator = np.log(new_Emax / new_Emin)

        denominator = np.log(Emax / Emin)

    else:

        numerator = np.power(new_Emin, 2 - spectral_index) - np.power(
            new_Emax, 2 - spectral_index
        )

        denominator = np.power(Emin, 2 - spectral_index) - np.power(
            Emax, 2 - spectral_index
        )

    return luminosity * (numerator / denominator)

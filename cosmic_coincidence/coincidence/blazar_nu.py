import os
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
from typing import List

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
        seed=1000,
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
            seed=seed,
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
        seed=1000,
        flux_factor: float = None,
        flare_only: bool = False,
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
            seed=seed,
        )

        # overwrite flux_factor if provided
        if flux_factor:
            for i in range(self._N):
                self._nu_param_servers[i].hese.connection["flux_factor"] = flux_factor
                self._nu_param_servers[i].ehe.connection["flux_factor"] = flux_factor

        # store choice for flare_only
        self._flare_only = flare_only

    def _blazar_nu_wrapper(self, nu_obs, bllac_pop, fsrq_pop):

        return BlazarNuConnection(
            nu_obs, bllac_pop, fsrq_pop, flare_only=self._flare_only
        )


class BlazarNuCoincidenceResults(object):
    """
    Load results from BlazarNuCoincidenceSim.
    """

    def __init__(self, file_name_list: List[str]):

        self._file_name_list = file_name_list

        self.bllac = OrderedDict()
        self.bllac["n_spatial"] = np.array([])
        self.bllac["n_variable"] = np.array([])
        self.bllac["n_flaring"] = np.array([])
        self.bllac["matched_flare_amplitudes"] = np.array([])

        self.fsrq = OrderedDict()
        self.fsrq["n_spatial"] = np.array([])
        self.fsrq["n_variable"] = np.array([])
        self.fsrq["n_flaring"] = np.array([])
        self.fsrq["matched_flare_amplitudes"] = np.array([])

        self.N = 0

        for file_name in self._file_name_list:

            self._load_from_h5(file_name)

    def _load_from_h5(self, file_name):

        with h5py.File(file_name, "r") as f:

            N_f = f.attrs["N"]

            bllac_n_spatial_f = np.zeros(N_f)
            bllac_n_variable_f = np.zeros(N_f)
            bllac_n_flaring_f = np.zeros(N_f)

            fsrq_n_spatial_f = np.zeros(N_f)
            fsrq_n_variable_f = np.zeros(N_f)
            fsrq_n_flaring_f = np.zeros(N_f)

            for i in range(N_f):

                bllac_group = f["survey_%i/blazar_nu_coincidence/bllac" % i]
                bllac_n_spatial_f[i] = bllac_group["n_spatial"][()]
                bllac_n_variable_f[i] = bllac_group["n_variable"][()]
                bllac_n_flaring_f[i] = bllac_group["n_flaring"][()]

                if bllac_n_flaring_f[i] >= 1:
                    bllac_flare_amps_i = bllac_group["matched_flare_amplitudes"][()]
                    self.bllac["matched_flare_amplitudes"] = np.append(
                        self.bllac["matched_flare_amplitudes"], bllac_flare_amps_i
                    )

                fsrq_group = f["survey_%i/blazar_nu_coincidence/fsrq" % i]
                fsrq_n_spatial_f[i] = fsrq_group["n_spatial"][()]
                fsrq_n_variable_f[i] = fsrq_group["n_variable"][()]
                fsrq_n_flaring_f[i] = fsrq_group["n_flaring"][()]

                if fsrq_n_flaring_f[i] >= 1:
                    fsrq_flare_amps_i = fsrq_group["matched_flare_amplitudes"][()]
                    self.fsrq["matched_flare_amplitudes"] = np.append(
                        self.fsrq["matched_flare_amplitudes"], fsrq_flare_amps_i
                    )

        self.bllac["n_spatial"] = np.append(self.bllac["n_spatial"], bllac_n_spatial_f)
        self.bllac["n_variable"] = np.append(
            self.bllac["n_variable"], bllac_n_variable_f
        )
        self.bllac["n_flaring"] = np.append(self.bllac["n_flaring"], bllac_n_flaring_f)

        self.fsrq["n_spatial"] = np.append(self.fsrq["n_spatial"], fsrq_n_spatial_f)
        self.fsrq["n_variable"] = np.append(self.fsrq["n_variable"], fsrq_n_variable_f)
        self.fsrq["n_flaring"] = np.append(self.fsrq["n_flaring"], fsrq_n_flaring_f)

        self.N += N_f

    @classmethod
    def load(cls, file_name_list: List[str]):

        return cls(file_name_list)


class BlazarNuConnectedResults(object):
    """
    Handle results from BlazarNuConnectedSim.
    """

    def __init__(self, file_name_list: List[str]):

        self._file_name_list = file_name_list

        self.bllac = OrderedDict()
        self.fsrq = OrderedDict()

        self._file_keys = ["n_alerts", "n_alerts_flare", "n_multi", "n_multi_flare"]

    def merge_over_flux_factor(self, flux_factors, write_to: str = None, delete=False):

        bllac_results = {}
        fsrq_results = {}

        for key in self._file_keys:

            bllac_results[key] = []
            bllac_results[key + "_tmp"] = []
            fsrq_results[key] = []
            fsrq_results[key + "_tmp"] = []

        for flux_factor, sub_file_name in zip(flux_factors, self._file_name_list):

            with h5py.File(sub_file_name, "r") as sf:

                N_f = sf.attrs["N"]

                for key in self._file_keys:
                    bllac_results[key + "_tmp"] = []
                    fsrq_results[key + "_tmp"] = []

                for i in range(N_f):

                    survey = sf["survey_%i/blazar_nu_connection" % i]
                    bllac_group = survey["bllac"]
                    fsrq_group = survey["fsrq"]

                    for key in self._file_keys:
                        bllac_results[key + "_tmp"].append(bllac_group[key][()])
                        fsrq_results[key + "_tmp"].append(fsrq_group[key][()])

            for key in self._file_keys:
                bllac_results[key].append(bllac_results[key + "_tmp"])
                fsrq_results[key].append(fsrq_results[key + "_tmp"])

        # write to single file
        if write_to:

            with h5py.File(write_to, "w") as f:

                f.create_dataset("flux_factors", data=flux_factors)

                bllac_group = f.create_group("bllac")
                fsrq_group = f.create_group("fsrq")

                for key in self._file_keys:
                    bllac_group.create_dataset(key, data=bllac_results[key])
                    fsrq_group.create_dataset(key, data=fsrq_results[key])

        # delete consolidated files
        if delete:

            for file_name in self._file_name_list:

                os.remove(file_name)


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

            # check flux_factor is equal
            if (
                hese_nu_params.connection["flux_factor"]
                != ehe_nu_params.connection["flux_factor"]
            ):
                raise ValueError(
                    "Flux factor not equal between HESE and EHE connections."
                )

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

            # For alternate energy range
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

                    # alternate energy range
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

            subgroup.create_dataset(
                "flux_factor",
                data=self._nu_obs._parameter_server.hese.connection["flux_factor"],
            )

            bllac_group = subgroup.create_group("bllac")

            # reduced info
            bllac_flare_sel = self.bllac_connection["src_flare"].astype(bool)
            bllac_group.create_dataset(
                "n_alerts", data=len(self.bllac_connection["nu_ras"])
            )
            bllac_group.create_dataset(
                "n_alerts_flare",
                data=len(self.bllac_connection["nu_ras"][bllac_flare_sel]),
            )
            unique, counts = np.unique(
                self.bllac_connection["src_id"], return_counts=True
            )
            bllac_group.create_dataset("n_multi", data=len(counts[counts > 1]))
            unique, counts = np.unique(
                self.bllac_connection["src_id"][bllac_flare_sel], return_counts=True
            )
            bllac_group.create_dataset("n_multi_flare", data=len(counts[counts > 1]))

            # for key, value in self.bllac_connection.items():

            #     bllac_group.create_dataset(key, data=value)

            fsrq_group = subgroup.create_group("fsrq")

            # redcued info
            fsrq_flare_sel = self.fsrq_connection["src_flare"].astype(bool)
            fsrq_group.create_dataset(
                "n_alerts", data=len(self.fsrq_connection["nu_ras"])
            )
            fsrq_group.create_dataset(
                "n_alerts_flare",
                data=len(self.fsrq_connection["nu_ras"][fsrq_flare_sel]),
            )
            unique, counts = np.unique(
                self.fsrq_connection["src_id"], return_counts=True
            )
            fsrq_group.create_dataset("n_multi", data=len(counts[counts > 1]))
            unique, counts = np.unique(
                self.fsrq_connection["src_id"][fsrq_flare_sel], return_counts=True
            )
            fsrq_group.create_dataset("n_multi_flare", data=len(counts[counts > 1]))

            # for key, value in self.fsrq_connection.items():

            #     fsrq_group.create_dataset(key, data=value)


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

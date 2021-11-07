import os
import h5py
import numpy as np
from collections import OrderedDict
from typing import List
from numpy.typing import ArrayLike
from astropy import units as u

from icecube_tools.neutrino_calculator import NeutrinoCalculator

from cosmic_coincidence.populations.popsynth_wrapper import PopsynthWrapper
from cosmic_coincidence.neutrinos.icecube import (
    IceCubeObsWrapper,
    IceCubeAlertsWrapper,
    _get_point_source,
    _run_sim_for,
)
from cosmic_coincidence.simulation import Results
from cosmic_coincidence.blazar_nu.base import BlazarNuSim, BlazarNuAction

erg_to_GeV = (1 * u.erg).to(u.GeV).value


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
        flux_factors: ArrayLike = None,
        flare_only: bool = False,
        det_only: bool = False,
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

        # for fixed flux factors
        if flux_factor:

            for i in range(self._N):

                self._nu_param_servers[i].hese.connection["flux_factor"] = flux_factor
                self._nu_param_servers[i].ehe.connection["flux_factor"] = flux_factor

        # for individual flux factors
        if flux_factors and len(flux_factors) == self._N:

            for i, ff in enumerate(flux_factors):

                self._nu_param_servers[i].hese.connection["flux_factor"] = ff
                self._nu_param_servers[i].ehe.connection["flux_factor"] = ff

        elif flux_factors:

            raise ValueError("Length of flux_factors must equal input N")

        # store choice for flare_only
        self._flare_only = flare_only

        # store choice for det_only
        self._det_only = det_only

    def _blazar_nu_wrapper(self, nu_obs, bllac_pop, fsrq_pop):

        return BlazarNuConnection(
            nu_obs,
            bllac_pop,
            fsrq_pop,
            flare_only=self._flare_only,
            det_only=self._det_only,
        )


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
        flare_only: bool = False,
        det_only: bool = False,
    ):

        self._bllac_connection = OrderedDict()

        self._fsrq_connection = OrderedDict()

        self._flare_only = flare_only

        self._det_only = det_only

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
            fsrq_group = subgroup.create_group("fsrq")

            if self._det_only:

                bllac_det_sel = self.bllac_connection["src_detected"].astype(bool)
                fsrq_det_sel = self.fsrq_connection["src_detected"].astype(bool)

            else:

                bllac_det_sel = np.tile(True, len(self.bllac_connection["nu_ras"]))
                fsrq_det_sel = np.tile(True, len(self.fsrq_connection["nu_ras"]))

            bllac_flare_sel = self.bllac_connection["src_flare"].astype(bool)
            fsrq_flare_sel = self.fsrq_connection["src_flare"].astype(bool)

            bllac_both_sel = bllac_det_sel & bllac_flare_sel
            fsrq_both_sel = fsrq_det_sel & fsrq_flare_sel

            # BL Lac
            bllac_group.create_dataset(
                "n_alerts", data=len(self.bllac_connection["nu_ras"][bllac_det_sel])
            )
            bllac_group.create_dataset(
                "n_alerts_flare",
                data=len(self.bllac_connection["nu_ras"][bllac_both_sel]),
            )
            unique, counts = np.unique(
                self.bllac_connection["src_id"][bllac_det_sel], return_counts=True
            )
            bllac_group.create_dataset("n_multi", data=len(counts[counts > 1]))
            unique, counts = np.unique(
                self.bllac_connection["src_id"][bllac_both_sel], return_counts=True
            )
            bllac_group.create_dataset("n_multi_flare", data=len(counts[counts > 1]))

            # FSRQ
            fsrq_group.create_dataset(
                "n_alerts", data=len(self.fsrq_connection["nu_ras"][fsrq_det_sel])
            )
            fsrq_group.create_dataset(
                "n_alerts_flare",
                data=len(self.fsrq_connection["nu_ras"][fsrq_both_sel]),
            )
            unique, counts = np.unique(
                self.fsrq_connection["src_id"][fsrq_det_sel], return_counts=True
            )
            fsrq_group.create_dataset("n_multi", data=len(counts[counts > 1]))
            unique, counts = np.unique(
                self.fsrq_connection["src_id"][fsrq_both_sel], return_counts=True
            )
            fsrq_group.create_dataset("n_multi_flare", data=len(counts[counts > 1]))


class BlazarNuConnectedResults(Results):
    """
    Handle results from BlazarNuConnectedSim.
    """

    def __init__(
        self,
        file_name_list: List[str],
        append_flux_factors: bool = False,
    ):

        self._append_flux_factors = append_flux_factors

        super().__init__(file_name_list=file_name_list)

    def _setup(self):

        self._file_keys = ["n_alerts", "n_alerts_flare", "n_multi", "n_multi_flare"]

        self._bllac = OrderedDict()
        self._fsrq = OrderedDict()

        if self._append_flux_factors:

            flux_factors = []
            for file_name in self._file_name_list:

                with h5py.File(file_name, "r") as f:

                    flux_factors.extend(f["flux_factors"][()])

            self.flux_factors = flux_factors

            for key in self._file_keys:

                self._bllac[key] = []
                self._fsrq[key] = []

        else:

            # check flux_factors are equal across files
            flux_factors = []
            for file_name in self._file_name_list:

                with h5py.File(file_name, "r") as f:

                    flux_factors.append(f["flux_factors"][()])

            if not all(ff.all() == flux_factors[0].all() for ff in flux_factors):

                raise ValueError("Flux factors are not equal across files")

            self.flux_factors = flux_factors[0]

            for key in self._file_keys:

                self._bllac[key] = [[] for _ in self.flux_factors]
                self._fsrq[key] = [[] for _ in self.flux_factors]

    def _load_from_h5(self, file_name):

        with h5py.File(file_name, "r") as f:

            bllac_group = f["bllac"]
            fsrq_group = f["fsrq"]

            for blazar, group in zip(
                [self._bllac, self._fsrq],
                [bllac_group, fsrq_group],
            ):

                for key in self._file_keys:

                    if self._append_flux_factors:
                        blazar[key].extend(group[key][()])

                    else:

                        for i in range(len(self.flux_factors)):
                            blazar[key][i].extend(group[key][()][i])

            if self._append_flux_factors:

                self.N += len(group[key][()])

            else:

                self.N += len(group[key][()][0])

    @property
    def bllac(self):

        for key in self._file_keys:
            self._bllac[key] = np.array(self._bllac[key])

        return self._bllac

    @property
    def fsrq(self):

        for key in self._file_keys:
            self._fsrq[key] = np.array(self._fsrq[key])

        return self._fsrq

    @staticmethod
    def merge_over_flux_factor(
        sub_file_names: List[str],
        flux_factors,
        write_to: str = None,
        delete=False,
    ):
        _file_keys = ["n_alerts", "n_alerts_flare", "n_multi", "n_multi_flare"]

        bllac_results = {}
        fsrq_results = {}

        for key in _file_keys:

            bllac_results[key] = []
            bllac_results[key + "_tmp"] = []
            fsrq_results[key] = []
            fsrq_results[key + "_tmp"] = []

        for flux_factor, sub_file_name in zip(flux_factors, sub_file_names):

            with h5py.File(sub_file_name, "r") as sf:

                N_f = sf.attrs["N"]

                for key in _file_keys:
                    bllac_results[key + "_tmp"] = []
                    fsrq_results[key + "_tmp"] = []

                for i in range(N_f):

                    try:

                        # look for survey
                        survey = sf["survey_%i/blazar_nu_connection" % i]
                        bllac_group = survey["bllac"]
                        fsrq_group = survey["fsrq"]

                        for key in _file_keys:
                            bllac_results[key + "_tmp"].append(bllac_group[key][()])
                            fsrq_results[key + "_tmp"].append(fsrq_group[key][()])

                    except KeyError:

                        # write nan if no survey found
                        for key in _file_keys:
                            bllac_results[key + "_tmp"].append(np.nan)
                            fsrq_results[key + "_tmp"].append(np.nan)

            for key in _file_keys:
                bllac_results[key].append(bllac_results[key + "_tmp"])
                fsrq_results[key].append(fsrq_results[key + "_tmp"])

        # write to single file
        if write_to:

            with h5py.File(write_to, "w") as f:

                f.create_dataset("flux_factors", data=flux_factors)

                bllac_group = f.create_group("bllac")
                fsrq_group = f.create_group("fsrq")

                for key in _file_keys:
                    bllac_group.create_dataset(key, data=bllac_results[key])
                    fsrq_group.create_dataset(key, data=fsrq_results[key])

        # delete consolidated files
        if delete:

            for file_name in sub_file_names:

                os.remove(file_name)

    @staticmethod
    def reorganise_file_structure(
        file_name: str,
        write_to: str = None,
        delete=False,
    ):

        _file_keys = ["n_alerts", "n_alerts_flare", "n_multi", "n_multi_flare"]

        bllac_results = {}
        fsrq_results = {}
        flux_factors = []

        for key in _file_keys:

            bllac_results[key] = []
            fsrq_results[key] = []

        with h5py.File(file_name, "r") as f:

            N_f = f.attrs["N"]

            for i in range(N_f):

                try:

                    # look for survey
                    survey = f["survey_%i/blazar_nu_connection" % i]
                    bllac_group = survey["bllac"]
                    fsrq_group = survey["fsrq"]

                    flux_factor = survey["flux_factor"][()]
                    flux_factors.append(flux_factor)

                    for key in _file_keys:
                        bllac_results[key].append(bllac_group[key][()])
                        fsrq_results[key].append(fsrq_group[key][()])

                except KeyError:

                    # write nan if no survey found
                    flux_factors.append(np.nan)

                    for key in _file_keys:
                        bllac_results[key].append(np.nan)
                        fsrq_results[key].append(np.nan)

        # write to new file
        if write_to:

            with h5py.File(write_to, "w") as f:

                f.create_dataset("flux_factors", data=flux_factors)

                bllac_group = f.create_group("bllac")
                fsrq_group = f.create_group("fsrq")

                for key in _file_keys:
                    bllac_group.create_dataset(key, data=bllac_results[key])
                    fsrq_group.create_dataset(key, data=fsrq_results[key])

        # delete consolidated files
        if delete:

            os.remove(file_name)


def _convert_energy_range(
    luminosity,
    spectral_index,
    Emin,
    Emax,
    new_Emin,
    new_Emax,
):
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

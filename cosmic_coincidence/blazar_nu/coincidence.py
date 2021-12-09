import numpy as np
import h5py
from typing import List
from collections import OrderedDict

from cosmic_coincidence.simulation import Results
from cosmic_coincidence.populations.popsynth_wrapper import PopsynthWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeObsWrapper
from cosmic_coincidence.coincidence import (
    check_spatial_coincidence,
    check_temporal_coincidence,
)
from cosmic_coincidence.blazar_nu.base import BlazarNuSim, BlazarNuAction


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

        (
            n_match_spatial,
            n_match_spatial_astro,
            spatial_match_inds,
        ) = check_spatial_coincidence(
            np.deg2rad(observation.ra),
            np.deg2rad(observation.dec),
            np.deg2rad(observation.ang_err),
            observation.source_label,
            np.deg2rad(survey.ra[survey.selection]),
            np.deg2rad(survey.dec[survey.selection]),
        )

        self.bllac_coincidence["n_spatial"] = n_match_spatial
        self.bllac_coincidence["n_spatial_astro"] = n_match_spatial_astro
        self.bllac_coincidence["spatial_match_inds"] = spatial_match_inds

        # FSRQs
        survey = self._fsrq_pop.survey

        (
            n_match_spatial,
            n_match_spatial_astro,
            spatial_match_inds,
        ) = check_spatial_coincidence(
            np.deg2rad(observation.ra),
            np.deg2rad(observation.dec),
            np.deg2rad(observation.ang_err),
            observation.source_label,
            np.deg2rad(survey.ra[survey.selection]),
            np.deg2rad(survey.dec[survey.selection]),
        )

        self.fsrq_coincidence["n_spatial"] = n_match_spatial
        self.fsrq_coincidence["n_spatial_astro"] = n_match_spatial_astro
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
            n_match_variable_astro,
            n_match_flaring,
            n_match_flaring_astro,
            matched_flare_amplitudes,
            matched_src_ras,
            matched_src_decs,
            matched_nu_ras,
            matched_nu_decs,
            matched_nu_ang_errs,
        ) = check_temporal_coincidence(
            observation.times,
            observation.source_label,
            observation.ra,
            observation.dec,
            observation.ang_err,
            self.bllac_coincidence["spatial_match_inds"],
            survey.ra[survey.selection],
            survey.dec[survey.selection],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
            survey.flare_amplitudes[survey.selection],
        )

        self.bllac_coincidence["n_variable"] = n_match_variable
        self.bllac_coincidence["n_variable_astro"] = n_match_variable_astro
        self.bllac_coincidence["n_flaring"] = n_match_flaring
        self.bllac_coincidence["n_flaring_astro"] = n_match_flaring_astro
        self.bllac_coincidence["matched_flare_amplitudes"] = matched_flare_amplitudes
        self.bllac_coincidence["matched_src_ras"] = matched_src_ras
        self.bllac_coincidence["matched_src_decs"] = matched_src_decs
        self.bllac_coincidence["matched_nu_ras"] = matched_nu_ras
        self.bllac_coincidence["matched_nu_decs"] = matched_nu_decs
        self.bllac_coincidence["matched_nu_ang_errs"] = matched_nu_ang_errs

        # FSRQs
        survey = self._fsrq_pop.survey

        (
            n_match_variable,
            n_match_variable_astro,
            n_match_flaring,
            n_match_flaring_astro,
            matched_flare_amplitudes,
            matched_src_ras,
            matched_src_decs,
            matched_nu_ras,
            matched_nu_decs,
            matched_nu_ang_errs,
        ) = check_temporal_coincidence(
            observation.times,
            observation.source_label,
            observation.ra,
            observation.dec,
            observation.ang_err,
            self.fsrq_coincidence["spatial_match_inds"],
            survey.ra[survey.selection],
            survey.dec[survey.selection],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
            survey.flare_amplitudes[survey.selection],
        )

        self.fsrq_coincidence["n_variable"] = n_match_variable
        self.fsrq_coincidence["n_variable_astro"] = n_match_variable_astro
        self.fsrq_coincidence["n_flaring"] = n_match_flaring
        self.fsrq_coincidence["n_flaring_astro"] = n_match_flaring_astro
        self.fsrq_coincidence["matched_flare_amplitudes"] = matched_flare_amplitudes
        self.fsrq_coincidence["matched_src_ras"] = matched_src_ras
        self.fsrq_coincidence["matched_src_decs"] = matched_src_decs
        self.fsrq_coincidence["matched_nu_ras"] = matched_nu_ras
        self.fsrq_coincidence["matched_nu_decs"] = matched_nu_decs
        self.fsrq_coincidence["matched_nu_ang_errs"] = matched_nu_ang_errs


class BlazarNuCoincidenceResults(Results):
    """
    Load results from BlazarNuCoincidenceSim.
    """

    def __init__(self, file_name_list: List[str]):

        self._file_keys = [
            "n_spatial",
            "n_spatial_astro",
            "n_variable",
            "n_variable_astro",
            "n_flaring",
            "n_flaring_astro",
            "matched_flare_amplitudes",
            "matched_src_ras",
            "matched_src_decs",
            "matched_nu_ras",
            "matched_nu_decs",
            "matched_nu_ang_errs",
        ]

        super().__init__(file_name_list=file_name_list)

    def _setup(self):

        self.bllac = OrderedDict()
        self.fsrq = OrderedDict()

        for key in self._file_keys:

            self.bllac[key] = np.array([])

            self.fsrq[key] = np.array([])

    def _load_from_h5(self, file_name):

        with h5py.File(file_name, "r") as f:

            N_f = f.attrs["N"]

            bllac_f = {}
            fsrq_f = {}

            for key in self._file_keys:

                if "matched" not in key:

                    bllac_f[key] = np.zeros(N_f)
                    fsrq_f[key] = np.zeros(N_f)

            for i in range(N_f):

                bllac_group = f["survey_%i/blazar_nu_coincidence/bllac" % i]
                fsrq_group = f["survey_%i/blazar_nu_coincidence/fsrq" % i]

                for key in self._file_keys:

                    if "matched" not in key:

                        bllac_f[key][i] = bllac_group[key][()]
                        fsrq_f[key][i] = fsrq_group[key][()]

                    else:

                        if bllac_f["n_flaring"][i] >= 1:
                            bllac_match_i = bllac_group[key][()]
                            self.bllac[key] = np.append(self.bllac[key], bllac_match_i)

                        if fsrq_f["n_flaring"][i] >= 1:
                            fsrq_match_i = bllac_group[key][()]
                            self.fsrq[key] = np.append(self.fsrq[key], fsrq_match_i)

        for key in self._file_keys:

            if "matched" not in key:

                self.bllac[key] = np.append(self.bllac[key], bllac_f[key])
                self.fsrq[key] = np.append(self.fsrq[key], fsrq_f[key])

        self.N += N_f

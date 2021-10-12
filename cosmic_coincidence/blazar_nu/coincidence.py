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


class BlazarNuCoincidenceResults(Results):
    """
    Load results from BlazarNuCoincidenceSim.
    """

    def __init__(self, file_name_list: List[str]):

        super().__init__(file_name_list=file_name_list)

    def _setup(self):

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

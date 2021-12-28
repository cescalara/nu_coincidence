import numpy as np
import h5py
from typing import List
from collections import OrderedDict

from nu_coincidence.simulation import Results
from nu_coincidence.populations.popsynth_wrapper import PopsynthWrapper
from nu_coincidence.neutrinos.icecube import IceCubeObsWrapper
from nu_coincidence.coincidence import (
    count_spatial_coincidence,
    check_temporal_coincidence,
)
from nu_coincidence.pop_nu.base import PopNuSim, PopNuAction


class PopNuCoincidenceSim(PopNuSim):
    """
    Set up and run simulations for popsynth--neutrino
    coincidences. Assumes populations and neutrinos have
    no underlying connection.
    """

    def __init__(
        self,
        file_name="output/test_coincidence_sim.h5",
        group_base_name="survey",
        N=1,
        pop_config: str = None,
        nu_config: str = None,
        nu_hese_config: str = None,
        nu_ehe_config: str = None,
        seed=1000,
    ):

        super().__init__(
            file_name=file_name,
            group_base_name=group_base_name,
            N=N,
            pop_config=pop_config,
            nu_config=nu_config,
            nu_hese_config=nu_hese_config,
            nu_ehe_config=nu_ehe_config,
            seed=seed,
        )

    def _pop_nu_wrapper(self, pop, nu_obs):

        return PopNuCoincidence(pop, nu_obs)


class PopNuCoincidence(PopNuAction):
    """
    Check for coincidences of interest.
    """

    def __init__(
        self,
        pop: PopsynthWrapper,
        nu_obs: IceCubeObsWrapper,
        name="pop_nu_coincidence",
    ):

        self._coincidence = OrderedDict()

        super().__init__(
            pop=pop,
            nu_obs=nu_obs,
            name=name,
        )

    def _run(self):

        self._check_spatial()

        # self._check_temporal()

    @property
    def coincidence(self):

        return self._coincidence

    def write(self):

        with h5py.File(self._file_name, "r+") as f:

            if self._group_name not in f.keys():

                group = f.create_group(self._group_name)

            else:

                group = f[self._group_name]

            subgroup = group.create_group(self.name)

            pop_group = subgroup.create_group("pop")

            for key, value in self._coincidence.items():

                if key != "spatial_match_inds":

                    pop_group.create_dataset(key, data=value)

    def _check_spatial(self):
        """
        Check for spatial coincidences between
        the *detected* blazar populations and
        neutrinos
        """

        observation = self._nu_obs.observation

        survey = self._pop.survey

        n_match_spatial, match_ids = count_spatial_coincidence(
            np.deg2rad(observation.ra),
            np.deg2rad(observation.dec),
            np.deg2rad(observation.ang_err),
            np.deg2rad(survey.ra[survey.selection]),
            np.deg2rad(survey.dec[survey.selection]),
        )

        self._coincidence["n_spatial"] = n_match_spatial
        self._coincidence["match_ids"] = match_ids

    def _check_temporal(self):
        """
        Check for temporal coincidences between
        the *detected* populations and
        neutrinos, which are also spatial
        coincidences.
        """

        observation = self._nu_obs.observation

        survey = self._pop.survey

        (
            n_match_variable,
            n_match_flaring,
            matched_flare_amplitudes,
        ) = check_temporal_coincidence(
            observation.times,
            self._coincidence["spatial_match_inds"],
            survey.variability[survey.selection],
            survey.flare_times[survey.selection],
            survey.flare_durations[survey.selection],
            survey.flare_amplitudes[survey.selection],
        )

        self._coincidence["n_variable"] = n_match_variable
        self._coincidence["n_flaring"] = n_match_flaring
        self._coincidence["matched_flare_amplitudes"] = matched_flare_amplitudes


class PopNuCoincidenceResults(Results):
    """
    Load results from PopNuCoincidenceSim.
    """

    def __init__(self, file_name_list: List[str]):

        super().__init__(file_name_list=file_name_list)

    def _setup(self):

        self.pop = OrderedDict()
        self.pop["n_spatial"] = np.array([])
        self.pop["match_ids"] = []
        # self.pop["n_variable"] = np.array([])
        # self.pop["n_flaring"] = np.array([])
        # self.pop["matched_flare_amplitudes"] = np.array([])

    def _load_from_h5(self, file_name):

        with h5py.File(file_name, "r") as f:

            N_f = f.attrs["N"]

            n_spatial_f = np.zeros(N_f)
            # n_variable_f = np.zeros(N_f)
            # n_flaring_f = np.zeros(N_f)

            for i in range(N_f):

                group = f["survey_%i/pop_nu_coincidence/pop" % i]
                n_spatial_f[i] = group["n_spatial"][()]
                self.pop["match_ids"].append(group["match_ids"][()])
                # n_variable_f[i] = group["n_variable"][()]
                # n_flaring_f[i] = group["n_flaring"][()]

                # if n_flaring_f[i] >= 1:
                #     flare_amps_i = group["matched_flare_amplitudes"][()]
                #     self.pop["matched_flare_amplitudes"] = np.append(
                #         self.pop["matched_flare_amplitudes"], flare_amps_i
                #     )

        self.pop["n_spatial"] = np.append(self.pop["n_spatial"], n_spatial_f)
        # self.pop["n_variable"] = np.append(self.pop["n_variable"], n_variable_f)
        # self.pop["n_flaring"] = np.append(self.pop["n_flaring"], n_flaring_f)

        self.N += N_f

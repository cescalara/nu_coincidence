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
from cosmic_coincidence.blazars.fermi_interface import VariableFermiPopParams
from cosmic_coincidence.blazars.bllac import VariableBLLacPopWrapper
from cosmic_coincidence.blazars.fsrq import VariableFSRQPopWrapper
from cosmic_coincidence.neutrinos.icecube import (
    IceCubeAlertsParams,
    IceCubeAlertsWrapper,
)
from cosmic_coincidence.simulation import Simulation
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


class BlazarNuSimulation(Simulation):
    """
    Set up and run simulations for blazar-neutrino
    coincidences.
    """

    def __init__(
        self,
        file_name="output/test_sim.h5",
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

            # BL Lacs
            bllac_param_server = VariableFermiPopParams(
                A=3.39e4,
                gamma1=0.27,
                Lstar=0.28e48,
                gamma2=1.86,
                zcstar=1.34,
                p1star=2.24,
                tau=4.92,
                p2=-7.37,
                alpha=4.53e-2,
                mustar=2.1,
                beta=6.46e-2,
                sigma=0.26,
                flux_boundary=4e-12,
                flux_sigma=1,
                hard_cut=True,
                variability_weight=0.05,
                flare_rate_min=1 / 7.5,
                flare_rate_max=15,
                flare_rate_index=1.5,
                obs_time=10,
            )

            bllac_param_server.seed = i
            bllac_param_server.file_name = self._file_name
            bllac_param_server.group_name = self._group_base_name + "_%i" % i

            self._bllac_param_servers.append(bllac_param_server)

            # FSRQs
            fsrq_param_server = VariableFermiPopParams(
                A=3.06e4,
                gamma1=0.21,
                Lstar=0.84e48,
                gamma2=1.58,
                zcstar=1.47,
                p1star=7.35,
                tau=0,
                p2=-6.51,
                alpha=0.21,
                mustar=2.44,
                beta=0,
                sigma=0.18,
                flux_boundary=4e-12,
                flux_sigma=1,
                hard_cut=True,
                variability_weight=0.4,
                flare_rate_min=1 / 7.5,
                flare_rate_max=15,
                flare_rate_index=1.5,
                obs_time=10,
            )

            fsrq_param_server.seed = i
            fsrq_param_server.file_name = self._file_name
            fsrq_param_server.group_name = self._group_base_name + "_%i" % i

            self._fsrq_param_servers.append(fsrq_param_server)

            # Neutrinos
            nu_param_server = IceCubeAlertsParams(
                hese_Emin=1e4,
                ehe_Emin=5e4,
                Emax=1e8,
                Enorm=1e5,
                hese_Emin_det=1e4,
                ehe_Emin_det=2.5e5,
                hese_atmo_flux_norm=4e-18,
                ehe_atmo_flux_norm=4e-18 / 3,
                atmo_index=3.7,
                hese_diff_flux_norm=2e-18,
                ehe_diff_flux_norm=2e-18 / 3,
                diff_index=2.6,
                obs_time=10,
                max_cosz=1,
            )

            nu_param_server.seed = i
            nu_param_server.file_name = self._file_name
            nu_param_server.group_name = self._group_base_name + "_%i" % i

            self._nu_param_servers.append(nu_param_server)

    def _bllac_pop_wrapper(self, param_server):

        return VariableBLLacPopWrapper(param_server)

    def _fsrq_pop_wrapper(self, param_server):

        return VariableFSRQPopWrapper(param_server)

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

import numpy as np
from scipy import optimize
from astropy import units as u

from cosmic_coincidence.blazars.fermi_interface import (
    LDDEFermiModel,
    FermiPopWrapper,
    _zpower,
    _sbpl,
)
from cosmic_coincidence.populations.sbpl_population import (
    SBPLZPowerCosmoPopulation,
)
from cosmic_coincidence.populations.aux_samplers import (
    VariabilityAuxSampler,
    FlareRateAuxSampler,
    FlareTimeAuxSampler,
    FlareDurationAuxSampler,
)


class BLLacLDDEModel(LDDEFermiModel):
    """
    BL Lac LDDE model from Ajello+14.
    """

    def __init__(self):

        super(BLLacLDDEModel, self).__init__(
            name="bllac_ldde_fermi",
        )

    def prep_pop(self):
        """
        Get necessary params to make popsynth.
        """

        self._get_dNdV_params()
        self._get_dNdL_params()

    def popsynth(self, seed=1234):

        Lambda = self._popt_dNdV[0] * (1 / u.Mpc ** 3)
        Lambda = Lambda.to(1 / u.Gpc ** 3).value * 4 * np.pi

        pop = SBPLZPowerCosmoPopulation(
            Lambda=Lambda,
            delta=self._popt_dNdV[1],
            Lmin=self.Lmin,
            alpha=self._popt_dNdL[2],
            Lbreak=self._popt_dNdL[1],
            beta=self._popt_dNdL[3],
            Lmax=self.Lmax,
            r_max=self.zmax,
            is_rate=False,
            seed=seed,
        )

        return pop

    def _get_dNdV_params(self):
        """
        Find params to approximate dNdV with
        a ZPowCosmoDistribution.
        """

        z = np.linspace(self.zmin, self.zmax)
        dNdV = self.dNdV(z)

        p0 = (max(dNdV), -5)
        bounds = ([0, -10], [1, 0])

        popt, pcov = optimize.curve_fit(
            self._wrap_func_dNdV,
            z,
            dNdV,
            p0=p0,
            bounds=bounds,
        )

        self._popt_dNdV = popt

    def _get_dNdL_params(self):
        """
        Find params to approximate dNdL with
        an SBPLDistribution.
        """

        L = 10 ** np.linspace(np.log10(self.Lmin), np.log10(self.Lmax))
        dNdL = self.dNdL(L)

        p0 = (1, 1e47, 1.5, 2.5)
        bounds = ([1e-1, 1e47, 1.0, 2.0], [10, 5e48, 2.0, 3.0])

        popt, pcov = optimize.curve_fit(
            self._wrap_func_dNdL,
            L,
            1e57 * dNdL,
            p0=p0,
            bounds=bounds,
        )

        popt[0] = popt[0] / 1e57
        self._popt_dNdL = popt

    def _wrap_func_dNdV(self, z, Lambda, delta):

        return _zpower(z, Lambda, delta)

    def _wrap_func_dNdL(self, L, A, Lbreak, a1, a2):

        return _sbpl(L, A, Lbreak, a1, a2)


class BLLacPopWrapper(FermiPopWrapper):
    """
    Wrapper for BL Lac type Fermi models.
    """

    def __init__(self, parameter_server):

        super().__init__(parameter_server)

    def _pop_type(self, **kwargs):

        return BLLacLDDEModel(**kwargs)


class VariableBLLacPopWrapper(BLLacPopWrapper):
    """
    Extended wrapper with added flare sampling.
    """

    def __init__(self, parameter_server):

        super().__init__(parameter_server)

    def _simulation_setup(self):

        variability = VariabilityAuxSampler()
        variability.weight = self._parameter_server.variability["variability_weight"]

        flare_rate = FlareRateAuxSampler()
        flare_rate.xmin = self._parameter_server.variability["flare_rate_min"]
        flare_rate.xmax = self._parameter_server.variability["flare_rate_max"]
        flare_rate.index = self._parameter_server.variability["flare_rate_index"]

        flare_times = FlareTimeAuxSampler()
        flare_times.obs_time = self._parameter_server.variability["obs_time"]

        flare_durations = FlareDurationAuxSampler()

        flare_rate.set_secondary_sampler(variability)
        flare_times.set_secondary_sampler(flare_rate)
        flare_durations.set_secondary_sampler(flare_times)

        self._popsynth.add_observed_quantity(flare_durations)

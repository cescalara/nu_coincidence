import numpy as np
from scipy import optimize
from astropy import units as u

from cosmic_coincidence.blazars.fermi_interface import (
    LDDEFermiModel,
    FermiPopWrapper,
    VariableFermiPopWrapper,
    _sfr,
    _sbpl,
)
from cosmic_coincidence.populations.sbpl_population import SBPLSFRPopulation


class FSRQLDDEModel(LDDEFermiModel):
    """
    FSRQ LDDE model from Ajello+2012.
    """

    def __init__(self, name="fsrq_ldde_fermi"):

        super(FSRQLDDEModel, self).__init__(name=name)

    def prep_pop(self):
        """
        Get necessary params to make popsynth.
        """

        self._get_dNdV_params()
        self._get_dNdL_params()

    def popsynth(self, seed=1234):

        r0 = self._popt_dNdV[0] * (1 / u.Mpc ** 3)
        r0 = r0.to(1 / u.Gpc ** 3).value * 4 * np.pi

        pop = SBPLSFRPopulation(
            r0=r0,
            a=1,
            rise=self._popt_dNdV[1],
            decay=self._popt_dNdV[2],
            peak=self._popt_dNdV[3],
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

        z = np.geomspace(self.zmin, self.zmax)
        dNdV = self.dNdV(z)

        popt, pcov = optimize.curve_fit(self._wrap_func_dNdV, z, dNdV)

        self._popt_dNdV = popt

    def _get_dNdL_params(self):

        L = 10 ** np.linspace(np.log10(self.Lmin), np.log10(self.Lmax))
        dNdL = self.dNdL(L)

        p0 = (5, 1e48, 1.1, 2.5)
        bounds = ([0.1, 1e48, 1.0, 2.2], [10, 5e48, 1.5, 2.7])

        popt, pcov = optimize.curve_fit(
            self._wrap_func_dNdL,
            L,
            1e58 * dNdL,
            p0=p0,
            bounds=bounds,
        )

        popt[0] = popt[0] / 1e58
        self._popt_dNdL = popt

    def _wrap_func_dNdV(self, z, r0, rise, decay, peak):

        return _sfr(z, r0, rise, decay, peak)

    def _wrap_func_dNdL(self, L, A, Lbreak, a1, a2):

        return _sbpl(L, A, Lbreak, a1, a2)


class FSRQPopWrapper(FermiPopWrapper):
    """
    Wrapper for FSRQ like Fermi models.
    """

    def __init__(self, parameter_server):

        super().__init__(parameter_server)

    def _pop_type(self, **kwargs):

        return FSRQLDDEModel(**kwargs)


class VariableFSRQPopWrapper(VariableFermiPopWrapper):
    """
    Extended FSRQPopWrapper with added flare sampling.
    """

    def __init__(self, parameter_server):

        super().__init__(parameter_server)

    def _pop_type(self, **kwargs):

        return FSRQLDDEModel(**kwargs)

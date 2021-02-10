from abc import abstractmethod
import numpy as np
from scipy import integrate, optimize
from astropy import units as u

from popsynth.distribution import Distribution, DistributionParameter
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
from cosmic_coincidence.populations.sbpl_population import SBPLZPowerCosmoPopulation
from cosmic_coincidence.distributions.sbpl_distribution import sbpl


class FermiModel(Distribution):
    """
    Base class for models from Fermi papers.
    """

    # 1e-13 Mpc^-3 erg^-1 s
    A = DistributionParameter(default=1, vmin=0)

    # erg s^-1
    Lstar = DistributionParameter(vmin=0)

    gamma1 = DistributionParameter()
    gamma2 = DistributionParameter()

    tau = DistributionParameter(default=0)

    mustar = DistributionParameter()
    beta = DistributionParameter(default=0)
    sigma = DistributionParameter(vmin=0)

    # erg s^-1
    Lmin = DistributionParameter(default=7e43, vmin=0)
    Lmax = DistributionParameter(default=1e52, vmin=0)

    Gmin = DistributionParameter(default=1.45)
    Gmax = DistributionParameter(default=2.8)

    zmin = DistributionParameter(default=0.03, vmin=0)
    zmax = DistributionParameter(default=6.0, vmin=0)

    def __init__(self, seed=1234, name="fermi", form=r"Phi(L, z, G)"):

        super(FermiModel, self).__init__(seed=seed, name=name, form=form)

    @abstractmethod
    def Phi(self):
        """
        Phi(L, z, G) = dN / dLdVdG.
        """

        pass

    @abstractmethod
    def popsynth(self):
        """
        Return equivalent popsynth model.
        """


class Ajello14PDEModel(FermiModel):
    """
    PDE model from Ajello+2014.
    """

    kstar = DistributionParameter()
    xi = DistributionParameter()

    def __init__(self):

        super(Ajello14PDEModel, self).__init__(
            name="Ajello14PDE",
        )

    def phi_L(self, L):

        f1 = self.A / (np.log(10) * L)

        f2 = np.power(
            (L / self.Lstar) ** self.gamma1 + (L / self.Lstar) ** self.gamma2, -1
        )

        return f1 * f2

    def phi_G(self, G, L=None):

        if self.beta == 0:

            mu = self.mustar

        else:

            mu = self.mustar + self.beta * (np.log10(L) - 46)

        return np.exp(-((G - mu) ** 2) / (2 * self.sigma ** 2))

    def phi_z(self, z, L=None):

        if self.tau == 0:

            k = self.kstar

        else:

            k = self.kstar + self.tau * (np.log10(L) - 46)

        return np.power(1 + z, k) * np.exp(z / self.xi)

    def Phi(self, L, z, G):

        return self.phi_L(L) * self.phi_G(G, L) * self.phi_z(z, L)

    def N(self):

        pass

    def local_density(self):

        if self.beta == 0 and self.tau == 0:

            I1, err = integrate.quad(self.phi_L, self.Lmin, self.Lmax)

            I2, err = integrate.quad(self.phi_G, self.Gmin, self.Gmax)

            r0 = I1 * I2 * 1e-13 * (1 / u.Mpc ** 3)

            return r0.to(1 / u.Gpc ** 3).value

        else:

            raise NotImplementedError

    def popsynth(self):

        if self.beta == 0 and self.tau == 0:

            r0 = self.local_density() * 4 * np.pi

            pop = SBPLZPowExpCosmoPopulation(
                r0=r0,
                k=self.kstar,
                xi=self.xi,
                Lmin=self.Lmin,
                alpha=-self.gamma1,
                Lbreak=self.Lstar,
                beta=-self.gamma2,
                Lmax=self.Lmax,
                r_max=self.zmax,
                is_rate=False,
            )

            return pop

        else:

            raise NotImplementedError


class Ajello14LDDEModel(FermiModel):
    """
    LDDE model from Ajello+14.
    """

    zcstar = DistributionParameter(vmin=0)
    alpha = DistributionParameter()
    p1star = DistributionParameter()
    p2 = DistributionParameter()

    def __init__(self):

        super(Ajello14LDDEModel, self).__init__(
            name="Ajello14LDDE",
        )

    def phi_L(self, L):

        f1 = self.A / (np.log(10) * L)

        f2 = np.power(
            (L / self.Lstar) ** self.gamma1 + (L / self.Lstar) ** self.gamma2, -1
        )

        return f1 * f2

    def phi_G(self, G, L):

        if self.beta == 0:

            mu = self.mustar

        else:

            mu = self.mustar + self.beta * (np.log10(L) - 46)

        return np.exp(-((G - mu) ** 2) / (2 * self.sigma ** 2))

    def phi_z(self, z, L):

        zc = self.zcstar * np.power(L / 1e48, self.alpha)

        p1 = self.p1star + self.tau * (np.log10(L) - 46)

        inner = (1 + z) / (1 + zc)

        return np.power(inner ** -p1 + inner ** -self.p2, -1)

    def Phi(self, L, z, G):

        return self.phi_L(L) * self.phi_G(G, L) * self.phi_z(z, L)

    def dNdV(self, z, approx=False):
        """
        Integrate Phi over L and G. In units of Mpc^-3.
        If approx, show appromximated version.
        """

        if approx:

            self._get_dNdV_params()

            return _wrap_func_dNdV(z, self._Lambda, self._delta)

        else:

            integral = np.zeros_like(z)

            L = 10 ** np.linspace(np.log10(self.Lmin), np.log10(self.Lmax), 1000)
            G = np.linspace(self.Gmin, self.Gmax, 1000)

            for i, z in enumerate(z):
                f = self.Phi(L[:, None], z, G) * 1e-13  # Mpc^-3 erg^-1 s
                integral[i] = integrate.simps(integrate.simps(f, G), L)

            return integral

    def dNdL(self, L, approx=False):
        """
        Integrate Phi over z and G.
        If approx, show approximated version.
        """

        if approx:

            self._get_dNdL_params()

            return _wrap_func_dNdL(
                L,
                self._LA,
                self._Lbreak,
                self._Lalpha,
                self._Lbeta,
            )

        else:

            integral = np.zeros_like(L)

            z = np.linspace(self.zmin, self.zmax, 1000)
            G = np.linspace(self.Gmin, self.Gmax, 1000)

            for i, L in enumerate(L):
                f = self.Phi(L, z[:, None], G) * 1e-13  # Mpc^-3 erg^-1 s
                integral[i] = integrate.simps(integrate.simps(f, G), z)

            return integral

    def popsynth(self):

        self._get_dNdV_params()
        self._get_dNdL_params()

        Lambda = self._Lambda * (1 / u.Mpc ** 3)
        Lambda = Lambda.to(1 / u.Gpc ** 3).value * 4 * np.pi

        pop = SBPLZPowerCosmoPopulation(
            Lambda=Lambda,
            delta=self._delta,
            Lmin=self.Lmin,
            alpha=self._Lalpha,
            Lbreak=self._Lbreak,
            beta=self._Lbeta,
            Lmax=self.Lmax,
            r_max=self.zmax,
            is_rate=False,
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
            _wrap_func_dNdV,
            z,
            dNdV,
            p0=p0,
            bounds=bounds,
        )

        self._Lambda = popt[0]
        self._delta = popt[1]

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
            _wrap_func_dNdL,
            L,
            1e57 * dNdL,
            p0=p0,
            bounds=bounds,
        )

        self._LA = popt[0] / 1e57
        self._Lbreak = popt[1]
        self._Lalpha = popt[2]
        self._Lbeta = popt[3]


def _wrap_func_dNdV(z, Lambda, delta):

    return Lambda * np.power(1 + z, delta)


def _wrap_func_dNdL(L, A, Lbreak, a1, a2):

    return A * sbpl(
        L,
        7e43,
        Lbreak,
        1e52,
        a1,
        a2,
    )

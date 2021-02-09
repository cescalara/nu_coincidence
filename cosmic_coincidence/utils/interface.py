from abc import abstractmethod
import numpy as np
from scipy import integrate
from astropy import units as u

from popsynth.distribution import Distribution, DistributionParameter
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation


class FermiModel(Distribution):
    """
    Base class for models from Fermi papers.
    """

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
    def local_density(self):
        """
        dV / dV at z=0 in Gpc^-3
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

    # 1e-13 Mpc^-3 erg^-1 s
    A = DistributionParameter(default=1, vmin=0)

    # erg s^-1
    Lstar = DistributionParameter(vmin=0)

    gamma1 = DistributionParameter()
    gamma2 = DistributionParameter()

    kstar = DistributionParameter(vmin=0)
    tau = DistributionParameter(default=0)
    xi = DistributionParameter()

    mustar = DistributionParameter()
    beta = DistributionParameter(default=0)
    sigma = DistributionParameter(vmin=0)

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

            r0 = self.local_density()

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
            )

            return pop

        else:

            raise NotImplementedError

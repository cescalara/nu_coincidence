import numpy as np

from popsynth.distribution import Distribution, DistributionParameter


class FermiModel:
    """
    Base class for models from Fermi papers.
    """

    def __init__(self):

        pass


class Ajello14PDEModel(Distribution):
    """
    Exact form of PDE model from Ajello+2014.
    """

    A = DistributionParameter(default=1, vmin=0)
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
            seed=1234, name="Ajello14PDE", form="Phi(L, z, G)"
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

    def get_N(self):
        """
        Integrate over L, z and G and dV/dz to get
        total number of objects.
        """

        pass

    def get_local_density(self, Lmin, Lmax, Gmin, Gmax):
        """
        Integrate over phi_L and phi_G to get factor
        appearing before phi_z in units of Mpc^-3.
        """

        if self.beta == 0:

            phi_L_int = x

        else:

            raise NotImplementedError

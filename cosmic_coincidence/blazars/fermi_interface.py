from abc import abstractmethod, ABCMeta
import numpy as np
from scipy import integrate
from astropy import units as u

from popsynth.distribution import Distribution, DistributionParameter
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
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

    def set_parameters(self, parameters):
        """
        Option to set parameters from dict.
        """

        self._parameter_storage = parameters

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


class LDDEFermiModel(FermiModel):
    """
    LDDE model used in Ajello+2012 and
    Ajello+2014.
    """

    zcstar = DistributionParameter(vmin=0)
    alpha = DistributionParameter()
    p1star = DistributionParameter()
    p2 = DistributionParameter()

    def __init__(self, seed=1234, name="ldde_fermi"):

        super(LDDEFermiModel, self).__init__(seed=seed, name=name)

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

            return self._wrap_func_dNdV(z, *self._popt_dNdV)

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

            return self._wrap_func_dNdL(L, *self._popt_dNdL)

        else:

            integral = np.zeros_like(L)

            z = np.linspace(self.zmin, self.zmax, 1000)
            G = np.linspace(self.Gmin, self.Gmax, 1000)

            for i, L in enumerate(L):
                f = self.Phi(L, z[:, None], G) * 1e-13  # Mpc^-3 erg^-1 s
                # f = f * cosmology.differential_comoving_volume(z) * 1e9
                integral[i] = integrate.simps(integrate.simps(f, G), z)

            return integral

    @abstractmethod
    def _get_dNdV_params(self):

        pass

    @abstractmethod
    def _get_dNdL_params(self):

        pass

    @abstractmethod
    def _wrap_func_dNdV(self):

        pass

    @abstractmethod
    def _wrap_func_dNdL(self):

        pass


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

        self.r0 = None

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

            self.r0 = r0.to(1 / u.Gpc ** 3).value * 4 * np.pi

            return r0.to(1 / u.Gpc ** 3).value

        else:

            raise NotImplementedError

    def popsynth(self, seed=1234):

        if self.beta == 0 and self.tau == 0:

            if not self.r0:
                self.r0 = self.local_density() * 4 * np.pi

            pop = SBPLZPowExpCosmoPopulation(
                r0=self.r0,
                k=self.kstar,
                xi=self.xi,
                Lmin=self.Lmin,
                alpha=-self.gamma1,
                Lbreak=self.Lstar,
                beta=-self.gamma2,
                Lmax=self.Lmax,
                r_max=self.zmax,
                is_rate=False,
                seed=seed,
            )

            return pop

        else:

            raise NotImplementedError


class FermiPopWrapper(object, metaclass=ABCMeta):
    """
    Parallel simulation wrapper for blazar
    populations defined through the Fermi interface.
    """

    def __init__(self, parameter_server):

        self._parameter_server = parameter_server

        self._pop_setup = self._pop_type()

        self._pop_setup.set_parameters(self._parameter_server.parameters)

        self._pop_setup.Lmax = self._parameter_server.Lmax

        self._pop_setup.prep_pop()

        self._popsynth = self._pop_setup.popsynth(seed=self._parameter_server.seed)

        # Optional further setup
        self._simulation_setup()

        # Run
        self._run()

    @abstractmethod
    def _pop_type(self):

        raise NotImplementedError()

    def _simulation_setup(self):

        pass

    def _run(self):

        survey = self._popsynth.draw_survey(**self._parameter_server.survey)

        survey.addto(
            self._parameter_server.file_path, self._parameter_server.group_name
        )

        del survey


class FermiPopParams(object):
    """
    Parameter server for blazar populations
    defined through the Fermi interface.
    """

    def __init__(
        self,
        A,
        gamma1,
        Lstar,
        gamma2,
        zcstar,
        p1star,
        tau,
        p2,
        alpha,
        mustar,
        beta,
        sigma,
        boundary,
        hard_cut,
    ):

        self._parameters = dict(
            A=A,
            gamma1=gamma1,
            Lstar=Lstar,
            gamma2=gamma2,
            zcstar=zcstar,
            p1star=p1star,
            tau=tau,
            p2=p2,
            alpha=alpha,
            mustar=mustar,
            beta=beta,
            sigma=sigma,
        )

        self._survey = dict(boundary=boundary, hard_cut=hard_cut)

        self._Lmax = 1e50

        self._file_path = None

        self._group_name = None

    @property
    def seed(self):

        return self._seed

    @seed.setter
    def seed(self, value: int):

        self._seed = value

    @property
    def file_path(self):

        return self._file_path

    @file_path.setter
    def file_path(self, value: str):

        self._file_path = value

    @property
    def group_name(self):

        return self._group_name

    @group_name.setter
    def group_name(self, value: str):

        self._group_name = value

    @property
    def Lmax(self):

        return self._Lmax

    @Lmax.setter
    def Lmax(self, value: float):

        self._Lmax = value

    @property
    def parameters(self):

        return self._parameters

    @property
    def survey(self):

        return self._survey


class VariableFermiPopParams(FermiPopParams):
    """
    Extended parameter server for variable
    populations.
    """

    def __init__(
        self,
        A,
        gamma1,
        Lstar,
        gamma2,
        zcstar,
        p1star,
        tau,
        p2,
        alpha,
        mustar,
        beta,
        sigma,
        boundary,
        hard_cut,
        variability_weight,
        flare_rate_min,
        flare_rate_max,
        flare_rate_index,
        obs_time,
    ):

        super().__init__(
            A=A,
            gamma1=gamma1,
            Lstar=Lstar,
            gamma2=gamma2,
            zcstar=zcstar,
            p1star=p1star,
            tau=tau,
            p2=p2,
            alpha=alpha,
            mustar=mustar,
            beta=beta,
            sigma=sigma,
            boundary=boundary,
            hard_cut=hard_cut,
        )

        self._variability = dict(
            variability_weight=variability_weight,
            flare_rate_min=flare_rate_min,
            flare_rate_max=flare_rate_max,
            flare_rate_index=flare_rate_index,
            obs_time=obs_time,
        )

    @property
    def variability(self):

        return self._variability


def _zpower(z, Lambda, delta):

    return Lambda * np.power(1 + z, delta)


def _sfr(z, r0, r, d, p):

    top = 1 + r * z
    bottom = 1 + np.power(z / p, d)

    return r0 * top / bottom


def _zpowerexp(z, r0, k, xi):

    return r0 * np.power(1 + z, k) * np.exp(z / xi)


def _sbpl(L, A, Lbreak, a1, a2):

    return A * sbpl(
        L,
        1e43,
        Lbreak,
        1e52,
        a1,
        a2,
    )

from abc import abstractmethod, ABCMeta
import numpy as np
from scipy import integrate
from astropy import units as u
import h5py

from popsynth.distribution import Distribution, DistributionParameter
from popsynth.selection_probability.flux_selectors import HardFluxSelection
from popsynth.utils.cosmology import cosmology

from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
from cosmic_coincidence.distributions.sbpl_distribution import sbpl
from cosmic_coincidence.utils.parameter_server import ParameterServer
from cosmic_coincidence.populations.aux_samplers import (
    VariabilityAuxSampler,
    FlareRateAuxSampler,
    FlareTimeAuxSampler,
    FlareDurationAuxSampler,
)


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

    def dNdV(self, z, approx=False, resolution=100):
        """
        Integrate Phi over L and G. In units of Mpc^-3.
        If approx, show appromximated version.
        """

        if approx:

            self._get_dNdV_params()

            return self._wrap_func_dNdV(z, *self._popt_dNdV)

        else:

            integral = np.zeros_like(z)

            l = np.linspace(np.log10(self.Lmin), np.log10(self.Lmax), resolution)
            L = 10 ** l

            G = np.linspace(self.Gmin, self.Gmax, resolution)

            for i, redshift in enumerate(z):
                f = (
                    self.Phi(L[:, None], redshift, G) * np.log(10) * L[:, None] * 1e-13
                )  # Mpc^-3 erg^-1 s
                integral[i] = integrate.simps(integrate.simps(f, G), l)

            return integral

    def dNdL(self, L, approx=False, resolution=100, cosmo=False):
        """
        Integrate Phi over z and G.
        If approx, show approximated version.
        """

        if approx:

            self._get_dNdL_params()

            return self._wrap_func_dNdL(L, *self._popt_dNdL)

        else:

            integral = np.zeros_like(L)

            z = np.linspace(self.zmin, self.zmax, resolution)
            G = np.linspace(self.Gmin, self.Gmax, resolution)

            for i, L in enumerate(L):
                f = self.Phi(L, z[:, None], G) * 1e-13  # Mpc^-3 erg^-1 s

                if cosmo:
                    f = f * cosmology.differential_comoving_volume(z) * 1e9

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

        flux_selector = HardFluxSelection()
        flux_selector.boundary = self._parameter_server.selection["flux_boundary"]
        self._popsynth.set_flux_selection(flux_selector)

        flux_sigma = self._parameter_server.selection["flux_sigma"]

        self._survey = self._popsynth.draw_survey(flux_sigma=flux_sigma)

    @property
    def survey(self):

        return self._survey

    def write(self):
        """
        A Lightweight version of the usual popsynth.Population.writeto()
        more suited to large numbers of simulations and relevant info.
        """

        with h5py.File(self._parameter_server.file_name, "r+") as f:

            if self._parameter_server.group_name not in f.keys():

                group = f.create_group(self._parameter_server.group_name)

            else:

                group = f[self._parameter_server.group_name]

            subgroup = group.create_group(self._pop_setup.name)

            # Attributes
            subgroup.attrs["name"] = np.string_(self._survey._name)
            subgroup.attrs["spatial_form"] = np.string_(self._survey._spatial_form)
            subgroup.attrs["lf_form"] = np.string_(self._survey._lf_form)
            subgroup.attrs["flux_sigma"] = self._survey._flux_sigma
            subgroup.attrs["r_max"] = self._survey._r_max
            subgroup.attrs["seed"] = int(self._survey._seed)

            # True distributions
            spatial_grp = subgroup.create_group("spatial_params")
            for k, v in self._survey._spatial_params.items():
                spatial_grp.create_dataset(k, data=np.array([v]), compression="lzf")

            lum_grp = subgroup.create_group("lf_params")
            for k, v in self._survey._lf_params.items():
                lum_grp.create_dataset(k, data=np.array([v]), compression="lzf")

            # Population objects
            subgroup.create_dataset(
                "distances",
                data=self._survey.distances,
                compression="lzf",
            )

            subgroup.create_dataset(
                "luminosities_latent",
                data=self._survey.luminosities_latent,
                compression="lzf",
            )

            # subgroup.create_dataset(
            #     "fluxes_latent",
            #     data=self._survey.fluxes_latent,
            #     compression="lzf",
            # )

            # subgroup.create_dataset(
            #     "fluxes_observed",
            #     data=self._survey.fluxes_observed,
            #     compression="lzf",
            # )

            subgroup.create_dataset(
                "selection",
                data=self._survey.selection,
                compression="lzf",
            )

            subgroup.create_dataset(
                "theta",
                data=self._survey._theta,
                compression="lzf",
            )

            subgroup.create_dataset(
                "phi",
                data=self._survey._phi,
                compression="lzf",
            )

            # Auxiliary quantities
            aux_grp = subgroup.create_group("auxiliary_quantities")
            for k, v in self._survey._auxiliary_quantities.items():
                aux_grp.create_dataset(k, data=v["true_values"], compression="lzf")


class VariableFermiPopWrapper(FermiPopWrapper):
    """
    Extended FermiPopWrapper with added flare sampling.
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


class FermiPopParams(ParameterServer):
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
        flux_boundary,
        flux_sigma,
        hard_cut,
    ):

        super().__init__()

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

        self._selection = dict(
            flux_boundary=flux_boundary,
            flux_sigma=flux_sigma,
            hard_cut=hard_cut,
        )

        self._Lmax = 1e50

    @property
    def Lmax(self):

        return self._Lmax

    @Lmax.setter
    def Lmax(self, value: float):

        self._Lmax = value

    @property
    def selection(self):

        return self._selection


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
        flux_boundary,
        flux_sigma,
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
            flux_boundary=flux_boundary,
            flux_sigma=flux_sigma,
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

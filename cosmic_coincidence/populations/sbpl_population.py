from popsynth.populations.spatial_populations import (
    ZPowerCosmoPopulation,
    SFRPopulation,
)

from .spatial_populations import ZPowExpCosmoPopulation
from ..distributions.sbpl_distribution import SBPLDistribution


class SBPLZPowerCosmoPopulation(ZPowerCosmoPopulation):
    def __init__(
        self,
        Lambda,
        delta,
        Lmin,
        alpha,
        Lbreak,
        beta,
        Lmax,
        r_max=5,
        seed=1234,
        is_rate=True,
    ):

        luminosity_distribution = SBPLDistribution()
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(SBPLZPowerCosmoPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )


class SBPLZPowExpCosmoPopulation(ZPowExpCosmoPopulation):
    def __init__(
        self,
        r0,
        k,
        xi,
        Lmin,
        alpha,
        Lbreak,
        beta,
        Lmax,
        r_max=5,
        seed=1234,
        is_rate=True,
    ):

        luminosity_distribution = SBPLDistribution()
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(SBPLZPowExpCosmoPopulation, self).__init__(
            r0=r0,
            k=k,
            xi=xi,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )


class SBPLSFRPopulation(SFRPopulation):
    def __init__(
        self,
        r0,
        rise,
        decay,
        peak,
        Lmin,
        alpha,
        Lbreak,
        beta,
        Lmax,
        r_max=5,
        seed=1234,
        is_rate=True,
    ):

        luminosity_distribution = SBPLDistribution()
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(SBPLSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )

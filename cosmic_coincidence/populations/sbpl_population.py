from .spatial_populations import ZPowExpCosmoPopulation
from ..distributions.sbpl_distribution import SBPLDistribution


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
        )

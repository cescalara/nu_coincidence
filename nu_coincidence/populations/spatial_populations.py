from popsynth.population_synth import PopulationSynth

from nu_coincidence.distributions.cosmological_distribution import (
    ZPowExpCosmoDistribution,
)


class ZPowExpCosmoPopulation(PopulationSynth):
    def __init__(
        self,
        r0,
        k,
        xi,
        r_max=5.0,
        seed=1234,
        luminosity_distribution=None,
        is_rate=True,
    ):
        """
        :param Lambda:
        :param k:
        :param xi:
        """

        spatial_distribution = ZPowExpCosmoDistribution(
            seed=seed,
            is_rate=is_rate,
        )
        spatial_distribution.r0 = r0
        spatial_distribution.k = k
        spatial_distribution.xi = xi
        spatial_distribution.r_max = r_max

        super(ZPowExpCosmoPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )

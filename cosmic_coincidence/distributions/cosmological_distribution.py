import numpy as np

from popsynth.distribution import DistributionParameter
from popsynth.distributions.cosmological_distribution import CosmologicalDistribution


class ZPowExpCosmoDistribution(CosmologicalDistribution):

    r0 = DistributionParameter(default=1, vmin=0)
    k = DistributionParameter(vmin=0)
    xi = DistributionParameter()

    def __init__(self, seed=1234, name="zpowexp_cosmo", is_rate=True):

        spatial_form = r"r_0 (1 + z)^k \exp{z/\xi}"

        super(ZPowExpCosmoDistribution, self).__init__(
            seed=seed,
            name=name,
            form=spatial_form,
            is_rate=is_rate,
        )

    def dNdV(self, distance):

        return self.r0 * np.power(1 + distance, self.k) * np.exp(distance / self.xi)

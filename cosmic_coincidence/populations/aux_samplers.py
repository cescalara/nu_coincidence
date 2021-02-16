import numpy as np
from scipy import stats

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class ParetoAuxSampler(AuxiliarySampler):
    """
    Sample from a Pareto distribution.
    """

    xmin = AuxiliaryParameter(vmin=0)
    index = AuxiliaryParameter(default=1)

    def __init__(self, name: str, observed: bool = False):

        super(ParetoAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        self._true_values = stats.pareto(self.index).rvs(size) * self.xmin


class VariabilityAuxSampler(AuxiliarySampler):
    """
    Sample whether a source is variable or not.
    Boolean outcome.
    """

    weight = AuxiliaryParameter(vmin=0, vmax=1)

    def __init__(self, name="variability", observed=False):

        super(VariabilityAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        self._true_values = np.random.choice(
            [True, False],
            p=[self.weight, 1 - self.weight],
            size=size,
        )


class FlareRateAuxSampler(ParetoAuxSampler):
    """
    Sample source flare rate given its variability.
    """

    def __init__(self, name="flare_rate", observed=False):

        super(FlareRateAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        rate = np.zeros(size)

        variability = self._secondary_samplers["variability"].true_values

        rate[variability == False] = 0

        rate[variability == True] = 1

        super(FlareRateAuxSampler, self).true_sampler(size)

        self._true_values = rate * self._true_values


class FlareNumAuxSampler(ParetoAuxSampler):
    """
    Sample number of flares for a given rate.
    """

    obs_time = AuxiliaryParameter(vmin=0, default=1)

    def __init__(self, name="flare_num", observed=False):

        super(FlareNumAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        rate = self._secondary_samplers["flare_rate"].true_values

        self._true_values = np.random.poisson(rate * self.obs_time)


class FlareTimeAuxSampler(AuxiliarySampler):

    pass

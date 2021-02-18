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


class BoundedPowerLawAuxSampler(AuxiliarySampler):
    """
    Sample from a bounded power law.
    """

    xmin = AuxiliaryParameter(vmin=0)
    xmax = AuxiliaryParameter()
    index = AuxiliaryParameter(default=1)

    def __init__(self, name: str, observed: bool = False):

        super(BoundedPowerLawAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        uniform_samples = np.random.uniform(0, 1, size)

        self._true_values = bounded_pl_inv_cdf(
            uniform_samples, self.xmin, self.xmax, self.index
        )


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


class FlareRateAuxSampler(BoundedPowerLawAuxSampler):
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
    """
    Sample flare times for each source give
    rate and total number of flares.
    """

    obs_time = AuxiliaryParameter(vmin=0, default=1)

    def __init__(self, name="flare_times", observed=False):

        super(FlareTimeAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        times = np.empty((size,), dtype=object)

        rate = self._secondary_samplers["flare_rate"].true_values

        for i, _ in enumerate(times):

            if rate[i] == 0:

                times[i] = []

            else:

                max_nflares = int(rate[i] * self.obs_time * 10)
                wait_times = stats.expon(loc=2 / 52, scale=1 / rate[i]).rvs(max_nflares)
                ts = np.cumsum(wait_times)
                times[i] = list(ts[ts < self.obs_time])

        self._true_values = times


class FlareDurationAuxSampler(AuxiliarySampler):
    """
    Sample flare durations given flare times
    """

    def __init__(self, name="flare_durations", observed=False):

        super(FlareDurationAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        durations = np.empty((size,), dtype=object)

        times = self._secondary_samplers["flare_times"].true_values

        obs_time = self._secondary_samplers["flare_times"].obs_time

        for i, _ in enumerate(durations):

            if times[i] == []:

                durations[i] = []

            else:

                max_durations = np.array(times[i][1:]) - np.array(times[i][:-1])
                max_durations = np.append(
                    max_durations, np.max([obs_time - times[i][-1], 2 / 52])
                )
                max_durations = max_durations - 1 / 52

                # for j, md in enumerate(max_durations):
                durations[i] = [
                    bounded_pl_inv_cdf(np.random.uniform(0, 1), 1 / 52, md, 1.5)
                    for md in max_durations
                ]
                # durations[i] = list(np.random.uniform(low=1 / 52, high=max_durations))

        self._true_values = durations


def bounded_pl_inv_cdf(x, xmin, xmax, index):
    """
    Bounded power law inverse CDF.
    """

    if index != 1.0:

        int_index = 1 - index
        norm = 1 / int_index * (xmax ** int_index - xmin ** int_index)
        norm = 1 / norm

        inv_cdf_factor = norm ** (-1) * int_index
        inv_cdf_const = xmin ** int_index
        inv_cdf_index = 1.0 / int_index

        return np.power((x * inv_cdf_factor) + inv_cdf_const, inv_cdf_index)

    else:

        norm = 1.0 / np.log(xmax / xmin)
        return xmin * np.exp(x / norm)

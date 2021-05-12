import numpy as np
from scipy import stats
import h5py

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter
from popsynth.aux_samplers.normal_aux_sampler import NormalAuxSampler


class SpectralIndexAuxSampler(NormalAuxSampler):
    """
    Sample the spectral index of a source
    with a simple power law spectrum.
    """

    _auxiliary_sampler_name = "SpectralIndexAuxSampler"

    def __init__(self, name="spectral_index", observed=True):

        super().__init__(name=name, observed=observed)


class ParetoAuxSampler(AuxiliarySampler):
    """
    Sample from a Pareto distribution.
    """

    _auxiliary_sampler_name = "ParetoAuxSampler"

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

    _auxiliary_sampler_name = "BoundedPowerLawAuxSampler"

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

    _auxiliary_sampler_name = "VariabilityAuxSampler"

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

    _auxiliary_sampler_name = "FlareRateAuxSampler"

    def __init__(self, name="flare_rate", observed=False):

        super(FlareRateAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        rate = np.zeros(size)

        variability = self._secondary_samplers["variability"].true_values

        rate[variability == False] = 0

        rate[variability == True] = 1

        super(FlareRateAuxSampler, self).true_sampler(size)

        self._true_values = rate * self._true_values


class FlareNumAuxSampler(AuxiliarySampler):
    """
    Sample number of flares for a given rate.
    """

    _auxiliary_sampler_name = "FlareNumAuxSampler"

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

    _auxiliary_sampler_name = "FlareTimeAuxSampler"

    obs_time = AuxiliaryParameter(vmin=0, default=1)

    def __init__(self, name="flare_times", observed=False):

        super(FlareTimeAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        dt = h5py.vlen_dtype(np.dtype("float64"))

        times = np.empty((size,), dtype=dt)

        rate = self._secondary_samplers["flare_rate"].true_values

        for i, _ in enumerate(times):

            if rate[i] == 0:

                times[i] = np.array([], dtype=np.dtype("float64"))

            else:

                n_flares = np.random.poisson(rate[i] * self.obs_time)

                time_samples = np.random.uniform(0, self.obs_time, size=n_flares)
                time_samples = np.sort(time_samples)

                times[i] = np.array(time_samples, dtype=np.dtype("float64"))

        self._true_values = times


class FlareDurationAuxSampler(AuxiliarySampler):
    """
    Sample flare durations given flare times
    """

    _auxiliary_sampler_name = "FlareDurationAuxSampler"

    index = AuxiliaryParameter(vmin=1, default=1.5)

    def __init__(self, name="flare_durations", observed=False):

        super(FlareDurationAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        dt = h5py.vlen_dtype(np.dtype("float64"))

        durations = np.empty((size,), dtype=dt)

        times = self._secondary_samplers["flare_times"].true_values

        obs_time = self._secondary_samplers["flare_times"].obs_time

        eps = 1e-3

        for i, _ in enumerate(durations):

            if times[i].size == 0:

                durations[i] = np.array([], dtype=np.dtype("float64"))

            else:

                # Difference between flare times
                max_durations = np.diff(times[i])

                # Add final flare duration, can go up until obs_time
                max_durations = np.append(max_durations, obs_time - times[i][-1])

                # Minimum duration of 1 week
                max_durations[max_durations < 1 / 52] = 1 / 52 + eps

                durations[i] = np.array(
                    [
                        bounded_pl_inv_cdf(
                            np.random.uniform(0, 1), 1 / 52, md, self.index
                        )
                        for md in max_durations
                    ],
                    dtype=np.dtype("float64"),
                )

        self._true_values = durations


class FlareAmplitudeAuxSampler(AuxiliarySampler):
    """
    Sample increase in luminosity of the flares
    as a multiplicative factor.
    """

    _auxiliary_sampler_name = "FlareAmplitudeAuxSampler"

    xmin = AuxiliaryParameter(vmin=0, default=1)
    index = AuxiliaryParameter(default=1)

    def __init__(self, name="flare_amplitudes", observed=False):

        super(FlareAmplitudeAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        dt = h5py.vlen_dtype(np.dtype("float64"))

        amplitudes = np.empty((size,), dtype=dt)

        times = self._secondary_samplers["flare_times"].true_values

        for i, _ in enumerate(amplitudes):

            if times[i].size == 0:

                amplitudes[i] = np.array([], dtype=np.dtype("float64"))

            else:

                n_flares = times[i].size

                samples = stats.pareto(self.index).rvs(n_flares) * self.xmin

                amplitudes[i] = np.array(samples, dtype=np.dtype("float64"))

        self._true_values = amplitudes


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

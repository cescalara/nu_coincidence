import numpy as np
from scipy import stats
import h5py

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter
from popsynth.aux_samplers.normal_aux_sampler import NormalAuxSampler
from popsynth.aux_samplers.plaw_aux_sampler import PowerLawAuxSampler, _sample_power_law
from popsynth.utils.cosmology import cosmology


class SpectralIndexAuxSampler(NormalAuxSampler):
    """
    Sample the spectral index of a source
    with a simple power law spectrum.
    """

    _auxiliary_sampler_name = "SpectralIndexAuxSampler"

    def __init__(self, name="spectral_index", observed=True):

        super().__init__(name=name, observed=observed)


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


class FlareRateAuxSampler(PowerLawAuxSampler):
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
    Sample flare durations given flare times.
    """

    _auxiliary_sampler_name = "FlareDurationAuxSampler"

    alpha = AuxiliaryParameter(default=-1.5)

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
                        _sample_power_law(1 / 52, md, self.alpha, 1)[0]
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
    alpha = AuxiliaryParameter(default=1)

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

                samples = stats.pareto(self.alpha).rvs(n_flares) * self.xmin

                amplitudes[i] = np.array(samples, dtype=np.dtype("float64"))

        self._true_values = amplitudes


class CombinedFluxIndexSampler(AuxiliarySampler):
    """
    Make a transformed parameter to perform a
    combined linear selection on energy flux and
    spectral index.

    Selection has the form:
    index = ``slope`` log10(flux) + ``intercept``

    So, here we transform to:
    -(index - ``slope`` log10(flux))
    such that a constant selection can be made
    on -``intercept``. This works with both
    :class:`HardSelection` and :class:`SoftSelection`

    See e.g. Fig. 4 in Ajello et al. 2020 (4LAC),
    default values are set to approximate this.
    """

    _auxiliary_sampler_name = "CombinedFluxIndexSampler"

    slope = AuxiliaryParameter(default=3)

    def __init__(
        self,
    ):

        # this time set observed=True
        super(CombinedFluxIndexSampler, self).__init__(
            "combined flux index",
            observed=False,
            uses_distance=True,
            uses_luminosity=True,
        )

    def true_sampler(self, size):

        # Calculate latent fluxes (ideally would use observed)
        dl = cosmology.luminosity_distance(self._distance)  # cm

        fluxes = self._luminosity / (4 * np.pi * dl ** 2)  # erg cm^-2 s^-1

        # Use observed spectral index
        spectral_index = self._secondary_samplers["spectral_index"].obs_values

        # Transformed based on desired selection
        true_values = spectral_index - self.slope * np.log10(fluxes)

        # Negative to use with SoftSelection
        self._true_values = -1 * true_values

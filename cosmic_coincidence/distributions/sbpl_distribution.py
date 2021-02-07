import numpy as np
from scipy import stats, interpolate
from popsynth.distribution import LuminosityDistribution, DistributionParameter


class SBPLDistribution(LuminosityDistribution):

    Lmin = DistributionParameter(vmin=0)
    alpha = DistributionParameter()
    Lbreak = DistributionParameter(vmin=0)
    beta = DistributionParameter()
    Lmax = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="sbpl"):

        lf_form = r"\[ (\frac{L}{L_{\rm break}})^\alpha + (\frac{L}{L_{\rm break}})^\beta) \]^{-1}"

        super(SBPLDistribution, self).__init__(
            seed=seed,
            name=name,
            form=lf_form,
        )

    def phi(self, L):

        C = integrate_sbpl(
            self.Lmin,
            self.Lbreak,
            self.Lmax,
            self.alpha,
            self.beta,
        )

        return (
            sbpl(
                L,
                self.Lmin,
                self.Lbreak,
                self.Lmax,
                self.alpha,
                self.beta,
            )
            / C
        )

    def draw_luminosity(self, size=1):

        u = stats.uniform(0, 1).rvs(size)

        samples = sample_sbpl(
            u,
            self.Lmin,
            self.Lbreak,
            self.Lmax,
            self.alpha,
            self.beta,
        )

        return samples


def sample_sbpl(u, x0, x1, x2, a1, a2, size=1):
    """
    Use inverse transform sampling to approx
    samples from an sbpl model.

    :param u: uniform samples between 0 and 1.
    All other params as in sbpl()
    """

    log10_diff = np.log10(x2) - np.log10(x0)

    if log10_diff > 8:

        raise ValueError(
            "Range (x0, x2) is too large to be sampled accurately without taking forever"
        )

    N = max(int(1e3), 10 ** int(log10_diff))

    x = np.linspace(x0, x2, N)

    y = sbpl(x, x0, x1, x2, a1, a2)

    cdf = np.cumsum(y)
    cdf = cdf / cdf.max()

    inv_cdf = interpolate.interp1d(cdf, x, fill_value="extrapolate")

    return inv_cdf(u)


def integrate_sbpl(x0, x1, x2, a1, a2):
    """
    Approx the sbpl integral using trapz.

    :param x0: lower bound
    :param x1: break point
    :param x2: upper bound
    :param a1: lower pl index
    :param a2: upper pl index
    """

    log10_diff = np.log10(x2) - np.log10(x0)

    if log10_diff > 8:

        raise ValueError(
            "Range (x0, x1) is too large to be integrated accurately without taking forever"
        )

    N = max(int(1e6), 10 ** int(log10_diff))

    x = np.linspace(x0, x2, N)

    f = sbpl(x, x0, x1, x2, a1, a2)

    C = np.trapz(f, x)

    return C


def sbpl(x, x0, x1, x2, a1, a2, limit=True):
    """
    Shape of sbpl distribution.

    :param x0: lower bound
    :param x1: break point
    :param x2: upper bound
    :param a1: lower pl index
    :param a2: upper pl index
    """

    f = np.power((x / x1) ** a1 + (x / x1) ** a2, -1)

    if limit:
        f[x < x0] = 0
        f[x > x2] = 0

    return f

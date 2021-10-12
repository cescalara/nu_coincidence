import abc

from popsynth.populations.bpl_population import BPLSFRPopulation
from cosmic_coincidence.populations.popsynth_wrapper import PopsynthWrapper


class ToyBLLacPopulation(BPLSFRPopulation):
    """
    A simplified BL Lac model with a fixed
    broken power law luminosity function and
    SFR-like evolution. Similar to "pure
    density evolution" (PDE) models discussed
    in the literature.
    """

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
        is_rate=False,
    ):

        super().__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            Lmin=Lmin,
            alpha=alpha,
            Lbreak=Lbreak,
            beta=beta,
            Lmax=Lmax,
            r_max=r_max,
            seed=seed,
            is_rate=False,
        )


class BlazarPopWrapper(PopsynthWrapper):
    """
    Parallel simulation wrapper for standard
    blazar populations.
    """

    def __init__(self, parameter_server):

        super().__init__(parameter_server)


class BlazarPopParams(object, metaclass=abc.ABCMeta):
    """
    Parallel simulation parameter server for
    standard blazar populations.
    """

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
        is_rate=False,
        **kwargs,
    ):

        self._parameters = dict(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            Lmin=Lmin,
            alpha=alpha,
            Lbreak=Lbreak,
            beta=beta,
            Lmax=Lmax,
            r_max=r_max,
            seed=seed,
            is_rate=is_rate,
        )

        for k, v in kwargs.items():

            self._parameters[k] = v

        self._file_path = None

    @property
    def parameters(self):

        return self._parameters

    def set_file_path(self, file_path):

        self._file_path = file_path

    @property
    def file_path(self):

        return self._file_path

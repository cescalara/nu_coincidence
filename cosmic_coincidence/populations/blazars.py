from popsynth.populations.bpl_population import BPLSFRPopulation


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


class PDEBlazarPopulation(BPLSFRPopulation):
    """
    A BL Lac-like population based on the
    results for the "luminosity-dependent
    density evolution" (LDDE) model reported
    in Ajello et al. 2014.
    """

    def __init__(self):

        pass


class FSRQPopulation:
    """
    Ajello et al. 2012.
    """

    def __init__(self):

        pass

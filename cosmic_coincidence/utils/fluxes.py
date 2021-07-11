import numpy as np


def flux_conv(index, E_min, E_max):
    """
    Convert from energy flux [erg cm^-2 s^-1]
    to number flux [cm^-2 s^-1] assuming
    a bounded power law spectrum.

    :param index: Spectral index
    :param E_min: Minimum energy
    :param E_max: Maximum energy
    """

    if index == 1.0:

        f1 = np.log(E_max) - np.log(E_min)

    else:

        f1 = (1 / (1 - index)) * (
            np.power(E_max, 1 - index) - np.power(E_min, 1 - index)
        )

    if index == 2.0:

        f2 = np.log(E_max) - np.log(E_min)

    else:

        f2 = (1 / (2 - index)) * (
            np.power(E_max, 2 - index) - np.power(E_min, 2 - index)
        )

    return f1 / f2

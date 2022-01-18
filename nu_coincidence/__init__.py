from nu_coincidence.populations.aux_samplers import (
    SpectralIndexAuxSampler,
    VariabilityAuxSampler,
    FlareRateAuxSampler,
    FlareTimeAuxSampler,
    FlareDurationAuxSampler,
    FlareAmplitudeAuxSampler,
    CombinedFluxIndexSampler,
)

from nu_coincidence.populations.selection import CombinedFluxIndexSelection

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

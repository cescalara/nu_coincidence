from popsynth.selection_probability.generic_selectors import SoftSelection
from popsynth.selection_probability.selection_probability import SelectionParameter


class CombinedFluxIndexSelection(SoftSelection):
    """
    Selection on :class:`CombinedFluxIndexSampler`,
    with the form:

    index = ``slope`` log10(flux) + ``intercept``

    :class:`CombinedFluxIndexSampler` transforms to:
    -(index - ``slope`` log10(flux))
    such that a constant selection can be made
    on -``intercept``.

    See e.g. Fig. 4 in Ajello et al. 2020 (4LAC),
    default values are set to approximate this.
    """

    _selection_name = "CombinedFluxIndexSelection"

    boundary = SelectionParameter(default=-37.5)
    strength = SelectionParameter(default=5, vmin=0)

    def __init__(
        self,
        name: str = "CombinedFluxIndexSelection",
        use_obs_value: bool = True,
    ):

        super(CombinedFluxIndexSelection, self).__init__(
            name=name, use_obs_value=use_obs_value
        )

    def draw(self, size: int, use_log: bool = False):
        """
        Override draw to not use log values.
        """

        super().draw(size, use_log)

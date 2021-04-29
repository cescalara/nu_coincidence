from astropy.coordinates import SkyCoord

from popsynth.selection_probability.spatial_selection import SpatialSelection
from popsynth.selection_probability.selection_probability import SelectionParameter


class GalacticPlaneSelection(SpatialSelection):

    _selection_name = "GalacticPlaneSelection"

    b_limit = SelectionParameter(vmin=0, vmax=90)

    def __init__(self, name="galactic plane selector"):
        """
        places a limit above the galactic plane for objects
        """
        super(GalacticPlaneSelection, self).__init__(name=name)

    def draw(self, size: int):

        c = SkyCoord(
            self._spatial_distribution.ra,
            self._spatial_distribution.dec,
            unit="deg",
            frame="icrs",
        )

        b = c.galactic.b.deg

        self._selection = (b >= self.b_limit) | (b <= -self.b_limit)

from __future__ import division, absolute_import, print_function

import os
import numpy as np
from pkg_resources import resource_filename

from .skymapcp import SkymapCP
from .utils import setdefaults

class SurveySkymapCP(SkymapCP):
    """
    """

    def draw_des17(self, **kwargs):
        """
        Draw the DES footprint
        """

        defaults = dict(color='blue', lw=2)
        setdefaults(kwargs, defaults)

        filename = resource_filename(__name__, 'data/footprints/des-round17-poly.txt')

        self.draw_polygon_file(filename, **kwargs)

class DESSkymapCP(SurveySkymapCP):
    """
    """

    def draw_footprint(self, **kwargs):
        self.draw_des17(**kwargs)

    @property
    def _default_rarange(self):
        return [-70.0, 110.0]

    @property
    def _default_decrange(self):
        return [-70.0, 10.0]

    @property
    def _default_xlocs(self):
        return [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]

    @property
    def _default_ylocs(self):
        return [-75.0, -60.0, -45.0, -30.0, -15.0, 0.0]


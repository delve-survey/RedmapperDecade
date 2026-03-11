
from __future__ import division, absolute_import, print_function

import os
import numpy as np
from pkg_resources import resource_filename

from .hpmap import HpMap

class DESHpMap(HpMap):
    def __init__(self, *args, **kwargs):
        filename = kwargs.pop('filename', None)

        self.load_footprint(filename)

        super(DESHpMap, self).__init__(*args, **kwargs)

    def load_footprint(self, filename=None):
        """
        """

        if filename is None:
            filename = resource_filename(__name__, 'data/footprints/des-round17-poly.txt')

        if not os.path.isfile(filename):
            raise IOError("Could not find footprint file %s" % (filename))

        self.footprint = np.genfromtxt(filename, names=['ra', 'dec'])

    def _default_xlocs(self):
        return [-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0]

    def _default_ylocs(self):
        return [-75.0, -60.0, -45.0, -30.0, -15.0, 0.0]

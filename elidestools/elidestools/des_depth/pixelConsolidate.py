from __future__ import division, absolute_import, print_function

import numpy as np
import healpy as hp
import fitsio
import os
import sys

def pixelConsolidate(pixFiles, outFile, nSide, nest=False):
    """
    Consolidate all the pixels into one file.

    Parameters
    ----------
    pixFiles: `list`
       List of files to consolidate
    outFile: `str`
       Output file
    nSide: `int`
       Nside of coarse map.
    nest: `bool`, optional
       Were the pixels nest pixels?  Default is False.
    """

    started=False

    for pixFile in pixFiles:
        tempCat = fitsio.read(pixFile,ext=1)

        if (not started):
            hdr=fitsio.FITSHDR()
            hdr['NSIDE'] = nSide
            hdr['NEST'] = nest
            fitsio.write(outFile,tempCat,clobber=True,header=hdr)
            fits=fitsio.FITS(outFile,mode='rw')
            started=True
        else:
            fits[1].append(tempCat)

    fits.close()

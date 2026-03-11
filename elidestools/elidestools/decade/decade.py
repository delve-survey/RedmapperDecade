import numpy as np
import healpy as hp
import fitsio
import esutil
import os
import glob
import re
import h5py
from tqdm import tqdm

class Y6A1Gold2_0Collator(object):
    def __init__(self, outBase, nside=8):
        self.outBase = outBase
        self.nside=nside

    def run(self, goldFiles):
        """
        """

        bands=['g', 'r', 'i', 'z']

        dtypeBasic = [
                      ('ra', 'f8'),
                      ('dec', 'f8')]

        dtypeBDF = [
                    ('flux_bdf_g', 'f4'),
                    ('flux_bdf_r', 'f4'),
                    ('flux_bdf_i', 'f4'),
                    ('flux_bdf_z', 'f4'),
                    ('fluxerr_bdf_g', 'f4'),
                    ('fluxerr_bdf_r', 'f4'),
                    ('fluxerr_bdf_i', 'f4'),
                    ('fluxerr_bdf_z', 'f4')]
        inCat = h5py.File(goldFiles, 'r')

        ipring = hp.ang2pix(self.nside, inCat['RA'], inCat['DEC'], lonlat=True)

        pixMin = ipring.min()
        pixMax = ipring.max()

        h, rev = esutil.stat.histogram(ipring, rev=True, min=pixMin, max=pixMax)
        hpix = np.arange(pixMin, pixMax+1, dtype='i8')

        for i in tqdm(range(h.size)):
            if (h[i] == 0):
                continue

            i1a=rev[rev[i]:rev[i+1]]
            print(i,i1a.size, flush=True)

            basicCat = np.zeros(i1a.size, dtype=dtypeBasic)

            basicCat['ra'] = inCat['RA'][:][i1a]
            basicCat['dec'] = inCat['DEC'][:][i1a]

            bdfCat = np.zeros(i1a.size, dtype=dtypeBDF)
            print(i,"band started", flush=True)

            for b in bands:
                bdfCat['flux_bdf_' + b] = inCat['BDF_FLUX_{0}_DERED_SFD98'.format(b.upper())][:][i1a]
                bdfCat['fluxerr_bdf_' + b] = inCat['BDF_FLUX_ERR_{0}_DERED_SFD98'.format(b.upper())][:][i1a]
                print(i,"band ", b, " done", flush=True)

            basicFile = '%s_pix%05d_basic.fits' % (self.outBase, hpix[i])
            bdfFile = '%s_pix%05d_bdf.fits' % (self.outBase, hpix[i])
            print(i,"Ready write", flush=True)

            if os.path.isfile(basicFile):
                fits=fitsio.FITS(basicFile, mode='rw')
                fits[1].append(basicCat)
                fits.close()

                fits=fitsio.FITS(bdfFile, mode='rw')
                fits[1].append(bdfCat)
                fits.close()
            else:
                fitsio.write(basicFile, basicCat, clobber=True)
                fitsio.write(bdfFile, bdfCat, clobber=True)
            print(i,"Done", flush=True)

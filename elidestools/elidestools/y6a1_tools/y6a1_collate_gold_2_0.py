import numpy as np
import healpy as hp
import fitsio
import esutil
import os
import glob
import re

class Y6A1Gold2_0Collator(object):
    def __init__(self, outBase, nside=8):
        self.outBase = outBase
        self.nside=nside

    def run(self, goldFiles):
        """
        """

        bands=['g', 'r', 'i', 'z', 'y']

        dtypeBasic = [('coadd_object_id', 'i8'),
                      ('ra', 'f8'),
                      ('dec', 'f8'),
                      ('tilename', 'a12'),
                      ('ext_coadd', 'i2'),
                      ('ext_fitvd', 'i2'),
                      ('ext_mash', 'i2'),
                      ('ebv_sfd98', 'f4'),
                      ('flags_gold', 'i4'),
                      ('flags_foreground', 'i4'),
                      ('flags_footprint', 'i4'),
                      ('badflag', 'i2')]

        dtypeBDF = [('coadd_object_id', 'i8'),
                    ('bdf_flags', 'i8'),
                    ('bdf_deblend_flags', 'i8'),
                    ('bdf_fracdev', 'f4'),
                    ('bdf_fracdev_err', 'f4'),
                    ('flux_bdf_g', 'f4'),
                    ('flux_bdf_r', 'f4'),
                    ('flux_bdf_i', 'f4'),
                    ('flux_bdf_z', 'f4'),
                    ('flux_bdf_y', 'f4'),
                    ('fluxerr_bdf_g', 'f4'),
                    ('fluxerr_bdf_r', 'f4'),
                    ('fluxerr_bdf_i', 'f4'),
                    ('fluxerr_bdf_z', 'f4'),
                    ('fluxerr_bdf_y', 'f4')]

        for f in goldFiles:
            print("Reading ", f)

            inCat = fitsio.read(f, ext=1, lower=True)

            ipring = hp.ang2pix(self.nside, inCat['ra'], inCat['dec'], lonlat=True)

            pixMin = ipring.min()
            pixMax = ipring.max()

            h, rev = esutil.stat.histogram(ipring, rev=True, min=pixMin, max=pixMax)
            hpix = np.arange(pixMin, pixMax+1, dtype='i8')

            for i in range(h.size):
                if (h[i] == 0):
                    continue

                i1a=rev[rev[i]:rev[i+1]]

                basicCat = np.zeros(i1a.size, dtype=dtypeBasic)

                basicCat['coadd_object_id'] = inCat['coadd_object_id'][i1a]
                basicCat['ra'] = inCat['ra'][i1a]
                basicCat['dec'] = inCat['dec'][i1a]
                basicCat['tilename'] = inCat['tilename'][i1a]
                basicCat['ext_coadd'] = inCat['ext_coadd'][i1a]
                basicCat['ext_fitvd'] = inCat['ext_fitvd'][i1a]
                basicCat['ext_mash'] = inCat['ext_mash'][i1a]
                basicCat['ebv_sfd98'] = inCat['ebv_sfd98'][i1a]
                basicCat['flags_gold'] = inCat['flags_gold'][i1a]
                basicCat['flags_foreground'] = inCat['flags_foreground'][i1a]
                basicCat['flags_footprint'] = inCat['flags_footprint'][i1a]

                bdfCat = np.zeros(i1a.size, dtype=dtypeBDF)
                bdfCat['coadd_object_id'] = inCat['coadd_object_id'][i1a]
                bdfCat['bdf_flags'] = inCat['bdf_flags'][i1a]
                bdfCat['bdf_deblend_flags'] = inCat['bdf_deblend_flags'][i1a]
                bdfCat['bdf_fracdev'] = inCat['bdf_fracdev'][i1a]
                bdfCat['bdf_fracdev_err'] = inCat['bdf_fracdev_err'][i1a]

                for b in bands:
                    bdfCat['flux_bdf_' + b] = inCat['bdf_flux_' + b][i1a]
                    bdfCat['fluxerr_bdf_' + b] = inCat['bdf_flux_err_' + b][i1a]

                basicFile = '%s_pix%05d_basic.fits' % (self.outBase, hpix[i])
                bdfFile = '%s_pix%05d_bdf.fits' % (self.outBase, hpix[i])

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

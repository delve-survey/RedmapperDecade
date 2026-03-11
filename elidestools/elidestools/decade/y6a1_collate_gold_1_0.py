from __future__ import division, absolute_import, print_function

import numpy as np
import healpy as hp
import fitsio
import esutil
import os
import glob
import re

class Y6A1Gold1_0Collator(object):
    def __init__(self, outBase, nside=8):
        self.outBase = outBase
        self.nside=nside

    def run(self, goldFiles):
        """
        """

        bands=['g','r','i','z']

        dtypeBasic = [('coadd_object_id', 'i8'),
                      ('ra', 'f8'),
                      ('dec', 'f8'),
                      ('tilename', 'a12'),
                      ('ext_mash', 'i2'),
                      ('ebv_sfd98', 'f4'),
                      ('ebv_planck13', 'f4'),
                      ('ebv_lenz17', 'f4'),
                      ('flags_gold', 'i4'),
                      ('flags_foreground', 'i4'),
                      ('flags_footprint', 'i4'),
                      ('badflag', 'i2')]

        dtypeMOF = [('coadd_object_id', 'i8'),
                    ('mof_flags', 'i8'),
                    ('mof_bd_flags', 'i8'),
                    ('mof_bd_fracdev', 'f4'),
                    ('mof_bd_fracdev_err', 'f4'),
                    ('flux_bd_mof_g', 'f4'),
                    ('flux_bd_mof_r', 'f4'),
                    ('flux_bd_mof_i', 'f4'),
                    ('flux_bd_mof_z', 'f4'),
                    ('fluxerr_bd_mof_g', 'f4'),
                    ('fluxerr_bd_mof_r', 'f4'),
                    ('fluxerr_bd_mof_i', 'f4'),
                    ('fluxerr_bd_mof_z', 'f4')]

        dtypeSOF = [('coadd_object_id', 'i8'),
                    ('sof_flags', 'i8'),
                    ('sof_bd_flags', 'i8'),
                    ('sof_bd_fracdev', 'f4'),
                    ('sof_bd_fracdev_err', 'f4'),
                    ('flux_bd_sof_g', 'f4'),
                    ('flux_bd_sof_r', 'f4'),
                    ('flux_bd_sof_i', 'f4'),
                    ('flux_bd_sof_z', 'f4'),
                    ('fluxerr_bd_sof_g', 'f4'),
                    ('fluxerr_bd_sof_r', 'f4'),
                    ('fluxerr_bd_sof_i', 'f4'),
                    ('fluxerr_bd_sof_z', 'f4')]

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
                basicCat['ext_mash'] = inCat['ext_mash'][i1a]
                basicCat['ebv_sfd98'] = inCat['ebv_sfd98'][i1a]
                basicCat['ebv_planck13'] = inCat['ebv_planck13'][i1a]
                basicCat['ebv_lenz17'] = inCat['ebv_lenz17'][i1a]
                basicCat['flags_gold'] = inCat['flags_gold'][i1a]
                basicCat['flags_foreground'] = inCat['flags_foreground'][i1a]
                basicCat['flags_footprint'] = inCat['flags_footprint'][i1a]

                mofCat = np.zeros(i1a.size, dtype=dtypeMOF)
                mofCat['coadd_object_id'] = inCat['coadd_object_id'][i1a]
                mofCat['mof_flags'] = inCat['mof_flags'][i1a]
                mofCat['mof_bd_flags'] = inCat['mof_bdf_flags'][i1a]
                mofCat['mof_bd_fracdev'] = inCat['mof_bdf_fracdev'][i1a]
                mofCat['mof_bd_fracdev_err'] = inCat['mof_bdf_fracdev_err'][i1a]

                for b in bands:
                    mofCat['flux_bd_mof_' + b] = inCat['mof_bdf_flux_' + b][i1a]
                    mofCat['fluxerr_bd_mof_' + b] = inCat['mof_bdf_flux_err_' + b][i1a]

                sofCat = np.zeros(i1a.size, dtype=dtypeSOF)
                sofCat['coadd_object_id'] = inCat['coadd_object_id'][i1a]
                sofCat['sof_flags'] = inCat['sof_flags'][i1a]
                sofCat['sof_bd_flags'] = inCat['sof_bdf_flags'][i1a]
                sofCat['sof_bd_fracdev'] = inCat['sof_bdf_fracdev'][i1a]
                sofCat['sof_bd_fracdev_err'] = inCat['sof_bdf_fracdev_err'][i1a]

                for b in bands:
                    sofCat['flux_bd_sof_' + b] = inCat['sof_bdf_flux_' + b][i1a]
                    sofCat['fluxerr_bd_sof_' + b] = inCat['sof_bdf_flux_err_' + b][i1a]

                basicFile = '%s_pix%05d_basic.fits' % (self.outBase, hpix[i])
                mofFile = '%s_pix%05d_mof.fits' % (self.outBase, hpix[i])
                sofFile = '%s_pix%05d_sof.fits' % (self.outBase, hpix[i])

                if os.path.isfile(basicFile):
                    fits=fitsio.FITS(basicFile, mode='rw')
                    fits[1].append(basicCat)
                    fits.close()

                    fits=fitsio.FITS(mofFile, mode='rw')
                    fits[1].append(mofCat)
                    fits.close()

                    fits=fitsio.FITS(sofFile, mode='rw')
                    fits[1].append(sofCat)
                    fits.close()
                else:
                    fitsio.write(basicFile, basicCat, clobber=True)
                    fitsio.write(mofFile, mofCat, clobber=True)
                    fitsio.write(sofFile, sofCat, clobber=True)

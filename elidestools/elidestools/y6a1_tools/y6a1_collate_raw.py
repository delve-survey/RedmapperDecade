from __future__ import division, absolute_import, print_function

import numpy as np
import healpy as hp
import fitsio
import esutil
import os
import glob
import re

class Y6A1RawCollator(object):
    def __init__(self, outBase, nside=8):
        self.outBase = outBase

        self.nside=nside

    def runSummary(self, coaddSummaryFiles):
        """
        Collate the summary tables
        """

        bands = ['g', 'r', 'i', 'z', 'y']

        dtypeBasic = [('coadd_object_id', 'i8'),
                      ('ra', 'f8'),
                      ('dec', 'f8'),
                      ('modest_class', 'i2'),
                      ('modest_mash', 'i2'),
                      ('ebv_sfd98', 'f4'),
                      ('badflag', 'i2')]

        dtypeAuto = [('coadd_object_id', 'i8'),
                     ('flux_auto_g', 'f4'),
                     ('flux_auto_r', 'f4'),
                     ('flux_auto_i', 'f4'),
                     ('flux_auto_z', 'f4'),
                     ('flux_auto_y', 'f4'),
                     ('fluxerr_auto_g', 'f4'),
                     ('fluxerr_auto_r', 'f4'),
                     ('fluxerr_auto_i', 'f4'),
                     ('fluxerr_auto_z', 'f4'),
                     ('fluxerr_auto_y', 'f4')]

        dtypeWavg = [('coadd_object_id', 'i8'),
                     ('flux_wavg_g', 'f4'),
                     ('flux_wavg_r', 'f4'),
                     ('flux_wavg_i', 'f4'),
                     ('flux_wavg_z', 'f4'),
                     ('flux_wavg_y', 'f4'),
                     ('fluxerr_wavg_g', 'f4'),
                     ('fluxerr_wavg_r', 'f4'),
                     ('fluxerr_wavg_i', 'f4'),
                     ('fluxerr_wavg_z', 'f4'),
                     ('fluxerr_wavg_y', 'f4')]

        for f in coaddSummaryFiles:
            print("Reading ", f)

            inCat = fitsio.read(f, ext=1, lower=True)

            gd, = np.where((inCat['flags_g'] <= 3) &
                           (inCat['flags_r'] <= 3) &
                           (inCat['flags_i'] <= 3) &
                           (inCat['flags_z'] <= 3) &
                           (inCat['imaflags_iso_g'] == 0) &
                           (inCat['imaflags_iso_r'] == 0) &
                           (inCat['imaflags_iso_i'] == 0) &
                           (inCat['imaflags_iso_z'] == 0) &
                           (inCat['niter_model_g'] > 0) &
                           (inCat['niter_model_r'] > 0) &
                           (inCat['niter_model_i'] > 0) &
                           (inCat['niter_model_z'] > 0))

            ipring = hp.ang2pix(self.nside, inCat['ra'][gd], inCat['dec'][gd], lonlat=True)

            pixMin = ipring.min()
            pixMax = ipring.max()

            h, rev = esutil.stat.histogram(ipring, rev=True, min=pixMin, max=pixMax)
            hpix = np.arange(pixMin, pixMax+1, dtype='i8')

            for i in range(h.size):
                if (h[i] == 0):
                    continue

                i1a=rev[rev[i]:rev[i+1]]

                basicCat = np.zeros(i1a.size, dtype=dtypeBasic)

                basicCat['coadd_object_id'] = inCat['coadd_object_id'][gd[i1a]]
                basicCat['ra'] = inCat['ra'][gd[i1a]]
                basicCat['dec'] = inCat['dec'][gd[i1a]]
                basicCat['ebv_sfd98'] = inCat['ebv_sfd98'][gd[i1a]]

                autoCat = np.zeros(i1a.size, dtype=dtypeAuto)
                autoCat['coadd_object_id'] = inCat['coadd_object_id'][gd[i1a]]

                for b in bands:
                    autoCat['flux_auto_'+b] = inCat['flux_auto_'+b][gd[i1a]]
                    autoCat['fluxerr_auto_'+b] = inCat['fluxerr_auto_'+b][gd[i1a]]

                wavgCat = np.zeros(i1a.size, dtype=dtypeWavg)
                wavgCat['coadd_object_id'] = inCat['coadd_object_id'][gd[i1a]]

                for b in bands:
                    wavgCat['flux_wavg_'+b] = inCat['wavg_flux_psf_'+b][gd[i1a]]
                    wavgCat['fluxerr_wavg_'+b] = inCat['wavg_fluxerr_psf_'+b][gd[i1a]]

                basicFile = '%s_pix%05d_basic.fits' % (self.outBase, hpix[i])
                autoFile = '%s_pix%05d_auto.fits' % (self.outBase, hpix[i])
                wavgFile = '%s_pix%05d_wavg.fits' % (self.outBase, hpix[i])

                if os.path.isfile(basicFile):
                    fits=fitsio.FITS(basicFile, mode='rw')
                    fits[1].append(basicCat)
                    fits.close()

                    fits=fitsio.FITS(autoFile, mode='rw')
                    fits[1].append(autoCat)
                    fits.close()

                    fits=fitsio.FITS(wavgFile, mode='rw')
                    fits[1].append(wavgCat)
                    fits.close()
                else:
                    fitsio.write(basicFile,basicCat, clobber=True)
                    fitsio.write(autoFile,autoCat, clobber=True)
                    fitsio.write(wavgFile,wavgCat, clobber=True)

    def runBd(self, bdFiles, name):
        """
        Run the BD fluxes
        """

        dtypeBd = [('coadd_object_id', 'i8'),
                   ('bd_flags', 'i8'),
                   ('flux_bd_%s_g' % (name), 'f4'),
                   ('flux_bd_%s_r' % (name), 'f4'),
                   ('flux_bd_%s_i' % (name), 'f4'),
                   ('flux_bd_%s_z' % (name), 'f4'),
                   ('fluxerr_bd_%s_g' % (name), 'f4'),
                   ('fluxerr_bd_%s_r' % (name), 'f4'),
                   ('fluxerr_bd_%s_i' % (name), 'f4'),
                   ('fluxerr_bd_%s_z' % (name), 'f4'),
                   ('flux_psf_%s_g' % (name), 'f4'),
                   ('flux_psf_%s_r' % (name), 'f4'),
                   ('flux_psf_%s_i' % (name), 'f4'),
                   ('flux_psf_%s_z' % (name), 'f4'),
                   ('fluxerr_psf_%s_g' % (name), 'f4'),
                   ('fluxerr_psf_%s_r' % (name), 'f4'),
                   ('fluxerr_psf_%s_i' % (name), 'f4'),
                   ('fluxerr_psf_%s_z' % (name), 'f4'),
                   ('bd_t', 'f8'),
                   ('bd_t_err', 'f8'),
                   ('flags', 'i8')]

        bands = ['g', 'r', 'i', 'z']

        # plan
        #  1) For each file, split into pixels
        #  2) Write out, with appending, to temporary file
        #  3) For each pixel file, do a final scan and match
        #  4) Write out to final file
        #  5) Delete temp file

        for f in bdFiles:
            print("Reading ", f)

            inCat = fitsio.read(f, ext=1, lower=True)

            ipring = hp.ang2pix(self.nside, inCat['ra'], inCat['dec'], lonlat=True)

            pixMin = ipring.min()
            pixMax = ipring.max()

            h, rev = esutil.stat.histogram(ipring, rev=True, min=pixMin, max=pixMax)
            hpix = np.arange(pixMin, pixMax + 1, dtype='i8')

            for i in range(h.size):
                if (h[i] == 0):
                    continue

                i1a=rev[rev[i]:rev[i+1]]

                bdCat = np.zeros(i1a.size, dtype=dtypeBd)

                bdCat['coadd_object_id'] = inCat['coadd_object_id'][i1a]
                bdCat['bd_flags'] = inCat['bdf_flags'][i1a]
                bdCat['bd_t'] = inCat['bdf_t'][i1a]
                bdCat['bd_t_err'] = inCat['bdf_t_err'][i1a]
                bdCat['flags'] = inCat['flags'][i1a]

                for b in bands:
                    bdCat['flux_bd_%s_%s' % (name, b)] = inCat['bdf_flux_' + b][i1a]
                    bdCat['fluxerr_bd_%s_%s' % (name, b)] = inCat['bdf_flux_err_' + b][i1a]
                    bdCat['flux_psf_%s_%s' % (name, b)] = inCat['psf_flux_' + b][i1a]
                    bdCat['fluxerr_psf_%s_%s' % (name, b)] = inCat['psf_flux_err_' + b][i1a]

                bdFile = '%s_pix%05d_%s_temp.fits' % (self.outBase, hpix[i], name)

                if os.path.isfile(bdFile):
                    fits = fitsio.FITS(bdFile, mode='rw')
                    fits[1].append(bdCat)
                    fits.close()
                else:
                    fitsio.write(bdFile, bdCat, clobber=True)

        tempFiles = sorted(glob.glob('%s_pix?????_%s_temp.fits' % (self.outBase, name)))

        for tempFile in tempFiles:
            m = re.search('_pix(\d\d\d\d\d)_', tempFile)
            hpix = int(m.groups()[0])

            print("Working on pixel %d" % (hpix))

            basicFile = '%s_pix%05d_basic.fits' % (self.outBase, hpix)
            bdFileTemp = tempFile
            bdFile = '%s_pix%05d_%s.fits' % (self.outBase, hpix, name)

            bCat = fitsio.read(basicFile, ext=1, lower=True, columns=['coadd_object_id'])
            bdCatTemp = fitsio.read(bdFileTemp, ext=1, lower=True)

            aa, bb = esutil.numpy_util.match(bCat['coadd_object_id'],
                                             bdCatTemp['coadd_object_id'])
            if aa.size != bCat.size:
                print("Warning! %d matches of %d (pixel %d)" % (aa.size, bCat.size, hpix))

            bdCat = np.zeros(bCat.size, dtype=dtypeBd)
            bdCat['bd_flags'][:] = 4294967295
            bdCat['flags'][:] = 4294967295

            bdCat[aa] = bdCatTemp[bb]

            fitsio.write(bdFile, bdCat, clobber=True)









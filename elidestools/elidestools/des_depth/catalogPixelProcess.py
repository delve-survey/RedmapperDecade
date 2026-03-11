from __future__ import division, absolute_import, print_function

from past.builtins import xrange

import numpy as np
import healpy as hp
import fitsio
import esutil
import re
import os
import sys

from .errorModels import calcErrorModel

def catalogPixelProcess(pixFileBase, outBase, typeName, fluxCol, errCol, nSide,
                        bandList = ['g', 'r', 'i', 'z', 'Y'], nTrial=100,
                        minGals=50, nSig=10.0, zp=30.0, s2nCut=5.0,
                        nSidePixFile=8, selectGalaxies=True,
                        bdSizeFileType=None, noGoldFlags=False,
                        checkPoint=100):
    """
    Process a set of single-pixel files to compute depth.

    Parameters
    ----------
    pixFileBase : `str`
        Base name of the pixel file (excluding _basic, _fluxtype, etc)
    outBase : `str`
        Output base name.
    typeName : `str`
        Type of flux to process.
    fluxCol : `str`
        Name of flux column
    errCol : `str`
        Name of flux error column
    nSide : `int`
        Nside of coarse depth map.
    bandList : `list`, optional.
        List of band names.  Default is ['g', 'r', 'i', 'z', 'Y'].
    nTrial : `int`, optional
        Number of trials for bootstrap errors.  Default is 100.
    minGals : `int`, optional
        Minimum number of galaxies to perform a fit.  Default is 50.
    nSig : `float`, optional
        Number of sigma to compute depth.  Default is 10.0
    zp : `float`, optional
        Zeropoint for fluxes.  Default is 30.0.
    s2nCut : `float`, optional
        Signal to noise cut to use in a fit.  Default is 5.0.
    nSidePixFile : `int`, optional
        Nside for input pixel file.  Default is 8.
    selectGalaxies : `bool`, optional
        Select galaxies with somewhat arbitrary selection?
        Default is True.  Must be True until we have a classification to
        select on.
    bdSizeFileType : `str`, optional
        Type of file to get bd_t sizes from.  Must be set if selectGalaxies is True.
        Default is None.
    noGoldFlags : `bool`, optional
        Set if the input catalog has no "gold" flags (for early runs).
        Default is False.
    checkPoint : `int`, optional
        Output results every checkPoint pixels.  Default is 100.
    """
    bands = np.array(bandList)
    nBands = bands.size

    const = 2.5/np.log(10.)

    # need pixFileBase
    m = re.search('pix([0-9]{5})', os.path.basename(pixFileBase))
    oPix = int(m.groups()[0])

    m=re.search('(.*)_pix%05d' % (oPix), pixFileBase)
    pixFileBaseNoPix = m.groups()[0]

    outFile = '%s_pix%05d.fits' % (outBase, oPix)

    allPars = None
    if (os.path.isfile(outFile)):
        allPars = fitsio.read(outFile, ext=1)

        if np.min(allPars['NGAL'][:, 0] >= 0):
            print("File %s is already done!" % (outFile))
            return

    pixFile = '%s_basic.fits' % (pixFileBase)
    fluxFile = '%s_%s.fits' % (pixFileBase, typeName)

    basicColumns = ['ra', 'dec']
    fluxColumns = []
    for band in bands:
        fluxColumns.append(fluxCol.lower() + '_' + band.lower())
        fluxColumns.append(errCol.lower() + '_' + band.lower())
    if not noGoldFlags:
        basicColumns.append('flags_gold')

    # do we have some ngmix flags or ext_mash?
    bdf_flag_name = None
    has_ext_mash = False
    with fitsio.FITS(fluxFile) as fits:
        if ('BD_FLAGS' in fits[1].get_colnames() or
            'bd_flags' in fits[1].get_colnames()):
            bdf_flag_name = 'bd_flags'
            has_bd_flags = True
        if ('BDF_FLAGS' in fits[1].get_colnames() or
            'bdf_flags' in fits[1].get_colnames()):
            bdf_flag_name = 'bdf_flags'
        if bdf_flag_name is not None:
            fluxColumns.append(bdf_flag_name)

    has_ext_mash = False
    with fitsio.FITS(pixFile) as fits:
        if ('ext_mash' in fits[1].get_colnames() or
            'EXT_MASH' in fits[1].get_colnames()):
            has_ext_mash = True
            basicColumns.append('ext_mash')

    if not has_ext_mash:
        if not selectGalaxies:
            raise RuntimeError("Must have selectGalaxies set to True until later runs when we have a star/galaxy selection field.")

        sizeColumns = []
        sizeColumns.append('bd_t')
        sizeColumns.append('bd_t_err')
        sizeFile = '%s_%s.fits' % (pixFileBase, bdSizeFileType)

    print("Reading pixfile: %s" % (pixFile))
    inCat = fitsio.read(pixFile, ext=1, columns=basicColumns, lower=True)
    print("Reading fluxfile: %s" % (fluxFile))
    inFluxCat = fitsio.read(fluxFile, ext=1, columns=fluxColumns, lower=True)
    if not has_ext_mash:
        print("Reading sizefile: %s" % (sizeFile))
        #inSizeCat = fitsio.read(sizeFile, ext=1, columns=sizeColumns, lower=True)

    # determine the neighboring pixels that we will need, at nside=nside/4
    nside_boundarypix = nSide // 4
    boundaries = hp.boundaries(nSidePixFile, oPix, step=nside_boundarypix // nSidePixFile)
    inhpix = []
    border = 0.5 # degrees
    for i in xrange(boundaries.shape[1]):
        pixint = hp.query_disc(nside_boundarypix, boundaries[:, i], np.radians(border), inclusive=True, fact=8)
        inhpix.extend(list(pixint))
    inhpix = np.unique(np.array(inhpix, dtype=np.int64))

    neighborPixels = hp.get_all_neighbours(nSidePixFile, oPix)
    for neighborPixel in neighborPixels:
        neighborFile = '%s_pix%05d_basic.fits' % (pixFileBaseNoPix, neighborPixel)
        neighborFluxFile = '%s_pix%05d_%s.fits' % (pixFileBaseNoPix, neighborPixel, typeName)
        if os.path.isfile(neighborFile):
            print("Additionally reading %s" % (neighborFile))
            tempCat = fitsio.read(neighborFile, ext=1, columns=basicColumns, lower=True)
            tempFluxCat = fitsio.read(neighborFluxFile, ext=1, columns=fluxColumns, lower=True)
            #if not has_ext_mash:
               # neighborSizeFile = '%s_pix%05d_%s.fits' % (pixFileBaseNoPix, neighborPixel, bdSizeFileType)
               # tempSizeCat = fitsio.read(neighborSizeFile, ext=1, columns=sizeColumns, upper=True)

            neighborPix = hp.ang2pix(nside_boundarypix, tempCat['ra'], tempCat['dec'], lonlat=True)
            aa, bb = esutil.numpy_util.match(inhpix, neighborPix)

            inCat = np.append(inCat, tempCat[bb])
            inFluxCat = np.append(inFluxCat, tempFluxCat[bb])
            #if not has_ext_mash:
                #inSizeCat = np.append(inSizeCat, tempSizeCat[bb])

    if not has_ext_mash:
        # select galaxies based on the extendedness
        #extendedClass = np.ones(inSizeCat.size, dtype=np.int32)

        #xs = np.array([22.0, 23.5])
        #ys = np.array([0.0, -0.025])
        #m = (ys[1] - ys[0]) / (xs[1] - xs[0])

        #flux = np.nan_to_num(inFluxCat[fluxCol.upper() + '_I'])
        #use, = np.where((flux > 0.0) & (np.isfinite(flux)))

        #imag = zp - 2.5 * np.log10(flux[use])

        #cut = m * (imag - xs[0]) + ys[0]

        #gals, = np.where(inSizeCat['bd_t'][use] > cut)
        #extendedClass[use[gals]] = 3
        pass
    else:
        pass
        #extendedClass = inCat['ext_mash']

    #galFlag = (extendedClass >= 2)
    

    #if (not noGoldFlags):
        # is not required to be complete everywhere
    #    gFlag = ((inCat['flags_gold'] & 255) == 0)
    #else:
    #    gFlag = np.ones(inCat.size, dtype=np.bool)

    #if bdf_flag_name is not None:
    #    print("Filtering bad %s" % (bdf_flag_name))
    #    cmFlag = (inFluxCat[bdf_flag_name] == 0)
    #else:
    #    print("No bd_flag filtering")
    #    cmFlag = np.ones(inFluxCat.size, dtype=np.bool)

    #uFlag = np.logical_and.reduce([galFlag,
    #                               gFlag,
    #                               cmFlag])
    #inCat = inCat[uFlag]
    #inFluxCat = inFluxCat[uFlag]

    cat = np.zeros(inCat.size, dtype=[('RA', 'f8'),
                                      ('DEC', 'f8'),
                                      ('HPIX', 'i8'),
                                      ('MAG', 'f4', nBands),
                                      ('MAG_ERR', 'f4', nBands),
                                      ('S2N', 'f4', nBands)])

    cat['RA'] = inCat['ra']
    cat['DEC'] = inCat['dec']
    cat['MAG'][:, :] = 99.0
    cat['MAG_ERR'][:, :] = 99.0
    cat['S2N'][:, :] = 0.0

    for i, band in enumerate(bands):
        flux = np.nan_to_num(inFluxCat[fluxCol.lower() + '_' + band.lower()][:])
        fluxErr = np.nan_to_num(inFluxCat[errCol.lower() + '_' + band.lower()][:])
        use, = np.where((flux > 0.0) & (np.isfinite(flux)))

        cat['MAG'][use, i] = zp - 2.5 * np.log10(flux[use])
        cat['MAG_ERR'][use, i] = const * fluxErr[use] / flux[use]

        cat['S2N'][use, i] = flux[use] / fluxErr[use]

    inCat = None
    inFluxCat = None
    inSizeCat = None

    theta = (90.0 - cat['DEC'])*np.pi/180.
    phi = cat['RA']*np.pi/180.

    cat['HPIX'][:] = hp.ang2pix(nSide, theta, phi)

    # want the unique list of pixels that are in the catalog
    uPix = np.unique(cat['HPIX'])

    minPix = uPix.min()
    hist, rev = esutil.stat.histogram(cat['HPIX'], rev=True, min=minPix)

    subPixels = minPix + np.arange(hist.size)

    # only consider pixels in the original pixel!
    tTheta, tPhi = hp.pix2ang(nSide, subPixels)
    testPixels = hp.ang2pix(nSidePixFile, tTheta, tPhi)

    gd, = np.where((testPixels == oPix) & (hist > 0))

    subPixels = gd + minPix

    # this is all possible pixels: will need to cut this down!
    if (allPars is None):
        allPars = np.zeros(gd.size,dtype=[('HPIX','i8'),
                                          ('LIMMAG','f4',nBands),
                                          ('EXPTIME','f4',nBands),
                                          ('LIMMAG_ERR','f4',nBands),
                                          ('EXPTIME_ERR','f4',nBands),
                                          ('NGAL','i4',nBands),
                                          ('NPIX_FIT','i4',nBands)])

        allPars['HPIX'][:] = subPixels
        # set the sentinal value
        allPars['NGAL'][:,0] = -1

    print('Subpixels: %d' % (subPixels.size))

    ctr=0

    for i in xrange(gd.size):
        if (allPars['NGAL'][i,0] >= 0):
            # we've already done this subpixel
            continue

        if ((i % 10) == 0):
            sys.stdout.write('%d' % (i))
        else:
            sys.stdout.write('.')
        sys.stdout.flush()
        # get the pixel number
        i1a = rev[rev[gd[i]]: rev[gd[i]+1]]
        i1aOrig = i1a.copy()

        thesePixels = np.array([subPixels[i]])

        gInd = i1a.copy()

        done = np.zeros(nBands, dtype=bool)
        embiggenLevel = 0
        # signal that this has been processed
        allPars['NGAL'][i, 0] = 0

        while (not done.min()):
            embiggen = False

            if (gInd.size < minGals):
                embiggen = True
            else:
                for j, band in enumerate(bands):
                    if (done[j]): continue

                    use, = np.where((cat['S2N'][gInd, j] >= s2nCut) &
                                    (np.isfinite(cat['MAG'][gInd, j])))

                    if (use.size < minGals):
                        embiggen=True
                    else:
                        pars, fail = calcErrorModel(cat['MAG'][gInd[use], j],
                                                    cat['MAG_ERR'][gInd[use], j],
                                                    nSig=nSig,
                                                    nTrial=nTrial,
                                                    calcErr=True,
                                                    snCut=s2nCut,
                                                    zp=zp)
                        #pars, fail, fig, ax = calcErrorModel(cat['MAG'][gInd[use],j],cat['MAG_ERR'][gInd[use],j],nSig=nSig,nTrial=nTrial,calcErr=True,snCut=s2nCut,zp=zp,doPlot=True)

                        if (fail):
                            embiggen=True
                        else:
                            #fig.savefig('testing_%s_%010d.png' % (band, allPars['HPIX'][i]))
                            done[j] = True
                            allPars['LIMMAG'][i, j] = pars['LIMMAG']
                            allPars['EXPTIME'][i, j] = pars['EXPTIME']
                            allPars['LIMMAG_ERR'][i, j] = pars['LIMMAG_ERR']
                            allPars['EXPTIME_ERR'][i, j] = pars['EXPTIME_ERR']
                            allPars['NPIX_FIT'][i, j] = thesePixels.size
                            allPars['NGAL'][i, j] = gInd.size

            if (embiggen):
                embiggenLevel += 1
                if (embiggenLevel > 2):
                    # just no good
                    test, = np.where(~done)
                    print("%d Failed on " % (subPixels[i]), test)
                    done[test] = True
                else:
                    # get the neighbors...(only once)
                    for p in thesePixels:
                        n = hp.get_all_neighbours(nSide, p)
                        thesePixels = np.append(thesePixels, n)

                    # unique!
                    # and cut out -1s which shouldn't be there...
                    ok, = np.where(thesePixels > 0)
                    thesePixels = np.unique(thesePixels[ok])

                    _, gInd = esutil.numpy_util.match(thesePixels, cat['HPIX'])

        if (i > 0) and ((i % checkPoint) == 0):
            # checkpoint save
            sys.stdout.write('!')
            sys.stdout.flush()
            fitsio.write(outFile, allPars, clobber=True)

    # and write out final version

    fitsio.write(outFile, allPars, clobber=True)
    print("\nDone!")

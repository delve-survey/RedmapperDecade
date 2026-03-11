from __future__ import division, absolute_import, print_function

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import healpy as hp
import fitsio
import esutil
import matplotlib.pyplot as plt
import subprocess
import warnings
import os


class MakeMap(object):
    """
    Make a high resolution depth map from a coarse depth map and some
    systematics maps.

    Parameters
    ----------
    depthFile: `str`
       The coarse depth file as an input
    sysPath: `str`
       Path for systematics maps
    sysTemplate: `str`
       Template for systematics map name.  Template should have two
       %s locations for the band and the systematics type
    outBase: `str`
       Output filename base, all files will start with this.
    minGalFit: `int`, optional
       Minimum number of galaxies in a pixel to use in the RF model.
       Default is 50.
    maglimType: `str`, optional
       Sub-name of the magnitude limit file.  Default is "maglim".
    sysTypes: `list`, optional
       List of systematics maps to use in model.  Default is
       ['FWHM.WMEAN', 'AIRMASS.WMEAN', 'EXPTIME.SUM', 'SKYBRITE.WMEAN'].
    bands: `list`, optional
       List of bands.  Default is ['g', 'r', 'i', 'z', 'Y'].
    npixFit: `list`, optional
       Number of neighboring sub-pixels used in coarse fit to use in RF
       model.  Default is [1, 1, 1, 1, 9].
    zpTemplate: `str`, optional
       Template for the zeropoint map name.  Template should have one
       %s location for the band.  Default is None.
    ebvFile: `str`, optional
       Map of E(B-V) if map should be extinction corrected.  Default is None.
    aLambda: `list`, optional
       List of A_lambda values for applying E(B-V).
       Default is [3.186, 2.140, 1.569, 1.196, 1.048].
    """
    def __init__(self, depthFile, sysPath, sysTemplate, outBase,
                 minGalFit=50,
                 maglimType='maglim',
                 sysTypes=['FWHM.WMEAN', 'AIRMASS.WMEAN', 'EXPTIME.SUM', 'SKYBRITE.WMEAN'],
                 bands=['g','r','i','z','Y'],
                 npixFit=[1,1,1,1,9],
                 zpTemplate=None,
                 ebvFile=None,
                 aLambda=[3.186, 2.140, 1.569, 1.196, 1.048],
                 maxTry=10):
        self.depthFile = depthFile
        self.sysPath = sysPath
        self.fullTemplate = '%s/%s' % (sysPath, sysTemplate)
        self.outBase = outBase
        self.maglimType = maglimType
        self.sysTypes = sysTypes
        self.bands = bands
        self.minGalFit = minGalFit
        self.npixFit = npixFit
        self.maxTry  = maxTry
        if zpTemplate is not None:
            self.fullZpTemplate = '%s/%s' % (sysPath, zpTemplate)
        else:
            self.fullZpTemplate = None
        self.ebvFile = ebvFile
        if self.ebvFile is not None:
            if not os.path.isfile(self.ebvFile):
                raise IOError("Could not find ebvFile: %s" % (self.ebvFile))
        self.aLambda = aLambda

        if not os.path.isfile(self.depthFile):
            raise IOError("Could not find depthFile: %s" % (self.depthFile))

        for band in self.bands:
            for sysType in sysTypes:
                fName = self.fullTemplate % (band, sysType)
                if not os.path.isfile(fName):
                    raise IOError("Could not find systematics file: %s" % (fName))
                fName = self.fullTemplate % (band, 'frac')
                if not os.path.isfile(fName):
                    raise IOError("Could not find frac file: %s" % (fName))
                fName = self.fullTemplate % (band, self.maglimType)
                if not os.path.isfile(fName):
                    raise IOError("Could not find maglim file: %s" % (fName))

    def run(self):
        """
        """

        depthStr,dHdr = fitsio.read(self.depthFile, ext=1, header=True)
        nSideDepth = dHdr['NSIDE']
        nestDepth = dHdr['NEST']

        if (nestDepth) :
            # convert to ring here...
            print("Converting depth structure to ring format...")
            ipring = hp.nest2ring(nSideDepth, depthStr['HPIX'])
            depthStr['HPIX'][:] = ipring

        testName = self.fullTemplate % (self.bands[0], 'frac')
        hdr = fitsio.read_header(testName,ext=1)
        nSide = hdr['NSIDE']

        # allocate memory for sysmaps
        sysMaps = np.zeros((len(self.sysTypes)*2 + 1, hp.nside2npix(nSide)), dtype=np.float32)
        sysMapsResamp = np.zeros((len(self.sysTypes)*2 + 1, hp.nside2npix(nSideDepth)), dtype=np.float32)

        # Read in ebvMap if necessary
        if self.ebvFile is not None:
            print("Reading EBV Map...")
            ebvMap = hp.read_map(self.ebvFile, nest=False, dtype=np.float32)

        for b, band in enumerate(self.bands):
            altBand = 'i'
            if (band == 'i'):
                altBand = 'r'

            print("Running on band %s (alt-band %s)" % (band, altBand))

            print("Reading systematics maps...")

            sysMaps.fill(hp.UNSEEN)
            sysMapsResamp.fill(hp.UNSEEN)

            print("Reading fracfile (%s)" % (band))
            fracFile = self.fullTemplate % (band, 'frac')
            frac = hp.read_map(fracFile, nest=False)

            print("Reading %s (%s)" % (self.maglimType, band))
            fName = self.fullTemplate % (band, self.maglimType)
            sysMaps[0,:] = hp.read_map(fName, dtype=np.float32, nest=False)

            ok, = np.where(sysMaps[0,:] > 0.0)

            if self.fullZpTemplate is not None:
                zpName = self.fullZpTemplate % (band)
                print("Reading %s (%s)" % (os.path.basename(zpName), band))
                sysMaps[0,ok] += hp.read_map(zpName, dtype=np.float32, nest=False)[ok]

            if self.ebvFile is not None:
                print("Applying EBV correction with A=%.3f" % (self.aLambda[b]))
                sysMaps[0,ok] -= self.aLambda[b] * ebvMap[ok]

            sysMapsResamp[0,:] = self.deresMap(sysMaps[0,:], nSideDepth,
                                               minFrac=0.8, minSub=16,
                                               nest=False, mapFrac=frac)
            
            # read in the main maps and resample

            for i,sysType in enumerate(self.sysTypes):
                print("Reading %s (%s)" % (sysType, band))
                ind = i+1
                fName = self.fullTemplate % (band, sysType)
                sysMaps[ind,:] = hp.read_map(fName, dtype=np.float32, nest=False)
                sysMapsResamp[ind,:] = self.deresMap(sysMaps[ind,:], nSideDepth,
                                                     minFrac=0.8, minSub=16,
                                                     nest=False, mapFrac=frac)


            # and the alternate maps
            for i,sysType in enumerate(self.sysTypes):
                print("Reading %s (%s)" % (sysType, altBand))
                ind = i+len(self.sysTypes)+1
                fName = self.fullTemplate % (altBand, sysType)
                sysMaps[ind,:] = hp.read_map(fName, dtype=np.float32, nest=False)
                sysMapsResamp[ind,:] = self.deresMap(sysMaps[ind,:], nSideDepth,
                                                     minFrac=0.8, minSub=16,
                                                     nest=False, mapFrac=frac)

            depthInd,=np.where(band == np.array(self.bands))[0]

            # now need to select things that are in range...
            sysCheck = sysMapsResamp.min(axis=0)
            try:
                u,=np.where((depthStr['LIMMAG'][:,depthInd] > 0.0) &
                            (depthStr['LIMMAG'][:,depthInd] < 30.0) &
                            (depthStr['NGAL'][:,depthInd] > self.minGalFit) &
                            (depthStr['NPIX_FIT'][:,depthInd] == self.npixFit[depthInd]) &
                            (sysCheck[depthStr['HPIX'][:]] > -10.0))
            except:
                u,=np.where((depthStr['LIMMAG'][:,depthInd] > 0.0) &
                            (depthStr['LIMMAG'][:,depthInd] < 30.0) &
                            (depthStr['NGAL'][:] > self.minGalFit) &
                            (depthStr['NPIX_FIT'][:,depthInd] == self.npixFit[depthInd]) &
                            (sysCheck[depthStr['HPIX'][:]] > -10.0))

            hpix=depthStr['HPIX'][u]

            # The X is all but the maglim map (index 0)
            X = sysMapsResamp[1:, hpix].T

            # The y is the residual after the maglim map (index 0)
            y = depthStr['LIMMAG'][u,depthInd].flatten() - sysMapsResamp[0, hpix]

            done = False
            maxTry = self.maxTry
            iteration = 0

            while (iteration < maxTry and not done):
                # Keep trying until a reasonable solution
                # (this is not ideal, but sometimes it goes CRAZY)

                xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.5)
                print("Training random forest (%d)..." % (iteration))
                rf=ensemble.RandomForestRegressor(n_estimators = 100, n_jobs = os.cpu_count())
                rf.fit(xTrain, yTrain)

                print("Testing random forest (%d)..." % (iteration))

                yPredict = rf.predict(xTest)
                err = yPredict - yTest

                # and the histogram
                fig=plt.figure(1)
                fig.clf()
                ax=fig.add_subplot(111)
                coeff = histoGauss(ax, err)
                ax.set_title("%s band" % (band))
                ax.legend()
                print(coeff[0], coeff[1], coeff[2])

                out, = np.where(np.abs(err) > coeff[2] * 4.0)
                if (out.size < 0.01 * err.size) :
                    done = True
                    print("Testing successful!")
                else:
                    print("Too many catastrophic outliers: %d" % (out.size))

                iteration += 1

            fig.savefig("%s_validation_histogram_%s.png" % (self.outBase, band))

            # and the importances...
            print("Importances:")
            for i in range(len(self.sysTypes)):
                ind = i
                print("%s: " % (self.sysTypes[i]), rf.feature_importances_[ind])
            for i in range(len(self.sysTypes)):
                ind = i+len(self.sysTypes)
                print("%s (alt): " % (self.sysTypes[i]), rf.feature_importances_[ind])

            print("Making high resolution map...")

            newMap = np.zeros(hp.nside2npix(nSide), dtype=np.float32) + hp.UNSEEN

            sysCheck = sysMaps.min(axis=0)
            gd,=np.where(sysCheck > -10.0)

            xMap = sysMaps[1:, gd].T

            print("Running RF predictor...")
            mPred = rf.predict(xMap)
            newMap[gd] = sysMaps[0, gd] + mPred.astype(np.float32)

            fig = plt.figure(1)
            fig.clf()
            ax = fig.add_subplot(111)
            ax.hist(newMap[gd] - sysMaps[0, gd], bins=100)
            ax.set_title("%s band" % (band))
            ax.set_xlabel("Model Map - Maglim")
            fig.savefig('%s_post_validation_histogram_%s.png' % (self.outBase, band))

            print("Writing maps...")

            # and convert to NEST
            newMapNest = hp.ud_grade(newMap, nside_out=nSide, order_in='RING', order_out='NEST')

            outFileNest = self.outBase+'_nside%d_nest_%s_depth.fits' % (nSide, band)
            hp.write_map(outFileNest, newMapNest, nest=True, coord='C', overwrite = True)
            subprocess.call('gzip -f ' + outFileNest, shell=True)

            outFileRing = self.outBase+'_nside%d_ring_%s_depth.fits' % (nSide, band)
            hp.write_map(outFileRing, newMap, nest=False, coord='C', overwrite = True)
            subprocess.call('gzip -f ' + outFileRing, shell=True)

    def deresMap(self, mapFine, nSide, minFrac=0.8, minSub=1, nest=False,
                 mapRange=[hp.UNSEEN, np.abs(hp.UNSEEN)], mapFrac=None):
        """
        De-res a healpix map to a coarser resolution.

        Parameters
        ----------
        mapFile: `np.array`
           Fine-scale healpix map.
        nSide: `int`
           Coarser nside to scale map
        minFrac: `float`, optional
           Minimum fracgood to use in a mean to scale the map.
           Default is 0.8.
        minSub: `int`, optional
           Minimum number of sub-pixels covered to include in the mean.
           Default is 1.
        nest: `bool`, optional
           Are the input/output map nest format?  Default is False.
        mapRange: `list`, optional
           Exclusive range of valid values to put into the mean.
           Default is [hp.UNSEEN, np.abs(hp.UNSEEN)].
        mapFrac: `np.array`, optional
           Map of fracgood to use to select which pixels go in mean.
           Default is None (ignore fracgood).

        Returns
        -------
        resampledMap: `np.array`
           Map resampled to coarse resolution.
        """

        # need a copy here
        _mapFine = mapFine.copy()

        if (mapFrac is not None):
            bad,=np.where((mapFrac < minFrac) & (mapFrac > 0.0))
            if bad.size > 0:
                _mapFine[bad] = hp.UNSEEN

        # generate pixels at old nside
        mapNSide = hp.npix2nside(_mapFine.size)

        thetaFine, phiFine = hp.pix2ang(mapNSide, np.arange(_mapFine.size), nest = nest)
        useMap, = np.where((_mapFine > mapRange[0]) &
                           (_mapFine < mapRange[1]))

        hpixOut = hp.ang2pix(nSide, thetaFine[useMap], phiFine[useMap], nest=nest)

        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            Ncounts   = np.bincount(hpixOut, minlength = hp.nside2npix(nSide))
            Numerator = np.bincount(hpixOut, minlength = hp.nside2npix(nSide), weights = _mapFine[useMap])
            mapOut    = Numerator / Ncounts
            mapOut    = np.where(Ncounts < minSub, hp.UNSEEN, mapOut)

        return mapOut

    def deresMap_deprecated(self, mapFine, nSide, minFrac=0.8, minSub=1, nest=False,
                 mapRange=[hp.UNSEEN, np.abs(hp.UNSEEN)], mapFrac=None):
        """
        De-res a healpix map to a coarser resolution.

        Parameters
        ----------
        mapFile: `np.array`
           Fine-scale healpix map.
        nSide: `int`
           Coarser nside to scale map
        minFrac: `float`, optional
           Minimum fracgood to use in a mean to scale the map.
           Default is 0.8.
        minSub: `int`, optional
           Minimum number of sub-pixels covered to include in the mean.
           Default is 1.
        nest: `bool`, optional
           Are the input/output map nest format?  Default is False.
        mapRange: `list`, optional
           Exclusive range of valid values to put into the mean.
           Default is [hp.UNSEEN, np.abs(hp.UNSEEN)].
        mapFrac: `np.array`, optional
           Map of fracgood to use to select which pixels go in mean.
           Default is None (ignore fracgood).

        Returns
        -------
        resampledMap: `np.array`
           Map resampled to coarse resolution.
        """

        # need a copy here
        _mapFine = mapFine.copy()

        if (mapFrac is not None):
            bad,=np.where((mapFrac < minFrac) & (mapFrac > 0.0))
            if bad.size > 0:
                _mapFine[bad] = hp.UNSEEN

        # generate pixels at old nside
        mapNSide = hp.npix2nside(_mapFine.size)

        thetaFine, phiFine = hp.pix2ang(mapNSide, np.arange(_mapFine.size), nest=nest)
        useMap, = np.where((_mapFine > mapRange[0]) &
                           (_mapFine < mapRange[1]))

        hpixOut = hp.ang2pix(nSide, thetaFine[useMap], phiFine[useMap], nest=nest)

        hpixOutU = np.unique(hpixOut)
        nPix = hpixOutU.size

        suba,subb = esutil.numpy_util.match(hpixOutU, hpixOut)

        fakeId = np.arange(nPix)+1
        h,rev=esutil.stat.histogram(fakeId[suba], min=0, rev=True)

        mapOut = np.zeros(hp.nside2npix(nSide), dtype=np.float32) + hp.UNSEEN

        for i in range(nPix):
            iid=fakeId[i]
            if (rev[iid] < rev[iid+1]):
                i1a=rev[rev[iid]:rev[iid+1]]

                pInd = subb[i1a]

                if (pInd.size >= minSub):
                    mapOut[hpixOutU[i]] = np.mean(_mapFine[useMap[pInd]])

        return mapOut

def gaussFunction(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2./(2.*sigma**2))

def histoGauss(ax,array):
    """
    """
    import scipy.optimize
    import matplotlib.pyplot as plt
    import esutil

    q13 = np.percentile(array,[25,75])
    binsize=2*(q13[1] - q13[0])*array.size**(-1./3.)

    hist=esutil.stat.histogram(array,binsize=binsize,more=True)

    p0=[array.size,
        np.median(array),
        np.std(array)]

    try:
        coeff,varMatrix = scipy.optimize.curve_fit(gaussFunction, hist['center'],
                                                   hist['hist'], p0=p0)
    except:
        # set to starting values...
        coeff = p0

    hcenter=hist['center']
    hhist=hist['hist']

    rangeLo = -10*coeff[2]
    rangeHi = 10*coeff[2]

    lo,=np.where(hcenter < rangeLo)
    ok,=np.where(hcenter > rangeLo)
    hhist[ok[0]] += np.sum(hhist[lo])

    hi,=np.where(hcenter > rangeHi)
    ok,=np.where(hcenter < rangeHi)
    hhist[ok[-1]] += np.sum(hhist[hi])

    ax.plot(hcenter[ok],hhist[ok],'b-',linewidth=3)
    ax.set_xlim(rangeLo,rangeHi)

    xvals=np.linspace(rangeLo,rangeHi,1000)
    yvals=gaussFunction(xvals,*coeff)

    ax.plot(xvals,yvals,'k--',linewidth=3,label='%.5f +/- %.5f' % (coeff[1], coeff[2]))
    ax.locator_params(axis='x',nbins=6)  # hmmm

    return coeff

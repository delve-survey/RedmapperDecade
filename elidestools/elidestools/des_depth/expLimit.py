from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import os
import numpy as np
import fitsio
import esutil
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def dataBinner(x,y,binSize,xRange,nTrial=100,xNorm=-1.0,minPerBin=5):
    """
    Bin data and compute errors via bootstrap resampling.  All median statistics.

    parameters
    ----------
    x: float array
       x values
    y: float array
       y values
    binSize: float
       Bin size
    xRange: float array [2]
       x limits for binning
    nTrial: float, optional, default=100
       Number of bootstraps
    xNorm: float, optional, default=-1
       Set the y value == 0 when x is equan to xNorm.  if -1.0 then no norm.
    minPerBin: int, optional, default=5
       Minimum number of points per bin

    returns
    -------
    binStruct: recarray, length nbins
       'X_BIN': Left edge of each bin
       'X': median x in bin
       'X_ERR': width of x distribution in bin
       'X_ERR_MEAN': error on the mean of the x's in bin
       'Y': mean y in bin
       'Y_WIDTH': width of y distribution in bin
       'Y_ERR': error on the mean of the y's in the bin

    """

    hist,rev=esutil.stat.histogram(x,binsize=binSize,min=xRange[0],max=xRange[1]-0.0001,rev=True)
    binStruct=np.zeros(hist.size,dtype=[('X_BIN','f4'),
                                        ('X','f4'),
                                        ('X_ERR_MEAN','f4'),
                                        ('X_ERR','f4'),
                                        ('Y','f4'),
                                        ('Y_WIDTH','f4'),
                                        ('Y_ERR','f4'),
                                        ('N','i4')])
    binStruct['X_BIN'] = np.linspace(xRange[0],xRange[1],hist.size)

    for i in xrange(hist.size):
        if (hist[i] >= minPerBin):
            i1a=rev[rev[i]:rev[i+1]]

            binStruct['N'][i] = i1a.size

            medYs=np.zeros(nTrial,dtype='f8')
            medYWidths=np.zeros(nTrial,dtype='f8')
            medXs=np.zeros(nTrial,dtype='f8')
            medXWidths=np.zeros(nTrial,dtype='f8')

            for t in xrange(nTrial):
                r=(np.random.random(i1a.size)*i1a.size).astype('i4')

                medYs[t] = np.median(y[i1a[r]])
                medYWidths[t] = 1.4826*np.median(np.abs(y[i1a[r]] - medYs[t]))

                medXs[t] = np.median(x[i1a[r]])
                medXWidths[t] = 1.4826*np.median(np.abs(x[i1a[r]] - medXs[t]))

            binStruct['X'][i] = np.median(medXs)
            binStruct['X_ERR'][i] = np.median(medXWidths)
            binStruct['X_ERR_MEAN'][i] = 1.4826*np.median(np.abs(medXs - binStruct['X'][i]))
            binStruct['Y'][i] = np.median(medYs)
            binStruct['Y_WIDTH'][i] = np.median(medYWidths)
            binStruct['Y_ERR'][i] = 1.4826*np.median(np.abs(medYs - binStruct['Y'][i]))

    if (xNorm >= 0.0) :
        ind=np.clip(np.searchsorted(binStruct['X_BIN'],xnorm),0,binStruct.size-1)
        binStruct['Y'] = binStruct['Y'] - binStruct['Y'][ind]

    return binStruct

def expLimit(depthFile, bands, npixMax, minNGal=50, pivot=23.0):
    """
    Calculate the relationship between the limiting magnitude and the effective exposure time.

    This will produce one plot per band.

    Parameters
    ----------
    depthFile: `str`
       Name of the depth file to use as an input
    bands: `list`
       String list of band names
    npixMax: `list`
       Maximum number of neighboring pixels used to obtain a model fit to
       use in computing mag/time relationship (one number per band).
    minNGal: `int`, optional
       Minimum number of galaxies in a pixel for the model fit to use
       in computing mag/time relationship.  Default is 50.
    pivot: `float`, optional
       Pivot magnitude for the mag/time relationship.  Default is 23.0
    """

    parts=os.path.basename(depthFile).split('.fit')
    outBase = parts[0]
    outDir  = os.path.dirname(depthFile) + '/'

    dep=fitsio.read(depthFile,ext=1)

    dtype = [('BAND','a2'),
             ('FIT', 'f4', 2)]

    teffStr = np.zeros(len(bands),dtype=dtype)

    for i,b in enumerate(bands):
        gd,=np.where((dep['NPIX_FIT'][:,i] <= npixMax[i]) &
                     (dep['NGAL'][:,i] >= minNGal))


        st=np.argsort(dep['LIMMAG'][gd,i])

        fitRange = np.array([dep['LIMMAG'][gd[st[int(0.01*st.size)]],i],
                             dep['LIMMAG'][gd[st[int(0.99*st.size)]],i]])

        use,=np.where((dep['LIMMAG'][gd,i] > fitRange[0]) &
                      (dep['LIMMAG'][gd,i] < fitRange[1]))


        binStruct = dataBinner(dep['LIMMAG'][gd[use],i],
                               np.log(dep['EXPTIME'][gd[use],i]),
                               (fitRange[1]-fitRange[0])/11.,fitRange)
        ok,=np.where(binStruct['Y_ERR'] > 0.0)
        binStruct=binStruct[ok]

        fit,cov = np.polyfit(binStruct['X_BIN']-pivot,
                             binStruct['Y'],
                             1.0,
                             w=(1./binStruct['Y_ERR'])**2.,
                             cov=True)

        f=plt.figure(1)
        f.clf()
        ax=f.add_subplot(111)

        ax.hexbin(dep['LIMMAG'][gd[use],i], np.log(dep['EXPTIME'][gd[use],i]), bins='log')
        ax.set_xlabel('limmag_'+b)
        ax.set_ylabel('log(exptime_'+b+')')
        ax.set_title('%s band, %s' % (b, os.path.basename(depthFile)))

        ax.errorbar(binStruct['X'],binStruct['Y'],binStruct['X_ERR'],binStruct['Y_ERR'],'k+')

        ax.plot(fitRange, fit[1] + fit[0]*(fitRange-pivot), 'k-')

        f.savefig(outDir + outBase+'_exp_vs_lim_'+b+'.png')

        # switch to be compatible with old IDL format (sorry)
        teffStr['BAND'][i] = b
        teffStr['FIT'][i,0] = fit[1]
        teffStr['FIT'][i,1] = fit[0]

    hdr = fitsio.FITSHDR()
    hdr['PIVOT'] = pivot
    fitsio.write(outDir + outBase+'_teff.fits', teffStr, clobber=True, header=hdr)

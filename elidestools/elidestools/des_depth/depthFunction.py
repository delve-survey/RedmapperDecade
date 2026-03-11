from __future__ import division, absolute_import, print_function

import numpy as np


class DepthFunction(object):
    """
    Class to implement function for fitting depth.

    """
    def __init__(self,mag,magErr,zp,nSig):
        """
        Instantiate DepthFunction class.

        Parameters
        ----------
        mag: `np.array`
           Float array of magnitudes
        magErr: `np.array`
           Float array of magnitude errors
        zp: `float`
           Reference zeropoint
        nSig: `float`
           Number of sigma to compute depth limit
        """
        self.const = 2.5/np.log(10.0)

        self.mag = mag
        self.magErr = magErr
        self.zp = zp
        self.nSig = nSig

        self.max_p1 = 1e10

    def __call__(self, x):
        """
        Compute total cost function for f(x)

        Parameters
        ----------
        x: `np.array`, length 2
           Float array of fit parameters

        Returns
        -------
        t: `float`
           Total cost function at parameters x
        """

        if ((x[1] < 0.0) or
            (x[1] > self.max_p1)):
            return 1e10

        f1lim = 10.**((x[0] - self.zp)/(-2.5))
        fsky1 = ((f1lim**2. * x[1])/(self.nSig**2.) - f1lim)

        if (fsky1 < 0.0):
            return 1e10

        tflux = x[1]*10.**((self.mag - self.zp)/(-2.5))
        err = np.sqrt(fsky1*x[1] + tflux)

        # apply the constant here, not to the magErr, which was dumb
        t=np.sum(np.abs(self.const*(err/tflux) - self.magErr))

        if not np.isfinite(t):
            t=1e10

        return t

from __future__ import division, absolute_import, print_function

import cartopy.crs as ccrs
import numpy as np
import healpy as hp

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .utils import setdefaults

class SkymapCP(object):
    """
    """
    def __init__(self, fig=None, rect=111):#, ra_range=None, dec_range=None):

        if fig is None:
            fig = plt.gcf()

        self.ax = fig.add_subplot(rect, projection=ccrs.Mollweide())

        #if ra_range is None:
        #    self.ra_range = self._default_rarange
        #else:
        #    self.ra_range = ra_range
        #if dec_range is None:
        #    self.dec_range = self._default_decrange

        self.im = None

    def draw_footprint(self, **kwargs):
        pass

    def draw_polygon_file(self, filename, **kwargs):
        """
        Draw a polygon footprint
        """

        defaults = dict(color='k', lw=2)
        setdefaults(kwargs, defaults)

        poly = np.loadtxt(filename, dtype=[('ra', float), ('dec', float)])
        return self.draw_polygon_radec(poly['ra'], poly['dec'], **kwargs)

    def draw_polygon_radec(self, ra, dec, **kwargs):
        """
        Draw a polygon from set of ra/dec
        """

        self.ax.plot(ra, dec, transform=ccrs.PlateCarree(), **kwargs)

    def draw_hpxmap(self, hpxmap, **kwargs):
        """
        """

        if not isinstance(hpxmap, np.ma.MaskedArray):
            mask = ~np.isfinite(hpxmap) | (hpxmap == hp.UNSEEN)
            hpxmap = np.ma.MaskedArray(hpxmap, mask=mask)

        xsize = kwargs.pop('xsize', 1600)
        perc_range = kwargs.pop('perc_range', [0.1, 99.9])

        vmin = kwargs.pop('vmin', np.percentile(hpxmap.compressed(), perc_range[0]))
        vmax = kwargs.pop('vmax', np.percentile(hpxmap.compressed(), perc_range[1]))
        xlocs = kwargs.pop('xlocs', self._default_xlocs)
        ylocs = kwargs.pop('ylocs', self._default_ylocs)
        ra_range = kwargs.pop('ra_range', self._default_rarange)
        dec_range = kwargs.pop('dec_range', self._default_decrange)

        # Get the ra/dec range
        #ra_range = self.ra_range
        #dec_range = self.dec_range

        nside = hp.get_nside(hpxmap.data)

        # If we don't have explicit ranges, we'll need a plot cushion
        ra_cushion = 0.0
        dec_cushion = 0.0

        if ra_range is None or dec_range is None:
            ipring, = np.where(hpxmap.data > hp.UNSEEN)
            ra_map, dec_map = hp.pix2ang(nside, ipring, lonlat=True)

            if dec_range is None:
                dec_range = [np.min(dec_map), np.max(dec_map)]
                dec_cushion = 0.05 * (dec_range[1] - dec_range[0])

            if ra_range is None:
                # Need to try various rotations...
                rot_tests = np.arange(0., 360., 10.)
                ra_ranges = np.zeros((rot_tests.size, 2))

                for i in range(rot_tests.size):
                    ra_temp = ra_map + rot_tests[i]
                    hi, = np.where(ra_temp > 360.0)
                    ra_temp[hi] -= 360.0
                    ra_ranges[i, 0] = np.min(ra_temp)
                    ra_ranges[i, 1] = np.max(ra_temp)

                ind = np.argmin(ra_ranges[:, 1] - ra_ranges[:, 0])

                ra_range = ra_ranges[ind, :] - rot_tests[ind]

                ra_cushion = 0.05 * (ra_range[1] - ra_range[0])


        ra_range[0] = np.clip(ra_range[0], -180.0, None)
        ra_range[1] = np.clip(ra_range[1], None, 180.0)
        dec_range[0] = np.clip(dec_range[0], -90.0, None)
        dec_range[1] = np.clip(dec_range[1], None, 90.0)

        # Compute the central RA for use as an offset for better
        # projections
        self._ra_offset = (ra_range[0] + ra_range[1]) / 2.

        # The extent will reverse the RA axis
        self._radec_extent = [np.clip(ra_range[1] + ra_cushion, None, 180.0),
                              np.clip(ra_range[0] - ra_cushion, -180.0, None),
                              np.clip(dec_range[0] - dec_cushion, -90.0, None),
                              np.clip(dec_range[1] + dec_cushion, None, 90.0)]
        self._radec_mid = [(self._radec_extent[0] + self._radec_extent[1]) / 2.,
                           (self._radec_extent[2] + self._radec_extent[3]) / 2.]

        print(self._radec_extent)

        ra_steps = np.linspace(ra_range[0], ra_range[1], xsize)
        aspect = (dec_range[1] - dec_range[0]) / np.cos(np.radians((dec_range[0] + dec_range[1]) / 2.)) / (ra_range[1] - ra_range[0])
        dec_steps = np.linspace(dec_range[0], dec_range[1], int(xsize * aspect))
        ra, dec = np.meshgrid(ra_steps, dec_steps)

        pix = hp.ang2pix(nside, ra, dec, lonlat=True)

        values = hpxmap[pix]
        self.im = self.ax.pcolormesh(ra, dec, values, vmin=vmin, vmax=vmax, rasterized=True, transform=ccrs.PlateCarree())

        # Set the x/y limits
        self.ax.set_xlim(left=ccrs.Mollweide().transform_point(self._radec_extent[0],
                                                               self._radec_mid[1],
                                                               ccrs.PlateCarree())[0],
                         right=ccrs.Mollweide().transform_point(self._radec_extent[1],
                                                                self._radec_mid[1],
                                                                ccrs.PlateCarree())[0])
        self.ax.set_ylim(bottom=ccrs.Mollweide().transform_point(self._radec_mid[0],
                                                                 self._radec_extent[2],
                                                                 ccrs.PlateCarree())[1],
                         top=ccrs.Mollweide().transform_point(self._radec_mid[0],
                                                              self._radec_extent[3],
                                                              ccrs.PlateCarree())[1])

        if xlocs is None:
            mn = MaxNLocator(nbins=7)
            self.xlocs = mn.tick_values(ra_range[0], ra_range[1])
        else:
            self.xlocs = xlocs
        if ylocs is None:
            mn = MaxNLocator(nbins=7)
            self.ylocs = mn.tick_values(dec_range[0], dec_range[1])
        else:
            self.ylocs = ylocs

        self.ax.gridlines(xlocs=self.xlocs,
                          ylocs=self.ylocs)

        self._plot_axislabels()
        self._plot_ticklabels()

        return self.im

    def _plot_axislabels(self):
        """
        """

        self.ax.text(-0.12, 0.5, 'Declination', ha='center',
                      va='center', rotation='vertical', rotation_mode='anchor',
                      transform=self.ax.transAxes)
        self.ax.text(0.5, -0.1, 'Right Ascension', ha='center',
                     va='center', transform=self.ax.transAxes)

    def _plot_ticklabels(self):
        """
        """

        # Get the extent in projected units
        extent = self.ax.get_extent()

        for xloc in self.xlocs:
            pt = ccrs.Mollweide().transform_point(xloc, self._radec_extent[2], ccrs.PlateCarree())
            if pt[0] > extent[0] and pt[0] < extent[1]:
                # The x axis is inverted here
                xpos_relative = 1.0 - (pt[0] - extent[0]) / (extent[1] - extent[0])
                self.ax.text(xpos_relative, -0.02,
                             '%.1f' % (xloc) + u'\u00B0', ha='center', va='top',
                             transform=self.ax.transAxes)

        for yloc in self.ylocs:
            # This has to be done in Mollweide units to keep along the axis
            pt = ccrs.Mollweide().transform_point(self._radec_extent[0], yloc,
                                                  ccrs.PlateCarree())
            if pt[1] > extent[2] and pt[1] < extent[3]:
                ypos_relative = (pt[1] - extent[2]) / (extent[3] - extent[2])
                self.ax.text(-0.01, ypos_relative,
                             '%.1f' % (yloc) + u'\u00B0', ha='right', va='center',
                             transform=self.ax.transAxes)

    def draw_inset_colorbar(self, format=None):
        """
        """

        if self.im is None:
            raise RuntimeError("Cannot draw a colorbar without an image with draw_hpxmap")

        cax = inset_axes(self.ax, width="25%", height="5%", loc=7)
        cmin, cmax = self.im.get_clim()
        cmed = (cmax + cmin)/2.
        delta = (cmax - cmin)/10.

        ticks = np.array([cmin + delta, cmed, cmax - delta])
        tmin = np.min(np.abs(ticks[0]))
        tmax = np.max(np.abs(ticks[1]))

        if (tmin < 1e-2) or (tmax > 1e3):
            format = '%.1e'
        elif (tmin > 0.1) and (tmax < 100):
            format = '%.1f'
        elif (tmax > 100):
            format = '%i'
        else:
            format = '%.2g'

        fig = self.ax.get_figure()
        cbar = fig.colorbar(self.im, cax=cax, orientation='horizontal',
                            ticks=ticks, format=format)
        cax.xaxis.set_ticks_position('top')
        cax.tick_params(axis='x', labelsize=10)

    @property
    def _default_rarange(self):
        return None

    @property
    def _default_decrange(self):
        return None

    @property
    def _default_xlocs(self):
        return None

    @property
    def _default_ylocs(self):
        return None

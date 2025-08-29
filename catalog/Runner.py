import redmapper
import numpy as np, fitsio, h5py, matplotlib.pyplot as plt
import healsparse as hsp, healpy as hp
import os, re, esutil, sys, time, glob, joblib, textwrap, shutil, yaml, pathlib
from tqdm import tqdm
from astropy.io import fits

sys.path.insert(0, os.path.dirname(__file__) + '/../') #Goes from catalog/ to mapper/ path

import mapper
import elidestools.elidestools as etools

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        print('='*100)
        print(f"Function <<{func.__name__}>> took {total_time:.5} seconds to run.")
        print('='*100)
        return result
    return wrapper


class BaseRunner:

    def __init__(self, outBase, seed = 42):
        self.outBase = outBase
        self.seed    = seed

    @timeit
    def go(self):

        self.prep_training_catalog(n_jobs = 24)
        self.make_all_maps()
        self.make_master_galaxy_catalog()
        self.make_spec_catalog()
        self.make_spec_catalog_redseq()
        self.make_config_yaml()
        self.run_redmapper_calibration()
        self.run_zred_pixel()
        self.run_zred_bkg()
        self.run_redmapper_pixel()
        self.consolidate_redmapper_pixel()
        self.make_randoms()
        self.make_redmagic()
        
    @timeit
    def prep_training_catalog(self, n_jobs):
        
        BANDS = 'GRIZ'
        keys  = []
        keys += [f'BDF_FLUX_{B}_DERED_SFD98'     for B in BANDS]
        keys += [f'BDF_FLUX_ERR_{B}_DERED_SFD98' for B in BANDS]
        keys += ['RA', 'DEC', 'COADD_OBJECT_ID']

        out   = {k : [] for k in keys + ['mask']}

        outpath = self.outBase + '.tmp_input_cat.hdf5'
        
        if os.path.isfile(outpath):
            print("FILE ALREADY EXISTS....SKIPPING....")
            return None

        for file in ['metacal_gold_combined_20240209.hdf', 'metacal_gold_combined_20241003.hdf']:
            
            footprint_flag = 0 if '20240209' in file else 1 #Cause we did a weird thing in the files :P
            with h5py.File(f'/project/chihway/data/decade/{file}', 'r') as f:
                GLD = ( (f['FLAGS_SG_BDF'][:] >= 2) & 
                        # (f['FLAGS_FOREGROUND'][:] == 0) &
                        (f['FLAGS_FOOTPRINT'][:] == footprint_flag) &
                        (f['FLAGS_BAD_COLOR'][:] == 0)
                        )
                for k in keys: out[k].append(f[k][:][GLD]); print("READ", k)
                out['mask'].append(GLD)

            print("FINISHED READING IN FILE", file)


        if os.path.isfile(outpath): print("FILE EXISTS AT", outpath, "... OVERWRITING IT....")
        with h5py.File(outpath, 'w') as f:

            for k in out.keys(): 
                f.create_dataset(name = k, data = np.concatenate(out[k]))
                print("FINISHED WRITING", k)

        del out
        print("OUTPUT WRITTEN TO", outpath)

        Collator = mapper.utils.DECADECollator(self.outBase, n_jobs = n_jobs)
        Collator.run(outpath)


    @timeit
    def make_all_maps(self):

        self.make_area_mask()
        self.make_depth_map()
        self.make_depth_str_map()

    
    @timeit
    def get_fracdet_map(self):

        return hsp.HealSparseMap.read("/project/chto/dhayaa/Redmapper/DR3_maglim_fracdet.hsp").fracdet_map(nside = 4096)
    
    @timeit
    def get_foreground_map(self):

        return hp.read_map("/project/chto/dhayaa/decade/GOLD_Ext0.2_Star5_MCs2_DESY6.fits")
    
    @timeit
    def make_area_mask(self):

        if os.path.isfile(self.outBase + '.decade_pixmask.hs'):
            print("PIX FILES ALREADY EXISTS. SKIPPING....")
            return None
        
        footprint  = self.get_fracdet_map()
        foreground = self.get_foreground_map()

        print("LOADED FOOTPRINT AND FOREGROUND MASKS")

        fp_valid_pixels = footprint.valid_pixels
        fp_fracgood     = footprint[fp_valid_pixels]

        #Zero out pixels with too low coverage
        #and pixels that are outside foreground mask
        fp_fracgood[fp_fracgood < 0.5] = hp.UNSEEN
        fp_fracgood[foreground[hp.nest2ring(hp.npix2nside(foreground.size), fp_valid_pixels)] > 0] = hp.UNSEEN

        print("APPLY FOREGROUND MASK AND FRACGOOD > 0.5")

        hsmask = hsp.HealSparseMap.make_empty(32, 4096, dtype = np.float32)
        hsmask[fp_valid_pixels] = fp_fracgood.astype(np.float32)

        # hp.mollview(hsmask.generate_healpix_map(4096), nest = True)
        hsmask.write(self.outBase + '.decade_pixmask.hs', clobber = True)
        hp.write_map(self.outBase + '.decade_pixmask.hpy', hsmask.generate_healpix_map(4096, nest = False), overwrite = True)

        print("WROTE TO DISK...")


    @timeit
    def make_depth_map(self, n_jobs = -1):

        self.compute_depth_pixel_process(n_jobs)
        self.prepare_sys_maps()
        self.compute_depth_map()
        self.compute_exp_limit()


    @timeit
    def prepare_sys_maps(self):

        for b in 'griz':
            for sys in ['airmass', 'fwhm', 'exptime', 'skybrite', 'maglim']:

                if os.path.isfile(self.outBase + f'_DR3_{b}_{sys}.hpy'):
                    print(f"FILE DR3_{b}_{sys}.hpy EXISTS. SKIPPING...")
                    continue
                
                if sys == 'exptime':
                    text = 'sum'
                else:
                    text = 'wmean'
                    
                DR31 = hsp.HealSparseMap.read(f"/project/jfrieman/chinyi/dr3_2_decasu_maps/{sys}/delve_dr32_{b}_{sys}_{text}.hsp")
                DR32 = hsp.HealSparseMap.read(f"/project/chihway/secco/decasu_outputs/{sys}/delve_dr311+dr312_{b}_{sys}_Nov28th.hsp")

                DR31 = DR31.generate_healpix_map(nside = 4096, nest = False)
                DR32 = DR32.generate_healpix_map(nside = 4096, nest = False)

                DR31 = np.where(DR31 == hp.UNSEEN, 0, DR31)
                DR32 = np.where(DR32 == hp.UNSEEN, 0, DR32)

                DR3  = DR31 + DR32
                DR3  = np.where(DR3 == 0, hp.UNSEEN, DR3)
                hp.write_map(self.outBase + f'_DR3_{b}_{sys}.hpy', DR3)
                
                print(f"FINISHED WRITING MAP {sys} IN BAND {b}", flush = True)


        if os.path.isfile(self.outBase + f'_DR3_z_frac.hpy'): return None
        FRACGOOD = hsp.HealSparseMap.read(self.outBase + '.decade_pixmask.hs')
        FRACGOOD = FRACGOOD.fracdet_map(nside=4096)
        FRACGOOD = FRACGOOD.generate_healpix_map(nside = 4096, nest = False)
        for b in 'griz': hp.write_map(self.outBase + f'_DR3_{b}_frac.hpy', FRACGOOD)

    @timeit
    def compute_depth_pixel_process(self, n_jobs = -1):

        filelist = sorted(glob.glob(self.outBase + '_pix*_bdf.fits')) #Get all files by checking for bdf fits
        filelist = [f[:-9] for f in filelist] #Remove the _bdf.fits part of the filename
        filelist = [filelist[i] for i in np.random.choice(len(filelist), len(filelist), replace = False)] #Randomize to balance the load
        
        if n_jobs == -1: n_jobs = np.min([os.cpu_count(), len(filelist)])

        print(f"RUNNING {len(filelist)} PIXELS USING {n_jobs} JOBS", flush = True)

        jobs = [joblib.delayed(self._single_step_pixel_process)(f) for f in filelist]
        out  = joblib.Parallel(n_jobs = n_jobs, verbose = 10)(jobs)

    def _single_step_pixel_process(self, path):
        
        if os.path.isfile(path.replace('_bdf.', '.')):
            print(f"PATH {path} EXISTS. SKIPPING PROCESS STEP....", flush = True)
        
        etools.des_depth.catalogPixelProcess(path, self.outBase, 'bdf', 'flux_bdf', 'fluxerr_bdf', 1024, 
                                             nSidePixFile = 8, bandList = ['g', 'r', 'i', 'z'], noGoldFlags = True, 
                                             s2nCut = 5, selectGalaxies = True, bdSizeFileType = None, nTrial = 3)



    @timeit
    def compute_depth_map(self):


        if os.path.isfile(self.outBase + '_nside4096_nest_z_depth.fits.gz'):
            print("FILES FOUND. SKIPPING....")
            return None

        files = glob.glob(self.outBase + '_pix?????.fits')
        print(f"FOUND {len(files)} PIXEL FILES. CONSOLATING.....")
        etools.des_depth.pixelConsolidate(files, self.outBase + '_coarse_depth.fits', nSide = 1024, nest = False)
        print("FINISHED CONSOLIDATING. MAKING DEPTH MAP....")

        mapMaker = etools.des_depth.MakeMap(self.outBase + '_coarse_depth.fits', os.path.dirname(self.outBase),
                                            os.path.basename(self.outBase) + "_DR3_%s_%s.hpy", self.outBase,
                                            sysTypes = ['fwhm', 'airmass', 'exptime', 'skybrite'],
                                            bands    = ['g', 'r', 'i', 'z'],
                                            aLambda  = [3.186, 2.140, 1.569, 1.196],
                                            npixFit  = [1, 1, 1, 1],
                                            maxTry   = 10)
        mapMaker.run()


    @timeit
    def compute_exp_limit(self):

        if os.path.isfile(self.outBase + '_coarse_depth_teff.fits'):
            print("FOUD COARSE_DEPTH_TEFF FILE. SKIPPING....")
            return None

        etools.des_depth.expLimit(self.outBase + '_coarse_depth.fits', ['g', 'r', 'i', 'z'], npixMax = [5,5,5,5])


    @timeit
    def make_depth_str_map(self):

        zp      = 30.0 #For all eternity in DES....
        hsmask  = hsp.HealSparseMap.read(self.outBase + '.decade_pixmask.hs')
        
        dtype   = [('exptime', 'f4'), ('limmag', 'f4'), ('m50', 'f4'), ('fracgood', 'f4')]

        teffstr, thdr = fitsio.read(self.outBase + '_coarse_depth_teff.fits', ext = 1, trim_strings = True, header = True, lower = True)
        pivot         = thdr['PIVOT']

        for b in ['g', 'r', 'i', 'z']:

            if os.path.isfile(self.outBase + f'-10{b}_bdf_depthstr.hs'):
                print("FOUND DEPTH_STR FILE", self.outBase + f'-10{b}_bdf_depthstr.hs', '. SKIPPING.....')
                continue

            depthfile = self.outBase + '_nside4096_nest_{0}_depth.fits.gz'.format(b)
            depthin   = hp.read_map(depthfile, nest = True)
            depth     = hsp.HealSparseMap.make_empty(32, 4096, dtype=np.float32)
            depth[:]  = depthin

            depth_pix = depth.valid_pixels
            ind       = list(teffstr['band']).index(b)

            dstr = np.zeros(depth_pix.size, dtype=dtype)
            dstr['limmag']   = depth[depth_pix]
            dstr['m50']      = depth[depth_pix]
            dstr['exptime']  = np.exp(teffstr[ind]['fit'][0] + teffstr[ind]['fit'][1]*(depth[depth_pix] - pivot))
            dstr['fracgood'] = hsmask[depth_pix]

            hsdepth = hsp.HealSparseMap.make_empty(nside_coverage = 32, nside_sparse = 4096, dtype = dtype, primary = 'fracgood')
            use,    = np.where(dstr['fracgood'] > 0.5)
            hsdepth[depth_pix[use]] = dstr[use]

            hdr = {}
            hdr['NSIG']  = 10.0
            hdr['ZP']    = zp
            hdr['NBAND'] = 1
            hdr['W']     = 0.0  # wall mode
            hdr['EFF']   = 1.0

            hsdepth.metadata = hdr
            hsdepth.write(self.outBase + f'-10{b}_bdf_depthstr.hs', clobber = True)


        
    @timeit
    def make_master_galaxy_catalog(self):


        if os.path.isfile(self.outBase + "_master_table.fit"):
            print("FOUND MASTER TABLE. SKIPPING MAKE CATALOG...")
            return None

        widebase  = self.outBase
        outbase   = self.outBase
        maskfile  = self.outBase + '.decade_pixmask.hs'
        depthfile = self.outBase + '-10z_bdf_depthstr.hs'

        mask  = hsp.HealSparseMap.read(maskfile)
        depth = hsp.HealSparseMap.read(depthfile)

        area  = np.sum(mask.get_values_pix(mask.valid_pixels), dtype=np.float64) * hp.nside2pixarea(mask.nside_sparse, degrees = True)

        print('Area = ', area)

        limmag_max = depth.get_values_pix(depth.valid_pixels)['limmag'].max()

        print('limmag_max = ', limmag_max)

        a_lambda = np.array([3.186, 2.140, 1.569, 1.196])

        zp       = 30.0
        errrange =[0.0, 100.0]

        b_array  = np.array([3.27e-12, 4.83e-12, 6.0e-12, 9.0e-12])
        bscale   = np.array(b_array) * 10.**(zp / 2.5)

        bands    = ['g', 'r', 'i', 'z']
        nmag     = len(bands)
        ref_ind  = bands.index('z')

        redmapper_dtype = [ ('id', 'i8'),              # galaxy id number (unique)
                            ('ra', '>f8'),             # right ascension (degrees)
                            ('dec', '>f8'),            # declination (degrees)
                            ('refmag', '>f4'),         # total magnitude in reference band
                            ('refmag_err', '>f4'),     # error in total reference mag
                            ('mag', '>f4', nmag),      # mag array
                            ('mag_err', '>f4', nmag),  # magnitude error array
                            ('ebv', '>f4')]            # E(B-V) (systematics checking)
        info_dict = {}
        info_dict['LIM_REF'] = limmag_max
        info_dict['REF_IND'] = ref_ind
        info_dict['AREA']    = area
        info_dict['NMAG']    = nmag
        info_dict['MODE']    = 'DES' # currently SDSS, DES, or LSST
        info_dict['ZP']      = zp
        info_dict['B']       = b_array # if magnitudes are actually luptitudes
        info_dict['G_IND']   = 0 # g-band index
        info_dict['R_IND']   = 1 # r-band index
        info_dict['I_IND']   = 2 # i-band index
        info_dict['Z_IND']   = 3 # z-band index

    
        widefiles = sorted(glob.glob('%s_pix*_basic.fits' % (widebase)))
        maker     = redmapper.GalaxyCatalogMaker(outbase, info_dict)

        offset    = 0
        for input_file in tqdm(widefiles, desc = 'collating files'):
            
            # make all 4 mags, combining
            #   sof (wide)
            #   luptitudification
            #   sfd98 reddening

            groups = re.search('_pix(\d+)_', os.path.basename(input_file))
            pixnum = int(groups[1])

            basic  = fitsio.read(input_file, ext=1, lower=True)
            bdf    = fitsio.read(input_file.replace('_basic.fits', '_bdf.fits'), ext=1, lower=True)

            galaxies         = np.zeros(basic.size, dtype = redmapper_dtype)
            galaxies['id']   = offset + np.arange(len(galaxies))
            offset          += len(galaxies)
            galaxies['ra']   = basic['ra']
            galaxies['dec']  = basic['dec']

            for i, band in enumerate(bands):

                flux    = bdf['flux_bdf_%s' % (band)].astype(np.float64)
                fluxerr = bdf['fluxerr_bdf_%s' % (band)].astype(np.float64)

                galaxies['mag'][:, i]     = 2.5 * np.log10(1.0 / b_array[i]) - np.arcsinh(0.5 * flux / bscale[i]) / (0.4 * np.log(10.0))
                galaxies['mag_err'][:, i] = 2.5 * fluxerr / (2.0 * bscale[i] * np.log(10.0) * np.sqrt(1.0 + (0.5 * flux / bscale[i])**2.))

                # Apply reddening
                #acorr = basic['ebv_sfd98'] * a_lambda[i]

            #galaxies['mag'][:, i] = galaxies['mag'][:, i] - acorr

            galaxies['refmag']     = galaxies['mag'][:, ref_ind]
            galaxies['refmag_err'] = galaxies['mag_err'][:, ref_ind]
            #galaxies['ebv'] = basic['ebv_sfd98']

            # Mark bad objects
            mark = np.ones(galaxies.size, dtype=bool)
            bad, = np.where((galaxies['refmag_err'] < errrange[0]) |
                            (galaxies['refmag_err'] > errrange[1]))
            mark[bad] = 0

            for i, band in enumerate(bands):
                bad, = np.where((galaxies['mag'][:, i] < errrange[0]) |
                                (galaxies['mag'][:, i] > errrange[1]) | 
                                (galaxies['mag_err'][:, i] < errrange[0]) |
                                (galaxies['mag_err'][:, i] > errrange[1]))
                mark[bad] = 0

            # Apply depth maps
            m50 = depth.get_values_pos(galaxies['ra'], galaxies['dec'], lonlat=True)['m50']

            gd, = np.where((mark) & (galaxies['refmag'] < m50))

            if gd.size == 0:
                print("All galaxies too faint in pixel %d" % (pixnum))
                continue

            # And append the galaxies
            maker.append_galaxies(galaxies[gd])
            # print(np.max(galaxies[gd]['mag']),np.max(galaxies[gd]['mag_err']), flush=True)

        maker.finalize_catalog()
    
    @timeit
    def make_spec_catalog(self):

        if os.path.isfile(self.outBase + ".all_specz.fits"):
            print("FOUND ALL SPECZ CATALOG. SKIPPING MAKE CATALOG...")
            return None

        specz = fitsio.read("/project/chto/dhayaa/decade/specz/BOSS_eBOSS.fits")
        specz_DESI1 = fitsio.read("/project/chto/dhayaa/decade/specz/DESI/BGS_BRIGHT-21.5_NGC_clustering.dat.fits")
        specz_DESI2 = fitsio.read("/project/chto/dhayaa/decade/specz/DESI/BGS_BRIGHT-21.5_SGC_clustering.dat.fits")
        specz_DESI3 = fitsio.read("/project/chto/dhayaa/decade/specz/DESI/LRG_NGC_clustering.dat.fits")
        specz_DESI4 = fitsio.read("/project/chto/dhayaa/decade/specz/DESI/LRG_SGC_clustering.dat.fits")
        specz_DESI5 = fitsio.read("/project/chto/dhayaa/decade/specz/DESI/ELG_LOPnotqso_NGC_clustering.dat.fits")
        specz_DESI6 = fitsio.read("/project/chto/dhayaa/decade/specz/DESI/ELG_LOPnotqso_SGC_clustering.dat.fits")


        spec_dtype = [('ra', 'f8'), ('dec', 'f8'), ('z', 'f4'), ('z_err', 'f4')]

        allspec    = np.zeros(specz.size + specz_DESI1.size + specz_DESI2.size + specz_DESI3.size + 
                              specz_DESI4.size + specz_DESI5.size + specz_DESI6.size, dtype = spec_dtype)
        for item1, item2 in zip(['ra','dec','z'], ['RA','DEC','Z']):
            allspec[item1] = np.r_[specz[item2], specz_DESI1[item2], specz_DESI2[item2],specz_DESI3[item2],
                                   specz_DESI4[item2], specz_DESI5[item2], specz_DESI6[item2]]

        allspec['z_err'] = 1e-4

        allspec = allspec[allspec['z'] > 0.01] #Remove very low redshifts. Use 0.01 to remove stars.
        fitsio.write(self.outBase + ".all_specz.fits", allspec, clobber = True)

    @timeit
    def make_spec_catalog_redseq(self):
        
        if os.path.isfile(self.outBase + ".train_specz.fits"):
            print("FOUND TRAIN SPECZ CATALOG. SKIPPING MAKE CATALOG...")
            return None
            
        galaxy  = redmapper.galaxy.GalaxyCatalog.from_galfile(self.outBase + "_master_table.fit")
        allspec = fitsio.read(self.outBase + ".all_specz.fits")

        print("LOADED ALL CATALOGS")
        
        id1, id2, dist = galaxy.match_many(allspec['ra'], allspec['dec'], 0.5/3600, maxmatch = 1)

        allspec = allspec[id1]
        galmag  = galaxy.mag[id2]

        #Now plot the sequences
        use, = np.where(galmag[:, 0] > 0.0)
        gmr = galmag[:, 0] - galmag[:, 1]
        rmi = galmag[:, 1] - galmag[:, 2]
        imz = galmag[:, 2] - galmag[:, 3]

        plt.hexbin(allspec['z'][use], gmr[use], extent=[0, 1, 0.0, 2.5], bins='log')
        plt.xlabel('z'); plt.ylabel('gmr')
        z1 = 0.05
        z2 = 0.4
        c1 = 0.6
        c2 = 1.6

        m = (c2 - c1) / (z2 - z1)
        cut_gmr = m * allspec['z'][use] - m * z1 + c1

        plt.plot(np.array([z1, z2]), m * np.array([z1, z2]) - m*z1 + c1, 'r--')
        plt.savefig(self.outBase + "_all_specz_z_gmr.png"); plt.close()

        plt.hexbin(allspec['z'][use], rmi[use], extent=[0, 1, 0.0, 2.0], bins='log')
        plt.xlabel('z'); plt.ylabel('rmi')

        z1 = 0.4
        z2 = 0.8
        c1 = 0.4
        c2 = 1.15

        m = (c2 - c1) / (z2 - z1)
        cut_rmi1 = m * allspec['z'][use] - m * z1 + c1

        plt.plot(np.array([z1, z2]), m * np.array([z1, z2]) - m*z1 + c1, 'r--')

        z1 = 0.8
        z2 = 1.0
        c1 = 1.15
        c2 = 0.85

        m = (c2 - c1) / (z2 - z1)
        cut_rmi2 = m * allspec['z'][use] - m * z1 + c1

        plt.plot(np.array([z1, z2]), m * np.array([z1, z2]) - m*z1 + c1, 'r--')
        plt.savefig(self.outBase + "_all_specz_z_rmi.png"); plt.close()


        plt.hexbin(allspec['z'][use], rmi[use], extent=[0, 1, 0.0, 1.5], bins='log')
        plt.xlabel('z'); plt.ylabel('imz')
        plt.savefig(self.outBase + "_all_specz_z_imz.png"); plt.close()

        print("PLOTTED ALLSPEC")

        reduse, = np.where(((allspec['z'][use] > 0.1) & (allspec['z'][use] < 0.4) & (gmr[use] > cut_gmr)) |
                           ((allspec['z'][use] > 0.4) & (allspec['z'][use] < 0.8) & (rmi[use] > cut_rmi1)) |
                           ((allspec['z'][use] > 0.8) & (rmi[use] > cut_rmi2)))
        

        print("GENERATED CUTS")

        plt.hexbin(allspec['z'][use[reduse]], gmr[use[reduse]], extent=[0, 1, 0.0, 2.5], bins='log')
        plt.xlabel('z'); plt.ylabel('gmr')
        plt.savefig(self.outBase + "_train_specz_z_gmr.png"); plt.close()

        plt.hexbin(allspec['z'][use[reduse]], rmi[use[reduse]], extent=[0, 1, 0.0, 2.5], bins='log')
        plt.xlabel('z'); plt.ylabel('rmi')
        plt.savefig(self.outBase + "_train_specz_z_rmi.png"); plt.close()

        plt.hexbin(allspec['z'][use[reduse]], imz[use[reduse]], extent=[0, 1, 0.0, 2.5], bins='log')
        plt.xlabel('z'); plt.ylabel('imz')
        plt.savefig(self.outBase + "_train_specz_z_imz.png"); plt.close()

        #Save the proper sample now
        fitsio.write(self.outBase + ".train_specz.fits", allspec[use[reduse]], clobber = True)

        print("SAVED CUT CATALOG")

    @timeit
    def make_calibration_hpix(self):

        specz       = fitsio.read(self.outBase + ".train_specz.fits")
        hpix_specz  = np.unique(hp.ang2pix(8, specz['ra'], specz['dec'], lonlat = True))

        print(f"{len(hpix_specz)} PIXELS HAVE SPECZ IN THEM")

        hpix_choice = sorted(glob.glob(self.outBase + '_pix?????.fits'))
        hpix_choice = [int(h[-10:-5]) for h in hpix_choice]
        hpix_choice = [h for h in hpix_choice if h in hpix_specz]
        np.random.default_rng(seed = self.seed).shuffle(hpix_choice)
        hpix_choice = hpix_choice[:int(len(hpix_choice) * 0.5)] #Use half of sample to calibrate

        return hpix_choice
    

    @timeit
    def make_config_yaml(self):

        if os.path.isfile(self.outBase + '_config.yaml'):
            print("CONFIG ALREADY EXISTS. SKIPPING...")
            return None
        
        hpix_choice = self.make_calibration_hpix()

        CONFIG = f"""
        # Galaxy file for input
        outpath: '{os.path.dirname(self.outBase)}'
        galfile: '{self.outBase + "_master_table.fit"}'

        redgalfile: '{self.outBase + "_zspec_redgals.fit"}'
        redgalmodelfile: '{self.outBase + "_zspec_redgals_model.fit"}'
        
        # Path to put plots in
        plotpath: '{os.path.dirname(self.outBase)}/plots'

        # Galaxy catalog has truth information (as with simulated catalogs)
        has_truth: False

        # Healpixel over which to run calibration
        # Set this and nside to 0 to run the full footprint
        hpix: {hpix_choice}
        # Nside of this healpix
        nside: 8
        # Should be set to 0 for calibration
        border: 0.0

        # Reference magnitude (band) name
        refmag: z

        # Redshift range [lo, hi]
        zrange: [0.1, 0.95]

        # Spectroscopic input catalog
        specfile: {self.outBase + ".all_specz.fits"}

        specfile_train: {self.outBase + ".train_specz.fits"}

        # All files will start with this name
        # If the name ends with "cal" it will be replaced with "run" for the runs
        outbase: my_decade

        # Maximum chi-squared to consider a possible member.  Default is 20.0
        chisq_max: 20.0

        # L* threshold for computing richness.  Default is optimal 0.2
        lval_reference: 0.2

        # Name of the "survey" for the m*(z) file
        # (see README.md for details)
        mstar_survey: des
        # Name of the "band" for the m*(z) file
        mstar_band: z03

        # Maximum weighted coverage area of a cluster to be masked before it
        # is removed from the catalog.  Default is 0.2
        max_maskfrac: 0.2

        # Number of sample galaxies to compute mask region.  Default is 6000
        maskgal_ngals: 6000
        # Number of different samples of maskgal_ngals (reduces biases at high redshift)
        maskgal_nsamples: 100
        # Filetype mode for the geometric mask.  Default is 0
        # Mode 3 is a healpix mask.  (only mode supported)
        mask_mode: 3
        # Name of the mask file.  See README.md for format.  Default is None
        maskfile: {self.outBase + '.decade_pixmask.hs'}

        # Name of the depth file.  See README.md for format.  Default is None
        depthfile: {self.outBase + f'-10z_bdf_depthstr.hs'}

        # chi-squared binsize for background.  Default is 0.5
        bkg_chisqbinsize: 0.5
        # reference mag binsize for background.  Default is 0.2
        bkg_refmagbinsize: 0.2
        # redshift binsize for background.  Default is 0.02
        bkg_zbinsize: 0.02
        # redshift binsize for zred background.  Default is 0.01
        bkg_zredbinsize: 0.01
        # Compute background down to the magnitude limit?  Default is False
        # This will be useful in the future when computing membership probabilities
        # for non-members (not supported yet).  If turned on, background
        # computation is slower (depending on depth).
        bkg_deepmode: False

        # Name of centering class for calibration first iteration.  Default is CenteringBCG
        firstpass_centerclass: CenteringBCG
        # Name of centering class for cluster runs.  Default is CenteringWcenZred
        centerclass: CenteringWcenZred

        # Number of iterations for calibration.  Default is 3
        calib_niter: 3
        # Number of cores on local machine for calibration (zred, background)
        # Default is 1
        calib_nproc: {8}
        # Number of cores on local machine for calibration cluster finder runs
        # Default is 1
        calib_run_nproc: {16}

        # Nsig for consistency with red sequence to be used in red-sequence calibration.  Default is 1.5
        #  Make this too wide, and blue galaxies contaminate the red sequence width computation.
        #  Make this too narrow, and there is not enough signal.  Empirically, 1.5 works well.
        calib_color_nsig: 1.5
        # Nsig for consistence with red sequence to be used as a training seed galaxy.  Default is 2.0
        calib_redspec_nsig: 2.0

        # Red-sequence color template file for initial guess of red sequence.
        calib_redgal_template: bc03_colors_des.fit
        # Redshift spline node spacing for red-sequence pivot magnitude
        # Default is 0.1
        calib_pivotmag_nodesize: 0.1
        # Redshift spline node spacing for each color.  Must be array of length nmag - 1
        # Recommended default is 0.05 for each color
        calib_color_nodesizes: [0.05, 0.05, 0.05]
        # Redshift spline node spacing for each slope.  Must be array of length nmag - 1
        # Recommended default is 0.1 for each color slope
        calib_slope_nodesizes: [0.1, 0.1, 0.1]
        # Maximum redshift for spline to use in fit.
        # Recommended is -1 (max redshift) for each color, unless a band is very shallow/blue
        # (for SDSS, only consider u-g in detail at z<0.4)
        calib_color_maxnodes: [0.8, -1, -1]
        # Maximum redshift for covariance spline to use in fit.
        # Recommended is -1 (max redshift) for each color, unless a band is very shallow/blue
        # (For SDSS, only consider u-g in detail in covmat at z<0.4)
        calib_covmat_maxnodes: [-1, -1, -1]
        # Redshift spline node spacing for covariance matrix.  Default is 0.15
        calib_covmat_nodesize: 0.25

        calib_covmat_constant: 0.9

        # Redshift spline node spacing for zred corrections.  Default is 0.05
        calib_corr_nodesize: 0.1
        # Redshift spline node spacing for zred slope corrections.  Default is 0.1
        calib_corr_slope_nodesize: 0.2

        # Use "pcol" (which excludes radial weight) in selecting galaxies to calibrate the
        #  red sequence.  Default is True, which is strongly recommended.
        calib_use_pcol: True
        # Membership probability cut to compute zred corrections.  Default is 0.9
        calib_corr_pcut: 0.8
        # Membership probability cut for pivotmag and median color computation.  Default is 0.7
        calib_color_pcut: 0.5

        # Membership probability cut for use in red-sequence calibration.  Default is 0.3
        calib_pcut: 0.3
        # Minimum richness for a cluster to be considered a calibration cluster.  Default is 5.0
        calib_minlambda: 5.0
        # Smoothing kernel on redshifts from calibration clusters.  Default is 0.003
        calib_smooth: 0.003

        # Luminosity function filter alpha parameter.
        calib_lumfunc_alpha: -1.0

        # Zeroth iteration color training parameters

        # Scale radius of radius/richness relation (r_lambda = r0 * (lambda/100)^beta)
        # Default is 0.5 (h^-1 Mpc)
        calib_colormem_r0: 0.5
        # Power-law slope of radius/richness relation
        # Default is 0.0
        calib_colormem_beta: 0.0
        # Smoothing kernel on redshifts from color calib clusters.  Default is 0.003
        calib_colormem_smooth: 0.003
        # Minimum richness to be used as a color-training cluster.  Default is 10.0
        calib_colormem_minlambda: 10.0
        # Color indices for training, redshift bounds, and assumed intrinsic scatter.
        # For the following settings:
        #  Low redshift: z<0.35, use color index 1 (g-r), intrinsic scatter 0.05
        #  Middle redshift: 0.5<z<0.72, use color index 2 (r-i), intrinsic scatter 0.03
        #  High redshift: z>0.72, use color index 3 (i-z), intrinsic scatter 0.03
        # (Note that the high redshift one is not used if zrange[1] = 0.60)
        calib_colormem_colormodes: [0, 1, 2]
        calib_colormem_zbounds: [0.35, 0.8]
        calib_colormem_sigint: [0.05, 0.03, 0.03]

        # Cluster photo-z z_lambda parameters

        # Redshift spline spacing for correcting cluster z_lambda.  Default is 0.04
        calib_zlambda_nodesize: 0.1
        # Redshift spline spacing for richness slope correction of cluster z_lambda.
        # Default is 0.1
        calib_zlambda_slope_nodesize: 0.1
        # Minimum richness for computing z_lambda correction.  Default is 20.0
        calib_zlambda_minlambda: 5.0
        # Number of z_lambda_e sigma an outlier can be to be included in correction
        # Default is 5.0
        calib_zlambda_clean_nsig: 3.5
        # Number of iterations for z_lambda correction algorithm.  Default is 3
        calib_zlambda_correct_niter: 3

        # Pivot richness for richness correction.  Default is 30.0
        zlambda_pivot: 30.0
        # Bin size for interpolating z_lambda correction.  Default is 0.002
        zlambda_binsize: 0.002
        # Tolerance for convergence when computing z_lambda.  Default is 0.0002
        zlambda_tol: 0.0002
        # Maximum number of iterations to converge to z_lambda.  Default is 20
        zlambda_maxiter: 20
        # Fraction of highest probability members to use to compute z_lambda.  Default is 0.7
        zlambda_topfrac: 0.7
        # Step size to fit a parabola to peak of z_lambda likelihood.  Default is 0.002
        zlambda_parab_step: 0.002
        # Epsilon size to compute change in richness as a function of redshift.  Default is 0.005
        zlambda_epsilon: 0.005

        # Centering parameters

        # Pivot richness for wcen model.  Default is 30.0
        wcen_pivot: 30.0
        # Minumum richness to use in calibrating wcen model.  Default is 10.0
        wcen_minlambda: 10.0
        # Maximum richness to use in calibrating wcen model.  Default is 100.0
        wcen_maxlambda: 100.0
        # Softening radius (h^-1 Mpc) in computing wcen.  Default is 0.05
        wcen_rsoft: 0.05
        # Richness range for calibrating wcen model.
        #  This should be within the volume-limited range of the catalog.
        wcen_cal_zrange: [0.2, 0.65]
        # Cluster finder: first pass

        # r0 in radius-richness relation (see above).  Default is 0.5 (h^-1 Mpc)
        firstpass_r0: 0.5
        # beta in radius-richness relation.  Default is 0.0
        firstpass_beta: 0.0
        # Number of iterations in first pass.  Default is 2
        firstpass_niter: 2
        # Minimum richness to pass cluster candidate to next step.  Default is 3.0
        firstpass_minlambda: 3.0

        # Cluster finder: Likelihood pass

        # r0 is radius-richness relation (see above).  Default is 1.0 (h^-1 Mpc)
        #  Note that these should typically be the same as used in percolation
        likelihoods_r0: 1.0
        # beta in radius-richness relation.  Default is 0.2
        likelihoods_beta: 0.2
        # Should likelihood use the zred in computing likelihood.  Default is True
        likelihoods_use_zred: True
        # Minimum richness to pass cluster candidate to next step.  Default is 3.0
        likelihoods_minlambda: 3.0

        # Cluster finder: Percolation pass

        # r0 is radius-richness relation (see above).  Default is 1.0 (h^-1 Mpc)
        #  Note that these should typically be the same as used in likelihood
        percolation_r0: 1.0
        # beta in radius-richness relation.  Default is 0.2
        percolation_beta: 0.2
        # rmask_0 in rmask = rmask_0 * (lambda / 100) ^ rmask_beta * (z_lambda / rmask_zpivot)^rmask_gamma
        #  relation.  This sets the radius to which member galaxies are masked in percolation.
        #  This should be >= percolation_r0.  Default is 1.5
        percolation_rmask_0: 1.5
        # beta in rmask relation
        percolation_rmask_beta: 0.2
        # gamma in rmask relation
        percolation_rmask_gamma: 0.0
        # zpivot in rmask relation
        percolation_rmask_zpivot: 0.3
        # Mask galaxies down to this luminosity cut (fainter than richness computation).  Default is 0.1 (L*)
        percolation_lmask: 0.1
        # Number of iterations to compute richness/redshift in percoltation.  Default is 2
        percolation_niter: 2
        # Minimum richness to save cluster.  Default is 3.0
        percolation_minlambda: 3.0
        # Minimum probability of being a bcg for a galaxy to be considered a center.  Default is 0.5
        percolation_pbcg_cut: 0.5
        # Maximum number of possible centrals to record.  Default is 5
        percolation_maxcen: 5


        vlim_depthfiles:
        - {self.outBase + f'-10i_bdf_depthstr.hs'}
        - {self.outBase + f'-10r_bdf_depthstr.hs'}
        - {self.outBase + f'-10g_bdf_depthstr.hs'}
        vlim_bands: ['i', 'r', 'g']
        vlim_nsigs: [5.0, 5.0, 3.0]
        """
        
        with open(self.outBase + '_config.yaml', 'w') as f:
            f.write(textwrap.dedent(CONFIG))

    @timeit
    def run_redmapper_calibration(self):

        calib = redmapper.calibration.RedmapperCalibrator(self.outBase + '_config.yaml')
        calib.run()


    @timeit
    def run_zred_pixel(self, n_jobs = -1):
    
        # config = redmapper.Configuration(os.path.dirname(self.outBase) + '_run/run_default.yml')
        # zredRunpix = redmapper.ZredRunPixels(config)
        # # This will use python multiprocessing to run on config.calib_nproc cores
        # zredRunpix.run()

        dirname  = os.path.dirname(self.outBase)
        basename = os.path.basename(self.outBase)
        outpath  = f"{dirname}_run/zreds/{basename}_zreds_master_table.fit"

        if os.path.isfile(outpath):
            print("FOUND ZRED MASTER TABLE AT", outpath, "SKIPPING....")
            return None
        

        filelist = sorted(glob.glob(self.outBase + '_pix*_bdf.fits')) #Get all files by checking for bdf fits
        pixlist  = [int(f[-14:-9]) for f in filelist] #Get only the pixel part of the name
        pixlist  = [pixlist[i] for i in np.random.choice(len(pixlist), len(pixlist), replace = False)] #Randomize to balance the load
        
        if n_jobs == -1: n_jobs = np.min([os.cpu_count(), len(filelist)])

        print(f"RUNNING {len(pixlist)} PIXELS USING {n_jobs} JOBS", flush = True)

        jobs = [joblib.delayed(self._single_step_zred_pixel)(p) for p in pixlist]
        out  = joblib.Parallel(n_jobs = n_jobs, verbose = 10)(jobs)

    
    def _single_step_zred_pixel(self, pix):

        runZredPixelTask = redmapper.pipeline.RunZredPixelTask(os.path.dirname(self.outBase) + '_run/run_default.yml', 
                                                               pix, 8, path = os.path.dirname(self.outBase))
        runZredPixelTask.run()


    @timeit
    def run_zred_bkg(self):

        config = redmapper.Configuration(os.path.dirname(self.outBase) + '_run/run_default.yml')
        zb     = redmapper.ZredBackgroundGenerator(config)
        zb.run()


    @timeit
    def run_redmapper_pixel(self, n_jobs = 16):

        filelist = sorted(glob.glob(self.outBase + '_pix*_bdf.fits')) #Get all files by checking for bdf fits
        pixlist  = [int(f[-14:-9]) for f in filelist] #Get only the pixel part of the name
        pixlist  = [pixlist[i] for i in np.random.choice(len(pixlist), len(pixlist), replace = False)] #Randomize to balance the load
        
        #Find which hpix we have zreds for. Even though we start with the full catalog
        #We subselect and only run on regions with foreground mask (and fracgood) set.
        #So this figures out which coarse pixels can still be used.
        hpixlist = sorted(glob.glob(f"{os.path.dirname(self.outBase)}_run/zreds/*zreds_???????.fit"))
        hpixlist = [int(f[-11:-4]) for f in hpixlist]
        hpixlist = hp.ang2pix(8, *hp.pix2ang(32, hpixlist))

        pixlist  = [p for p in pixlist if p in hpixlist]

        np.save('/scratch/midway3/dhayaa/hpix8.npy', np.array(pixlist))
        if n_jobs == -1: n_jobs = np.min([os.cpu_count(), len(filelist)])

        print(f"RUNNING {len(pixlist)} PIXELS (OUT OF POSSIBLE {len(filelist)}) USING {n_jobs} JOBS", flush = True)

        jobs = [joblib.delayed(self._single_step_run_redmapper_pixel)(p) for p in pixlist]
        out  = joblib.Parallel(n_jobs = n_jobs, verbose = 10)(jobs)

    
    def _single_step_run_redmapper_pixel(self, pix):

        if os.path.isfile(os.path.dirname(self.outBase) + f'/my_decade_run_8_{pix:05d}_final_catalog.fit'):
            print("HPIX RUN EXISTS. SKIPPING.....")
            return None
        
        runRedmapperPixelTask = redmapper.pipeline.RunRedmapperPixelTask(os.path.dirname(self.outBase) + '_run/run_default.yml', 
                                                                         pix, 8, path = os.path.dirname(self.outBase))
        runRedmapperPixelTask.run()


    @timeit
    def consolidate_redmapper_pixel(self):

        if os.path.isfile(os.path.dirname(self.outBase) + "/my_decade_run_redmapper_v0.8.7_lgt05_vl02_catalog.fit"):
            print("FILE EXISTS. SKIPPING.....")
            return None
        
        consolidate = redmapper.pipeline.RedmapperConsolidateTask(os.path.dirname(self.outBase) + '_run/run_default.yml')
        consolidate.run()


    @timeit
    def make_randoms(self):

        config = os.path.dirname(self.outBase) + '/zmask02/zmask02.yml'

        self.setup_random_run(config)
        self.generate_randoms(config)
        self.redmapper_randoms_zmask(config)
        self.weight_randoms(config)


    @timeit
    def setup_random_run(self, config_path):

        if os.path.isfile(os.path.dirname(self.outBase) + "/zmask02/my_decade_run_redmapper_v0.8.7_vl02_vlim_zmask.fit"):
            print("FILES EXIST. SKIPPING....")
            return None
        
        #Make a new directory and duplicate config files
        os.makedirs(os.path.dirname(config_path), exist_ok = True)
        run_config  = os.path.dirname(self.outBase) + '_run/run_default.yml'
        rand_config = config_path
        shutil.copy(run_config, rand_config)
        
        #Rewrite lines in config
        path = pathlib.Path(rand_config)
        data = yaml.safe_load(path.read_text())
        data["catfile"]  = os.path.dirname(self.outBase) + "/my_decade_run_redmapper_v0.8.7_lgt05_vl02_catalog.fit"
        data["randfile"] = os.path.dirname(self.outBase) + "/zmask02/my_decade_randoms_master_table.fit"
        path.write_text(yaml.safe_dump(data, sort_keys=False))

        config = redmapper.Configuration(rand_config)

        #Move zmask file over
        shutil.copy(os.path.dirname(self.outBase) + "/my_decade_run_redmapper_v0.8.7_vl02_vlim_zmask.fit", 
                    os.path.dirname(self.outBase) + "/zmask02/")


    @timeit
    def generate_randoms(self, config):

        if os.path.isfile(os.path.dirname(self.outBase) + "/zmask02/my_decade_randoms_master_table.fit"):
            print("FILES EXIST. SKIPPING....")
            return None
        
        #Run Step 6.1
        generateRandoms = redmapper.GenerateRandoms(redmapper.Configuration(config))
        generateRandoms.generate_randoms(50_000_000)


    @timeit
    def redmapper_randoms_zmask(self, config, n_jobs = -1):

        filelist = sorted(glob.glob(os.path.dirname(config) + '/*randoms_???????.fit')) #Get all files by checking for bdf fits
        pixlist  = [int(f[-11:-4]) for f in filelist] #Get only the pixel part of the name
        pixlist  = [pixlist[i] for i in np.random.choice(len(pixlist), len(pixlist), replace = False)] #Randomize to balance the load
        
        if n_jobs == -1: n_jobs = np.min([os.cpu_count(), len(filelist)])

        print(f"RUNNING {len(pixlist)} PIXELS (OUT OF POSSIBLE {len(filelist)}) USING {n_jobs} JOBS", flush = True)

        jobs = [joblib.delayed(self._single_step_run_zmask_pixel)(p, config) for p in pixlist]
        out  = joblib.Parallel(n_jobs = n_jobs, verbose = 10)(jobs)

        if os.path.isfile(os.path.dirname(self.outBase) + '/my_decade_run_redmapper_v0.8.7_randoms_zmask_catalog.fit'):
            print("Consolidated file exists. Skipping.....")
            return None

        consolidate = redmapper.pipeline.RuncatConsolidateTask(config)
        consolidate.run(do_plots = False, match_spec = False, consolidate_members = False, cattype = 'randoms_zmask')

    
    def _single_step_run_zmask_pixel(self, pix, config):

        if os.path.isfile(os.path.dirname(self.outBase) + f'/my_decade_run_32_{pix:05d}_randoms_zmask_catalog.fit'):
            return None
        
        runRedmapperPixelTask = redmapper.pipeline.RunZmaskPixelTask(config, pix, 32, path = os.path.dirname(self.outBase))
        runRedmapperPixelTask.run()


    @timeit
    def weight_randoms(self, config):

        if os.path.isfile(os.path.dirname(self.outBase) + '/my_decade_run_redmapper_v0.8.7_weighted_randoms_z010-095_lgt020_vl02_area.fit'):
            print("FILES EXIST. SKIPPING...")
            return None
        
        randfile = os.path.dirname(self.outBase) + '/my_decade_run_redmapper_v0.8.7_randoms_zmask_catalog.fit'

        config   = redmapper.Configuration(config)
        weigher  = redmapper.RandomWeigher(config, randfile)
        for lambda_cut in [5, 20]:
            wt_randfile, wt_areafile = weigher.weight_randoms(lambda_cut)

            print("Made weighted random file %s" % (wt_randfile))
            print("Made weighted area file %s" % (wt_areafile))


    @timeit
    def make_redmagic(self):

        self.calibrate_redmagic()
        self.generate_redmagic()


    @timeit
    def calibrate_redmagic(self):

        if os.path.isfile(os.path.dirname(self.outBase) + '/redmagic/my_decade_run_redmagic_calib.fit'):
            print("File exists. Skipping....")
            return None

        #Make a new directory and duplicate config files
        os.makedirs(os.path.dirname(self.outBase) + '/redmagic', exist_ok = True)
        run_config  = os.path.dirname(self.outBase) + '_run/run_default.yml'
        rmgc_config = os.path.dirname(self.outBase) + '/redmagic/run_default.yml'
        shutil.copy(run_config, rmgc_config)
        
        #Rewrite lines in config
        path = pathlib.Path(rmgc_config)
        data = yaml.safe_load(path.read_text())
        data["outpath"]  = os.path.dirname(self.outBase) + "/redmagic/"
        data["catfile"]  = os.path.dirname(self.outBase) + "/my_decade_run_redmapper_v0.8.7_lgt20_vl02_catalog.fit"
        data["redmagic_etas"]   = [0.5, 1.0]
        data["redmagic_n0s"]    = [10.0, 4.0]
        data["redmagic_names"]  = ['highdens', 'highlum']
        data["redmagic_zmaxes"] = [0.75, 0.95]
        data["redmagicfile"]    = self.outBase + '_redmagic_calib.fit'
        data["hpix"]            = self.make_calibration_hpix()
        data["nside"]           = 8

        path.write_text(yaml.safe_dump(data, sort_keys=False))

        calib = redmapper.redmagic.RedmagicCalibrator(rmgc_config)
        calib.run(do_run = False)

    @timeit
    def generate_redmagic(self):

        rmgc_config  = os.path.dirname(self.outBase) + '/redmagic/run_default_run.yaml'
        
        #Rewrite lines in config
        path = pathlib.Path(rmgc_config)
        data = yaml.safe_load(path.read_text())
        data["redmagicfile"]  = os.path.dirname(self.outBase) + '/redmagic/my_decade_run_redmagic_calib.fit'
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        
        run_redmagic = redmapper.redmagic.RunRedmagicTask(rmgc_config)
        run_redmagic.run()


    @timeit
    def generate_zscan(self, outbase, catfile):
        
        run_config = os.path.dirname(self.outBase) + '_run/run_default.yml'
        zsn_config = os.path.dirname(self.outBase) + '_run/run_zscan.yml'

        shutil.copy(run_config, zsn_config)

        #Rewrite lines in config
        path = pathlib.Path(zsn_config)
        data = yaml.safe_load(path.read_text())
        data["catfile"] = catfile
        data["outbase"] = outbase
        path.write_text(yaml.safe_dump(data, sort_keys=False))

        runzscan   = redmapper.RunZScan(zsn_config)
        runzscan.run()
        runzscan.output(savemembers = True, withversion = True)


class CombinedRunner(BaseRunner):

    @timeit
    def prep_training_catalog(self, n_jobs):
        
        BANDS = 'GRIZ'
        keys  = []
        keys += [f'BDF_FLUX_{B}_DERED_SFD98'     for B in BANDS]
        keys += [f'BDF_FLUX_ERR_{B}_DERED_SFD98' for B in BANDS]
        keys += ['RA', 'DEC', 'COADD_OBJECT_ID']

        out   = {k : [] for k in keys + ['mask']}

        outpath = self.outBase + '.tmp_input_cat.hdf5'
        
        if os.path.isfile(outpath):
            print("FILE ALREADY EXISTS....SKIPPING....")
            return None
        else:
            print("STARTING TO MAKE OUTPUT")

        #Silly thing I need to do because of how I saved gold catalog from DELVE
        def read(f, k): return f[k][k][:]

        with h5py.File(f'/project/kadrlica/dhayaa/delve_dr3/delve_dr3_gold.hdf5', 'r') as f:

            GLDf = ((read(f, 'EXT_MASH') >= 2) & 
                    (read(f, 'FLAGS_FOREGROUND') == 0) &
                    (read(f, 'FLAGS_FOOTPRINT') > 0) & 
                    (read(f, 'FLAGS_GOLD') == 0)
                    )
    
            with fits.open('/project/kadrlica/dhayaa/DES/Data/y6_gold_2.2.fits') as g:
                g    = g[1].data
                GLDg = ((g['EXT_MASH'][:] >= 2) & 
                        (g['FLAGS_FOREGROUND'][:] == 0) &
                        (g['FLAGS_FOOTPRINT'][:] > 0) & 
                        (g['FLAGS_GOLD'][:] == 0)
                        )
                
                print("FINISHED FIRST MASK")
                MSK  = hsp.HealSparseMap.read('/scratch/midway3/dhayaa/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz')
                MSK  = MSK.get_values_pos(g['RA'][:], g['DEC'][:], lonlat = True) > 0
                GLDg = GLDg & MSK

                del MSK

                if os.path.isfile(outpath): print("FILE EXISTS AT", outpath, "... OVERWRITING IT....")
                with h5py.File(outpath, 'w') as h:
                    
                    for k in ['RA', 'DEC', 'COADD_OBJECT_ID']: 
                        h.create_dataset(name = k, data = np.concatenate([read(f,k)[GLDf], g[k][:][GLDg]]))
                        print("FINISHED WRITING", k)

                    A = dict(G = 3.186, R = 2.140, I = 1.568, Z = 1.196)
                    E = np.concatenate([read(f,'EBV_SFD98')[GLDf], g['EBV_SFD98'][:][GLDg]])
                    for b in 'GRIZ':
                        Ab = np.power(10, 0.4 * E * A[b])

                        for ki in ['BDF_FLUX_%s_DERED_SFD98', 'BDF_FLUX_ERR_%s_DERED_SFD98']:
                            k   = ki % b
                            kin = k.replace('_DERED_SFD98', '_CORRECTED')
                            h.create_dataset(name = k, data = np.concatenate([read(f,kin)[GLDf], g[kin][:][GLDg]]) * Ab)
                            print("FINISHED WRITING", k)
                    
        print("OUTPUT WRITTEN TO", outpath)

        Collator = mapper.utils.DECADECollator(self.outBase, n_jobs = n_jobs)
        Collator.run(outpath)


    @timeit
    def get_fracdet_map(self):

        return hsp.HealSparseMap.read("/project/chto/dhayaa/Redmapper/All_maglim_fracdet.hsp").fracdet_map(nside = 4096)
    
    
    @timeit
    def get_foreground_map(self):

        return hp.read_map("/project/chto/dhayaa/decade/GOLD_Ext0.2_Star5_MCs2.fits")
    

    @timeit
    def prepare_sys_maps(self):

        for b in 'griz':
            for sys in ['airmass', 'fwhm', 'exptime', 'skybrite', 'maglim']:

                if os.path.isfile(self.outBase + f'_DR3_{b}_{sys}.hpy'):
                    print(f"FILE DR3_{b}_{sys}.hpy EXISTS. SKIPPING...")
                    continue
                
                if sys == 'exptime':
                    text = 'sum'
                else:
                    text = 'wmean'
                    
                DR31 = hsp.HealSparseMap.read(f"/project/jfrieman/chinyi/dr3_2_decasu_maps/{sys}/delve_dr32_{b}_{sys}_{text}.hsp")
                DR32 = hsp.HealSparseMap.read(f"/project/chihway/secco/decasu_outputs/{sys}/delve_dr311+dr312_{b}_{sys}_Nov28th.hsp")
                Y6   = hsp.HealSparseMap.read(f"/project/kadrlica/dhayaa/DES/sys_maps/y6a2_decasu_{b}_{sys}_{text}.hs")

                DR31 = DR31.generate_healpix_map(nside = 4096, nest = False)
                DR32 = DR32.generate_healpix_map(nside = 4096, nest = False)
                Y6   = Y6.generate_healpix_map(nside = 4096,   nest = False)

                DR31 = np.where(DR31 == hp.UNSEEN, 0, DR31)
                DR32 = np.where(DR32 == hp.UNSEEN, 0, DR32)
                Y6   = np.where(DR32 == hp.UNSEEN, 0, Y6)

                DR3  = DR31 + DR32
                DR3  = np.where(DR3 != 0, DR3, Y6)
                DR3  = np.where(DR3 == 0, hp.UNSEEN, DR3)
                hp.write_map(self.outBase + f'_DR3_{b}_{sys}.hpy', DR3)
                
                print(f"FINISHED WRITING MAP {sys} IN BAND {b}", flush = True)

        if os.path.isfile(self.outBase + f'_DR3_z_frac.hpy'): return None
        FRACGOOD = hsp.HealSparseMap.read(self.outBase + '.decade_pixmask.hs')
        FRACGOOD = FRACGOOD.fracdet_map(nside=4096)
        FRACGOOD = FRACGOOD.generate_healpix_map(nside = 4096, nest = False)
        for b in 'griz': hp.write_map(self.outBase + f'_DR3_{b}_frac.hpy', FRACGOOD)


class DESRunner(CombinedRunner):

    @timeit
    def prep_training_catalog(self, n_jobs):
        
        BANDS = 'GRIZ'
        keys  = []
        keys += [f'BDF_FLUX_{B}_DERED_SFD98'     for B in BANDS]
        keys += [f'BDF_FLUX_ERR_{B}_DERED_SFD98' for B in BANDS]
        keys += ['RA', 'DEC', 'COADD_OBJECT_ID']

        out   = {k : [] for k in keys + ['mask']}

        outpath = self.outBase + '.tmp_input_cat.hdf5'
        
        if os.path.isfile(outpath):
            print("FILE ALREADY EXISTS....SKIPPING....")
            return None
        else:
            print("STARTING TO MAKE FILE..")

        with fits.open('/project/kadrlica/dhayaa/DES/Data/y6_gold_2.2.fits') as hdu:
            f   = hdu[1].data
            GLD = ( (f['EXT_MASH'][:] >= 2) & 
                    (f['FLAGS_FOREGROUND'][:] == 0) &
                    (f['FLAGS_FOOTPRINT'][:] > 0) & 
                    (f['FLAGS_GOLD'][:] == 0)
                    )
            
            print("FINISHED MASK")

            MSK = hsp.HealSparseMap.read('/scratch/midway3/dhayaa/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz')
            MSK = MSK.get_values_pos(f['RA'][:], f['DEC'][:], lonlat = True) > 0
            GLD = GLD & MSK

            print("FINISHED MASK 2")
            
            out['RA'].append(f['RA'][:][GLD]); print("FINISHED RA")
            out['DEC'].append(f['DEC'][:][GLD]); print("FINISHED DEC")
            out['COADD_OBJECT_ID'].append(f['COADD_OBJECT_ID'][:][GLD]); print("FINISHED COADD_OBJECT_ID")

            out['BDF_FLUX_G_DERED_SFD98'].append(f['BDF_FLUX_G_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 3.186)); print("FINISHED FLUX G")
            out['BDF_FLUX_R_DERED_SFD98'].append(f['BDF_FLUX_R_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 2.140)); print("FINISHED FLUX R")
            out['BDF_FLUX_I_DERED_SFD98'].append(f['BDF_FLUX_I_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 1.568)); print("FINISHED FLUX I")
            out['BDF_FLUX_Z_DERED_SFD98'].append(f['BDF_FLUX_Z_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 1.196)); print("FINISHED FLUX Z")

            out['BDF_FLUX_ERR_G_DERED_SFD98'].append(f['BDF_FLUX_ERR_G_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 3.186)); print("FINISHED FLUX ERROR G")
            out['BDF_FLUX_ERR_R_DERED_SFD98'].append(f['BDF_FLUX_ERR_R_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 2.140)); print("FINISHED FLUX ERROR R")
            out['BDF_FLUX_ERR_I_DERED_SFD98'].append(f['BDF_FLUX_ERR_I_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 1.568)); print("FINISHED FLUX ERROR I")
            out['BDF_FLUX_ERR_Z_DERED_SFD98'].append(f['BDF_FLUX_ERR_Z_CORRECTED'][:][GLD] * np.power(10, 0.4 * f['EBV_SFD98'][:][GLD] * 1.196)); print("FINISHED FLUX ERROR Z")

            out['mask'].append(GLD)

        
        if os.path.isfile(outpath): print("FILE EXISTS AT", outpath, "... OVERWRITING IT....")
        with h5py.File(outpath, 'w') as f:

            for k in out.keys(): 
                f.create_dataset(name = k, data = np.concatenate(out[k]))
                print("FINISHED WRITING", k)

        del out
        print("OUTPUT WRITTEN TO", outpath)

        Collator = mapper.utils.DECADECollator(self.outBase, n_jobs = n_jobs)
        Collator.run(outpath)


    @timeit
    def get_fracdet_map(self):

        return hsp.HealSparseMap.read("/project/chto/dhayaa/Redmapper/All_maglim_fracdet.hsp").fracdet_map(nside = 4096)
    
    
    @timeit
    def get_foreground_map(self):

        return hsp.HealSparseMap.read('/scratch/midway3/dhayaa/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz').generate_healpix_map(nside = 4096, nest = False)


class DECADERunner(BaseRunner):

    @timeit
    def prep_training_catalog(self, n_jobs):
        
        BANDS = 'GRIZ'
        keys  = []
        keys += [f'BDF_FLUX_{B}_DERED_SFD98'     for B in BANDS]
        keys += [f'BDF_FLUX_ERR_{B}_DERED_SFD98' for B in BANDS]
        keys += ['RA', 'DEC', 'COADD_OBJECT_ID']

        out   = {k : [] for k in keys + ['mask']}

        outpath = self.outBase + '.tmp_input_cat.hdf5'
        
        if os.path.isfile(outpath):
            print("FILE ALREADY EXISTS....SKIPPING....")
            return None

        #Silly thing I need to do because of how I saved gold catalog from DELVE
        def read(f, k): return f[k][k][:]

        with h5py.File(f'/project/kadrlica/dhayaa/delve_dr3/delve_dr3_gold.hdf5', 'r') as f:

            GLD = ( (read(f, 'EXT_MASH') >= 2) & 
                    (read(f, 'FLAGS_FOREGROUND') == 0) &
                    (read(f, 'FLAGS_FOOTPRINT') > 0) & 
                    (read(f, 'FLAGS_GOLD') == 0)
                    )
            
            out['RA'].append(read(f,'RA')[GLD])
            out['DEC'].append(read(f,'DEC')[GLD])
            out['COADD_OBJECT_ID'].append(read(f,'COADD_OBJECT_ID')[GLD])

            out['BDF_FLUX_G_DERED_SFD98'].append(read(f,'BDF_FLUX_G_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 3.186))
            out['BDF_FLUX_R_DERED_SFD98'].append(read(f,'BDF_FLUX_R_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 2.140))
            out['BDF_FLUX_I_DERED_SFD98'].append(read(f,'BDF_FLUX_I_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 1.568))
            out['BDF_FLUX_Z_DERED_SFD98'].append(read(f,'BDF_FLUX_Z_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 1.196))

            out['BDF_FLUX_ERR_G_DERED_SFD98'].append(read(f,'BDF_FLUX_ERR_G_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 3.186))
            out['BDF_FLUX_ERR_R_DERED_SFD98'].append(read(f,'BDF_FLUX_ERR_R_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 2.140))
            out['BDF_FLUX_ERR_I_DERED_SFD98'].append(read(f,'BDF_FLUX_ERR_I_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 1.568))
            out['BDF_FLUX_ERR_Z_DERED_SFD98'].append(read(f,'BDF_FLUX_ERR_Z_CORRECTED')[GLD] * np.power(10, 0.4 * read(f,'EBV_SFD98')[GLD] * 1.196))

            out['mask'].append(GLD)

            
        if os.path.isfile(outpath): print("FILE EXISTS AT", outpath, "... OVERWRITING IT....")
        with h5py.File(outpath, 'w') as f:

            for k in out.keys(): 
                f.create_dataset(name = k, data = np.concatenate(out[k]))
                print("FINISHED WRITING", k)

        del out
        print("OUTPUT WRITTEN TO", outpath)

        Collator = mapper.utils.DECADECollator(self.outBase, n_jobs = n_jobs)
        Collator.run(outpath)


    @timeit
    def get_fracdet_map(self):
        return hsp.HealSparseMap.read("/project/chto/dhayaa/Redmapper/All_maglim_fracdet.hsp").fracdet_map(nside = 4096)
    
    
    @timeit
    def get_foreground_map(self):
        return hp.read_map("/project/chto/dhayaa/decade/GOLD_Ext0.2_Star5_MCs2_DESY6.fits")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--DECADE",   action = 'store_true')
    parser.add_argument("--DES",      action = 'store_true')
    parser.add_argument("--Combined", action = 'store_true')
    parser.add_argument("--outdir",   action = 'store', type = str, required = True)

    args = vars(parser.parse_args())

    assert (args['DECADE'] + args['DES'] + args['Combined']) == 1, "Use only one of DECADE, DES, or Combined"

    os.environ['TMPDIR'] = args['outdir']

    if args['DECADE']:
        Runner = DECADERunner
    elif args['DES']:
        Runner = DESRunner
    elif args['Combined']:
        Runner = CombinedRunner

    os.makedirs(args['outdir'], exist_ok = True)
    Runner(outBase = os.environ['TMPDIR'] + '/Eli').go()

    # BaseRunner(outBase = os.environ['TMPDIR'] + '/Eli').generate_zscan('my_decade_20250726', '/project/chto/chto/fordhayaa/wide_cluster_catalog_01may25.fits')
    # BaseRunner(outBase = os.environ['TMPDIR'] + '/Eli').generate_zscan('my_decade_rands_20250726', '/project/chto/chto/fordhayaa/wide_random_coords_01may25.fits')
    # BaseRunner(outBase = os.environ['TMPDIR'] + '/Eli').go()
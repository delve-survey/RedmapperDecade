import numpy as np, fitsio, h5py
import redmapper
import healsparse as hsp, healpy as hp
import os, re, esutil, sys, time, glob, joblib

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

    def __init__(self, outBase):
        self.outBase = outBase
        pass

    @timeit
    def go(self):

        # self.prep_training_catalog()
        self.make_all_maps()
        
    @timeit
    def prep_training_catalog(self):
        
        BANDS = 'GRIZ'
        keys  = []
        keys += [f'BDF_FLUX_{B}_DERED_SFD98'     for B in BANDS]
        keys += [f'BDF_FLUX_ERR_{B}_DERED_SFD98' for B in BANDS]
        keys += ['RA', 'DEC', 'COADD_OBJECT_ID']

        out   = {k : [] for k in keys + ['mask']}
        for file in ['metacal_gold_combined_20240209.hdf', 'metacal_gold_combined_20241003.hdf']:

            with h5py.File(f'/project/chihway/data/decade/{file}', 'r') as f:
                GLD = ( (f['FLAGS_SG_BDF'][:] >= 4) & 
                        (f['FLAGS_FOREGROUND'][:] == 0) &
                        (f['FLAGS_FOOTPRINT'][:] == 0) &
                        (f['FLAGS_BAD_COLOR'][:] == 0)
                        )
                for k in keys: out[k].append(f[k][:][GLD]); print("READ", k)
                out['mask'].append(GLD)

            print("FINISHED READING IN FILE", file)


        outpath = self.outBase + '.tmp_input_cat.hdf5'
        if os.path.isfile(outpath): print("FILE EXISTS AT", outpath, "... OVERWRITING IT....")
        with h5py.File(outpath, 'w') as f:

            for k in out.keys(): 
                f.create_dataset(name = k, data = np.concatenate(out[k]))
                print("FINISHED WRITING", k)

        print("OUTPUT WRITTEN TO", outpath)


        Collator = mapper.utils.DECADECollator(self.outBase, n_jobs = -1)
        Collator.run(outpath)


    @timeit
    def make_all_maps(self):

        # self.make_area_mask()
        self.make_depth_map()
    
    @timeit
    def get_fracdet_map(self):

        return hsp.HealSparseMap.read("/project/chto/dhayaa/Redmapper/DR3_maglim_fracdet.hsp").fracdet_map(nside = 4096)
    
    @timeit
    def get_foreground_map(self):

        return hp.read_map("/project/chto/dhayaa/decade/GOLD_Ext0.2_Star5_MCs2_DESY6.fits")
    
    @timeit
    def make_area_mask(self):

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
        hp.write_map(self.outBase + '.decade_pixmask.hpy', hsmask.generate_healpix_map(4096), nest = True, overwrite = True)

        print("WROTE TO DISK...")

    @timeit
    def make_depth_map(self, n_jobs = -1):

        self.compute_depth_pixel_process(n_jobs)
        self.prepare_sys_maps()
        self.compute_depth_map()
        # self.compute_exp_limit()


    @timeit
    def prepare_sys_maps(self):

        for b in 'griz':
            for sys in ['airmass', 'fwhm', 'exptime', 'skybrite', 'maglim']:

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
        
        etools.des_depth.catalogPixelProcess(path, self.outBase, 'bdf', 'flux_bdf', 'fluxerr_bdf', 1024, 
                                             nSidePixFile = 8, bandList = ['g', 'r', 'i', 'z'], noGoldFlags = True, 
                                             s2nCut = 5, selectGalaxies = True, bdSizeFileType = None, nTrial = 3)



    @timeit
    def compute_depth_map(self):

        files = glob.glob(self.outBase + '_pix?????.fits')
        print(f"FOUND {len(files)} PIXEL FILES. CONSOLATING.....")
        etools.des_depth.pixelConsolidate(files, self.outBase + '_coarse_depth.fits', nSide = 1024, nest = False)
        print("FINISHED CONSOLIDATING. MAKING DEPTH MAP....")


        mapMaker = etools.des_depth.MakeMap(self.outBase + '_coarse_depth.fits', os.path.dirname(self.outBase),
                                            os.path.basename(self.outBase) + "_DR3_%s_%s.hpy", self.outBase,
                                            sysTypes = ['fwhm', 'airmass', 'exptime', 'skybrite'],
                                            bands    = ['g', 'r', 'i', 'z'],
                                            aLambda  = [3.186, 2.140, 1.569, 1.196],
                                            npixFit  = [1, 1, 1, 1])
        mapMaker.run()


    @timeit
    def compute_exp_limit(self):

        etools.des_depth.expLimit(self.outBase + '_coarse_depth.fits', ['g', 'r', 'i', 'z'], npixMax = [5,5,5,5])


if __name__ == '__main__':

    os.environ['TMPDIR'] = '/scratch/midway3/dhayaa/RedMaPPer_TMP'

    BaseRunner(outBase = os.environ['TMPDIR'] + '/Eli').go()



    # import time
    # from contextlib import contextmanager

    # @contextmanager
    # def timer(label: str = "elapsed"):
    #     start = time.perf_counter()
    #     try:
    #         yield                # run the body of the with-block
    #     finally:
    #         dt = time.perf_counter() - start
    #         print(f"{label}: {dt:.6f} s")
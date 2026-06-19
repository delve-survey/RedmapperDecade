import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import yaml
import numpy as np
import fitsio
import matplotlib.pyplot as plt
from astropy.table import Table
from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density, get_boost
import tqdm


params = {'stellarmass': 'MASS_CG',
 'outmeasurement': '/home/chto/code/decade_redmapper/decade/decade_new_v1/dsigma/measurement_dsigma_boost.npy',
 'outcov': '/home/chto/code/decade_redmapper/decade/decade_new_v1/dsigma/cov_dsigma_boost.npy',
 'weight': True,
 'mcut': [8.2, 9.2, 9.2, 10.2],
 'table_ln': '/project2/kicp/chto/decades_saga_full_withindex.hdf5',
 'doboost': True,
 'fracflux': True}

doerror = params.get('doerror', True)
scalar_shear_response_correction = params.get('scalar_shear_response_correction', False)

class BaseRunner:

    def __init__(self, table_ln, table_ln_random, Npatches = 100, maxiter = 600, 
                 lbin = [20, 30, 45, 60, 1000],
                 zbin = [0.2, 0.35, 0.5],
                 sys  = None,
                 sysr = None):
        
        self.NPATCHES = Npatches
        self.table_ln = table_ln
        self.table_ln_random = table_ln_random
        self.lbin     = lbin
        self.zbin     = zbin
        self.sys      = sys
        self.sysr     = sysr
        self.maxiter  = maxiter
        
    
    def get_result(self, table_lin, table_r_lin, n_jobs):
        print(len(table_lin ), len(table_r_lin ))
        kwargs = {'return_table': True,
                'scalar_shear_response_correction': scalar_shear_response_correction,
                'matrix_shear_response_correction': True,
                'random_subtraction': True,'table_r': table_r_lin}
        result = excess_surface_density(table_lin, **kwargs)
        print("start jackknife", flush=True)
        kwargs['return_table'] = False

        cov = jackknife_resampling(excess_surface_density, table_lin ,njobs=n_jobs, **kwargs)
        result['ds_err'] = np.sqrt(np.diag(cov))
        result['boost'] = get_boost(table_lin, table_r_lin, rp=result['rp'], method='L-BFGS-B', maxiter = self.maxiter)
        result['boost_err']=np.sqrt(np.diag(jackknife_resampling(
            lambda x, table_r: get_boost(x,table_r,result['rp'],method='L-BFGS-B'), table_lin, table_r_lin)))
        result['ds_b'] = result['ds'] * result['boost']
        result['ds_err_b'] = result['ds_err'] * result['boost']
        return result

    def abundancematchedlbin(self, lbin, zbin, table_l):
        lcut = []
        for l in lbin:
            lcuin=[]
            for zcut in tqdm.tqdm(zip(zbin[:-1], zbin[1:])):
                m=(table_l['z']>zcut[0])&(table_l['z']<zcut[1])
                Ncl = len(table_l[(table_l['lambda']>l)&m])
                print(len(table_l[m]['lambda_cyln']), Ncl)

                if Ncl<len(table_l[m]['lambda_cyln']):
                    Ncl+=1
                lcuin.append(np.sort(table_l[m]['lambda_cyln'])[-Ncl])
                print(len(table_l[m][table_l[m]['lambda_cyln']>lcuin[-1]]), Ncl)
            lcut.append(lcuin)
        lcut.append([10000]*(len(zbin)-1))
        return np.array(lcut).T


    def cut_table(self, table, sys):
        raise NotImplementedError("Implement a cut_table method. You're using a BaseClass right now")
    

    def run(self):

        centers = compute_jackknife_fields(table_ln, self.NPATCHES, weights=np.sum(table_ln['sum 1'], axis=1), distance_threshold=2)
        compute_jackknife_fields(table_ln_random, centers, weights=np.sum(table_ln['sum 1'], axis=1), distance_threshold=2)

        results_lambda = {}
        with tqdm.tqdm(total = (len(self.zbin) - 1) * (len(self.lbin) - 1)) as p:
            for zind in range(len(self.zbin)-1):
                for lind in range(len(self.lbin)-1):
                    table_lin   = table_ln[(table_ln['z'] > self.zbin[zind]) & (table_ln['z'] < self.zbin[zind+1]) & 
                                           (table_ln['lambda'] > self.lbin[lind]) & (table_ln['lambda'] < self.lbin[lind+1])]
                    table_r_lin = table_ln_random[(table_ln_random['z'] > self.zbin[zind]) & (table_ln_random['z'] < self.zbin[zind+1]) &
                                                  (table_ln_random['lambda'] > self.lbin[lind]) & (table_ln_random['lambda'] < self.lbin[lind+1])]
                    table_lin   = self.cut_table(table_lin, self.sys)
                    table_r_lin = self.cut_table(table_r_lin, self.sysr)
                    result      = self.get_result(table_lin, table_r_lin, n_jobs = 1)      
                    results_lambda["{0}_{1}".format(lind, zind)] = result
                
                    p.update(1)
        

        lcutlist = self.abundancematchedlbin(self.lbin[:-1], self.zbin, table_ln)

        results_lambda_cyln = {}
        with tqdm.tqdm(total = (len(self.zbin) - 1) * (len(self.lbin) - 1)) as p:
            for zind in range(len(self.zbin)-1):
                lcutin = lcutlist[zind]
                for lind in range(len(self.lbin)-1):
                    table_lin   = table_ln[(table_ln['z'] > self.zbin[zind]) & 
                                           (table_ln['z'] < self.zbin[zind+1]) & 
                                           (table_ln['lambda_cyln'] > lcutin[lind]) & 
                                           (table_ln['lambda_cyln'] < lcutin[lind+1])]
                    table_r_lin = table_ln_random[(table_ln_random['z'] > self.zbin[zind]) & 
                                                  (table_ln_random['z'] < self.zbin[zind+1]) &
                                                  (table_ln_random['lambda'] > self.lbin[lind]) & 
                                                  (table_ln_random['lambda'] < self.lbin[lind+1])]
                    table_lin   = self.cut_table(table_lin, self.sys)
                    table_r_lin = self.cut_table(table_r_lin, self.sysr)
                    result      = self.get_result(table_lin, table_r_lin, n_jobs = 1)
                    results_lambda_cyln["{0}_{1}".format(lind, zind)]=result
                    
                    p.update(1)


        return results_lambda, results_lambda_cyln
    

class BaseSystematicHigh(BaseRunner):

    key = 'maglim_i'
    def cut_table(self, table, sys):
        flag = sys[self.key] > np.median(sys[self.key])
        return table[flag[table['index']]]
    

class BaseSystematicLow(BaseRunner):

    key = 'maglim_i'
    def cut_table(self, table, sys):
        flag = sys[self.key] < np.median(sys[self.key])
        return table[flag[table['index']]]
    

class Fiducial(BaseRunner):
    def cut_table(self, table, sys):
        return table


class MaglimLow(BaseSystematicLow):
    key = 'maglim_i'

class MaglimHigh(BaseSystematicHigh):
    key = 'maglim_i'

class PSFLow(BaseSystematicLow):
    key = 'fwhm_i'

class PSFHigh(BaseSystematicHigh):
    key = 'fwhm_i'

class AirmassLow(BaseSystematicLow):
    key = 'airmass_i'

class AirmassHigh(BaseSystematicHigh):
    key = 'airmass_i'

class SkyBrightLow(BaseSystematicLow):
    key = 'skybrite_i'

class SkyBrightHigh(BaseSystematicHigh):
    key = 'skybrite_i'

class ExptimeLow(BaseSystematicLow):
    key = 'exptime_i'

class ExptimeHigh(BaseSystematicHigh):
    key = 'exptime_i'

class StellarDensityLow(BaseSystematicLow):
    key = 'stellar_density'

class StellarDensityHigh(BaseSystematicHigh):
    key = 'stellar_density'

class ExtinctionLow(BaseSystematicLow):
    key = 'extinction'

class ExtinctionHigh(BaseSystematicHigh):
    key = 'extinction'


if __name__ == "__main__":

    BASE            = "/scratch/midway3/dhayaa/TMP/V3_"
    if BASE[-1] == '/': raise ValueError("BASE CANNOT END WITH FORWARD SLASH")

    maxiter         = 1200
    NPATCHES        = 100
    MODELS          = [Fiducial, MaglimLow, MaglimHigh, PSFLow, PSFHigh, AirmassLow, AirmassHigh, 
                       SkyBrightLow, SkyBrightHigh, ExptimeLow, ExptimeHigh,
                       StellarDensityLow, StellarDensityHigh, ExtinctionLow, ExtinctionHigh,
                       ]
    # MODELS          = [Fiducial]
    table_ln        = Table.read("/project/kicp/chto/decade/decade_new_v1/dsigma/redmapper_decade_cylinder_with_index.hdf5", path='catalog')
    table_ln_random = Table.read("/project/kicp/chto/decade/decade_new_v1/dsigma/redmapper_decade_full_randoms_with_index.hdf5", path='catalog')
    sys  = fitsio.read('/project/kadrlica/dhayaa/Redmapper/DECADEEli_20260314/Files/my_decade_run_redmapper_v0.8.7_lgt20_vl02_catalog_sysmaps.fit')
    sysr = fitsio.read('/project/kadrlica/dhayaa/Redmapper/DECADEEli_20260314/Files/my_decade_run_redmapper_v0.8.7_weighted_randoms_z010-095_lgt020_vl02_sysmaps.fit')
    
    for runner in MODELS:
        print(runner.__name__)

        if os.path.isfile(f'{BASE}DECADE_{runner.__name__}_lambda_cyln.npy'): 
            print("FOUND FILE. SKIPPING....")
            continue
        
        RUNNER     = runner(table_ln, table_ln_random, Npatches = NPATCHES, sys = sys, sysr = sysr, maxiter = maxiter)
        RES1, RES2 = RUNNER.run()

        np.save(f'{BASE}DECADE_{runner.__name__}_lambda.npy',       RES1, allow_pickle = True)
        np.save(f'{BASE}DECADE_{runner.__name__}_lambda_cyln.npy',  RES2, allow_pickle = True)


    del table_ln, table_ln_random, sys, sysr

    table_ln        = Table.read("/project/kicp/chto/decade/decade_new_v1/dsigma/redmapper_des_cylinder_with_index.hdf5", path='catalog')
    table_ln_random = Table.read("/project/kicp/chto/decade/decade_new_v1/dsigma/redmapper_des_full_randoms_with_index.hdf5", path='catalog')
    sys  = fitsio.read('/project/kadrlica/dhayaa/Redmapper/DESEli_20260314/Files/my_decade_run_redmapper_v0.8.7_lgt20_vl02_catalog_sysmaps.fit')
    sysr = fitsio.read('/project/kadrlica/dhayaa/Redmapper/DESEli_20260314/Files/my_decade_run_redmapper_v0.8.7_weighted_randoms_z010-095_lgt020_vl02_sysmaps.fit')

    for runner in MODELS:
        RUNNER = runner(table_ln, table_ln_random, Npatches = NPATCHES, sys = sys, sysr = sysr, maxiter = maxiter,
                        zbin = [0.2, 0.35, 0.5, 0.65])
        RES1, RES2 = RUNNER.run()

        if os.path.isfile(f'{BASE}DES_{runner.__name__}_lambda_cyln.npy'): continue

        np.save(f'{BASE}DES_{runner.__name__}_lambda.npy',       RES1, allow_pickle = True)
        np.save(f'{BASE}DES_{runner.__name__}_lambda_cyln.npy',  RES2, allow_pickle = True)



import numpy as np
import fitsio
import healsparse as hsp, healpy as hp


#You may want to use Lambda > 5 (instead of default Lambda > 20), for completeness considerations.
#You will almost NEVER want to use volume limited catalog. Always use the flux-limited one.
def select_des_redmapper(lambda_cut_5 = False, volume_limited = False, cosmo_redshifts = False):
    
    PATH = '/project/kadrlica/dhayaa/Redmapper/DESEli_20260314//Files/my_decade_run_redmapper_v0.8.7_lgt20_vl02_catalog.fit'
    if lambda_cut_5:   PATH = PATH.replace('lgt20', 'lgt5')
    if volume_limited: PATH = PATH.replace('vl02', 'vl05')
    
    MASK = hsp.HealSparseMap.read('/project/kadrlica/dhayaa/Masks_for_Manu/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz')
    CAT  = fitsio.read(PATH)
    MSK  = MASK.get_values_pos(CAT['ra'], CAT['dec']) > 0
    CAT  = CAT[MSK]

    if cosmo_redshifts:
        MSK = (CAT['z_lambda'] > 0.2) & (CAT['z_lambda'] < 0.65)
        CAT = CAT[MSK]

    return CAT


def select_des_redmagic():

    PATH = '/project/kadrlica/dhayaa/Redmapper/DESEli_20260314//Files/redmagic/my_decade_run_redmapper_v0.8.7_redmagic_highdens.fit'
    MASK = hsp.HealSparseMap.read('/project/kadrlica/dhayaa/Masks_for_Manu/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz')
    CAT  = fitsio.read(PATH)
    MSK  = MASK.get_values_pos(CAT['ra'], CAT['dec']) > 0 #Mask is fracdet mask so > 0 means more than 0% of pixel was observed in DES
    CAT  = CAT[MSK]

    return CAT


def select_decade_redmapper(lambda_cut_5 = False, volume_limited = False, cosmo_redshifts = False):
    
    PATH = '/project/kadrlica/dhayaa/Redmapper/DECADEEli_20260314//Files/my_decade_run_redmapper_v0.8.7_lgt20_vl02_catalog.fit'
    if lambda_cut_5:   PATH = PATH.replace('lgt20', 'lgt5')
    if volume_limited: PATH = PATH.replace('vl02', 'vl05')
    
    MASK = hp.read_map('/project/kadrlica/dhayaa/Masks_for_Manu/GOLD_Ext0.2_Star5_MCs2_DESY6.fits')
    CAT  = fitsio.read(PATH)
    MSK  = MASK[hp.ang2pix(4096, CAT['ra'], CAT['dec'], lonlat = True)] == 0 #Proper foreground mask, so 0 means pixel is uncontaminated
    CAT  = CAT[MSK]

    if cosmo_redshifts:
        MSK = (CAT['z_lambda'] > 0.2) & (CAT['z_lambda'] < 0.65)
        CAT = CAT[MSK]

    return CAT


def select_decade_redmagic():

    PATH = '/project/kadrlica/dhayaa/Redmapper/DECADEEli_20260314//Files/redmagic/my_decade_run_redmapper_v0.8.7_redmagic_highdens.fit'
    MASK = hp.read_map('/project/kadrlica/dhayaa/Masks_for_Manu/GOLD_Ext0.2_Star5_MCs2_DESY6.fits')
    CAT  = fitsio.read(PATH)
    MSK  = MASK[hp.ang2pix(4096, CAT['ra'], CAT['dec'], lonlat = True)] == 0
    CAT  = CAT[MSK]

    return CAT





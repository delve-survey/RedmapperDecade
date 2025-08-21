import healsparse as hsp, numpy as np

intersection = None

for b in 'griz':
    DR31 = hsp.HealSparseMap.read(f"/project/chihway/secco/decasu_outputs/maglim/delve_dr311+dr312_{b}_maglim_Nov28th.hsp")
    DR32 = hsp.HealSparseMap.read(f"/project/jfrieman/chinyi/dr3_2_decasu_maps/maglim/delve_dr32_{b}_maglim_wmean.hsp")
    
    Y6   = hsp.HealSparseMap.read(f"/project/kadrlica/dhayaa/DES/sys_maps/y6a2_decasu_{b}_maglim_wmean.hs")
    Y6mk = hsp.HealSparseMap.read('/scratch/midway3/dhayaa/maglim_joint_lss-shear_mask_nside16384_NEST_v4.hsp.gz')
    
    
    assert Y6.nside_sparse == Y6mk.nside_sparse and Y6.nside_coverage == Y6mk.nside_coverage, "Reproject/ud_grade the mask to match Y6 before proceeding."

    mpix = Y6mk.valid_pixels
    mval = Y6mk.get_values_pix(mpix) > 0
    keep_pix_mask = mpix[mval]
    
    ypix      = Y6.valid_pixels
    good_pix  = np.intersect1d(ypix, keep_pix_mask)
    Y6_masked = hsp.HealSparseMap.make_empty(nside_coverage=Y6.nside_coverage,nside_sparse=Y6.nside_sparse,dtype=Y6.dtype,)
    Y6_masked.update_values_pix(good_pix, Y6.get_values_pix(good_pix))
    Y6 = Y6_masked * 1

    assert DR32.nside_sparse == Y6mk.nside_sparse and DR32.nside_coverage == Y6mk.nside_coverage, "Reproject/ud_grade the mask to match Y6 before proceeding."

    mpix = Y6mk.valid_pixels
    mval = Y6mk.get_values_pix(mpix) > 0
    keep_pix_mask = mpix[mval]
    
    ypix      = DR32.valid_pixels
    good_pix  = ypix[np.isin(ypix, keep_pix_mask, invert = True)]
    Y6_masked = hsp.HealSparseMap.make_empty(nside_coverage=DR32.nside_coverage,nside_sparse=DR32.nside_sparse,dtype=DR32.dtype,)
    Y6_masked.update_values_pix(good_pix, DR32.get_values_pix(good_pix))
    DR32 = Y6_masked * 1

    del Y6_masked, ypix, good_pix, mpix, mval, keep_pix_mask, Y6mk
    
    # DR3  = hsp.sum_union([DR31, DR32])
    # DR3.write(f"/project/chto/dhayaa/Redmapper/DR3_maglim_{b}.hsp", clobber = True, nocompress = True)

    DR3  = hsp.sum_union([DR31, DR32, Y6]); del DR31, DR32, Y6
    # DR3  = DR3.degrade(nside_out = 8192, reduction = 'mean', weights = None)
    DR3.write(f"/project/chto/dhayaa/Redmapper/All_maglim_{b}.hsp", clobber = True, nocompress = True)

    print("Created DR3 in band", b, flush = True)
    if intersection is None:
        intersection = DR3
    else:
        intersection = hsp.sum_intersection([intersection, DR3])

    print("Intersected DR3 with band", b, flush = True)
    

# intersection.write(f"/project/chto/dhayaa/Redmapper/DR3_maglim_fracdet.hsp", clobber = True, nocompress = True)
intersection.write(f"/project/chto/dhayaa/Redmapper/All_maglim_fracdet.hsp", clobber = True, nocompress = True)


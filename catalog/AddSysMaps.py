import numpy as np, pandas as pd
import healsparse as hsp, healpy as hp
import fitsio

# base   = '/project/kadrlica/dhayaa/Redmapper/DECADE_20260210'
bases   = ['/project/kadrlica/dhayaa/Redmapper/DECADEEli_20260314',
           '/project/kadrlica/dhayaa/Redmapper/DESEli_20260314']


for base in bases:
    for append in ['my_decade_run_redmapper_v0.8.7_lgt20_vl02_catalog.fit',
                   'my_decade_run_redmapper_v0.8.7_weighted_randoms_z010-095_lgt020_vl02.fit']:
        
        p      = base + '/Files/' + append
        CAT    = fitsio.read(p)
        ra,dec = CAT['ra'], CAT['dec']

        VALUES = pd.DataFrame()
        if 'mem_match_id' in CAT.dtype.names: VALUES['mem_match_id'] = CAT['mem_match_id']
        VALUES['ra']           = CAT['ra']
        VALUES['dec']          = CAT['dec']

        for b in 'griz':
            for sys in ['airmass', 'fwhm', 'exptime', 'skybrite', 'maglim']:

                if sys == 'exptime':
                    text = 'sum'
                else:
                    text = 'wmean'

                string   = f'{sys}_{b}'

                if 'DECADE' in base:

                    MAP  = hsp.HealSparseMap.read(f"/project/jfrieman/chinyi/dr3_2_decasu_maps/{sys}/delve_dr32_{b}_{sys}_{text}.hsp")
                    OUT1 = MAP.get_values_pos(ra, dec, lonlat=True)
                    
                    MAP  = hsp.HealSparseMap.read(f"/project/chihway/secco/decasu_outputs/{sys}/delve_dr311+dr312_{b}_{sys}_Nov28th.hsp")
                    OUT2 = MAP.get_values_pos(ra, dec, lonlat=True)

                    OUTF = np.where(OUT1 == hp.UNSEEN, OUT2, OUT1)
                    VALUES[string] = OUTF
                
                elif 'DES' in base:
                    MAP  = hsp.HealSparseMap.read(f"/project/kadrlica/dhayaa/DES/sys_maps/y6a2_decasu_{b}_{sys}_{text}.hs")
                    OUTF = MAP.get_values_pos(ra, dec, lonlat=True)
                    VALUES[string] = OUTF
                else:
                    raise ValueError("Huh?")
                

                print(f"DONE WITH {string}")
                

        MAP = hp.read_map('/project/kadrlica/dhayaa/foreground_mask/data/stellar_density/gaia_stellar_density_G21_equ_n128_v0.fits')
        VALUES['stellar_density'] = MAP[hp.ang2pix(128, ra, dec, lonlat = True)]

        MAP = hp.read_map('/project/kadrlica/dhayaa/foreground_mask/data/extinction/ebv_sfd98_fullres_nside_4096_ring_equatorial.fits')
        VALUES['extinction'] = MAP[hp.ang2pix(4096, ra, dec, lonlat = True)]

        # Optional but often necessary: convert object/string columns to fixed-width byte strings
        for col in VALUES.columns:
            if VALUES[col].dtype == "object":
                maxlen = VALUES[col].astype(str).str.len().max()
                VALUES[col] = VALUES[col].astype(f"S{maxlen}")

        arr = VALUES.to_records(index=False)

        fitsio.write(p.replace('.fit', '_sysmaps.fit'), arr, clobber = True)
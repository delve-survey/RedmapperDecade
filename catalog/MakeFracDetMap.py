import healsparse as hsp

intersection = None

for b in 'griz':
    DR31 = hsp.HealSparseMap.read(f"/project/jfrieman/chinyi/dr3_2_decasu_maps/maglim/delve_dr32_{b}_maglim_wmean.hsp")
    DR32 = hsp.HealSparseMap.read(f"/project/chihway/secco/decasu_outputs/maglim/delve_dr311+dr312_{b}_maglim_Nov28th.hsp")

    DR3  = hsp.sum_union([DR31, DR32])
    DR3.write(f"/project/chto/dhayaa/Redmapper/DR3_maglim_{b}.hsp", clobber = True, nocompress = True)

    print("Created DR3 in band", b, flush = True)
    if intersection is None:
        intersection = DR3
    else:
        intersection = hsp.sum_intersection([intersection, DR3])

    print("Intersected DR3 with band", b, flush = True)
    

intersection.write(f"/project/chto/dhayaa/Redmapper/DR3_maglim_fracdet.hsp", clobber = True, nocompress = True)


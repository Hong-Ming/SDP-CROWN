# Introduction of pkl files
The pkl files have columns as follow:

'Idx': idx of images 

'classify_correct': 1 means classified correct, 0 means classified wrong

'pgd_in_bab_success': 1 means attack success, 0 means failure. (This is the pgd attack we used in bab code)

'dpgd_success': 1 means diverse pgd attack success, 0 means failure.

'pgd_success': 1 means pgd attack success, 0 means failure.

'auto_success': 1 means auto_attack success, 0 means failure.

'bab_verified': 1 means bab verification success, 0 means failure.

'pgd_margin': a tensor indicates the margin after pgd attack we used in bab code, None means classified wrong, so we 
do not perform any attack

'mip_status': verified result of mip. 1 means mip-safe, 3 means mip-unknown, 2 means mip-unsafe, -1 means no record for mip verification.


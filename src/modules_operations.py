import numpy as np
from scipy.linalg import solve
from src import constraints_functions as con_fun, useful_functions as utils, functions_eta as fun_eta, \
    field_structures as fields


def my_mod_update(Mod):
    if not 'SKS' in Mod:
        Mod['SKS'] = utils.my_new_SKS(Mod)
    
    if '0' in Mod:
        (x, p) = (Mod['0'], Mod['mom'].flatten())
        Mod['cost'] = Mod['coeff'] * np.dot(p, np.dot(Mod['SKS'], p)) / 2
    
    if 'x,R' in Mod:
        (x, R) = Mod['x,R']
        N = x.shape[0]
        Mod['Amh'] = con_fun.my_Amh(Mod, Mod['h']).flatten()
        Mod['lam'] = solve(Mod['SKS'], Mod['Amh'], sym_pos=True)
        Mod['mom'] = np.tensordot(Mod['lam'].reshape(N, 3),
                                  fun_eta.my_eta().transpose(), axes=1)
        Mod['cost'] = Mod['coeff'] * np.dot(Mod['Amh'], Mod['lam']) / 2
    return


def my_init_from_mod(Mod):
    if '0' in Mod:
        nMod = {'sig': Mod['sig'], 'coeff': Mod['coeff']}
    
    if 'x,R' in Mod:
        nMod = {'sig': Mod['sig'], 'C': Mod['C'], 'coeff': Mod['coeff']}
        if 'nu' in Mod:
            nMod['nu'] = Mod['nu']
    return nMod


def my_mod_init_from_Cot(Mod, nCot):
    if '0' in Mod:
        nMod = my_init_from_mod(Mod)
        nx0 = nCot['0'][0][0]
        nMod['0'] = nx0
        nMod['SKS'] = utils.my_new_SKS(nMod)
        nMod['mom'] = solve(nMod['coeff'] * nMod['SKS'],
                            fields.my_VsToV(fields.my_CotToVs(nCot, Mod['sig']), nx0, 0).flatten(),
                            sym_pos=True).reshape(nx0.shape)
        my_mod_update(nMod)  # compute cost0
    
    # updating h1
    if 'x,R' in Mod:
        nMod = my_init_from_mod(Mod)
        (nx1, nR) = nCot['x,R'][0][0]
        nMod['x,R'] = (nx1, nR)
        nMod['SKS'] = utils.my_new_SKS(nMod)
        dv = fields.my_VsToV(fields.my_CotToVs(nCot, Mod['sig']), nx1, 1)
        S = np.tensordot((dv + np.swapaxes(dv, 1, 2)) / 2, fun_eta.my_eta())
        tlam = solve(nMod['coeff'] * nMod['SKS'], S.flatten(), sym_pos=True)
        (Am, AmKiAm) = con_fun.my_new_AmKiAm(nMod)
        nMod['h'] = solve(AmKiAm, np.dot(tlam, Am), sym_pos=True)
        my_mod_update(nMod)  # will compute the new lam, Amh, mom and cost
    
    return nMod

import Forward.Hamiltonianderivatives as HamDer


def exchange_value_in_cot(Cot):
    nCot = dict()
    nCot['0'] = []
    nCot['x,R'] = []
    if '0' in Cot:
        N0 = len(Cot['0'])
        for i in range(N0):
            (x0, p0) = Cot['0'][i]
            nCot['0'].append((p0, x0))
    
    if 'x,R' in Cot:
        N0 = len(Cot['x,R'])
        for i in range(N0):
            ((x0, R0), (p0, PR0)) = Cot['x,R'][i]
            nCot['x,R'].append(((p0, PR0), (x0, R0)))
    
    return nCot


def exchange_value_in_cot_mult(Cot):
    nCot = dict()
    nCot['0'] = []
    nCot['x,R'] = []
    if '0' in Cot:
        N0 = len(Cot['0'])
        for i in range(N0):
            (x0, p0) = Cot['0'][i]
            nCot['0'].append((-p0, x0))
    
    if 'x,R' in Cot:
        N0 = len(Cot['x,R'])
        for i in range(N0):
            ((x0, R0), (p0, PR0)) = Cot['x,R'][i]
            nCot['x,R'].append(((-p0, -PR0), (x0, R0)))
    
    return nCot


def backward_step(Mod, eps, grad):  # tested
    grad_exch = grad.copy()
    grad_exch.Cot = exchange_value_in_cot_mult(grad.Cot)
    grad_exch.updatefromCot()
    grad_exch.mult_Cot_scal(eps)
    GD_0 = Mod.GD.copy_full()  # n
    GD_1 = Mod.GD.copy_full()  # b
    
    GD_0.add_cot(grad_exch.Cot)
    grad_exch.mult_Cot_scal(-1.)
    GD_1.add_cot(grad_exch.Cot)
    
    Mod_tmp = Mod.copy()
    Mod_tmp.fill_GD(GD_0)
    Mod_tmp.update()
    Mod_tmp.GeodesicControls_curr(Mod_tmp.GD)
    dxH_0 = HamDer.dxH(Mod_tmp)
    dpH_0 = HamDer.dpH(Mod_tmp)
    # in dxH the momentum value is -\partial_x
    dxH_0.Cot = exchange_value_in_cot_mult(dxH_0.Cot)
    dxH_0.updatefromCot()
    # in dpH the GD value is \partial_p
    dpH_0.Cot = exchange_value_in_cot(dpH_0.Cot)
    dpH_0.updatefromCot()
    
    Mod_tmp.fill_GD(GD_1)
    Mod_tmp.update()
    Mod_tmp.GeodesicControls_curr(Mod_tmp.GD)
    dxH_1 = HamDer.dxH(Mod_tmp)
    dpH_1 = HamDer.dpH(Mod_tmp)
    # in dxH the momentum value is -\partial_x
    dxH_1.Cot = exchange_value_in_cot_mult(dxH_1.Cot)
    dxH_1.updatefromCot()
    # in dpH the GD value is \partial_p
    dpH_1.Cot = exchange_value_in_cot(dpH_1.Cot)
    dpH_1.updatefromCot()
    
    dxH_1.mult_Cot_scal(-1.)
    dpH_1.mult_Cot_scal(-1.)
    
    dxH_0.add_cot(dxH_1.Cot)
    dpH_0.add_cot(dpH_1.Cot)
    
    eps1 = 2 * eps
    dxH_0.mult_Cot_scal(1. / eps1)
    dpH_0.mult_Cot_scal(1. / eps1)
    
    out = dxH_0.copy_full()
    out.add_cot(dpH_0.Cot)
    
    return out


def backward_shoot_rk2(Modlist, grad_1, eps):
    """
    Modlist is the "trajectory" of the module
    grad_1 is a GD corresponding to the derivative of an attachment term
    eps in the step for finite differences
    """
    # Modlist.reverse()
    N_tot = len(Modlist)
    N = int((N_tot - 1) / 2)
    h = 1. / N
    count = N_tot - 1
    cgrad = grad_1.copy_full()
    
    for i in range(N):
        count -= 1
        Mod = Modlist[count].copy_full()
        
        dGD_np = backward_step(Mod, eps, cgrad)
        dGD_np.mult_Cot_scal(h)
        
        count -= 1
        Mod = Modlist[count].copy_full()
        
        dGD = backward_step(Mod, eps, dGD_np)
        dGD.mult_Cot_scal(h / 2)
        
        cgrad.add_cot(dGD.Cot)
        cgrad.add_cot(dGD_np.Cot)
    
    #Modlist.reverse()
    return cgrad

import src.Forward.Hamiltonianderivatives as HamDer


def backward_step(Mod, eps, grad):  # tested
    
    GD_0 = Mod.GD.copy_full()  # n
    GD_1 = Mod.GD.copy_full()  # b
    
    grad_c = grad.copy_full()
    grad_c.mult_tan_scal(eps)
    grad_c.mult_cotan_scal(eps)
    
    # GD_0_p = Mod.GD.p  + eps grad.x
    GD_0.add_tantocotan(grad_c)
    
    # GD_1_x = Mod.GD.x  + eps grad.p
    GD_1.add_cotantoGD(grad_c)
    
    # GD_0_x = Mod.GD.x  - eps grad.p
    grad_c.mult_cotan_scal(-1.)
    GD_0.add_cotantoGD(grad_c)
    
    # GD_1_p = Mod.GD.p  - eps grad.x
    grad_c.mult_tan_scal(-1.)
    GD_1.add_tantocotan(grad_c)
    
    # Mod with GD_0 and corresponding geodesic controles
    Mod_tmp = Mod.copy()
    Mod_tmp.fill_GD(GD_0)
    Mod_tmp.update()
    Mod_tmp.GeodesicControls_curr(Mod_tmp.GD)
    
    # Hamiltonian derivatives for GD_0
    dxH_0 = HamDer.dxH(Mod_tmp)  # put in cotan, it is -\partial_x
    dxH_0.mult_cotan_scal(-1.)  # so that in cotan there is \partial_x
    dxH_0.exchange_tan_cotan()  # now \partial_x H in tan
    dpH_0 = HamDer.dpH(Mod_tmp)  # put in tan
    dpH_0.exchange_tan_cotan()  # now \partial_p H in cotan
    
    # Mod with GD_1 and corresponding geodesic controles
    Mod_tmp = Mod.copy()
    Mod_tmp.fill_GD(GD_1)
    Mod_tmp.update()
    Mod_tmp.GeodesicControls_curr(Mod_tmp.GD)
    
    # Hamiltonian derivatives for GD_0
    dxH_1 = HamDer.dxH(Mod_tmp)  # put in cotan, it is -\partial_x
    dxH_1.exchange_tan_cotan()  # now -\partial_x H in tan
    dpH_1 = HamDer.dpH(Mod_tmp)  # put in tan
    dpH_1.exchange_tan_cotan()  # now \partial_p H in cotan
    dpH_1.mult_cotan_scal(-1.)  # -now \partial_p H in cotan
    
    dxH_0.add_tan(dxH_1)  # in tan : (\partial_x H (GD1) - \partial_x H (GD0))
    dpH_0.add_cotan(dpH_1)  # in cotan : (\partial_p H (GD1) - \partial_p H (GD0))
    
    eps1 = 2 * eps
    dxH_0.mult_tan_scal(1. / eps1)
    dpH_0.mult_cotan_scal(1. / eps1)
    
    out = dxH_0.copy_full()
    out.add_cotan(dpH_0)
    
    # in tan : d (\partial_x H (GD) ). (-grad.p, grad.x)
    # in cotan : d (\partial_p H (GD) ). (-grad.p, grad.x)
    
    return out


def backward_shoot_rk2(Modlist, grad_1, eps):
    """
    Modlist is the "trajectory" of the module
    grad_1 is a GD corresponding to the gradient of an attachment term
    It is seen as a tangent element of the cotangent, 
    the tangent of GD is in tan and the tangent of cotan is in cotan (vector space)
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
        
        dnu_np = backward_step(Mod, eps, cgrad)
        dnu_np.mult_tan_scal(h)
        dnu_np.mult_cotan_scal(h)
        
        count -= 1
        Mod = Modlist[count].copy_full()
        
        dnu = backward_step(Mod, eps, dnu_np)
        dnu.mult_tan_scal(h / 2)
        dnu.mult_cotan_scal(h / 2)
        
        cgrad.add_tan(dnu)
        cgrad.add_tan(dnu_np)
        cgrad.add_cotan(dnu)
        cgrad.add_cotan(dnu_np)
    
    # Modlist.reverse()
    return cgrad

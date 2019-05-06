import numpy as np
from scipy.linalg import solve

import implicitmodules.numpy.DeformationModules.Abstract as ab
import implicitmodules.numpy.GeometricalDescriptors.x_R as GeoDescr
import implicitmodules.numpy.StructuredFields.StructuredField_p as stru_fiep
from implicitmodules.numpy.Kernels import ScalarGaussian as ker
from implicitmodules.numpy.Utilities import FunctionsEta as fun_eta


class ElasticOrder1(ab.DeformationModule):
    """
     Elastic module of order 0
    """

    def __init__(self, sigma, N_pts, dim, coeff, C, nu):
        """
        sigma is the scale of the rkhs of generated vector fields
        N_pts is the number of landmarks
        dim is the dimension of the ambient space
        """
        self.sig = sigma
        self.N_pts = N_pts
        self.dim = dim
        self.coeff = coeff
        self.C = C.copy()
        self.nu = nu
        self.dimR = 3
        self.dimcont = C.shape[2]
        self.GD = GeoDescr.GD_xR(N_pts, dim)
        self.SKS = np.zeros([self.N_pts * self.dimR, self.N_pts * self.dimR])
        self.Mom = np.zeros([self.N_pts, self.dim, self.dim])
        self.lam = np.zeros([self.N_pts * self.dimR])
        self.Amh = []
        self.Cont = np.zeros([self.dimcont])
        self.cost = 0.

    def copy(self):
        return ElasticOrder1(self.sig, self.N_pts, self.dim, self.coeff, self.C, self.nu)

    def copy_full(self):
        Mod = ElasticOrder1(self.sig, self.N_pts, self.dim, self.coeff, self.C, self.nu)
        Mod.GD = self.GD.copy_full()
        Mod.SKS = self.SKS.copy()
        Mod.Mom = self.Mom.copy()
        Mod.Cont = self.Cont.copy()
        Mod.cost = self.cost
        Mod.Amh = self.Amh
        Mod.lam = self.lam
        return Mod

    def fill_GD(self, GD):
        self.GD = GD.copy_full()
        self.SKS = np.zeros([self.N_pts * self.dimR, self.N_pts * self.dimR])

    def fill_Cont(self, Cont):
        self.Cont = Cont.copy()
        self.compute_mom_from_cont_curr()

    def Compute_SKS_curr(self):
        # We supposes that values of GD have been filled
        x = self.GD.get_points()
        self.SKS = ker.my_K(x, x, self.sig, 1)
        self.SKS += self.nu * np.eye(self.N_pts * self.dimR)

    def update(self):
        """
        Computes SKS so that it is done only once.
        Supposes that values of GD have been filled
        """

        self.Compute_SKS_curr()

    def Amh_curr(self, h):
        R = self.GD.get_R()
        eta = fun_eta.my_eta()
        return np.einsum('nli, nik, k, nui, niv, lvt->nt', R, self.C, h, np.tile(np.eye(self.dim), [self.N_pts, 1, 1]), np.swapaxes(R, 1, 2), eta)

    def AmKiAm_curr(self):
        lam = np.zeros((self.dimcont, self.dimR * self.N_pts))
        Am = np.zeros((self.dimR * self.N_pts, self.dimcont))

        for i in range(self.dimcont):
            h = np.zeros((self.dimcont))
            h[i] = 1.
            Am[:, i] = self.Amh_curr(h).flatten()
            lam[i, :] = solve(self.SKS, Am[:, i], sym_pos=True)
        return (Am, np.dot(lam, Am))

    def GeodesicControls_curr(self, GDCot):
        """
        Supposes that SKS has been computed and values of GD filled
        Supposes that GDCot has Cot filled
        """
        x = self.GD.get_points()

        vs = GDCot.Cot_to_Vs(self.sig)
        dvsx = vs.Apply(x, 1)
        dvsx_sym = (dvsx + np.swapaxes(dvsx, 1, 2)) / 2
        S = np.tensordot(dvsx_sym, fun_eta.my_eta())

        tlam = solve(self.coeff * self.SKS, S.flatten(), sym_pos=True)
        (Am, AmKiAm) = self.AmKiAm_curr()

        Am_s_tlam = np.dot(tlam, Am)

        self.Cont = solve(AmKiAm, Am_s_tlam, sym_pos=True)
        self.compute_mom_from_cont_curr()

    def compute_mom_from_cont_curr(self):
        self.Amh = self.Amh_curr(self.Cont).flatten()
        self.lam = solve(self.SKS, self.Amh, sym_pos=True)
        self.Mom = np.tensordot(self.lam.reshape(self.N_pts, 3),
                                fun_eta.my_eta().transpose(), axes=1)

    def field_generator_curr(self):
        v = stru_fiep.StructuredField_p(self.sig, self.N_pts, self.dim)
        param = (self.GD.get_points(), self.Mom)
        v.fill_fieldparam(param)
        return v

    def Cost_curr(self):
        self.cost = self.coeff * np.dot(self.Amh, self.lam) / 2

    def DerCost_curr(self):
        out = self.GD.copy_full()
        out.fill_zero_tan()
        x = self.GD.get_points()
        R = self.GD.get_R()
        v = self.field_generator_curr()

        # der = self.my_pSmV(v.dic,v.dic,1)
        der = self.p_Ximv_curr(v, 1)
        cotx = -self.coeff * der.cotan[0]

        tP = self.coeff * self.Mom

        cotR = 2. * np.einsum('kul, kli, kij, kj, kiv->kuv', tP, R, self.C, np.tile(self.Cont, [x.shape[0], 1]), np.tile(np.eye(2), [x.shape[0], 1, 1]))

        out.cotan = (cotx, cotR)

        return out

    def cot_to_innerprod_curr(self, GDCot, j):  #
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with cotan filled (tan=0)
        """
        
        
        if j==0:
            vsr = GDCot.Cot_to_Vs(self.sig)
            out = self.p_Ximv_curr(vsr, j)
        if j == 1:
            
            vsr = GDCot.Cot_to_Vs(self.sig)
    
            # der wrt x as support of generated vector field by self
            der0 = self.p_Ximv_curr(vsr, j)
    
            x = self.GD.get_points()
            R = self.GD.get_R()
    
            dvx = vsr.Apply(x, 1)
            dvx_sym = (dvx + np.swapaxes(dvx, 1, 2)) / 2
            S = np.tensordot(dvx_sym, fun_eta.my_eta())
    
            tlam = solve(self.SKS, S.flatten(), sym_pos=True)
            tP = np.tensordot(tlam.reshape(S.shape), fun_eta.my_eta().transpose(), axes=1)
    
            tVs = stru_fiep.StructuredField_p(self.sig, self.N_pts, self.dim)
            tVs.fill_fieldparam((x.copy(), tP.copy()))
    
            der1 = self.p_Ximv_curr(tVs, 1)
            der1.mult_cotan_scal(-1)
    
            # artifial module in order to use p_Ximv_curr : needs to be changed 
            Mod_t = self.copy()
            Mod_t.GD = self.GD.copy_full()
            Mod_t.Mom = tP.copy()
            der2 = Mod_t.p_Ximv_curr(self.field_generator_curr(), 1)
            der2.mult_cotan_scal(-1)
    
            der0.add_cotan(der1)
            der0.add_cotan(der2)
    
            cotR = 2. * np.einsum('kul, kli, kij, kj, kiv->kuv', tP, R, self.C, np.tile(self.Cont, [x.shape[0], 1]), np.tile(np.eye(2), [x.shape[0], 1, 1]))
    
            out = self.GD.copy_full()
            out.fill_zero_tan()
            out.fill_zero_cotan()
    
            cotx0, cotR0 = der0.cotan
            out.cotan = (cotx0.copy(), cotR.copy())

        return out

    def p_Ximv_curr(self, vs, j):
        """
        Put in Module because it uses the link between GD and support 
        of vector fields      
        The derivative is done wrt to the support of the generated field
        which are the points of self.GD
        """
        v_curr = self.field_generator_curr()
        x = v_curr.support
        P = v_curr.mom

        if j == 0:
            out = 0.
            dvx = vs.Apply(x, j + 1)
            dvx = (dvx + np.swapaxes(dvx, 1, 2)) / 2
            # TODO: remove loop
            out += np.sum(np.asarray([np.tensordot(P[i], dvx[i])
                                      for i in range(x.shape[0])]))

        elif j == 1:
            out = self.GD.copy_full()
            out.fill_zero_tan()
            out.fill_zero_cotan()
            ddvx = vs.Apply(x, j + 1)
            ddvx = (ddvx + np.swapaxes(ddvx, 1, 2)) / 2
            der = np.einsum('kij, kijl->kl', P, ddvx)

            out.cotan = (der.copy(), np.zeros([self.N_pts, self.dim, self.dim]))

        return out


import numpy as np
from scipy.linalg import solve

import old.StructuredFields.StructuredFields_p
from old import GeometricalDescriptors, functions_eta as fun_eta, kernels as ker, field_structures as fields, \
    pairing_structures as pair
from old.DeformationModules.DeformationModules import DeformationModule


class ElasticOrder1(DeformationModule):
    """
     Elastic module of order 1
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
        self.C = C.copy()
        self.nu = nu
        self.dimR = 3
        self.coeff = coeff
        self.GD = old.GeometricalDescriptors.GD_xR.GD_xR(N_pts, dim, C)
        self.SKS = np.zeros([self.N_pts * self.dimR, self.N_pts * self.dimR])
        self.Mom = np.zeros([self.N_pts, self.dim, self.dim])
        self.lam = np.zeros([self.N_pts * self.dimR])
        self.Amh = []
        self.Cont = np.zeros([1])
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
    
    def add_cot(self, GD):
        self.GD.add_cot(GD.Cot)
    
    def Compute_SKS_curr(self):
        """
        Supposes that values of GD have been filled
        """
        try:
            x = self.GD.get_points()
            self.SKS = ker.my_K(x, x, self.sig, 1)
            self.SKS += self.nu * np.eye(self.N_pts * self.dimR)
        
        except:
            raise NameError('Need to fill landmark points before computing SKS')
    
    def update(self):
        """
        Computes SKS so that it is done only once.
        Supposes that values of GD have been filled
        """
        
        self.Compute_SKS_curr()
    
    def Amh_curr(self, h):
        R = self.GD.get_R()
        eta = fun_eta.my_eta()
        out = np.asarray([np.tensordot(np.dot(R[i],
                                              np.dot(np.diag(np.dot(self.GD.C[i], h)),
                                                     R[i].transpose())), eta, axes=2) for i in range(self.N_pts)])
        return out
    
    def AmKiAm_curr(self):
        dimh = self.GD.C.shape[2]
        
        lam = np.zeros((dimh, 3 * self.N_pts))
        Am = np.zeros((3 * self.N_pts, dimh))
        
        for i in range(dimh):
            h = np.zeros((dimh))
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
        
        v = old.StructuredFields.StructuredFields_p.StructuredField_p(self.sig, self.dim)
        param = (self.GD.get_points(), self.Mom)
        v.fill_fieldparam(param)
        return v
    
    def Cost_curr(self):
        self.cost = self.coeff * np.dot(self.Amh, self.lam) / 2
    
    def DerCost_curr(self):
        out = self.GD.copy()
        x = self.GD.get_points()
        R = self.GD.get_R()
        v = self.field_generator_curr()
        
        der = pair.my_pSmV(v.dic, v.dic, 1)
        dx = -self.coeff * der['p'][0][1]
        
        tP = self.coeff * self.Mom
        dR = 2 * np.asarray([np.dot(np.dot(tP[i], R[i]), np.diag(np.dot(self.GD.C[i], self.Cont)))
                             for i in range(x.shape[0])])
        
        out.Cot['x,R'] = [((dx, dR), (np.zeros(dx.shape), np.zeros(dR.shape)))]
        
        return out
    
    def cot_to_innerprod_curr(self, GDCot, j):
        """
         Transforms the GD (with Cot filled) GDCot into vsr and computes
         the inner product (or derivative wrt self.GD) with the
         field generated by self.
         The derivative is a returned as a GD with Cot filled (mom=0)
        """
        
        vsr = GDCot.Cot_to_Vs(self.sig)
        v = self.field_generator_curr()
        
        if j == 0:
            out = v.p_Ximv(vsr, j)
        if j == 1:
            out = self.GD.copy()
            x = self.GD.get_points()
            R = self.GD.get_R()
            
            Vsr1 = fields.my_CotToVs(GDCot.Cot, self.sig)
            der = pair.my_pSmV(v.dic, Vsr1, 1)
            dx = der['p'][0][1]
            
            dvx = fields.my_VsToV(Vsr1, x, 1)
            dvx_sym = (dvx + np.swapaxes(dvx, 1, 2)) / 2
            S = np.tensordot(dvx_sym, fun_eta.my_eta())
            
            tlam = solve(self.SKS, S.flatten(), sym_pos=True)
            tP = np.tensordot(tlam.reshape(S.shape), fun_eta.my_eta().transpose(), axes=1)
            
            tVs = {'p': [(x, tP)], 'sig': self.sig}
            
            der = pair.my_pSmV(tVs, v.dic, 1)
            dx += - der['p'][0][1]
            der = pair.my_pSmV(v.dic, tVs, 1)
            dx += - der['p'][0][1]
            
            dR = 2 * np.asarray([np.dot(np.dot(tP[i], R[i]), np.diag(np.dot(self.GD.C[i], self.Cont)))
                                 for i in range(x.shape[0])])
            
            out.Cot['x,R'] = [((dx, dR), (np.zeros(dx.shape), np.zeros(dR.shape)))]
        
        return out

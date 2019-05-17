import numpy as np

import implicitmodules.numpy.Manifolds.Abstract as ab
import implicitmodules.numpy.StructuredFields.StructuredField_0 as stru_fie0
import implicitmodules.numpy.StructuredFields.StructuredField_m as stru_fiem
import implicitmodules.numpy.StructuredFields.Sum as stru_fie_sum


class Stiefel(ab.Manifold):
    def __init__(self, N_pts, dim):  # 
        """
        The GD and Mom are arrays of size N_pts x dim.
        """
        self.Cot = {'x,R': []}
        self.N_pts = N_pts
        self.dim = dim
        self.GDshape = [self.N_pts, self.dim]
        self.Rshape = [self.N_pts, self.dim, self.dim]
        self.GD = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape))
        self.tan = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape))
        self.cotan = (np.zeros([self.N_pts, self.dim]), np.zeros(self.Rshape))
        
        self.dimGD = self.N_pts * self.dim + self.N_pts * self.dim * self.dim
        self.dimMom = self.N_pts * self.dim + self.N_pts * self.dim * self.dim
    
    def copy(self):  # 
        return Stiefel(self.N_pts, self.dim)
    
    def copy_full(self):  # 
        GD = Stiefel(self.N_pts, self.dim)
        x, R = self.GD
        dx, dR = self.tan
        cotx, cotR = self.cotan
        GD.GD = (x.copy(), R.copy())
        GD.tan = (dx.copy(), dR.copy())
        GD.cotan = (cotx.copy(), cotR.copy())
        return GD
    
    def fill_zero(self):
        self.GD = (np.zeros(self.GDshape), np.zeros(self.Rshape))
        self.tan = (np.zeros(self.GDshape), np.zeros(self.Rshape))
        self.cotan = (np.zeros(self.GDshape), np.zeros(self.Rshape))
    
    def fill_zero_GD(self):
        self.GD = (np.zeros(self.GDshape), np.zeros(self.Rshape))
    
    def fill_zero_tan(self):
        self.tan = (np.zeros(self.GDshape), np.zeros(self.Rshape))
    
    def fill_zero_cotan(self):
        self.cotan = (np.zeros(self.GDshape), np.zeros(self.Rshape))
    
    def updatefromCot(self):
        pass
    
    def get_points(self):  # 
        return self.GD[0]
    
    def get_R(self):  # 
        return self.GD[1]
    
    def get_mom(self):  # 
        cotx, cotR = self.cotan
        return (cotx.copy(), cotR.copy())
    
    def fill_cot_from_param(self, param):  # 
        self.GD = (param[0][0].copy(), param[0][1].copy())
        self.cotan = (param[1][0].copy(), param[1][1].copy())
    
    def Cot_to_Vs(self, sig):  #
        x = self.GD[0].copy()
        R = self.GD[1].copy()
        px = self.cotan[0].copy()
        pR = self.cotan[1].copy()
        
        v0 = stru_fie0.StructuredField_0(sig, self.N_pts, self.dim)
        v0.fill_fieldparam((x, px))
        
        vm = stru_fiem.StructuredField_m(sig, self.N_pts, self.dim)
        #  P = np.einsum('nik, nkj->nij', pR, R.swapaxes(1, 2))
        P = np.einsum('nik, njk->nij', pR, R)
        vm.fill_fieldparam((x, P))
        
        return stru_fie_sum.Summed_field([v0, vm])

    def action(self, v):  #
        pts = self.get_points()
        R = self.get_R()
        vx = v.Apply(pts, 0)
        dvx = v.Apply(pts, 1)
        S = (dvx - np.swapaxes(dvx, 1, 2)) / 2
        
        vR = np.einsum('nik, nkj->nij', S, R)
        out = self.copy_full()
        out.fill_zero_cotan()
        out.tan = (vx.copy(), vR.copy())
        
        return out
    
    def dCotDotV(self, vs):  # ,
        """
        Supposes that Cot has been filled
        """
        x = self.get_points()
        R = self.get_R()
        px, pR = self.get_mom()
        
        dvx = vs.Apply(x, 1)
        ddvx = vs.Apply(x, 2)
        
        skew_dvx = (dvx - np.swapaxes(dvx, 1, 2)) / 2
        skew_ddvx = (ddvx - np.swapaxes(ddvx, 1, 2)) / 2
        
        dx = np.einsum('nu, nuv->nv', px, dvx) + np.einsum('nvu, niu, nvik->nk', pR, R, skew_ddvx)
        dR = np.einsum('nik, nkj->nij', -skew_dvx, pR)
        
        GD = self.copy()
        GD.fill_zero_tan()
        GD.cotan = (dx.copy(), dR.copy())
        return GD
    
    def inner_prod_v(self, v):  # 
        vGD = self.action(v)
        vx, vR = vGD.tan
        px, pR = self.get_mom()
        out = np.dot(px.flatten(), vx.flatten())
        return out + np.einsum('nij, nij->', pR, vR)
    
    def add_GD(self, GDCot):
        x, R = self.GD
        xGD, RGD = GDCot.GD
        self.GD = (x + xGD, R + RGD)
    
    def add_tan(self, GDCot):
        dx, dR = self.tan
        dxGD, dRGD = GDCot.tan
        self.tan = (dx + dxGD, dR + dRGD)
    
    def add_cotan(self, GDCot):
        cotx, cotR = self.cotan
        cotxGD, cotRGD = GDCot.cotan
        self.cotan = (cotx + cotxGD, cotR + cotRGD)
    
    def mult_GD_scal(self, s):
        x, R = self.GD
        self.GD = (s * x, s * R)
    
    def mult_tan_scal(self, s):
        dx, dR = self.tan
        self.tan = (s * dx, s * dR)
    
    def mult_cotan_scal(self, s):
        cotx, cotR = self.cotan
        self.cotan = (s * cotx, s * cotR)
    
    def add_speedGD(self, GDCot):
        x, R = self.GD
        dx, dR = GDCot.tan
        self.GD = (x + dx, R + dR)
    
    def add_tantocotan(self, GDCot):
        dxGD, dRGD = GDCot.tan
        cotx, cotR = self.cotan
        self.cotan = (cotx + dxGD, cotR + dRGD)
    
    def add_cotantotan(self, GDCot):
        dx, dR = self.tan
        cotxGD, cotRGD = GDCot.cotan
        self.tan = (dx + cotxGD, dR + cotRGD)
    
    def add_cotantoGD(self, GDCot):
        x, R = self.GD
        cotxGD, cotRGD = GDCot.cotan
        self.GD = (x + cotxGD, R + cotRGD)
    
    def exchange_tan_cotan(self):
        (dx, dR) = self.tan
        (cotx, cotR) = self.cotan
        self.tan = (cotx.copy(), cotR.copy())
        self.cotan = (dx.copy(), dR.copy())
    
    def get_GDinVector(self):
        x, R = self.GD
        return np.concatenate([x.flatten(), R.flatten()])
    
    def get_cotaninVector(self):
        cotx, cotR = self.cotan
        return np.concatenate([cotx.flatten(), cotR.flatten()])
    
    def get_taninVector(self):
        tanx, tanR = self.tan
        return np.concatenate([tanx.flatten(), tanR.flatten()])
    
    def fill_from_vec(self, PX, PMom):
        x = PX[:self.N_pts * self.dim]
        x = x.reshape([self.N_pts, self.dim])
        R = PX[self.N_pts * self.dim:]
        R = R.reshape([self.N_pts, self.dim, self.dim])
        
        cotx = PMom[:self.N_pts * self.dim]
        cotx = cotx.reshape([self.N_pts, self.dim])
        cotR = PMom[self.N_pts * self.dim:]
        cotR = cotR.reshape([self.N_pts, self.dim, self.dim])
        
        param = ((x, R), (cotx, cotR))
        self.fill_cot_from_param(param)

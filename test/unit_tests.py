import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 1)

import unittest
import numpy as np


class NumpyUnitTestCase(unittest.TestCase):
    sig = 0.9
    
    def test_old_vs_new(self):
        from src.DeformationModules.Combination import CompoundModules
        from src.DeformationModules.ElasticOrder1 import ElasticOrder1
        from src.DeformationModules.SilentLandmark import SilentLandmark
        import src.Utilities.Rotation as rot
        import src.Forward.Shooting as shoot
        
        # %%
        xmin, xmax, ymin, ymax = -5, 5, -5, 5
        nx, ny = 5, 5
        
        X0 = np.linspace(xmin, xmax, nx)
        Y0 = np.linspace(ymin, ymax, ny)
        Z0 = np.meshgrid(X0, Y0)
        
        Z = np.reshape(np.swapaxes(Z0, 0, 2), [-1, 2])
        Z_c = np.concatenate([np.array([X0, np.zeros([nx]) + ymin]).transpose(),
                              np.array([np.zeros([ny]) + xmax, Y0]).transpose(),
                              np.array([np.flip(X0), np.zeros([nx]) + ymax]).transpose(),
                              np.array([np.zeros([ny]) + xmin, np.flip(Y0)]).transpose()])
        # %%
        x1 = Z.copy()
        xs = Z_c.copy()
        
        # %% parameter for module of order 1
        th = 0.25 * np.pi * np.ones(x1.shape[0])
        R = np.asarray([rot.my_R(cth) for cth in th])
        for i in range(x1.shape[0]):
            R[i] = rot.my_R(th[i])
        
        C = np.zeros((x1.shape[0], 2, 1))
        
        def define_C0(x, y):
            return 1
        
        def define_C1(x, y):
            return np.ones(y.shape)
        
        C[:, 1, 0] = define_C1(x1[:, 0], x1[:, 1]) * define_C0(x1[:, 0], x1[:, 1])
        C[:, 0, 0] = 0. * C[:, 1, 0]
        
        # %%
        coeffs = [5., 0.05]
        sig1 = 5
        nu = 0.001
        dim = 2
        
        Sil = SilentLandmark(xs.shape[0], dim)
        Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
        Mod_el_init = CompoundModules([Sil, Model1])
        
        # %%
        ps = np.zeros(xs.shape)
        ps[nx + ny:2 * nx + ny, 1] = 0.5
        
        (p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
        param_sil = (xs, ps)
        param_1 = ((x1, R), (p1, PR))
        param = [param_sil, param_1]
        
        # %%
        Mod_el_init.GD.fill_cot_from_param(param)
        Mod_el = Mod_el_init.copy_full()
        
        [xx, xy] = np.meshgrid(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))
        grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()
        
        Sil_grid = SilentLandmark(grid_points.shape[0], dim)
        
        param_grid = (grid_points, np.zeros(grid_points.shape))
        Sil_grid.GD.fill_cot_from_param(param_grid)
        
        Mod_tot = CompoundModules([Sil_grid, Mod_el])
        Modlist_opti_tot = shoot.shooting_traj(Mod_tot, 5)
        
        # plot mom at t = i
        i = 5
        xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
        xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
        ps_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].cotan
        x1_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[1].GD[0]
        
        xgrid_tem = np.array([[-1.00144815e+01, -1.00144815e+01],
                              [-7.76795743e+00, -1.02637595e+01],
                              [-5.39236898e+00, -1.06552803e+01],
                              [-2.94228248e+00, -1.11249775e+01],
                              [-4.71610937e-01, -1.16014957e+01],
                              [1.99169272e+00, -1.20600950e+01],
                              [4.44767582e+00, -1.25126637e+01],
                              [6.85328452e+00, -1.29117670e+01],
                              [9.06865405e+00, -1.31097740e+01],
                              [1.09908265e+01, -1.30093956e+01],
                              [1.26456961e+01, -1.26456961e+01],
                              [-1.02637595e+01, -7.76795743e+00],
                              [-8.00683215e+00, -8.00683215e+00],
                              [-5.56408708e+00, -8.44831736e+00],
                              [-3.01847906e+00, -9.00942606e+00],
                              [-4.51343094e-01, -9.58334765e+00],
                              [2.09769205e+00, -1.01280660e+01],
                              [4.64720857e+00, -1.06791580e+01],
                              [7.17595476e+00, -1.12100847e+01],
                              [9.48694365e+00, -1.15086543e+01],
                              [1.14159142e+01, -1.14159142e+01],
                              [1.30093956e+01, -1.09908265e+01],
                              [-1.06552803e+01, -5.39236898e+00],
                              [-8.44831736e+00, -5.56408708e+00],
                              [-5.99469996e+00, -5.99469996e+00],
                              [-3.40597001e+00, -6.58846532e+00],
                              [-7.91850375e-01, -7.20961313e+00],
                              [1.79724280e+00, -7.79363946e+00]])
        
        xs_i_tem = np.array([[-4.99301095, -4.99301095],
                             [-1.72059117, -5.77480676],
                             [1.52675102, -6.51743471],
                             [4.77231399, -7.2656818],
                             [8.06670136, -8.06670136],
                             [8.06670136, -8.06670136],
                             [7.2656818, -4.77231399],
                             [6.51743471, -1.52675102],
                             [5.77480676, 1.72059117],
                             [4.99301095, 4.99301095],
                             [4.99301095, 4.99301095],
                             [1.72059117, 5.77480676],
                             [-1.52675102, 6.51743471],
                             [-4.77231399, 7.2656818],
                             [-8.06670136, 8.06670136],
                             [-8.06670136, 8.06670136],
                             [-7.2656818, 4.77231399],
                             [-6.51743471, 1.52675102],
                             [-5.77480676, -1.72059117],
                             [-4.99301095, -4.99301095]])
        
        ps_i_tem = np.array([[0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0.09352299, 0.40673046],
                             [0.09640031, 0.40746337],
                             [0.09012355, 0.40670259],
                             [0.09896043, 0.40652798],
                             [0.08805103, 0.41026997],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.],
                             [0., 0.]])
        
        x1_i_tem = np.array([[-4.99301095e+00, -4.99301095e+00],
                             [-5.77480676e+00, -1.72059117e+00],
                             [-6.51743471e+00, 1.52675102e+00],
                             [-7.26568180e+00, 4.77231399e+00],
                             [-8.06670136e+00, 8.06670136e+00],
                             [-1.72059117e+00, -5.77480676e+00],
                             [-2.50174974e+00, -2.50174974e+00],
                             [-3.24837276e+00, 7.47624234e-01],
                             [-3.98186741e+00, 3.98186741e+00],
                             [-4.77231399e+00, 7.26568180e+00],
                             [1.52675102e+00, -6.51743471e+00],
                             [7.47624234e-01, -3.24837276e+00],
                             [1.72778355e-16, -5.17891577e-16],
                             [-7.47624234e-01, 3.24837276e+00],
                             [-1.52675102e+00, 6.51743471e+00],
                             [4.77231399e+00, -7.26568180e+00],
                             [3.98186741e+00, -3.98186741e+00],
                             [3.24837276e+00, -7.47624234e-01],
                             [2.50174974e+00, 2.50174974e+00],
                             [1.72059117e+00, 5.77480676e+00],
                             [8.06670136e+00, -8.06670136e+00],
                             [7.26568180e+00, -4.77231399e+00],
                             [6.51743471e+00, -1.52675102e+00],
                             [5.77480676e+00, 1.72059117e+00],
                             [4.99301095e+00, 4.99301095e+00]])
        
        self.assertTrue(np.allclose(xgrid[0:28, :], xgrid_tem, atol=1e-6))
        self.assertTrue(np.allclose(ps_i, ps_i_tem, atol=1e-6))
        self.assertTrue(np.allclose(xs_i, xs_i_tem, atol=1e-6))
        self.assertTrue(np.allclose(x1_i, x1_i_tem, atol=1e-6))


if __name__ == '__main__':
    unittest.main()

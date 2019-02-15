import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 1)

import unittest
import numpy as np


class NumpyUnitTestCase(unittest.TestCase):
    sig = 0.9
    
    def test_matching_simple(self):
        import pickle
        
        import numpy as np
        import scipy.optimize
        
        import src.DeformationModules.Combination as comb_mod
        import src.DeformationModules.ElasticOrder0 as defmod0
        import src.DeformationModules.ElasticOrder1 as defmod1
        import src.DeformationModules.SilentLandmark as defmodsil
        import src.Forward.Shooting as shoot
        import src.Optimisation.ScipyOpti as opti
        from src.Utilities import Rotation as rot
        
        # Source
        with open('./data/basi1b.pkl', 'rb') as f:
            _, lx = pickle.load(f)
        
        nlx = np.asarray(lx).astype(np.float32)
        (lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
        scale = 38. / (lmax - lmin)
        
        nlx[:, 1] = 38.0 - scale * (nlx[:, 1] - lmin)
        nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0]))
        
        # %% target
        with open('./data/basi1t.pkl', 'rb') as f:
            _, lxt = pickle.load(f)
        
        nlxt = np.asarray(lxt).astype(np.float32)
        (lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
        scale = 100. / (lmax - lmin)
        
        nlxt[:, 1] = 38.0 - scale * (nlxt[:, 1] - lmin)
        nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))
        
        xst = nlxt[nlxt[:, 2] == 2, 0:2]
        
        #  common options
        nu = 0.001
        dim = 2
        
        # %% Silent Module
        xs = nlx[nlx[:, 2] == 2, 0:2]
        xs = np.delete(xs, 3, axis=0)
        Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
        ps = np.zeros(xs.shape)
        param_sil = (xs, ps)
        
        # %% Modules of Order 0
        sig0 = 6
        x0 = nlx[nlx[:, 2] == 1, 0:2]
        Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, 1., nu)
        p0 = np.zeros(x0.shape)
        param_0 = (x0, p0)
        
        # %% Modules of Order 0
        sig00 = 200
        x00 = np.array([[0., 0.]])
        Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, 0.1, nu)
        p00 = np.zeros([1, 2])
        param_00 = (x00, p00)
        
        # %% Modules of Order 1
        sig1 = 30
        x1 = nlx[nlx[:, 2] == 1, 0:2]
        C = np.zeros((x1.shape[0], 2, 1))
        K, L = 10, 38
        a, b = -2 / L ** 3, 3 / L ** 2
        C[:, 1, 0] = K * (a * (L - x1[:, 1]) ** 3 + b * (L - x1[:, 1]) ** 2)
        C[:, 0, 0] = 1. * C[:, 1, 0]
        Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, 0.01, C, nu)
        
        th = 0 * np.pi * np.ones(x1.shape[0])
        R = np.asarray([rot.my_R(cth) for cth in th])
        
        (p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
        param_1 = ((x1, R), (p1, PR))
        
        # %% Full model
        Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
        Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
        P0 = opti.fill_Vector_from_GD(Module.GD)
        
        # %%
        lam_var = 10.
        sig_var = 30.
        N = 10
        args = (Module, xst, lam_var, sig_var, N, 1e-7)
        
        res = scipy.optimize.minimize(opti.fun, P0,
                                      args=args,
                                      method='L-BFGS-B',
                                      jac=opti.jac,
                                      bounds=None,
                                      tol=None,
                                      callback=None,
                                      options={
                                          'maxcor': 10,
                                          'ftol': 1.e-09,
                                          'gtol': 1e-03,
                                          'eps': 1e-08,
                                          'maxfun': 6,
                                          'maxiter': 3,
                                          'iprint': -1,
                                          'maxls': 5
                                      })
        
        P1 = res['x']
        opti.fill_Mod_from_Vector(P1, Module)
        Modules_list = shoot.shooting_traj(Module, N)
        
        Modules_list_tem0 = np.array([[-2.21646979, 38.57068066],
                                      [-4.83115561, 36.39164195],
                                      [-6.89468678, 34.48598193],
                                      [-9.06333937, 32.02407938],
                                      [-10.59586459, 29.1805394],
                                      [-12.0893006, 26.36452847],
                                      [-13.39670657, 23.17491767],
                                      [-14.78636988, 19.65720968],
                                      [-16.28594697, 15.62258403],
                                      [-17.16586795, 11.05887767],
                                      [-18.48793881, 6.52167917],
                                      [-19.02088741, 1.03100368],
                                      [-18.98623281, -4.8987317],
                                      [-18.33203777, -10.72828894],
                                      [-16.78457955, -15.63310512],
                                      [-14.07522784, -21.5189159],
                                      [-10.869284, -27.93764784],
                                      [-6.70864436, -31.44623012],
                                      [-4.04493798, -36.11294763],
                                      [-3.06567041, -39.02225204],
                                      [1.97142437, -39.71109242],
                                      [6.03566358, -39.29212565],
                                      [5.02782737, -36.4894709],
                                      [5.83608211, -31.71462459],
                                      [10.82909502, -24.69130414],
                                      [13.9100608, -15.23621841],
                                      [15.51822993, -7.1144884],
                                      [15.07841038, 1.61306357],
                                      [14.32452248, 7.94812632],
                                      [13.24846987, 13.00451562],
                                      [12.44965777, 17.62246337],
                                      [11.39486544, 21.68441742],
                                      [9.94786, 24.98582738],
                                      [8.41835073, 27.89836421]])
        
        Modules_list_tem1 = np.array([[-2.22073448, -39.70241681]])
        
        Modules_list_tem2 = np.array([[-4.19662402e+00, 3.58872205e+01],
                                      [-3.20024125e-01, 3.61027131e+01],
                                      [-7.95089309e+00, 3.15345709e+01],
                                      [-4.08590979e+00, 3.19992084e+01],
                                      [-3.02971463e-03, 3.23278222e+01],
                                      [3.68265483e+00, 3.23567034e+01],
                                      [-1.07048325e+01, 2.69696260e+01],
                                      [-4.07589302e+00, 2.74465061e+01],
                                      [2.23740559e-01, 2.75323184e+01],
                                      [5.95689000e+00, 2.75271378e+01],
                                      [-1.20110969e+01, 2.11659329e+01],
                                      [-4.25359871e+00, 2.12218140e+01],
                                      [1.27439714e+00, 2.17006879e+01],
                                      [8.71614539e+00, 2.22061140e+01],
                                      [-1.47595915e+01, 1.42293629e+01],
                                      [-6.94267575e+00, 1.41082985e+01],
                                      [2.54018742e+00, 1.48076623e+01],
                                      [1.06884379e+01, 1.53261235e+01],
                                      [-1.61699412e+01, 4.40704720e+00],
                                      [-7.54081331e+00, 4.91603449e+00],
                                      [2.30006519e+00, 4.05440067e+00],
                                      [1.12377820e+01, 6.19171627e+00],
                                      [-1.39430943e+01, -7.22214520e+00],
                                      [-3.45879570e+00, -7.13668276e+00],
                                      [5.43030089e+00, -6.21278629e+00],
                                      [1.28152413e+01, -5.23665663e+00],
                                      [-1.07220737e+01, -1.77074052e+01],
                                      [-6.04816514e-01, -1.81170066e+01]])
        
        Modules_list_tem31 = np.array([[-4.19662402e+00, 3.58872205e+01],
                                       [-3.20024125e-01, 3.61027131e+01],
                                       [-7.95089309e+00, 3.15345709e+01],
                                       [-4.08590979e+00, 3.19992084e+01],
                                       [-3.02971463e-03, 3.23278222e+01],
                                       [3.68265483e+00, 3.23567034e+01],
                                       [-1.07048325e+01, 2.69696260e+01],
                                       [-4.07589302e+00, 2.74465061e+01],
                                       [2.23740559e-01, 2.75323184e+01],
                                       [5.95689000e+00, 2.75271378e+01],
                                       [-1.20110969e+01, 2.11659329e+01],
                                       [-4.25359871e+00, 2.12218140e+01],
                                       [1.27439714e+00, 2.17006879e+01],
                                       [8.71614539e+00, 2.22061140e+01],
                                       [-1.47595915e+01, 1.42293629e+01],
                                       [-6.94267575e+00, 1.41082985e+01],
                                       [2.54018742e+00, 1.48076623e+01],
                                       [1.06884379e+01, 1.53261235e+01],
                                       [-1.61699412e+01, 4.40704720e+00],
                                       [-7.54081331e+00, 4.91603449e+00],
                                       [2.30006519e+00, 4.05440067e+00],
                                       [1.12377820e+01, 6.19171627e+00],
                                       [-1.39430943e+01, -7.22214520e+00],
                                       [-3.45879570e+00, -7.13668276e+00],
                                       [5.43030089e+00, -6.21278629e+00],
                                       [1.28152413e+01, -5.23665663e+00],
                                       [-1.07220737e+01, -1.77074052e+01],
                                       [-6.04816514e-01, -1.81170066e+01]])
        
        Modules_list_tem32 = np.array([[[0.99971065, 0.02418204], [-0.02418204, 0.99971065]],
                                       [[0.99996505, -0.0083005], [0.0083005, 0.99996505]],
                                       [[0.9975893, 0.06968298], [-0.06968298, 0.9975893]],
                                       [[0.99960612, 0.02825157], [-0.02825157, 0.99960612]],
                                       [[0.99987325, - 0.01590845], [0.01590845, 0.99987325]],
                                       [[0.99850203, - 0.0548006], [0.0548006, 0.99850203]],
                                       [[0.99357498, 0.11367944], [-0.11367944, 0.99357498]],
                                       [[0.99943499, 0.03386045], [-0.03386045, 0.99943499]],
                                       [[0.99970135, - 0.0245064], [0.0245064, 0.99970135]],
                                       [[0.99542883, - 0.09584995], [0.09584995, 0.99542883]],
                                       [[0.9899716, 0.14201453], [-0.14201453, 0.9899716]],
                                       [[0.99919496, 0.04042412], [-0.04042412, 0.99919496]],
                                       [[0.99905805, - 0.04362357], [0.04362357, 0.99905805]],
                                       [[0.99016615, - 0.14054658], [0.14054658, 0.99016615]],
                                       [[0.98419081, 0.17813439], [-0.17813439, 0.98419081]],
                                       [[0.99688275, 0.07946446], [-0.07946446, 0.99688275]],
                                       [[0.99809376, - 0.0621291], [0.0621291, 0.99809376]]])
        
        self.assertTrue(np.allclose(Modules_list[11].GD.GD_list[0].GD[0:34], Modules_list_tem0, atol=1e-6))
        self.assertTrue(np.allclose(Modules_list[11].GD.GD_list[1].GD, Modules_list_tem1, atol=1e-6))
        self.assertTrue(np.allclose(Modules_list[11].GD.GD_list[2].GD[0:28], Modules_list_tem2, atol=1e-6))
        self.assertTrue(np.allclose(Modules_list[11].GD.GD_list[3].GD[0][0:28], Modules_list_tem31, atol=1e-6))
        self.assertTrue(np.allclose(Modules_list[11].GD.GD_list[3].GD[1][0:17, :, :], Modules_list_tem32, atol=1e-6))
    
    def test_geodesic_rectangle(self):
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

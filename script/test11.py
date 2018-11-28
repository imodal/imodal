 
##  Exp basipetal  (Sandbox to test backward dynamics)
##
##
import scipy.optimize
def my_plot3(Mod0, Mod1, Cot, xst, fig, nx, ny, name, i):            
    x0 = Mod0['0']
    plt.axis('equal')
    x0 = my_close(x0)
    
    (x1,R) = Mod1['x,R']
    plt.plot(x1[:,0],x1[:,1],'b.')
    plt.plot(x0[:,0],x0[:,1],'r.')
    
    xs = Cot['0'][1][0] #[nx*ny:,:]
    xs = my_close(xs)
    plt.plot(xs[:,0], xs[:,1],'g')
    
    xst = my_close(xst)
    plt.plot(xst[:,0], xst[:,1],'m')
    
    plt.ylim(-40,80)
    plt.show()  
    plt.savefig(name + '{:02d}'.format(i) +'.png')
    return
    
def my_plot4(Mod0, Mod1, Cot, xst, fig, nx, ny, name, i):            
    x0 = Mod0['0']
    plt.axis('equal')
    x0 = my_close(x0)
    
    
    xs = Cot['0'][1][0][0:nx*ny,:]
    xsx = xs[:,0].reshape((nx,ny))
    xsy = xs[:,1].reshape((nx,ny))
    plt.plot(xsx, xsy,color = 'lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(),color = 'lightblue')
    (x1,R) = Mod1['x,R']
    plt.plot(x1[:,0],x1[:,1],'b.')
    plt.plot(x0[:,0],x0[:,1],'r.')
    
    xs = Cot['0'][1][0][nx*ny:,:]
    xs = my_close(xs)
    plt.plot(xs[:,0], xs[:,1],'g')
    
    xst = my_close(xst)
    plt.plot(xst[:,0], xst[:,1],'m')
    
    
    plt.ylim(-40,80)
    plt.show()  
    plt.savefig(name + '{:02d}'.format(i) +'.png')
    return
##
    # Importing the model
    import pickle
    with open('basi1b.pkl', 'rb') as f:
        img, lx = pickle.load(f)
    
    nlx = np.asarray(lx).astype(np.float64)
    (lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
    scale = 38./(lmax-lmin)
    
    nlx[:,1]  = 38.0 - scale*(nlx[:,1]-lmin)
    nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))            
    
    # importing the target
    with open('basi1t.pkl', 'rb') as f:
        img, lxt = pickle.load(f)
    
    nlxt = np.asarray(lxt).astype(np.float64)
    (lmint, lmaxt) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
    scale = 100./(lmaxt-lmint)
    
    #nlxt[:,1]  = 100.0 - scale*(nlxt[:,1]-lmint)-30.
    nlxt[:,1]  = 100.0 - scale*(nlxt[:,1]-lmint)-10.
    nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))            
    
    (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expf_', [5., 0.05], 0.001, 10., 30., 10., 10.) 
    # (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expa_', [1., 10.], 0.001, 10., 30. , 10., 10.) 
    # (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expb_', [1., 10.], 0.001,
    # 10., 30.,  1., 10.)
    # (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expe_', [5., 0.05], 0.001, 10., 30., 1., 10.)
    # (name, coeffs, nu, sin0, sig1, lam_var, sig_var) = ('basi_expd_', [5., 0.05],0.001, 30., 30., 1., 10.)
    (name, coeffs, nu, sin0, sig1, lam_var, sig_var) = ('basi_expc_', [1., 0.05],0.001, 300., 30., 10., 30.) 
    # (name, coeffs, nu, sig0, sig1, lam_var, sig_var) = ('basi_expf_', [500., 0.05], 0.001, 10., 30., 10., 10.) 
            
    x0 = nlx[nlx[:,2]==0,0:2]
    x1 = nlx[nlx[:,2]==1,0:2]
    xs = nlx[nlx[:,2]==2,0:2]
    xst = nlxt[nlxt[:,2]==2,0:2]
    
    nx, ny = (5,11)
    (a,b,c,d) = (-10., 10., -3., 40.)
    [xx, xy] = np.meshgrid(np.linspace(a,b,nx), np.linspace(c,d,ny))
    (nx,ny) = xx.shape
    
    nxs = np.asarray([xx.flatten(), xy.flatten()]).transpose()
    nps = np.zeros(nxs.shape)
    
    th = 0*np.pi
    p0 = np.zeros(x0.shape)
    Mod0 ={ '0': x0, 'sig':sig0, 'coeff':coeffs[0]}
    
    th = th*np.ones(x1.shape[0])
    R = np.asarray([my_R(cth) for cth in th])
    
    for  i in range(x1.shape[0]):
        R[i] = my_R(th[i])
        
    Mod1 = {'x,R':(x1,R), 'sig':sig1, 'coeff' :coeffs[1], 'nu' : nu}
    
    ps = np.zeros(xs.shape)
    ps[0:4,1] = 1.5
    ps[22:26,1] = 1.5
    
    
    C = np.zeros((x1.shape[0],2,1))
    K = 10
    C[:,0,0] = K*(38. - x1[:,1])/38.
    C[:,1,0] = K*(38. - x1[:,1])/38.
    
    L = 38.
    a, b = (2.)/(2*L), -(2.)/(2*L*L)
    C[:,0,0] = 0.85*K*(b*(38. - x1[:,1])**2/2 + a*(38. - x1[:,1]))
    C[:,1,0] = K*(b*(38. - x1[:,1])**2/2 + a*(38. - x1[:,1]))
    
    L = 38.
    a, b = -2/L**3, 3/L**2
    C[:,1,0] = K*(a*(38. - x1[:,1])**3 + b*(38. - x1[:,1])**2)
    C[:,0,0] = 1.*C[:,1,0]
    
    Mod1['C'] = C
    
    (p1,pR) = (np.zeros((x1.shape)), np.zeros((x1.shape[0],2,2)))
    
    Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,pR))]}
    
   
##

    N = 7
    X = my_X(x0, xs, x1, R)
    nX = (x0.shape[0], xs.shape[0], x1.shape[0])
    (sig0, sig1, coeff0, coeff1) = (Mod0['sig'], Mod1['sig'], Mod0['coeff'], Mod1['coeff'])
    args = (X, nX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N)
    
    P0 = my_P(p0, ps, p1, pR)
    
    res= scipy.optimize.minimize(my_fun, P0, 
        args = (X, nX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N),
        method='L-BFGS-B', jac=my_jac, bounds=None, tol=None, callback=None,    
        options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
        'eps': 1e-08, 'maxfun': 100, 'maxiter': 100, 'iprint': -1, 'maxls': 20})

    fig = plt.figure(5)
    plt.clf()
    P0 = res['x']
    tp0, tps, (tp1, tpR) = my_splitP(P0,nX)
    axs = np.concatenate((nxs,xs), axis = 0)
    aps = np.concatenate((nps,tps), axis = 0)
    aX = my_X(x0, axs, x1, R)
    anX = (x0.shape[0], axs.shape[0], x1.shape[0])
    aargs = (aX, anX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N)
    
    aP0 = my_P(tp0, aps, tp1, tpR)
    Traj = my_fun_Traj(aP0,*aargs)
    
    for i in range(N+1):
        plt.clf()
        (tMod0, tMod1, tCot) = Traj[2*i]
        my_plot4(tMod0,tMod1,tCot,xst, fig,nx,ny, name, i)
    
    filepng = name + "*.png"
    os.system("convert " + filepng + " -set delay 0 -reverse " + filepng + " -loop 0 " + name + ".gif")  

  ## 
    n0  = (2, 2)   # number of order 0
    n1  = (4,5)   # number of point of order 1
    dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
    sig = (0.25, 0.7) # sigma's for K0 and K1
    th = 0*np.pi
    h =[0.,0.5]
    Mod0, Mod1 = my_new_exp0(dom, n0, n1, th, sig)
    (x,P) = my_hToP(Mod1,h)
    
    Vs = {'p':[(x,P)], 'sig' : Mod1['sig']}
    v = my_VsToV(Vs, x, 0)
    
    fig=plt.figure(1)
    fig.clf()
            
    plt.axis('equal')
    Q = plt.quiver(x[:,0], x[:,1],v[:,0], v[:,1],
        units='xy',scale = 1.,zorder=0,headwidth=5.,
        width=0.005,headlength=5.,color='blue')
        
    xb = my_border(dom, n1)
    vxb = my_VsToV(Vs, xb, 0)
    
    plt.plot(xb[:,0]+vxb[:,0],xb[:,1]+vxb[:,1],color='red')
    
    
    plt.ylim(-1,3)
    plt.show()  
    
    ## my_CotDotV et my_dCotDotV (tested !!!)
    
    n0  = (2, 2)   # number of order 0
    n1  = (4,5)   # number of point of order 1
    dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
    sig = (0.25, 0.7) # sigma's for K0 and K1
    th = 0*np.pi
    h =[0.,0.5]
    Mod0, Mod1 = my_new_exp0(dom, n0, n1, th, sig)
    
    Vs = {'0':[(rd.normal(0,1,(12,2)),rd.normal(0,1,(12,2)))], 'sig':0.3}
    Cot = {'0':[(rd.normal(0,1,(12,2)),rd.normal(0,1,(12,2)))]}
    
    z = rd.normal(0,1,(12,2))
    p = rd.normal(0,1,(12,2))
    zp = z+ 0.00001*rd.normal(0,1,(12,2))
    Dz = zp -z
    Cot = {'0':[(z,p)]}
    Cotp = {'0':[(zp,p)]}
    
    De = my_CotDotV(Cotp,Vs)-my_CotDotV(Cot,Vs)
    der = my_dCotDotV(Cot,Vs)
    
    out = 0
    for (x,d) in der['0']:
        for i in range(x.shape[0]):
            out += np.dot(d[i],Dz[i])
    
    print(out)
    print(De)
    
    
    th = rd.normal(0,1,(12,1))
    R = np.asarray([my_R(cth) for cth in th])
    P = rd.normal(0,1,(z.shape[0],2,2))
    
    Cot = { 'x,R':[((z,R),(p,P))]}
    Cotp = { 'x,R':[((zp,R),(p,P))]}
    Vs = {'0':[(rd.normal(0,1,(12,2)),rd.normal(0,1,(12,2)))], 'sig':0.3}
    
    De = my_CotDotV(Cotp,Vs)-my_CotDotV(Cot,Vs)
    der = my_dCotDotV(Cot,Vs)

    
    out = 0
    for ((x,R),(dedx,dedR)) in der['x,R']:
        for i in range(z.shape[0]):
            out += np.dot(dedx[i],Dz[i])
    
    print(out)
    print(De)

    thp = th + 0.0001*rd.normal(0,1,th.shape)
    Rp = np.asarray([my_R(cth) for cth in thp])  
    Cotp = { 'x,R':[((z,Rp),(p,P))]}
    DR = Rp-R
    
    De = my_CotDotV(Cotp,Vs)-my_CotDotV(Cot,Vs)
    der = my_dCotDotV(Cot,Vs)

    out = 0
    for ((x,R),(dedx,dedR)) in der['x,R']:
        for i in range(z.shape[0]):
            out += np.tensordot(dedR[i],DR[i])
    print(out)
    print(De)
    
    ## my_pSmV  (passed !!)
    
    x0 = rd.normal(0,1,(12,2))
    p = rd.normal(0,1,(12,2))
    x0p = x0+ 0.00001*rd.normal(0,1,(12,2))
    Dx0 = x0p -x0
    
    x1 = rd.normal(0,1,(12,2))
    P = rd.normal(0,1,(12,2,2))
    x1p = x1+ 0.00001*rd.normal(0,1,(12,2))
    Dx1 = x1p -x1
    
    Vsl = {'0':[(x0,p)], 'p':[(x1,P)], 'sig':0.3}
    Vslp = {'0':[(x0p,p)], 'p':[(x1p,P)], 'sig':0.3}
    
    Vsr = {'0':[(rd.normal(0,1,(14,2)),rd.normal(0,1,(14,2)))],
            'p':[(x1,P)], 'sig':0.4}
    De = my_pSmV(Vslp,Vsr,0)-my_pSmV(Vsl,Vsr,0)
    der = my_pSmV(Vsl,Vsr,1)
    
    out = 0.
    if '0' in der:
        for (x,d) in der['0']:
            out += np.sum(np.asarray([np.dot(d[i],Dx0[i]) 
                for i in range(x.shape[0])]))
            
    if 'p' in der:
        for (x,d) in der['p']:
            out += np.sum(np.asarray([np.dot(d[i],Dx1[i]) 
                for i in range(x.shape[0])]))

    print(out)
    print(De)
            
    
## my_update my_new_ham
    n0  = (2, 2)   # number of order 0
    n1  = (3,5)   # number of point of order 1
    dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
    sig = (0.25, 0.7) # sigma's for K0 and K1
    th = 0*np.pi
    h =[0.,0.5]
    coeffs = [0.1, 1]
    Mod0, Mod1 = my_new_exp0(dom, n0, n1, th, sig)
    x0 = Mod0['0']
    Mod0['coeff'] = 0.1
    Mod1['coeff'] = 1.
    Mod0['mom'] = rd.normal(0,1,x0.shape)
    Mod1['h']= h
    
    x0, p0 = Mod0['0'], rd.normal(0,1,x0.shape)
    (x1, R) = Mod1['x,R']
    (p1,PR) = rd.normal(0,1,x1.shape), rd.normal(0,1,(x1.shape[0],2,2))
    
    xs, ps = rd.normal(0,1,(5,2)), rd.normal(0,1,(5,2))
    
    Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}
    print(my_new_ham(Mod0, Mod1, Cot))
##  (passed)
    K= 0.
    x0p = x0+ 0.0001*rd.normal(0,1,x0.shape)
    xsp = xs+ 0.0001*rd.normal(0,1,xs.shape)
    x1p = x1+ 0.00001*rd.normal(0,1,x1.shape)
    th = rd.normal(0,1,(x1.shape[0],1))
    R = np.asarray([my_R(cth) for cth in th])
    thp = th + 0.000001*rd.normal(0,1,th.shape)
    Rp = np.asarray([my_R(cth) for cth in thp])  


    
    Cot = { '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}
    Cotp = { '0':[(x0p,p0), (xsp,ps)], 'x,R':[((x1p,Rp),(p1,PR))]}
    
    # Cot = { '0':[(x0,p0), (xs,ps)]}
    # Cotp = { '0':[(x0p,p0), (xs,ps)]}
    
    Mod0p = my_init_from_mod(Mod0)
    Mod0p['0'] = x0p
    Mod0p['mom'] = Mod0['mom']
    
    Mod1p = my_init_from_mod(Mod1)
    
    Mod1['x,R'] = (x1,R)
    Mod1p['x,R'] = (x1p,Rp)
    Mod1p['h'] = Mod1['h']
    
    Dx0 = x0p - x0
    Dxs = xsp - xs
    Dx1 = x1p - x1
    DR = Rp-R
    
    De = my_new_ham(Mod0p, Mod1p, Cotp)- my_new_ham(Mod0, Mod1, Cot)
    der = my_dham(Mod0,Mod1,Cot)
    
    out = 0.
    (x,dxH) = der['0'][0]
    out = np.sum(np.asarray([np.dot(dxH[i],Dx0[i]) 
        for i in range(x.shape[0])]))
    
    (x,dxH) = der['0'][1]
    out += np.sum(np.asarray([np.dot(dxH[i],Dxs[i]) 
        for i in range(x.shape[0])]))
    
        
    ((x,R),(d,dR)) = der['x,R'][0]
    out += np.sum(np.asarray([np.dot(d[i],Dx1[i]) 
        for i in range(x.shape[0])]))
    
    ((x,R),(d,dR)) = der['x,R'][0]
    out += np.sum(np.asarray([np.tensordot(dR[i],DR[i]) 
        for i in range(x.shape[0])]))
    print(out)
    print(De)

    
## test my_AmKiAm (tested)
    AKiA= my_new_AmKiAm(Mod1)
    p = np.dot(AKiA,h)
    
    u = rd.normal(0,1,(2))
    
    print(np.dot(p,u)-
    np.dot(Mod1['lam'],my_new_Amh(Mod1,u).flatten()))
    
##
def my_close(x):
    N = x.shape[0]
    z = np.zeros((N+1,2))
    z[0:N,:] = x
    z[N,:] = x[0,:]
    return z

def my_plot(Mod0, Mod1, Cot, fig, nx, ny, name, i):
            
    x0 = Mod0['0']
    plt.axis('equal')
    # Q = plt.quiver(x[:,0], x[:,1],v[:,0], v[:,1],
    #     units='xy',scale = 1.,zorder=0,headwidth=5.,
    #     width=0.005,headlength=5.,color='blue')
    #     
    # xb = my_border(dom, n1)
    # vxb = my_VsToV(Vs, xb, 0)
    
    x0 = my_close(x0)
    
    
    xs = Cot['0'][1][0]
    xsx = xs[:,0].reshape((nx,ny))
    xsy = xs[:,1].reshape((nx,ny))
    plt.plot(xsx, xsy,color = 'lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(),color = 'lightblue')
    (x1,R) = Mod1['x,R']
    plt.plot(x1[:,0],x1[:,1],'bo')
    plt.plot(x0[:,0],x0[:,1],'r.')
    
    
    plt.ylim(-3,5)
    # plt.xlim(-2,4)
    plt.show()  
    plt.savefig(name + '{:02d}'.format(i) +'.png')
    return
##  forward
    for  s in \
    [([1, 10], 0.1, 'expb1_10__'),
    ([1, 0.1], 0.01, 'expb1_0p1_'),
    ([1, 0.5], 0.02, 'expb1_0p5_'),
    ([1, 1], 0.03, 'expb1_1_'),
    ([1, 2], 0.04, 'expb1_2_'),
    ([1, 4], 0.05, 'expb1_4_')]:
        (coeffs, dt, name) = s
        n0  = (3,5)   # number of order 0
        n1  = (3,5)   # number of point of order 1
        dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
        sig = (0.5, 0.7) # sigma's for K0 and K1
        th = 0*np.pi
        #h =[0.,0.5]
        #coeffs = [1, 0.1]
        #name = 'coeff1_0p1_'
        Mod0, Mod1 = my_new_exp0(dom, n0, n1, th, sig)
        x0 = Mod0['0']
        Mod0['coeff'] = coeffs[0]
        Mod1['coeff'] = coeffs[1]
        Mod0['mom'] = rd.normal(0,1,x0.shape)
        #Mod1['h']= h
        
        (a,b,c,d) = dom
        [xx, xy] = np.meshgrid(np.linspace(a,b,5), np.linspace(c,d,11))
        (nx,ny) = xx.shape
        
        xs = np.asarray([xx.flatten(), xy.flatten()]).transpose()
        ps = np.zeros(xs.shape)
        
        x0, p0 = Mod0['0'], np.zeros(x0.shape)
        p0[6:9,1] =1.
        p0[0:3,1] =-1.
        (x1, R) = Mod1['x,R']
        (p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
        
        #xs, ps = x0, p0
        
        Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}
        
        (sig0, sig1) = sig
        # initialisation des parametres en fonctions des p
        Mod0['SKS'] = my_new_SKS(Mod0)
        Mod0['mom'] = solve(Mod0['coeff']*Mod0['SKS'],
            my_VsToV(my_CotToVs(Cot,sig0),
            x0,0).flatten(),sym_pos = True).reshape(x0.shape)
        my_mod_update(Mod0) # compute cost0
        
        Mod1['SKS'] = my_new_SKS(Mod1)
        dv = my_VsToV(my_CotToVs(Cot,sig1),x1,1)
        S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
        tlam = solve(Mod1['coeff']*Mod1['SKS'], S.flatten(), sym_pos = True)
        (Am, AmKiAm) = my_new_AmKiAm(Mod1)
        Mod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
        my_mod_update(Mod1) # will compute the new lam, mom and cost
        
        
        print(my_new_ham(Mod0, Mod1, Cot))
        
        fig = plt.figure(5)
        plt.clf()
        my_plot(Mod0,Mod1,Cot,fig, nx, ny, name, 0)
    
        for i in range(10):
            (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
            plt.clf()
            my_plot(Mod0,Mod1,Cot,fig,nx,ny, name, i+1)
        filepng = name + "*.png"
        os.system("convert " + filepng + " -set delay 10 -reverse " + filepng + " -loop 0 " + name + ".gif")  
        
##
##  forward
   
def my_plot2(Mod0, Mod1, Cot, fig, nx, ny, name, i):
            
    x0 = Mod0['0']
    plt.axis('equal')
    # Q = plt.quiver(x[:,0], x[:,1],v[:,0], v[:,1],
    #     units='xy',scale = 1.,zorder=0,headwidth=5.,
    #     width=0.005,headlength=5.,color='blue')
    #     
    # xb = my_border(dom, n1)
    # vxb = my_VsToV(Vs, xb, 0)
    
    coeffs = [1., 0.1]
    dt = 0.01
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
    #xs = np.concatenate((xs,xs[0,:].reshape(1,2)), axis = 0)
    plt.plot(xs[:,0], xs[:,1],'g')
    
    
    plt.ylim(-40,80)
    plt.show()  
    plt.savefig(name + '{:02d}'.format(i) +'.png')
    return

##
 #  ('leaf_10_1', 0.1, [10,1])
   for s in [('leaf1c', 0.03, [1,10])]:
        (name, dt, coeffs) = s
        
        with open('objs.pkl','rb') as f:
            x0, x1, xs = pickle.load(f)
            
        K  = 30
        x0 = K*x0
        x1 = K*x1
        xs = K*xs
    
        nx, ny = (5,11)
        (a,b,c,d) = (-0.6, 0.8, -1.5, 1.5)
        [xx, xy] = np.meshgrid(np.linspace(a,b,nx), np.linspace(c,d,ny))
        (nx,ny) = xx.shape
        
        nxs = np.asarray([xx.flatten(), xy.flatten()]).transpose()
        nps = np.zeros(nxs.shape)
            
        sig = (1, 0.7) # sigma's for K0 and K1
        
        th = 0*np.pi
        p0 = np.zeros(x0.shape)
        Mod0 ={ '0': x0, 'sig':sig[0], 'coeff':coeffs[0]}
        
        th = th*np.ones(x1.shape[0])
        R = np.asarray([my_R(cth) for cth in th])
        
        for  i in range(x1.shape[0]):
            R[i] = my_R(th[i])
            
        Mod1 = {'x,R':(x1,R), 'sig':sig[1], 'coeff' :coeffs[1]}
        
        ps = np.zeros(xs.shape)
        ps[7:12,1] = 1.
        
        xs = np.concatenate((nxs,xs), axis = 0)
        ps = np.concatenate((nps,ps), axis = 0)
        C = np.zeros((x1.shape[0],2,1))
        C[:,0,0] = -1
        C[:,1,0] = 1
        
        Mod1['C'] = C
        
    
        (p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
        
        #xs, ps = x0, p0
        
        Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}
        
        (sig0, sig1) = sig
        # initialisation des parametres en fonctions des p
        Mod0['SKS'] = my_new_SKS(Mod0)
        Mod0['mom'] = solve(Mod0['coeff']*Mod0['SKS'],
            my_VsToV(my_CotToVs(Cot,sig0),
            x0,0).flatten(),sym_pos = True).reshape(x0.shape)
        my_mod_update(Mod0) # compute cost0
        
        Mod1['SKS'] = my_new_SKS(Mod1)
        dv = my_VsToV(my_CotToVs(Cot,sig1),x1,1)
        S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
        tlam = solve(Mod1['coeff']*Mod1['SKS'], S.flatten(), sym_pos = True)
        (Am, AmKiAm) = my_new_AmKiAm(Mod1)
        Mod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
        my_mod_update(Mod1) # will compute the new lam, mom and cost
        
        
        print(my_new_ham(Mod0, Mod1, Cot))
    
        fig = plt.figure(5)
        plt.clf()
        my_plot2(Mod0,Mod1,Cot,fig, nx, ny, name, 0)
    
    
        (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
        plt.clf()
        my_plot2(Mod0,Mod1,Cot,fig, nx, ny, name, 0)
    
        for i in range(17):
            (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
            plt.clf()
            my_plot2(Mod0,Mod1,Cot,fig,nx,ny, name, i+1)
    
        filepng = name + "*.png"
        os.system("convert " + filepng + " -set delay 10 -reverse " + filepng + " -loop 0 " + name + ".gif")  

##  Random shooting
 #  ('leaf_10_1', 0.1, [10,1])
   for s in [('leaf1c', 0.02, [1,0.1])]:
        (name, dt, coeffs) = s
        
        with open('objs.pkl','rb') as f:
            x0, x1, xs = pickle.load(f)
            
        K  = 30
        x0 = K*x0
        x1 = K*x1
        xs = K*xs
    
        nx, ny = (5,11)
        (a,b,c,d) = (-0.6, 0.8, -1.5, 1.5)
        [xx, xy] = np.meshgrid(np.linspace(a,b,nx), np.linspace(c,d,ny))
        (nx,ny) = xx.shape
        
        nxs = np.asarray([xx.flatten(), xy.flatten()]).transpose()
        nps = np.zeros(nxs.shape)
            
        sig = (1, 0.7) # sigma's for K0 and K1
        
        th = 0*np.pi
        p0 = np.zeros(x0.shape)
        Mod0 ={ '0': x0, 'sig':sig[0], 'coeff':coeffs[0]}
        
        th = th*np.ones(x1.shape[0])
        R = np.asarray([my_R(cth) for cth in th])
        
        for  i in range(x1.shape[0]):
            R[i] = my_R(th[i])
            
        Mod1 = {'x,R':(x1,R), 'sig':sig[1], 'coeff' :coeffs[1]}
        
        #ps = rd.normal(0,1,xs.shape)/np.sqrt(xs.shape[0])
        ps = np.zeros(xs.shape)
        ps[7:12,1] = 1.
        
        xs = np.concatenate((nxs,xs), axis = 0)
        ps = np.concatenate((nps,ps), axis = 0)
        C = np.zeros((x1.shape[0],2,1))
        C[:,0,0] = -1
        C[:,1,0] = 1
        
        Mod1['C'] = C
        
    
        (p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
        
        #xs, ps = x0, p0
        
        Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}
        
        (sig0, sig1) = sig
        # initialisation des parametres en fonctions des p
        Mod0['SKS'] = my_new_SKS(Mod0)
        Mod0['mom'] = solve(Mod0['coeff']*Mod0['SKS'],
            my_VsToV(my_CotToVs(Cot,sig0),
            x0,0).flatten(),sym_pos = True).reshape(x0.shape)
        my_mod_update(Mod0) # compute cost0
        
        Mod1['SKS'] = my_new_SKS(Mod1)
        dv = my_VsToV(my_CotToVs(Cot,sig1),x1,1)
        S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
        tlam = solve(Mod1['coeff']*Mod1['SKS'], S.flatten(), sym_pos = True)
        (Am, AmKiAm) = my_new_AmKiAm(Mod1)
        Mod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
        my_mod_update(Mod1) # will compute the new lam, mom and cost
        
        
        print(my_new_ham(Mod0, Mod1, Cot))
    
        fig = plt.figure(5)
        plt.clf()
        my_plot2(Mod0,Mod1,Cot,fig, nx, ny, name, 0)
    
    
        (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
        plt.clf()
        my_plot2(Mod0,Mod1,Cot,fig, nx, ny, name, 0)
    
        for i in range(20):
            (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
            plt.clf()
            my_plot2(Mod0,Mod1,Cot,fig,nx,ny, name, i+1)
    
        filepng = name + "*.png"
        os.system("convert " + filepng + " -set delay 20 -reverse " + filepng + " -loop 0 " + name + ".gif")  
            
##  Exp basipetal
def my_close(x):
    N = x.shape[0]
    z = np.zeros((N+1,2))
    z[0:N,:] = x
    z[N,:] = x[0,:]
    return z
    
def my_plot3(Mod0, Mod1, Cot, xst, fig, nx, ny, name, i):
            
    x0 = Mod0['0']
    plt.axis('equal')
    x0 = my_close(x0)
    
    
    # xs = Cot['0'][1][0][0:nx*ny,:]
    # xsx = xs[:,0].reshape((nx,ny))
    # xsy = xs[:,1].reshape((nx,ny))
    # plt.plot(xsx, xsy,color = 'lightblue')
    # plt.plot(xsx.transpose(), xsy.transpose(),color = 'lightblue')
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
    
##
    # Importing the model
    import pickle
    with open('basi1.pkl', 'rb') as f:
        img, lx = pickle.load(f)
    
    nlx = np.asarray(lx).astype(np.float32)
    (lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
    scale = 38./(lmax-lmin)
    
    nlx[:,1]  = 38.0 - scale*(nlx[:,1]-lmin)
    nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))            
    
    # importing the target
    with open('basi1t.pkl', 'rb') as f:
        img, lxt = pickle.load(f)
    
    nlxt = np.asarray(lx).astype(np.float32)
    (lmint, lmaxt) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
    scale = 100./(lmaxt-lmint)
    
    nlxt[:,1]  = 100.0 - scale*(nlxt[:,1]-lmint)-30.
    nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))            
    
    
    # (name, dt, coeffs, nu) = ('basi_expe_', 0.1, [8., 0.05], 0.001)
    (name, dt, coeffs, nu) = ('basi_expf_', 0.1, [5., 0.05], 0.001)
        
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
        
    sig = (10., 30.) # sigma's for K0 and K1
    
    th = 0*np.pi
    p0 = np.zeros(x0.shape)
    Mod0 ={ '0': x0, 'sig':sig[0], 'coeff':coeffs[0]}
    
    th = th*np.ones(x1.shape[0])
    R = np.asarray([my_R(cth) for cth in th])
    
    for  i in range(x1.shape[0]):
        R[i] = my_R(th[i])
        
    Mod1 = {'x,R':(x1,R), 'sig':sig[1], 'coeff' :coeffs[1], 'nu' : nu}
    
    #ps = rd.normal(0,1,xs.shape)/np.sqrt(xs.shape[0])
    ps = np.zeros(xs.shape)
    ps[0:4,1] = 2.
    ps[22:26,1] = 2.
    
    
    # xs = np.concatenate((nxs,xs), axis = 0)
    # ps = np.concatenate((nps,ps), axis = 0)
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
    C[:,0,0] = 0.9*C[:,1,0]
    
    Mod1['C'] = C
    
    
    (p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
    
    #xs, ps = x0, p0
    
    Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}
    
    (sig0, sig1) = sig
    # initialisation des parametres en fonctions des p
    Mod0['SKS'] = my_new_SKS(Mod0)
    Mod0['mom'] = solve(Mod0['coeff']*Mod0['SKS'],
        my_VsToV(my_CotToVs(Cot,sig0),
        x0,0).flatten(),sym_pos = True).reshape(x0.shape)
    my_mod_update(Mod0) # compute cost0
    
    Mod1['SKS'] = my_new_SKS(Mod1)
    dv = my_VsToV(my_CotToVs(Cot,sig1),x1,1)
    S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
    tlam = solve(Mod1['coeff']*Mod1['SKS'], S.flatten(), sym_pos = True)
    (Am, AmKiAm) = my_new_AmKiAm(Mod1)
    Mod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
    my_mod_update(Mod1) # will compute the new lam, mom and cost
    
    ## Test
    my_eps = 0.00001
    grad ={'0':[(rd.normal(0,1,x0.shape), rd.normal(0,1,x0.shape)),
        (rd.normal(0,1,xs.shape), rd.normal(0,1,xs.shape))], 
        'x,R':[((rd.normal(0,1,x1.shape), rd.normal(0,1,R.shape)),
            (rd.normal(0,1,x1.shape), rd.normal(0,1,R.shape)))]}
    ngrad = my_sub_bckwd(Mod0, Mod1, Cot, grad, my_eps)
    ##    
    N =3
    Traj = my_fd_shoot(Mod0, Mod1, Cot,N)
##
    cCot = Traj[-1][2]
    # xsf = cCot['0'][1][0][nx*ny:]
    xsf = cCot['0'][1][0]
    (cost, dxcost) = my_dxvar_cost(xsf, xst, 10.)
    grad ={'0':[(np.zeros(x0.shape), np.zeros(x0.shape)),
        (np.concatenate((np.zeros((nx*ny,2)),dxcost),
            axis = 0),np.zeros(xs.shape))], 
        'x,R':[((np.zeros(x1.shape), np.zeros(R.shape)),
            (np.zeros(x1.shape), np.zeros(R.shape)))]}
    grad ={'0':[(np.zeros(x0.shape), np.zeros(x0.shape)),
        (dxcost,np.zeros(xs.shape))], 
        'x,R':[((np.zeros(x1.shape), np.zeros(R.shape)),
            (np.zeros(x1.shape), np.zeros(R.shape)))]}
    ngrad = my_bck_shoot(Traj, grad, 0.0001)
    
    ##
    fig = plt.figure(5)
    plt.clf()
    for i in range(N+1):
        plt.clf()
        (tMod0, tMod1, tCot) = Traj[2*i]
        my_plot3(tMod0,tMod1,tCot,xst, fig,nx,ny, name, i)
    
    filepng = name + "*.png"
    #os.system("convert " + filepng + " -set delay 20 -reverse " + filepng + " -loop 0 " + name + ".gif")  
    os.system("convert " + filepng + " -set delay 0 -reverse " + filepng + " -loop 0 " + name + ".gif")  
##
    (x0,p0),(xs,ps) = Cot['0']
    ((x1,R),(p1,pR)) = Cot['x,R'][0]
    r= 0.00001
    Dt = 0.1
    tx0 = x0 + 0*rd.normal(0,1,x0.shape)
    tp0 = p0 + 0*rd.normal(0,1,p0.shape)
    txs = xs + 0*rd.normal(0,1,xs.shape)
    tps = ps + 0*rd.normal(0,1,ps.shape)
    tx1 = x1 + 0*rd.normal(0,1,x1.shape)
    tp1 = p1 + 0*rd.normal(0,1,p1.shape)
    dv = rd.normal(0,1,R.shape)
    S= (dv - np.swapaxes(dv, 1,2))/2
    tR = R+r*np.asarray([np.dot(S[i], R[i]) for i in range(S.shape[0])])
    # tR = R + r*rd.normal(0,1,R.shape)
    tpR = pR + 0*rd.normal(0,1,pR.shape)
    
    tCot = {'0':[(tx0,tp0), (txs, tps)], 'x,R':[((tx1,tR),(tp1,tpR))]}
    ex0 = rd.normal(0,1,x0.shape)
    ep0 = rd.normal(0,1,p0.shape)
    exs = rd.normal(0,1,xs.shape)
    eps = rd.normal(0,1,ps.shape)
    ex1 = rd.normal(0,1,x1.shape)
    ep1= rd.normal(0,1,p1.shape)
    dv = rd.normal(0,1,R.shape)
    S= (dv - np.swapaxes(dv, 1,2))/2
    eR = np.asarray([np.dot(S[i], R[i]) for i in range(S.shape[0])])
    epR = rd.normal(0,1,pR.shape)
    
    tMod0 = my_mod_init_from_Cot(Mod0, tCot)
    tMod1 = my_mod_init_from_Cot(Mod1, tCot)
    
    (nMod0, nMod1, nCot) = my_forward(Mod0, Mod1, Cot, Dt)
    (ntMod0, ntMod1, ntCot) = my_forward(tMod0, tMod1, tCot, Dt)
    (nx0,np0),(nxs,nps) = nCot['0']
    ((nx1,nR),(np1,npR)) = nCot['x,R'][0]
    (ntx0,ntp0),(ntxs,ntps) = ntCot['0']
    ((ntx1,ntR),(ntp1,ntpR)) = ntCot['x,R'][0]

##
    Dc = np.sum((ntx0-nx0)*ex0)+np.sum((ntp0-np0)*ep0)\
        +np.sum((ntxs-nxs)*exs)+np.sum((ntps-nps)*eps)\
        +np.sum((ntx1-nx1)*ex1)+np.sum((ntp1-np1)*ep1)\
        +np.sum((ntR-nR)*eR) +np.sum((ntpR-npR)*epR)
        
    grad = {'0':[(ex0,ep0),(exs,eps)], 'x,R':[((ex1,eR),(ep1,epR))]}
    ngrad = my_sub_bckwd(Mod0, Mod1, Cot, grad, 0.0001)
    ngrad = my_add_grad(my_mult_grad(ngrad, Dt), grad)
    (nex0,nep0),(nexs,neps) = ngrad['0']
    ((nex1,neR),(nep1,nepR)) = ngrad['x,R'][0]
    
    
    
    dc = np.sum(nex0*(tx0-x0))+np.sum(nep0*(tp0-p0))\
        + np.sum(nexs*(txs-xs))+np.sum(neps*(tps-ps))\
        + np.sum(nex1*(tx1-x1))+np.sum(nep1*(tp1-p1))\
        + np.sum(neR*(tR-R))+np.sum(nepR*(tpR-pR))
        
    print(Dc)
    print(dc)
    
##
# ##
#     nx1 = x1 + 0.00001*rd.normal(0,1,x1.shape)
#     (cost, dcost) = my_dxvar_cost(x1, xs,0.4)
#     (ncost, tmp) = my_dxvar_cost(nx1, xs, 0.4)
#     Dcost = ncost -cost
#     Dx1 = nx1 -x1
#     e= 0
#     for i in range(Dx1.shape[0]):
#         e += np.dot(dcost[i],Dx1[i])
#     print(e)
#     print(Dcost)
# ##

    print(my_new_ham(Mod0, Mod1, Cot))

    fig = plt.figure(5)
    plt.clf()
    my_plot2(Mod0,Mod1,Cot,fig, nx, ny, name, 0)


    (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
    plt.clf()
    my_plot2(Mod0,Mod1,Cot,fig, nx, ny, name, 0)

    for i in range(20):
        (Mod0, Mod1, Cot) = my_forward(Mod0,Mod1,Cot, dt)
        plt.clf()
        my_plot2(Mod0,Mod1,Cot,fig,nx,ny, name, i+1)

    filepng = name + "*.png"
    #os.system("convert " + filepng + " -set delay 20 -reverse " + filepng + " -loop 0 " + name + ".gif")  
    os.system("convert " + filepng + " -set delay 0 -reverse " + filepng + " -loop 0 " + name + ".gif")  
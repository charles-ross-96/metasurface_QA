# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:30:38 2021

@author: cr26
"""
def plot_binary(M,N,Angle_t,Angle_p,opt_code,ti):
    import numpy as np
    import matplotlib.pyplot as plt
    E_r_Num = M;
    E_c_Num = N; 
    from matplotlib.patches import Rectangle

    theta = np.linspace(0,90,181)/180*np.pi
    phi = np.linspace(0,180,181)/180*np.pi
    pi = np.pi
    wavelength=0.035
    d = wavelength
    plt.imshow(np.reshape(opt_code,[M,N]))

    k = 2*pi/wavelength

    X = k*d/(2*np.pi)*np.sin(theta)*np.reshape(np.cos(phi),(181,1))
    Y = k*d/(2*np.pi)*(np.sin(theta)*np.reshape(np.sin(phi),(181,1))-np.sin(ti))
    
    Ef_t = np.cos(theta)*np.reshape(np.sin(phi),(181,1))*np.sinc(X)*np.sinc(Y)
    Ef_p = np.sinc(X)*np.sinc(Y)*np.reshape(np.cos(phi),(181,1))
    
    Ef_t=abs(Ef_t)**2
    Ef_p=abs(Ef_p)**2
    
    Ef = Ef_t+Ef_p
    
    [xx,yy] = np.meshgrid(theta,phi);
    
    Afr1=0;
    Afi1=0;
    
    Ising_Value = np.ravel(opt_code)

    k = 2*pi/wavelength
    kx = k*np.sin(theta)*np.reshape(np.cos(phi),(181,1))
    ky = k*(np.sin(theta)*np.reshape((np.sin(phi)),(181,1))-np.sin(ti))
    
    for i in range(0,E_r_Num):
        for j in range(0,E_c_Num):
            Ising_index = (i)*E_c_Num+j
            ejkd_real = np.cos(kx*d*(i))*np.cos(ky*d*(j))-np.sin(kx*d*(i))*np.sin(ky*d*(j));
            ejkd_imag = np.cos(kx*d*(i))*np.sin(ky*d*(j))+np.sin(kx*d*(i))*np.cos(ky*d*(j));
            
            Afr1 = Afr1 + Ising_Value[Ising_index]*ejkd_real;
            Afi1 = Afi1 + Ising_Value[Ising_index]*ejkd_imag;
    
    
    Af1=abs(Afr1+1j*Afi1)**2; 
    
    ff1=Af1*Ef;
    # 3d plot
    #fig = plt.figure(figsize=(12,10))
    #ax = fig.add_subplot(111, projection='3d') 
    #ax.plot_surface(xx*180/np.pi, yy*180/np.pi, ff1, cmap=cm.coolwarm);
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.tick_params(axis='x',which='both', bottom=False, top=False) 
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    plt.xticks([0, 60, 120, 180], [0, 30, 60, 90])
    plt.yticks([0, 60, 120, 180],[0,60,120,180])
    rect1 = Rectangle((2*Angle_t-2,Angle_p-2),4,4, edgecolor='r', facecolor="none")
    #rect2 = Rectangle((2*Angle_t1-2,Angle_p1-2),4,4, edgecolor='r', facecolor="none")
    
    ax.add_patch(rect1)
    #ax.add_patch(rect2)
    
    ax.imshow(ff1)
def plot_quadriphase(M,N,Angle_t,Angle_p,total,ti):
    import numpy as np
    import matplotlib.pyplot as plt
    E_r_Num = M;
    E_c_Num = N; 
    from matplotlib.patches import Rectangle
    
    theta = np.linspace(0,90,181)/180*np.pi
    phi = np.linspace(0,180,181)/180*np.pi
    pi = np.pi
    wavelength=0.035
    d = wavelength
    k = 2*pi/wavelength
    plt.imshow(np.reshape(np.angle(total),[M,N]),cmap='gray')

    X = k*d/(2*np.pi)*np.sin(theta)*np.reshape(np.cos(phi),(181,1))
    Y = k*d/(2*np.pi)*(np.sin(theta)*np.reshape(np.sin(phi),(181,1))-np.sin(ti))
    
    Ef_t = np.cos(theta)*np.reshape(np.sin(phi),(181,1))*np.sinc(X)*np.sinc(Y)
    Ef_p = np.sinc(X)*np.sinc(Y)*np.reshape(np.cos(phi),(181,1))
    
    
    Ef_t=abs(Ef_t)**2
    Ef_p=abs(Ef_p)**2
    
    Ef = Ef_t+Ef_p
    
    [xx,yy] = np.meshgrid(theta,phi);
    
    Afr1=0;
    Afi1=0;
    
    #print(Ising_Value)
    k = 2*np.pi/wavelength
    kx = k*np.sin(theta)*np.reshape(np.cos(phi),(181,1))
    ky = k*(np.sin(theta)*np.reshape((np.sin(phi)),(181,1))-np.sin(ti))
        
    for i in range(0,E_r_Num):
        for j in range(0,E_c_Num):
            Ising_index = (i)*E_c_Num+j
            ejkd_real = np.cos(kx*d*(i))*np.cos(ky*d*(j))-np.sin(kx*d*(i))*np.sin(ky*d*(j));
            ejkd_imag = np.cos(kx*d*(i))*np.sin(ky*d*(j))+np.sin(kx*d*(i))*np.cos(ky*d*(j));
                
            I_c= total[Ising_index]*complex(1/2,-1/2) 
            
            Afr1 = Afr1 + I_c.real*ejkd_real-I_c.imag*ejkd_imag;
            Afi1 = Afi1 + I_c.real*ejkd_imag+I_c.imag*ejkd_real;
        
        
    Af1=abs(Afr1+1j*Afi1)**2; 
        
    ff1=Af1*Ef*4*np.pi*(d*d/wavelength)**2;
    
    
    
    
    
    
    fig = plt.figure(figsize=(3,3))
#    ax = fig.add_subplot(111, projection='3d') 
#    ax.plot_surface(xx*180/np.pi, yy*180/np.pi, ff1/(wavelength**2), cmap=cm.coolwarm);
    
    ax = fig.add_axes([0, 0, 1, 1])
    ax.tick_params(axis='x',which='both', bottom=False, top=False) 
        
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    plt.xticks([0, 60, 120, 180], [0, 30, 60, 90])
    plt.yticks([0, 60, 120, 180],[0,60,120,180])
    rect1 = Rectangle((2*Angle_t-2,Angle_p-2),4,4, edgecolor='r', facecolor="none")
      
    ax.add_patch(rect1)
    ax.imshow(ff1)
def plot_binary_2_Rx(M,N,Angle_t1,Angle_p1,Angle_t2,Angle_p2,opt_code,ti):
    import numpy as np
    import matplotlib.pyplot as plt
    E_r_Num = M;
    E_c_Num = N; 
    from matplotlib.patches import Rectangle
    
    plt.imshow(np.reshape(opt_code,[M,N]))

    E_r_Num = M;
    E_c_Num = N; 
    
    theta = np.linspace(0,90,181)/180*np.pi
    phi = np.linspace(0,180,181)/180*np.pi
    pi = np.pi
    wavelength=0.035
    d = wavelength
    

    k = 2*pi/wavelength

    X = k*d/(2*np.pi)*np.sin(theta)*np.reshape(np.cos(phi),(181,1))
    Y = k*d/(2*np.pi)*(np.sin(theta)*np.reshape(np.sin(phi),(181,1))-np.sin(ti))
    
    Ef_t = np.cos(theta)*np.reshape(np.sin(phi),(181,1))*np.sinc(X)*np.sinc(Y)
    Ef_p = np.sinc(X)*np.sinc(Y)*np.reshape(np.cos(phi),(181,1))
    
    Ef_t=abs(Ef_t)**2
    Ef_p=abs(Ef_p)**2
    
    Ef = Ef_t+Ef_p
    
    [xx,yy] = np.meshgrid(theta,phi);
    
    Afr1=0;
    Afi1=0;
    
    Ising_Value = np.ravel(opt_code)
    #print(Ising_Value)
    k = 2*pi/wavelength
    kx = k*np.sin(theta)*np.reshape(np.cos(phi),(181,1))
    ky = k*(np.sin(theta)*np.reshape((np.sin(phi)),(181,1))-np.sin(ti))
    
    for i in range(0,E_r_Num):
        for j in range(0,E_c_Num):
            Ising_index = (i)*E_c_Num+j
            ejkd_real = np.cos(kx*d*(i))*np.cos(ky*d*(j))-np.sin(kx*d*(i))*np.sin(ky*d*(j));
            ejkd_imag = np.cos(kx*d*(i))*np.sin(ky*d*(j))+np.sin(kx*d*(i))*np.cos(ky*d*(j));
            
            Afr1 = Afr1 + Ising_Value[Ising_index]*ejkd_real;
            Afi1 = Afi1 + Ising_Value[Ising_index]*ejkd_imag;
    
    
    Af1=abs(Afr1+1j*Afi1)**2; 
    
    ff1=Af1*Ef*4*np.pi*(d*d/wavelength)**2;
    #plt.imshow(ff1)
    

    
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    
    ax.tick_params(axis='x',which='both', bottom=False, top=False) 
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    plt.xticks([0, 60, 120, 180], [0, 30, 60, 90])
    plt.yticks([0, 60, 120, 180],[0,60,120,180])
    
    rect1 = Rectangle((2*Angle_t1-2,Angle_p1-2),4,4, edgecolor='r', facecolor="none")
    rect2 = Rectangle((2*Angle_t2-2,Angle_p2-2),4,4, edgecolor='r', facecolor="none")
    
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    ax.imshow(ff1)
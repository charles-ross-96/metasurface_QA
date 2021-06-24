# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:59:11 2021

@author: 17248
"""

# Optimization of M x N binary metasurface for scattering in single direction
# with some elements being broken with random probability
import neal
import collections
import numpy as np
import itertools
from joblib import Parallel, delayed
import multiprocessing
from plotting import plot_binary

M = 8
N = 8
ti = 0
Angle_p = 20
Angle_t = 20
wavelength = 0.035
d = 0.035

ti = ti * np.pi/180
Ising_num=M*N

h={}
for i in range(0,Ising_num):
    m_str = str(i)
    h[m_str] = 0
    

    
J = {}

def Jmn(m1,n1,m2,n2):
    import numpy as np
    from scipy import integrate
    pi = np.pi
    k = 2*pi/wavelength
    kd = k*d

    kx = lambda t,p: kd*np.sin(t)*np.cos(p)
    ky = lambda t,p: kd*(np.sin(t)*np.sin(p)-np.sin(ti))

    P = lambda t,p: (np.cos(t)**2*np.sin(p)**2+np.cos(p)**2)*np.sinc(kd*np.sin(t)*np.cos(p)/(2*pi))**2*np.sinc(kd*(np.sin(t)*np.sin(p)-np.sin(ti))/(2*pi))**2     
    integrand = lambda t,p: -2*(np.cos((m2-m1)*kx(t,p)+(n2-n1)*ky(t,p))*P(t,p))*np.sin(t)
        
    Angle_p_L=(Angle_p-2)*pi/180
    Angle_p_H=(Angle_p+2)*pi/180

    Angle_t_L=(Angle_t-2)*pi/180
    Angle_t_H=(Angle_t+2)*pi/180
            
    J1 = integrate.dblquad(integrand,Angle_p_L,Angle_p_H,Angle_t_L,Angle_t_H,epsabs=1.0e-12) 
    
    return J1[0]

str_mn = []
x = np.linspace(0,M*N-1,M*N)
y = np.linspace(0,M*N-1,M*N)
for (xi, yi) in itertools.product(x, y):
    #if(xi!=yi):
    str_mn.append((str(int(xi)), str(int(yi))))

comb = itertools.product(np.linspace(0,M-1,M),np.linspace(0,N-1,N),np.linspace(0,M-1,M),np.linspace(0,N-1,N))
num_cores = multiprocessing.cpu_count()
# parallel process for calculating Jmn
def coupling_coeff(c):
    m1 = c[0]
    n1 = c[1]
    m2 = c[2]
    n2 = c[3]
    if(m1*N+n1 < m2*N+n2):
         return (Jmn(m1,n1,m2,n2))
    else:
        return 0
   
results = Parallel(n_jobs=num_cores)(delayed(coupling_coeff)(c) for c in comb)

i=0
for s in str_mn:
    J[s] = results[i]
    i+=1
# Now imagine if certain array elements are broken, which we represent 
# as hard constraint that a few random elements must be -1



    
# Annealing step

sampleset = neal.SimulatedAnnealingSampler().sample_ising(h, J,num_reads=1000,num_sweeps=1000)
mat_dict = sampleset.first[0]
int_dict = {float(k) : v for k, v in mat_dict.items()}
int_dict = collections.OrderedDict(sorted(int_dict.items()))
opt_code = list(int_dict.values())

plot_binary(M,N,Angle_t,Angle_p,opt_code,ti)

break_chance = 0.1
for elem in h:
    if(np.random.rand() < break_chance):
        print(str(elem) + ' is broken')
        h[elem] = 100    

sampleset = neal.SimulatedAnnealingSampler().sample_ising(h, J,num_reads=1000,num_sweeps=1000)
mat_dict = sampleset.first[0]
int_dict = {float(k) : v for k, v in mat_dict.items()}
int_dict = collections.OrderedDict(sorted(int_dict.items()))
opt_code = list(int_dict.values())

plot_binary(M,N,Angle_t,Angle_p,opt_code,ti)


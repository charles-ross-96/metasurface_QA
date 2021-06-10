# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:53:53 2021

@author: cr26
"""

import neal
import collections
import numpy as np
import itertools
from joblib import Parallel, delayed
import multiprocessing
from plotting import plot_binary
from pyqubo import Array

M = 1
N = 12
ti = 0
Angle_p = 90
Angle_t = 20
wavelength = 0.035
d = 0.035

ti = ti * np.pi/180
Ising_num=M*N

h={}
for i in range(0,Ising_num):
    m_str = str(i)
    h[m_str] = 0
J = []

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
    if(xi!=yi):
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
J = np.zeros((M*N)**2)
i=0
for r in results:
    J[i] = r
    i+=1
J = J[J != 0]
    
s = Array.create('s', shape=(M*N), vartype='SPIN')
four_bodies = itertools.product(s,repeat=4)
two_bodies = itertools.combinations(s,2)
H2 = sum(n*s[0]*s[1] for n,s in zip(J,two_bodies))
model = H2.compile()
h,J,offset = model.to_ising()
sampleset_ising = neal.SimulatedAnnealingSampler().sample_ising(h,J,num_reads=1000,num_sweeps=1000)

opt_code = np.zeros(M*N)
mat_dict = sampleset_ising.first[0]
for i in range(0,M*N):
    str_mn = 's[' + str(i) + ']'
    opt_code[i] = mat_dict[str_mn]

plot_binary(M,N,Angle_t,Angle_p,opt_code,ti)

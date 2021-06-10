import neal
import collections
import numpy as np
import time
import itertools
from joblib import Parallel, delayed
import multiprocessing
from plotting import plot_quadriphase

M = 4
N = 4
ti = 0 
Angle_p = 30
Angle_t = 30
max_dist = 1000

wavelength = 0.035
d = 0.035
start_time = time.time()
ti = ti * np.pi/180


Ising_num=M*N
h={}
for i in range(0,Ising_num):
    m_str = str(i)
    h[m_str] = 0

J = {}
J_arr_2 = np.zeros(shape=(Ising_num,Ising_num))

def Jmn_RR(m,n,i,v):
    import numpy as np
    from scipy import integrate
    pi = np.pi
    k = 2*pi/wavelength
    kd = k*d
 
    kx = lambda t,p: kd*np.sin(t)*np.cos(p)
    ky = lambda t,p: kd*(np.sin(t)*np.sin(p)-np.sin(ti))

    P = lambda t,p: (np.cos(t)**2*np.sin(p)**2+np.cos(p)**2)*np.sinc(kd*np.sin(t)*np.cos(p)/(2*pi))**2*np.sinc(kd*(np.sin(t)*np.sin(p)-np.sin(ti))/(2*pi))**2     
    integrand = lambda t,p: -np.cos((i-m)*kx(t,p)+(v-n)*ky(t,p))*P(t,p)*np.sin(t)
    
    Angle_p_L=(Angle_p-2)*pi/180
    Angle_p_H=(Angle_p+2)*pi/180

    Angle_t_L=(Angle_t-2)*pi/180
    Angle_t_H=(Angle_t+2)*pi/180

    J1 = integrate.dblquad(integrand,Angle_p_L,Angle_p_H,Angle_t_L,Angle_t_H,epsabs=1.0e-12) 

    return J1[0] 

def Jmn_RI(m,n,i,v):
    import numpy as np
    from scipy import integrate
    pi = np.pi
    k = 2*pi/wavelength
    kd = k*d
    
    kx = lambda t,p: kd*np.sin(t)*np.cos(p)
    ky = lambda t,p: kd*(np.sin(t)*np.sin(p)-np.sin(ti))

    P = lambda t,p: (np.cos(t)**2*np.sin(p)**2+np.cos(p)**2)*np.sinc(kd*np.sin(t)*np.cos(p)/(2*pi))**2*np.sinc(kd*(np.sin(t)*np.sin(p)-np.sin(ti))/(2*pi))**2     
    
    integrand = lambda t,p: np.sin((i-m)*kx(t,p)+(v-n)*ky(t,p))*P(t,p)*np.sin(t)
    
    Angle_p_L=(Angle_p-2)*pi/180
    Angle_p_H=(Angle_p+2)*pi/180
    
    Angle_t_L=(Angle_t-2)*pi/180
    Angle_t_H=(Angle_t+2)*pi/180
        
    J1 = integrate.dblquad(integrand,Angle_p_L,Angle_p_H,Angle_t_L,Angle_t_H,epsabs=1.0e-12) 
      
    return J1[0] 
    
Ising_num = M * N
h={}
for i in range(0,2*Ising_num):
    m_str = str(i)
    h[m_str] = 0
    
J = {}

J_arr = np.zeros(shape=(2*Ising_num,2*Ising_num))    

J_RR = np.zeros(shape=(Ising_num,Ising_num))
J_RI = np.zeros(shape=(Ising_num,Ising_num))  
J_IR = np.zeros(shape=(Ising_num,Ising_num))  
J_II = np.zeros(shape=(Ising_num,Ising_num))

comb = itertools.product(np.linspace(0,M-1,M),np.linspace(0,N-1,N),np.linspace(0,M-1,M),np.linspace(0,N-1,N))
num_cores = multiprocessing.cpu_count()

def coupling_coeff_1(c):
    m1 = c[0]
    n1 = c[1]
    m2 = c[2]
    n2 = c[3]
    #str_mn = (str(m1*N+n1),str(m2*N+n2))
    if(int(m1*N+n1) != int(m2*N+n2) and m1*N+n1 < m2*N+n2):
        rr = (Jmn_RR(m1,n1,m2,n2))
        ri = (Jmn_RI(m1,n1,m2,n2))
        ir = -(Jmn_RI(m1,n1,m2,n2))
        ii = (Jmn_RR(m1,n1,m2,n2))        
    #J[str_mn] = J_arr_2[int(m1*N+n1),int(m2*N+n2)]
        return rr,ri,ir,ii
    else:
        return 0,0,0,0

results = Parallel(n_jobs=num_cores)(delayed(coupling_coeff_1)(c) for c in comb)
arr_results = np.array(results)
J_RR = np.reshape(arr_results[:,0],[M*N,M*N])
J_RI = np.reshape(arr_results[:,1],[M*N,M*N])
J_IR = np.reshape(arr_results[:,2],[M*N,M*N])
J_II = np.reshape(arr_results[:,3],[M*N,M*N])

J_arr[0:M*N,0:M*N]  =  J_RR
J_arr[0:M*N,M*N:2*M*N] = J_RI
J_arr[M*N:2*M*N,0:M*N] = J_IR
J_arr[M*N:2*M*N,M*N:2*M*N] = J_II   

for Ising_index_1 in range(0,2*Ising_num):
    for Ising_index_2 in range(0,2*Ising_num):                    
        if(Ising_index_1 < Ising_index_2):    
            str_mn = (str(Ising_index_1),str(Ising_index_2))                      
            J[str_mn] = J_arr[Ising_index_1,Ising_index_2]
                                     
    
sampleset = neal.SimulatedAnnealingSampler().sample_ising(h, J,num_reads=1000,num_sweeps=1000) 
mat_dict = sampleset.first[0]
int_dict = {int(k) : v for k, v in mat_dict.items()}
int_dict = collections.OrderedDict(sorted(int_dict.items()))
opt_code = list(int_dict.values())

Ising_Value = np.ravel(opt_code)
sRR_1 = Ising_Value[0:M*N]
sII_1 = Ising_Value[M*N:2*M*N]
total = sRR_1 + 1j*sII_1 

plot_quadriphase(M,N,Angle_t,Angle_p,total,ti)
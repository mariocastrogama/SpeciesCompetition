# -*- coding: utf-8 -*-
"""
% Corresponds to Figure 4 of
%
% Huisman J. and Weissing F.J. (1999) 
% Biodiversity of plankton by species oscillations and chaos.
% Letters to Nature 402(November)407-410. 

Created on Sun Oct 27 19:20:25 2021

@author: Castro-Gama Mario
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def get_KC(id_problem):
    # K: Half-saturation constant for resource j,
    # C: is the content of resource j in species i
    if id_problem == 3: # case 3, 6 species, 3 resources
        K = [[1.00, 0.90, 0.30, 1.04, 0.34, 0.77], 
             [0.30, 1.00, 0.90, 0.71, 1.02, 0.76], 
             [0.90, 0.30, 1.00, 0.46, 0.34, 1.07]]
         
        C = [[00.04, 0.07, 0.04, 0.10, 0.03, 0.02],
             [0.08, 0.08, 0.10, 0.10, 0.05, 0.17], 
             [0.14, 0.10, 0.10, 0.16, 0.06, 0.14]]
    elif id_problem == 4: #case 4, 9 species, 3 resources
        K = [[1.00, 0.75, 0.25, 0.70, 0.20, 0.65, 0.68, 0.38, 0.46],
             [0.25, 1.00, 0.75, 0.20, 1.01, 0.55, 0.83, 1.10, 0.85],
             [0.75, 0.25, 1.00, 1.10, 0.70, 0.95, 0.60, 0.50, 0.77]]

        C = [[0.10, 0.20, 0.15, 0.05, 0.01, 0.40, 0.30, 0.20, 0.25],
             [0.15, 0.10, 0.20, 0.15, 0.30, 0.35, 0.25, 0.02, 0.35],
             [0.20, 0.15, 0.10, 0.25, 0.05, 0.20, 0.40, 0.15, 0.10]]
    elif id_problem == 5: #case 5, 12 species, 5 resources
        K = np.array([[0.39, 0.34, 0.30, 0.24, 0.23, 0.41, 0.20, 0.45, 0.14, 0.15, 0.38, 0.28],
             [0.22, 0.39, 0.34, 0.30, 0.27, 0.16, 0.15, 0.05, 0.38, 0.29, 0.37, 0.31],
             [0.27, 0.22, 0.39, 0.34, 0.30, 0.07, 0.11, 0.05, 0.38, 0.41, 0.24, 0.25],
             [0.30, 0.24, 0.22, 0.39, 0.34, 0.28, 0.12, 0.13, 0.27, 0.33, 0.04, 0.41],
             [0.34, 0.30, 0.22, 0.20, 0.39, 0.40, 0.50, 0.26, 0.12, 0.29, 0.09, 0.16]])

        C = np.array([[0.04, 0.04, 0.07, 0.04, 0.04, 0.22, 0.10, 0.08, 0.02, 0.17, 0.25, 0.03],
             [0.08, 0.08, 0.08, 0.10, 0.08, 0.14, 0.22, 0.04, 0.18, 0.06, 0.20, 0.04],
             [0.10, 0.10, 0.10, 0.10, 0.14, 0.22, 0.24, 0.12, 0.03, 0.24, 0.17, 0.01],
             [0.05, 0.03, 0.03, 0.03, 0.03, 0.09, 0.07, 0.06, 0.03, 0.03, 0.11, 0.05],
             [0.07, 0.09, 0.07, 0.07, 0.07, 0.05, 0.24, 0.05, 0.08, 0.10, 0.02, 0.04]])

    return K, C

def competition_species(t,NR,
                        n,
                        k,
                        D,
                        r,
                        m,
                        S,
                        C,
                        K):

    
    #model of competition of species  
    N = NR[:n].copy().reshape((n,))   # Species values (population abundance)
    R = NR[n:].copy().reshape((k,))  # Resources values (resource available)
      
    miu = np.zeros((n,));
    dNRdt = np.zeros((n+k,));
    
    for i in range(n):
        miu[i]   = specific_growthrate(r[i],R,K[:,i]);
        dNRdt[i] = N[i]*(miu[i] - m[i]);
     
      
    for j in range(k):
        dNRdt[n+j] = D*(S[j] - R[j]) - sum(C[j][:n]*miu*N);
    
    return dNRdt


def specific_growthrate(r,R,K):
    #  Liebig's "law of the minimum"
    # Huisman and Weissing (199) Biodiversity ... Nature
    #
    # 
    # miu = specific growth rate of species i as function of resource
    # availabilities
    # ri  = maximum specific growth rate of species i
    # Kji = half-saturation constant for resource j of species i
    # Rj  = Resource j available 

    miu = min(r*R/(K+R))
    return miu

def plot_results(axs,t,NR,n,k):
    
    t_factor = 1.0/365.25
    for ii in range(n):
        # if ii in [ 1,  2,  4]:
        #     axs[0].plot(t*t_factor,NR[ii,:])
        # elif ii in [ 0,  3,  6]:
        #     axs[1].plot(t*t_factor,NR[ii,:])
        # elif ii in [ 5,  7,  8]:
        #     axs[2].plot(t*t_factor,NR[ii,:])
        # elif ii in [ 9, 10, 11]:
        #     axs[3].plot(t*t_factor,NR[ii,:])
        
        if ii in [0, 1, 2, 3, 4, 5]:
            axs[0].plot(t*t_factor,NR[ii,:])
        elif ii in [6, 7, 8, 9, 10, 11]:
            axs[1].plot(t*t_factor,NR[ii,:])
            
    axs[-1].set_xlim(0,50)
    return 'plot finished'


id_problem = 5                       # 12 species, 5 resources 

## Enter species 1-5 first part run for 1000 days
n = 5                              # Number of Species   i = 1,...,n
k = 5                              # Number of Resources j = 1,...,k
D = 0.25                           # [d^-1] System turnover rate -> T = 1/D ~ 4days;
r = np.ones((n,))                  # [%] maximum specific growth rate of species i
m = 0.25 + np.zeros((n,))          # [%] Specific mortality rate of species i
K, C = get_KC(id_problem)          # K: Half-saturation constant for resource j, 
                                   # C: is the content of resource j in species i
S = np.array([6.00, 10.00, 14.00, 4.00, 9.00]) # Supply concentration of resource j 

# Initial Condition
tmp1 = 0.1 + 0.01*(np.arange(1,n+1))
tmp2 = 0.1 + np.zeros((7,))
No  = np.concatenate((tmp1, tmp2),axis=0)
Ro  = S.copy()
NRo = np.concatenate((tmp1, Ro.copy()),axis=0)

# Integrate the Species Competition equations
tmin = 0
tmax = 1000
soln = solve_ivp(competition_species, 
                 (tmin, tmax), 
                 (NRo), 
                 args = (n, k, D, r, m, S, C, K),
                 dense_output=True)
# Interpolate solution onto the time grid, t.
t1 = np.linspace(0, tmax, tmax)
NR1 = soln.sol(t1)
print('finished 1-5')

### Enter species 6-8 into the ecosystem at t=1000, this runs for 2000 days
n = 8
r = np.ones((n,))                  # [%] maximum specific growth rate of species i 
m = 0.25 + np.zeros((n,))          # [%] Specific mortality rate of species i
tmp1 = NR1[:6,-1].copy()           # final values of previous simulation are initial values for this one
tmp2 = NR1[6:,-1].copy()           # final values of resources are the initial for next simulation
NRo = np.concatenate((tmp1,No[5:n],tmp2),axis=0)

# Integrate the Species Competition equations
tmin = 0
tmax = 2000
soln = solve_ivp(competition_species, 
                 (tmin, tmax), 
                 (NRo), 
                 args = (n, k, D, r, m, S, C, K),
                 dense_output=True)
# Interpolate solution onto the time grid, t.
t2 = np.linspace(0, tmax, tmax)
NR2 = soln.sol(t2)
print('finished 6-8')

### Enter species 9-10 into the ecosystem at t=3000, and run for 2000 days
n = 10
r = np.ones((n,))                  # [%] maximum specific growth rate of species i 
m = 0.25 + np.zeros((n,))          # [%] Specific mortality rate of species i
tmp1 = NR2[:9,-1].copy()           # final values of previous simulation are initial values for this one
tmp2 = NR2[9:,-1].copy()           # final values of resources are the initial for next simulation
NRo = np.concatenate((tmp1,No[8:n],tmp2),axis=0)

# Integrate the Species Competition equations
tmin = 0
tmax = 2000
soln = solve_ivp(competition_species, 
                 (tmin, tmax), 
                 (NRo), 
                 args = (n, k, D, r, m, S, C, K),
                 dense_output=True)
# Interpolate solution onto the time grid, t.
t3 = np.linspace(0, tmax, tmax)
NR3 = soln.sol(t3)
print('finished 9-10')

### Enter species 11-12 into the ecosystem at t=5000, and run for 13,257 days
n = 12
r = np.ones((n,))                  # [%] maximum specific growth rate of species i 
m = 0.25 + np.zeros((n,))          # [%] Specific mortality rate of species i
tmp1 = NR3[:11,-1].copy()          # final values of previous simulation are initial values for this one
tmp2 = NR3[11:,-1].copy()          # final values of resources are the initial for next simulation
NRo = np.concatenate((tmp1,No[10:n],tmp2),axis=0)

# Integrate the Species Competition equations
tmin = 0
tmax = 13257
soln = solve_ivp(competition_species, 
                 (tmin, tmax), 
                 (NRo), 
                 args = (n, k, D, r, m, S, C, K),
                 dense_output=True)
# Interpolate solution onto the time grid, t.
t4 = np.linspace(0, tmax, tmax)
NR4 = soln.sol(t4)
print('finished 11-12')


# plot the results Figure 4 of Huisman and Weissing (1999)
fig, axs = plt.subplots(num=1, 
                        nrows = 2, 
                        ncols = 1, 
                        sharex=True, 
                        figsize=(10,6))
plot_results(axs,t1,NR1[:5,:],5,k)
plot_results(axs,t2+1000.0,NR2[:8,:],8,k)
plot_results(axs,t3+3000.0,NR3[:10,:],10,k)
plot_results(axs,t4+5000.0,NR4[:12,:],12,k)

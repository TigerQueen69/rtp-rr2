"""
Created on Fri Nov 21 17:01 2025

@author: feyza.beziane
"""

## ----------- Libraries -----------
import numpy as np
import matplotlib.pyplot as plt
import ast
import time
from scipy.stats import linregress
from simcode.src import core as c
from simcode.src import aux  as x
from simcode.src import vars as var

## ----------- From file -----------

def traj_from_file(FileName):
    '''Takes a text file as an argument and fill an array with the values in it
    returns an array of 2 dimension with times and value of msd at each of these times'''
    rel_x, rel_y, abs_x, abs_y = [], [], [], []
    with open(FileName, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            vals = line.split()
            x, y, X, Y = map(float, vals)
            rel_x.append(x)
            rel_y.append(y)
            abs_x.append(X)
            abs_y.append(Y)
    trajrel, trajabs = np.array((rel_x, rel_y)), np.array((abs_x, abs_y))
    return trajrel, trajabs

def msd_from_file(FileName):
    '''Takes a text file as an argument and fill an array with the values in it
    returns an array of 2 dimension with times and value of msd at each of these times'''
    file = open(FileName)
    time = []
    msd = []
    for line in file:
        if not line.startswith('#'):
            line = line.strip()
            t, m = line.split("\t",1)
            try:    
                t = ast.literal_eval(t) # str to int,bool,list etc.
                m = ast.literal_eval(m)
            except (ValueError, SyntaxError):
                pass
            time.append(t)
            msd.append(m)
    return time, msd

## ----------- Initialization -----------

def init(par, traj, msd, rac):
    var.parfile = var.path + rac + '.par'
    var.trajfile = var.path + rac + '.traj'
    var.msdfile  = var.path + rac + '.msd'
    x.sim_init(var.parfile, par, traj, msd)
    var.p = x.particle_create(par)
    x.particle_init(var.p,par)
    return

## ----------- Trajectory -----------

def traj_plot(par,trajrel,trajabs, Ttrajmin, Ttrajmax):
    ''' Takes parameters (instance of class par_c), relative and absolute trajectories
    and initial and final time as arguments and return the plots of the relative (within
    a box which size is given by par) and absolute trajectories of the particle'''
    #nmin, nmax = int(Ttrajmin/par.dT), int(Ttrajmax/par.dT)
        # Convert times to indices
    nmin = max(0, int(Ttrajmin / par.dT))
    nmax = min(trajrel.shape[1], int(Ttrajmax / par.dT))  # ⚡ never exceed array size

    if nmax <= nmin:
        raise ValueError(f"Ttrajmax={Ttrajmax} is too small or trajectory too short")
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(trajrel[0,nmin:nmax], trajrel[1,nmin:nmax],'.', lw=1)
    ax[0].scatter(trajrel[0,nmin], trajrel[1,nmin], color='green', label="Start")
    ax[0].scatter(trajrel[0,nmax-1], trajrel[1,nmax-1], color='red', label="End")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_xlim(0,par.L)
    ax[0].set_ylim(0,par.L)
    ax[0].set_title("Relative trajectory")
    ax[0].legend()
    #ax[0].axis("equal")
    ax[1].plot(trajabs[0,nmin:nmax], trajabs[1,nmin:nmax],'.', lw=1)
    ax[1].scatter(trajabs[0,nmin], trajabs[1,nmin], color='green', label="Start")
    ax[1].scatter(trajabs[0,nmax-1], trajabs[1,nmax-1], color='red', label="End")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].set_title("Absolute trajectory")
    ax[1].legend()
    ax[1].axis("equal")
    plt.show()

def traj_plot_step(par, trajrel, trajabs, Ttrajmin, Ttrajmax, step=1):
    '''
    Takes parameters (instance of class par_c), relative and absolute trajectories
    and initial and final time as arguments and returns the plots of the relative
    (within a box which size is given by par) and absolute trajectories of the particle.
    Points are numbered every "step".
    '''
    nmin, nmax = int(Ttrajmin / par.dT), int(Ttrajmax / par.dT)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(trajrel[0, nmin:nmax],trajrel[1, nmin:nmax],'.',lw=1)
    ax[0].scatter(trajrel[0, nmin], trajrel[1, nmin], color='green', label="Start")
    ax[0].scatter(trajrel[0, nmax-1], trajrel[1, nmax-1], color='red', label="End")
    for i in range(nmin, nmax, step):
        ax[0].annotate(str(i),(trajrel[0, i], trajrel[1, i]),fontsize=8,xytext=(3, 3),textcoords="offset points")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_xlim(0, par.L)
    ax[0].set_ylim(0, par.L)
    ax[0].legend()
    ax[1].plot(trajabs[0, nmin:nmax],trajabs[1, nmin:nmax],'.',lw=1)
    ax[1].scatter(trajabs[0, nmin], trajabs[1, nmin], color='green', label="Start")
    ax[1].scatter(trajabs[0, nmax-1], trajabs[1, nmax-1], color='red', label="End")
    for i in range(nmin, nmax, step):
        ax[1].annotate(str(i),(trajabs[0, i], trajabs[1, i]),fontsize=8,xytext=(3, 3),textcoords="offset points")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

## ----------- MSD -----------

def msdthbal(p,t):
    '''Theoretical behaviour of MSD for balistic case'''
    v = p.mode.v
    return v**2*t**2

def msdthabp(p,t):
    '''Theoretical behaviour of MSD for ABP case'''
    pm = p.mode
    dr = pm.dr
    v = pm.v
    return 2 * v**2/dr**2 * (np.exp(-dr * t) - 1 + dr * t)

def msdthrtp(p, t):
    tau = p.mode.runtau
    result = 2*p.mode.v**2*tau*(t+(-1+np.exp(-t*(p.mode.dr+1/tau)))*tau+p.mode.dr*t*tau)/(1+p.mode.dr*tau)**2
    return result

def msd_plot(p,par,msd,filename):
    '''Take an instance of parameters class, time and values an arra'''
    t, msdsim = msd_from_file(filename)
    t = np.array(t)
    if   p.mode.runtype == 1 and p.mode.drind == 0 and p.mode.tumind == 0:  # balistic
        msdth = msdthbal(p,t)
    elif p.mode.runtype == 1 and p.mode.drind == 1 and p.mode.tumind == 0:  # ABP
        msdth = msdthabp(p,t)
    elif p.mode.runtype == 1 and p.mode.tumind == 1 and p.mode.tumtype == 1 and par.moden == 1:     # unimodal RTP
        msdth = msdthrtp(p,t)
    else:
        print('unidentified mode')
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f"Plot MSD and Behaviour with T = {par.T}", fontsize=14)
        ax.plot(t, msdsim, marker='.', markersize=1, linestyle='--', label='simulation')
        ax.set_xlabel("Time")
        ax.set_ylabel("MSD")
        ax.legend()
        plt.show()
        return
    n = int(msd.Tmax/msd.dT)
    msdsim, msdth, t = msdsim[1:n], msdth[1:n], t[1:n]
    fig, ax = plt.subplots(1,2)
    fig.suptitle(f"Plot MSD and difference with T = {par.T}", fontsize=14)
    ax[0].plot(t, msdsim, marker='.', markersize=1, linestyle='--', label='simulation')
    ax[0].plot(t, msdth, marker='.', markersize=1, linestyle='--',label=r'theoretical')
    difference = (msdsim - msdth)/msdth * 100
    if np.abs(max(difference)) <= 1:
        print(f"The test is passed with a maximum absolute relative difference equal to {np.abs(max(difference))}")
    else : 
        print(f"The test is not passed with a maximum absolute relative difference equal to {np.abs(max(difference))}")
    ax[1].plot(t, difference, marker='.', markersize=1, linestyle='--', label=r'Difference')
    for i in range(2):
        ax[i].set_xlabel("Time")
        ax[i].legend()
    ax[0].set_ylabel("MSD")
    ax[1].set_ylabel("MSD (%)")
    plt.show()

# def msd_plot_rrot(p,par,msd,filename):
#     '''Take an instance of particle, an instance of parameters class, and '''
#     t, msdsim = msd_from_file(filename)
#     t = np.array(t)
#     n = int(msd.Tmax/msd.dT)
#     msdsim, t = msdsim[1:n], t[1:n]
#     D = msdsim/(4*t)
#     fig, ax = plt.subplots(1,2)
#     fig.suptitle(f"MSD and diffusion with T = {par.T}", fontsize=14)
#     ax[0].plot(t, msdsim, marker='.', markersize=1, linestyle='--', label='simulation')
#     ax[0].set_xlabel("Time")
#     ax[0].set_ylabel("MSD")
#     ax[1].plot(t, D, marker='.', markersize=1, linestyle='--', label='simulation')
#     ax[1].set_xlabel("Time")
#     ax[1].set_ylabel("D")
#     plt.show()

def msd_plot_rrot(p, par, msd, filename):
    '''Take an instance of particle, an instance of parameters class'''
    t, msdsim = msd_from_file(filename)
    t = np.array(t)
    n = int(msd.Tmax / msd.dT)
    msdsim, t = msdsim[1:n], t[1:n]
    D = msdsim / (4 * t)
    n_plateau = max(1, int(0.2 * len(D)))
    D_plateau = np.mean(D[-n_plateau:])
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(f"MSD and diffusion with T = {par.T}", fontsize=14)
    ax[0].plot(t, msdsim, marker='.', markersize=1, linestyle='--', label='simulation')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("MSD")
    ax[1].plot(t, D, marker='.', markersize=1, linestyle='--', label='simulation')
    ax[1].axhline(D_plateau, color='r', linestyle='-', label=f'plateau ≈ {D_plateau:.3e}')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("D")
    ax[1].legend()
    delta = 0.1
    y_min = D_plateau * (1 - delta)
    y_max = D_plateau * (1 + delta)
    ax[1].set_ylim(y_min, y_max)
    plt.show()

def plotloglog(p,par,msd,filename):
    '''Take an instance of parameters class, time and values an arra'''
    t_init = time.time()
    t, msdsim = msd_from_file(filename)
    t = np.array(t)
    if p.mode.runtype == 1 and p.mode.drind == 0 and p.mode.tumind == 0:
        msdth = msdthbal(p,par)
    elif p.mode.runtype == 1 and p.mode.drind == 1 and p.mode.tumind == 0:
        msdth = msdthabp(p,t)
    elif p.mode.runtype == 1 and p.mode.tumind == 1 and par.moden == 1:
        msdth = msdthrtp(p,t)
    else:
        print('unidentified mode')
        fig, ax = plt.subplots(1,1)
        fig.suptitle(f"Plot MSD and difference with T = {par.T}", fontsize=14)
        ax.plot(t, msdsim, marker='.', markersize=1, linestyle='--', label='simulation')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Time")
        ax.set_ylabel("MSD")
        ax.legend()
        plt.show()
        return
    n = int(msd.Tmax/msd.dT)
    msdsim, msdth, t = msdsim[:n], msdth[:n], t[:n]
    fig, ax = plt.subplots(1,2)
    fig.suptitle(f"Plot MSD and Behaviour with T = {par.T}", fontsize=14)
    ax[0].plot(t[1:], msdsim[1:], marker='.', markersize=1, linestyle='-', label='simulation')
    ax[0].plot(t[1:], msdth[1:], marker='.', markersize=1, linestyle='--',label=r'theoretical')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    difference = np.abs(msdsim - msdth)/msdth                            
    ax[1].plot(t[1:], difference[1:], marker='.', markersize=1, linestyle='', label=r'Difference')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    for i in range(2):
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("MSD")
        ax[i].legend()
    plt.show()
    t_plotlog = time.time()
    print(f"Time for ploting MSD in loglog : {t_plotlog - t_init }s")

## ----------- Diffusion -----------

def diffusion(v, lamb, mu, om):
    return v**2 * mu * ((lamb + mu)**2 + om**2)/(2 * lamb * (lamb + mu) * om**2)

def diff_check(p,par,msd,filename):
    '''Take an instance of particle, an instance of parameters class, and '''
    t, msdsim = msd_from_file(filename)
    t = np.array(t)
    n = int(msd.Tmax/msd.dT)
    msdsim, t = msdsim[1:n], t[1:n]
    D = msdsim/(4*t)
    fig, ax = plt.subplots(1,1)
    fig.suptitle(f"Plot MSD and difference with T = {par.T}", fontsize=14)
    ax.plot(t, D, marker='.', markersize=1, linestyle='--', label='simulation')
    ax.set_xlabel("Time")
    ax.set_ylabel("D")
    plt.show()

def diff_from_msd(p,par,msd,filename):
    t, msdsim = msd_from_file(filename)
    t = np.array(t)
    msdsim = np.array(msdsim)
    tstart,tend = msd.Tmax-msd.Tmax/2, msd.Tmax-1
    start, end = int(tstart/msd.dT), int(tend/msd.dT)
    t_lin = t[start:end]
    msd_lin = msdsim[start:end]
    Dplot = msdsim/(4*t)
    if np.abs((Dplot[start] - Dplot[end])/Dplot[end])*100 < 1:
        slope, intercept, r_value, p_value, std_err = linregress(t_lin, msd_lin)
        D = slope / 4
        return D

def diff_from_msd_plot(p,par,msd,filename):
    t, msdsim = msd_from_file(filename)
    t = np.array(t)
    msdsim = np.array(msdsim)
    valid = t > 0
    t = t[valid]
    msdsim = msdsim[valid]
    tstart,tend = msd.Tmax-msd.Tmax/2, msd.Tmax-1
    start, end = int(tstart/msd.dT), int(tend/msd.dT)
    t_lin = t[start:end]
    msd_lin = msdsim[start:end]
    Dplot = msdsim/(4*t)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(t, Dplot, marker='.', markersize=1, linestyle='--', label='simulation')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("D")
    ax[1].plot(t, msdsim, label='MSD simulation')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("MSD")
    if np.abs((Dplot[start] - Dplot[end])/Dplot[end])*100 < 1:
        print(f"plateau with an error of {np.abs(Dplot[start] - Dplot[end])*100} %")
        slope, intercept, r_value, p_value, std_err = linregress(t_lin, msd_lin)
        D = slope / 4
        print(f"The diffusive behaviour is reached with a diffusion coefficient from the simulation D={D}")
        ax[1].plot(t_lin, intercept + slope*t_lin, 'r--', label=f'Linear fit, D={D:.3f}')
        return D
    else:
        print("The diffusive regime is not reached yet...")
    plt.legend()
    plt.show()







# def diff_from_msd_plot(tstart, tend, p, par, msd, filename, plateau_tol=0.001, window=50): 
#     """ This function plots the evolution of MSD/(4t) and the MSD over time
#     if MSD/(4t) reaches a plateau (an evolution lower than 5% in a specific window)
#     then the diffusive behaviour is reached and the MSD is fited
#     to obtain the diffusion coefficient"""        
#     t, msdsim = msd_from_file(filename)
#     t = np.array(t)
#     msdsim = np.array(msdsim)
#     valid = t > 0
#     t = t[valid]
#     msdsim = msdsim[valid]
#     Dplot = msdsim / (4 * t)
#     if len(Dplot) < window:
#         print("Not enough point to detect the plateau")
#         return None
#     rolling_mean = np.convolve(Dplot, np.ones(window)/window, mode='valid')
#     rel_var = np.abs(np.diff(rolling_mean)) / rolling_mean[:-1]
#     plateau_indices = np.where(rel_var < plateau_tol)[0]
#     if len(plateau_indices) == 0:
#         print("No plateau → no fit.")
#         do_fit = False
#     else:
#         print("Plateau detected → linear fit.")
#         do_fit = True
#     if do_fit:
#         start = int(tstart / msd.dT)
#         end = int(tend / msd.dT)
#         t_lin = t[start:end]
#         msd_lin = msdsim[start:end]
#         slope, intercept, r_value, p_value, std_err = linregress(t_lin, msd_lin)
#         D = slope / 4
#         print(f"The diffusion from the simulation is D={D}")
#     else:
#         D = None
#     fig, ax = plt.subplots(1, 2)
#     ax[0].plot(t, Dplot, marker='.', markersize=1, linestyle='--', label='simulation')
#     ax[0].set_xlabel("Time")
#     ax[0].set_ylabel("D(t)")
#     ax[1].plot(t, msdsim, label='MSD simulation')
#     if do_fit:
#         ax[1].plot(t_lin, intercept + slope * t_lin, 'r--', label=f'Linear fit, D={D:.3f}')
#     ax[1].set_xlabel("Time")
#     ax[1].set_ylabel("MSD")
#     ax[0].legend()
#     ax[1].legend()
#     plt.show()
#     return D





def comparison(Dth,Dsim):
    if Dsim != None:
        r = np.abs(Dth-Dsim)/Dth
        if r <= 0.1:
            print(f"The simulation fits the prediction with a relative error equal to {r}")
        else:
            print(f"The simulation does not fit the prediction, the relative error is {r}")
    else:
        print("No simulated diffusion coefficient computed")

def comparison_plot(v, lamb, om, par_list, Dsim_list):
    Dth = np.array([diffusion(v, lamb, mu, om) for mu in par_list])
    fig, ax = plt.subplots(1,2)
    fig.suptitle(f"Diffusion coefficient", fontsize=14)
    ax[0].plot(par_list, Dth, marker='.', markersize=1, linestyle='--', label='simulation')
    ax[0].plot(par_list, Dsim_list, marker='.', markersize=1, linestyle='--',label=r'theoretical')
    difference = (Dsim_list - Dth)/Dth * 100
    ax[1].plot(par_list, difference, marker='.', markersize=1, linestyle='--', label=r'Difference')
    for i in range(2):
        ax[i].set_xlabel(r"$\mu")
        ax[i].legend()
    ax[0].set_ylabel("D")
    ax[1].set_ylabel("Relative difference (%)")
    plt.show()
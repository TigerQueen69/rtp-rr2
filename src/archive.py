"""
Created on Thue Dec 2 09:35 2025

@author: feyza.beziane
"""

## ------- Libraries -------
import numpy as np
import matplotlib.pyplot as plt
import simcode.src.aux as x
import simcode.src.core as c
import time

## ------- Initialization -------

def parameters_from_file(FileName, section_to_read: str):
    '''Takes a text file and a section to read as arguments and fill an empty dict
    par with the keys and values in the wanted section of the file'''
    # read parameters from a file into a dictionnary
    file = open(FileName)
    par = {} # empty dictionnary
    in_section = False
    for line in file:
        line = line.strip() # erases useless spaces and tabs
        if not line or line.startswith("#"): # verify if we enter a new section
            if line.lower().startswith(f"#{section_to_read.lower()}"): 
            # if it's the wanted section
                in_section = True
            else:
                if in_section:
                    break # if we enter an other section
            continue
        # if we are in the following lines of the good section
        if in_section:
            if "=" in line:
                key, value = line.split("=",1)      
                # makes 1 split (seperator "=") of the line into two lists of strings
                try:    
                    value = ast.literal_eval(value) # str to int,bool,list etc.
                except (ValueError, SyntaxError):
                    pass
                par[key.strip()] = value            # fill the dict with keys and values
    # convert read parameters into desired variable
    #par["mode"]          = str  (par["mode"])
    par["emode"]         = str  (par["emode"])
    par["v"]             = float(par["v"])
    par["dim"]           = int  (par["dim"])
    par["T"]             = float(par["T"])
    par["msdT"]          = float(par["msdT"])
    par["dT"]            = float(par["dT"])
    par["msddT"]         = float(par["msddT"])
    par["L"]             = float(par["L"])
    par["Dr"]            = float(par["Dr"])
    par["Dt"]            = float(par["Dt"])
    par["tumbling_rate"] = float(par["tumbling_rate"])
    par["path"]          = str  (par["path"])
    par["msdfile"]       = str  (par["msdfile"])
    par["trajfile"]      = str  (par["trajfile"])
    par["seed"]          = int  (par["seed"])
    return par

## ------- Trajectory -------

def move(p, par):
    '''Takes a particle and parameters as arguments and
    applies to the particle an elementary move depending on the strategy
    of displacement
    Applies to balistic and ABP'''
    mode = par.mode
    if mode == 'bal':
        p.x += par.v * par.dT * p.vec
    elif mode == 'abp':
        gausst = np.random.normal(0,1,2)
        gaussr = np.random.normal(0,1)
        vec = np.array([np.cos(p.angle),np.sin(p.angle)])
        p.x += par.v * par.dT * vec + par.Dtnoise*gausst
        p.angle += par.Drnoise*gaussr
    else:
        print("unidentified mode")
    return

def generate_trajrelabs(p,par):
    '''Generates a trajectory obtained with the relative and
     absolute positions of a particle given some parameters
     Applies to balistic and ABP'''
    t_init = time.time()
    trajrel = np.zeros((par.dim, par.n_step))
    trajabs = np.zeros((par.dim, par.n_step))
    trajrel[:, 0] = p.x
    trajabs[:, 0] = x.absolute_position(p, par)
    for t in range(1,par.n_step):
        move(p,par)
        x.apply_boundaryconditions(p, par)
        trajrel[:,t] = p.x
        trajabs[:,t] = x.absolute_position(p,par)
    t_traj = time.time()
    print(f"Time of generating relative and absolute trajectory = {t_traj - t_init }s")
    return trajrel, trajabs

# rtp1 had a problem with np.arange --> trajectory going back to zero randomly 
# before continuing the expected trajectory
def rtp1(p,par):
    '''Takes a particle and parameters as arguments and applies 
    rt strategy and returns the relative and absolute trajectory of the particle'''
    trajrel, trajabs = np.ones((par.dim, par.n_step)), np.zeros((par.dim, par.n_step))
    trajrel[:, 0] = p.x
    trajabs[:, 0] = x.absolute_position(p, par)
    if par.mode == 'rtp':
        p.t_tumble = x.generate_run_time(par)            # initialization of t_tumble in an other value than 100T
    for t in np.arange(0,par.T - par.dT, par.dT):
        c.evolve(p, par, t, t + par.dT)
        trajrel[:,int((t+par.dT)/par.dT)] = p.x
        trajabs[:,int((t+par.dT)/par.dT)] = x.absolute_position(p,par)
    return trajrel, trajabs

## Modification in the saving of position in trajectory arrays
# Creation of a function traj_save
def rtp2(p,par):
    '''Takes a particle and parameters as arguments and applies 
    rt strategy and returns the relative and absolute trajectory of the particle'''
    trajrel, trajabs = np.ones((par.dim, par.n_step)), np.zeros((par.dim, par.n_step))
    x.apply_boundaryconditions(p,par)
    trajrel[:, 0] = p.x
    trajabs[:, 0] = x.absolute_position(p, par)
    if par.mode == 'rtp':
        p.t_tumble = c.generate_run_time(par)            # initialization of t_tumble in an other value than 100T
    for n in range(1,par.n_step): # loop on step
        t1, t2 = (n - 1) * par.dT, n * par.dT
        c.evolve(p, par, t1, t2)
        trajrel[:,n] = p.x
        trajabs[:,n] = x.absolute_position(p,par)
    return trajrel, trajabs

## ------- Computations -------

def traj_msd_arch(traj, par):
    '''Computes the MSD at several time. Returns a list of values of MSD for each time'''
    n_step = par.n_step
    msd = np.zeros((2,n_step))
    for t in range(n_step):
        for d in range(par.dim):
            msd[1,t] += (traj[d,t]-traj[d,0])**2
        msd[0,t] = t * par.dT
    return msd

def traj_msd(traj,par):
    '''Computes msd with many intervals'''
    t_init = time.time()
    n = par.n_traj
    nlagmax = int(par.msdT/par.dT)
    msd = np.zeros((par.dim,n))
    for nlag in range(0,nlagmax):
        msd[0,nlag] = nlag * par.dT
        S = 0.
        for nstart in range(n-nlag):
            for d in range(par.dim):
                S += (traj[d,nstart+nlag] - traj[d,nstart])**2
        msd[1,nlag] = S/(n-nlag)
    t_msd = time.time()
    print(f"Time of computation of msd : {t_msd - t_init }s")
    return msd

## ------- Analysis -------

def traj_plot(par,trajrel,trajabs):
    ''' Takes a particle (instance of class particle_c) and parameters (instance of class par_c)
    as arguments and return the relative (within a box which size is given by par)
    and absolute trajectories of the particle'''
    fig, ax = plt.subplots(1,2)
    ax[0].plot(trajrel[0,:100], trajrel[1,:100],'-', lw=1)
    ax[0].scatter(trajrel[0,0], trajrel[1,0], color='green', label="Start")
    ax[0].scatter(trajrel[0,99], trajrel[1,99], color='red', label="End")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_xlim(0,par.L)
    ax[0].set_ylim(0,par.L)
    ax[0].legend()
    #ax[0].axis("equal")
    ax[1].plot(trajabs[0,:100], trajabs[1,:100],'-', lw=1)
    ax[1].scatter(trajabs[0,0], trajabs[1,0], color='green', label="Start")
    ax[1].scatter(trajabs[0,99], trajabs[1,99], color='red', label="End")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].legend()
    #ax[1].axis("equal")
    plt.show()
"""
Created on Thue Dec 2 09:20 2025

@author: feyza.beziane
"""
## ------- Libraries -------
import numpy as np
import ast
import os
import random
import time
from decimal import Decimal
from dataclasses import dataclass, fields
from . import classes as cl
from . import core as c

'''This module contains the function to get the right parameters, 
the initialization functions, '''

## ------- Get parameters -------

def safe_int(x):
    '''convert x into int if possible'''
    try:
        return int(x)
    except (ValueError, TypeError):
        return x

def safe_float(x):
    '''convert x into float if possible'''
    try:
        return float(x)
    except (ValueError, TypeError):
        return x

def safe_bool(x):
    '''convert x into bool if possible'''
    try:
        return bool(int(x))
    except (ValueError, TypeError):
        return x
    
def par_generate_rr(rac, par, par_list, temp):
    """
    Generate par files by varying one parameter.
    par: name of parameter to vary ("om", "rotau", "runtau", etc.)
    par_list: list of values
    """
    output_dir = os.path.abspath(os.path.join(os.getcwd(), "2-par", "par"))
    os.makedirs(output_dir, exist_ok=True)
    template_path = os.path.join(os.getcwd(), "2-par", temp)
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    for i, p in enumerate(par_list):
        current_rac = f"{rac}{i:02d}"
        filename = os.path.join(output_dir, current_rac)
        values = {
            "rac": current_rac,
            "v": 1.0,
            #"v2": 1.0,
            #"v3": 1.0,
            "runtau": 1,
            "rotau": 1,
            "om": 0.0,
            "tumtype1": 0,
            "tumtype2": 0,
            "tumtype3": 0,
        }
        if "rr" in rac:
            values["om"] = 1.0
        elif "rt" in rac:
            values["om"] = 0.0
        if par in ["rotau", "runtau"]:
            values[par] = 1 / p
        else:
            values[par] = p
        content = template.format(**values)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"{len(par_list)} files generated in {output_dir}")

def params_from_file(filename, par, traj, msd):
    '''This function fills the basic parameters of each instances par, traj, msd with values directly taken from the par file'''
    mod1 = cl.mode_c(id = -1, runtype = -1, v = -1., dr = -1., drind = -1, om = -1., oms = -1, rho = -1,tumind = -1, tumtype = -1,runtau = -1., rundis=-1, turndis=-1)
    mod2 = cl.mode_c(id = -1, runtype = -1, v = -1., dr = -1., drind = -1, om = -1., oms = -1, rho = -1,tumind = -1, tumtype = -1,runtau = -1., rundis=-1, turndis=-1)
    mod3 = cl.mode_c(id = -1, runtype = -1, v = -1., dr = -1., drind = -1, om = -1., oms = -1, rho = -1,tumind = -1, tumtype = -1,runtau = -1., rundis=-1, turndis=-1)
    param = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#")[0]   # everything before comments
            values = line.split()       # values splited with sep = " "
            param.append(values)
    tab = np.array(param, dtype=object)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            try:
                tab[i,j] = ast.literal_eval(tab[i,j])
            except (ValueError, SyntaxError):
                pass
    l  = 0 ; par.path                                           = str       (tab[l,0])
    l += 1 ; par.rac                                            = str       (tab[l,0])
    l += 1 ; par.L                                              = safe_float(tab[l,0])
    l += 1 ; par.moden                                          = safe_int  (tab[l,0])
    l += 1 ; par.modswitch                                      = safe_int  (tab[l,0])
    l += 1 ; mod1.id,       mod2.id,        mod3.id             = safe_int  (tab[l,0]), safe_int  (tab[l,1]), safe_int  (tab[l,2])
    l += 1 ; mod1.runtype,  mod2.runtype,   mod3.runtype        = safe_int  (tab[l,0]), safe_int  (tab[l,1]), safe_int  (tab[l,2])
    l += 1 ; mod1.v,        mod2.v,         mod3.v              = safe_float(tab[l,0]), safe_float(tab[l,1]), safe_float(tab[l,2])
    l += 1 ; mod1.dr,       mod2.dr,        mod3.dr             = safe_float(tab[l,0]), safe_float(tab[l,1]), safe_float(tab[l,2])
    l += 1 ; mod1.drind,    mod2.drind,     mod3.drind          = safe_bool (tab[l,0]), safe_bool (tab[l,1]), safe_bool (tab[l,2])
    l += 1 ; mod1.om,       mod2.om,        mod3.om             = safe_float(tab[l,0]), safe_float(tab[l,1]), safe_float(tab[l,2])
    l += 1 ; mod1.oms,      mod2.oms,       mod3.oms            = safe_int  (tab[l,0]), safe_int  (tab[l,1]), safe_int  (tab[l,2])
    l += 1 ; mod1.tumind,   mod2.tumind,    mod3.tumind         = safe_bool (tab[l,0]), safe_bool (tab[l,1]), safe_bool (tab[l,2])
    l += 1 ; mod1.tumtype,  mod2.tumtype,   mod3.tumtype        = safe_int  (tab[l,0]), safe_int  (tab[l,1]), safe_int  (tab[l,2])
    l += 1 ; mod1.runtau,   mod2.runtau,    mod3.runtau         = safe_float(tab[l,0]), safe_float(tab[l,1]), safe_float(tab[l,2])
    l += 1 ; mod1.rundis,   mod2.rundis,    mod3.rundis         = safe_int  (tab[l,0]), safe_int  (tab[l,1]), safe_int  (tab[l,2])
    l += 1 ; mod1.turndis,  mod2.turndis,   mod3.turndis        = safe_int  (tab[l,0]), safe_int  (tab[l,1]), safe_int  (tab[l,2])
    l += 1 ; par.T, par.dT                                      = safe_float(tab[l,0]), safe_float(tab[l,1])
    l += 1 ; par.seed                                           = safe_int  (tab[l,0])
    l += 1 ; traj.n, traj.T                                     = safe_int  (tab[l,0]), safe_float(tab[l,1])
    l += 1 ; msd.Tmax, msd.dT                                   = safe_float(tab[l,0]), safe_float(tab[l,1])
    par.modetab = []
    par.modetab.append(mod1)
    par.modetab.append(mod2)
    par.modetab.append(mod3)
    return

## ------- Instance par -------

def par_create():
    '''Creates par object'''
    defaultinitval = -137
    par = cl.par_c (
        L         = defaultinitval,
        moden     = defaultinitval,
        modswitch = defaultinitval,
        modetab   = [],
        T         = defaultinitval,
        dT        = defaultinitval,
        seed      = defaultinitval,
        path      = 'pb',
        rac       = 'pb',
        pathrac   = 'pb'
        )
    return par

def par_init(par):
    '''Initializes par object'''
    for i in range(2):
        if type(par.modetab[i].v) != str and type(par.modetab[i].om) != str:
            if par.modetab[i].om >= 1e-10:
                par.modetab[i].rho = par.modetab[i].v/par.modetab[i].om
    par.pathrac = par.path + par.rac
    return

## ------- Instance particle -------

def particle_create(par):    
    '''Creates particle object'''
    defaultinitval = -137
    p = cl.particle_c(
        pos     = defaultinitval * np.ones(2),
        box     = defaultinitval * np.ones(2),
        theta   = defaultinitval,
        ctheta  = defaultinitval,
        stheta  = defaultinitval,
        vec     = defaultinitval * np.ones(2),
        mode    = cl.mode_c,
        tumt    = 10 * par.T    
    )
    return p

def particle_init(p,par):
    '''Initializes particle object'''
    p.pos   = np.array([par.L/2,par.L/2])           # initialization of the position of the particle in the center of the box
    p.box   = np.zeros(2)                           # particle initially in box (0,0)
    p.theta = 0                                     # initial angle of particle =0 with x-axis
    c.update_csthetavec(p)
    first_mode_generate(p,par)                      # initialize 1st mode randomly with appropriate probabilities
    if p.mode.tumind:                               # if there is a tumble
        p.tumt = 0  
        c.runtime_generate(p)
    if p.mode.runtype == 2:
        p.mode.oms = np.random.choice([-1,1])
    return

## ------- Instance traj -------
    
def traj_create():
    '''Creates traj object'''
    traj = cl.traj_c(
        traj = np.ones(2),
        time = np.ones(2),
        n    = -1,
        s    = -1,
        p    = -1,
        T    = -1,
        file = 'pb'
        )
    return traj

def traj_init(par,traj):
    '''Initializes traj object'''
    traj.s      = int(par.T/par.dT)
    traj.p      = traj.s + 1
    traj.traj   = np.ones((2,cl.dim,traj.p+1))
    traj.time   = np.ones((traj.p+1))
    traj.file   = par.pathrac + '.traj'
    return

def traj_save(p,par,traj,n):
    '''Save the relative and absolute position of the particle at step n of the trajectory'''
    if n * par.dT < traj.T:
        traj.time[n]     = n * par.dT
        traj.traj[0,:,n] = p.pos                        
        traj.traj[1,:,n] = c.absolute_position(p,par) 
    return

## ------- Instance msd -------

def msd_create():
    '''Creates msd object'''
    msd  = cl.msd_c (
        res          = np.ones(2),
        Tmax         = -1,
        dT           = -1,
        mmax         = -1,
        nT           = -1,
        qmsddToverdT = -1,
        p            = -1,
        file         = 'pb'
        )
    return msd

def msd_init(par, msd):
    '''Initializes msd object'''
    msd.mmax         = int(msd.Tmax/msd.dT)
    msd.nT           = int(par.T/msd.dT)
    msd.qmsddToverdT = int(msd.dT/par.dT)
    msd.p            = msd.mmax + 1
    msd.res          = np.ones((2,msd.p))
    msd.file         = par.pathrac + '.msd'
    return

## ------- Initialization of simulation -------

def seed_init(par):
    with open('/home/fbeziane/all/études/1-runandrotate/simulations/2-par/seed.dat', 'r', encoding='UTF-8') as seed_file:
        input_seed = int( seed_file.readlines()[ par.seed ] )        # get seed from parameters and seed.dat file
        np.random.seed(input_seed)                                   # set seed for NumPy RNG
    return

def adjust_dTmsd(dT, dTmsd_init):
    k = round(dTmsd_init / dT)
    dTmsd_new = k * dT
    return dTmsd_new

def sim_init(filename, par, traj, msd):
    '''Initialize objects u'''
    seed_init       (par)
    params_from_file(filename, par, traj, msd)            # read parameters in filename
    par_init        (par)                                 # fills the instance par with derived values
    traj_init       (par, traj)                           # fills the instance traj with derived values
    msd_init        (par, msd)                            # fills the instance msd with derived values
    initialmsddT = msd.dT
    a = Decimal(f'{msd.dT}')                              # The following allows to take into account time steps that are not multiple
    b = Decimal(f'{par.dT}')
    res = a % b
    if res != 0:
        msd.dT = adjust_dTmsd(par.dT,msd.dT)
        print(f"The time step for the computation of msd has been changed from {initialmsddT} to {msd.dT} because it needs to be a multiple of the simulation time step equal to {par.dT}")
    return


## ---------- Initialization of first mode -------------

def first_mode_generate(p, par):
    """Initialize 1st mode randomly with appropriate probabilities."""
    # unimodal
    if par.moden == 1:
        p.mode = par.modetab[0]
        return
    # multimodal
    mod1 = par.modetab[0]
    mod2 = par.modetab[1]
    p1 = 1/(2 * np.pi) * (1/mod2.runtau) / ((1/mod1.runtau) + (1/mod2.runtau))
    rand = np.random.uniform()
    if par.moden == 2:                                  # bimodal case
        if rand <= p1:
            p.mode = mod1
        else:
            p.mode = mod2
    if par.moden == 3:                                  # trimodal case
        mod3 = par.modetab[2]
        if rand <= p1:
            p.mode = mod1
        else:
            p.mode = np.random.choice([mod2, mod3])
    return

    
## ---------- Write results in file -------------

def msd_write(par, msd):
    '''saves msd in resmsd'''
    os.makedirs(os.path.dirname(msd.file), exist_ok=True)
    data = np.column_stack((msd.res))
    np.savetxt(msd.file, data, fmt="%.6f", delimiter="\t", header=f"Time\tMSD")
    t_wmsd = time.time()

def traj_write(par, traj):
    '''saves trajectories in restraj'''
    nmax = int(traj.T/par.dT)
    os.makedirs(os.path.dirname(traj.file), exist_ok=True)
    x_rel = traj.traj[0][0][:nmax]
    y_rel = traj.traj[0][1][:nmax]
    x_abs = traj.traj[1][0][:nmax]
    y_abs = traj.traj[1][1][:nmax]
    data = np.column_stack((x_rel,y_rel,x_abs,y_abs))
    header = (
        f"{'Relative':>12}{'':12}{'Absolute':>12}{'':12}\n"
        f"{'x':>5}{'y':>12}{'X':>12}{'Y':>12}"
    )
    np.savetxt(traj.file, data, fmt="%.6f", delimiter="\t", header=header, comments='')
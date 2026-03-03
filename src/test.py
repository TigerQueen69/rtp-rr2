"""
Created on Wed Dec 17 13:58 2025

@author: feyza.beziane
"""
## ------- Libraries -------
import numpy as np
import ast
import os
import random
import time
from dataclasses import dataclass, fields

## ------- Classes -------

@dataclass
class mode_c:
    id:int          # n° of the mode
    runtype:int     # = 1 for balistic, 2 for circular
    v:float         # velocity of the particle
    dr:float        # rotational diffusion of the particle
    drind:bool      # indicator of rot dif
    om:float        # absolute value of the angular velocity
    oms:int         # gives the sense of rotation during a rotation
    rho:float       # radius of the circle of rotation in a mode rotate (derived value)
    tumind:bool     # indicator of tumble
    tumtype:int     # 1 for isotropic, 2 for reverse etc.
    runtau:float    # mean run time = 1/tumbling_rate
    #rundis
    #turndis
@dataclass
class par_c :
    L:float             # length of the box
    moden:int           # number of modes
    modetab:np.ndarray  # 1D array of all modes instances
    T:float             # total time of the simulation
    dT:float            # simulation step
    seed:int            # seed line from seed file
    path:str            # path where results will be written
    rac:str             # root to construct filenames where the different results will be written
@dataclass
class particle_c :
    pos:np.ndarray  # of 2 relative coordinates
    box:np.ndarray  # of 2 coordinates of the box
    theta:float     # angle between the particle’s orientation and the x-axis
    ctheta:float    # cos(theta)
    stheta:float    # sin(theta)
    vec:float       # unit vector with angle theta with x-axis
    mode:mode_c     # instance of current mode of class mode_c
    tumt:float      # instant of next tumble
@dataclass
class traj_c :
    traj:np.ndarray  # array of 2 columns first one for rel pos 2nd for abs pos
    n:int            # number of trajectories
    s:int            # number of intervals of a trajectory (derived value)
    p:int            # total number of points of a trajectory (derived value)
    file:str         # filename of result file for traj
@dataclass
class msd_c :
    msd: np.ndarray # time and values of msd computed at each time t
    T:float         # largest time for computation of MSD (large enough to see the different behaviour)
    dT:float        # time step for the computation of msd
    p:int           # total nb of points for computation of msd (derived value)
    lagmax:int      # total number of intervals for the computation of msd (derived value)
    file:str        # filename of result file for msd

## ------- Generic -------

def parameters_from_file(filename):
    # Get values from par file and put it in an array
    param = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#")[0] # everything before comments
            values = line.split()   # values splited with sep = " "
            param.append(values)
    m = np.array(param, dtype=object)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            try:
                m[i,j] = ast.literal_eval(m[i,j])
            except (ValueError, SyntaxError):
                pass
    # get values from the array and put it in an instance with appropriate attributes
    par = par_c(m[0,0], m[1,0], [], m[12,0], m[12,1], m[13,0],
                 m[16,0],m[17,0])
    mode = mode_c(id = 0, runtype = 0, v = 0., dr = 0., drind = 0, om = 0., oms = 1, rho = 0,tumind = 1, tumtype = 1,runtau = 1.)
    for i in range(0,2) :
        mode.id = m[2,i]
        mode.runtype = m[3,i]
        mode.v = m[4,i]
        mode.dr = m[5,i]
        mode.drind = m[6,i]
        mode.om = m[7,i]
        mode.oms = m[8,i]
        mode.rho = mode.v/mode.om
        mode.tumind = m[9,i]
        mode.tumtype = m[10,i]
        mode.runtau = m[11,i]
        par.modetab.append(mode) # add the instance in the list modetab
    T, dT = m[15,0], m[15,1]
    p1 = int(T/dT)+1
    msd = msd_c(msd=-137*np.ones((2,p1)), T=T, dT=dT, p=p1, lagmax=p1-1, file = par.rac+'.msd')
    p2 = int(par.T/par.dT)+1
    traj = traj_c(traj=-137*np.ones((2,2,p2)), n = m[14,0], s = p2-1, p = p2 , file = par.rac+'.traj')
    return par, traj, msd


filename = '2-par/testrtA01'
#print(parameters_from_file(filename))


def sim_init(FileName):
    par, traj, msd = parameters_from_file(FileName)
    return par, traj, msd

#print(sim_init(filename))


def apply_boundaryconditions(p, par):
    '''Applies boudary condition of a squared box of lenght L'''
    p.box += np.floor(p.pos/par.L)      # Quotient de la division
    p.pos    = np.mod  (p.pos,par.L)      # Reste de la division
    return

def absolute_position(p, par):
    '''Returns the absolute coordinates of the particle'''
    X = p.pos + par.L*p.box
    return X

def update_csthetavec(p):
    p.ctheta, p.stheta = np.cos(p.theta), np.sin(p.theta)
    p.vec = np.array([p.ctheta, p.stheta])
    return

def generate_run_time(p):
    '''generates a poissonian distribution'''
    return np.random.exponential(p.mode.runtau)

## ------- Core -------

def tumble(p,par):
    p.theta = 2*np.pi*np.random.uniform()
    update_csthetavec(p)
    m = par.moden//p.mode.id
    p.mode = par.modetab[m-1]
    p.tumt += np.random.exponential(p.mode.runtau)
    return

def emove(p,par,t1,t2):
    '''makes the particle move from position at t1 to position at t2'''
    assert p.vec.shape == (2,), f"vec AVANT emove: {p.vec}, shape={p.vec.shape}"
    limit=10*par.dT
    deltat = t2-t1
    pm = p.mode         # instance of the current mode
    if pm.runtype == 1: # linear
        p.pos += pm.v * deltat * p.vec
    if pm.runtype == 2: # circular
        p.theta += pm.oms * pm.om * deltat
        update_csthetavec(p)
        if par.dT<limit:
            p.pos += pm.v * deltat * p.vec
        else:
            posc = p.pos + pm.rho * pm.oms * np.array([-p.stheta, p.ctheta])
            p.pos = posc + np.dot(np.array([[p.ctheta, -p.stheta],[p.stheta,p.ctheta]]), (p.pos-posc))
    if pm.drind :       # rot dif
        p.theta += np.sqrt(2 * pm.dr * deltat) * np.random.normal()
        update_csthetavec(p)
        assert p.vec.shape == (2,), f"vec après tumble: {p.vec}, shape={p.vec.shape}"
    apply_boundaryconditions(p,par)
    assert p.vec.shape == (2,), f"vec APRÈS boundary: {p.vec}, shape={p.vec.shape}"
    return

def evolve(p,par,t1,t2):
    '''makes the particle evolve (possible tumble) from position at t1 to position at t2'''
    tumt = p.tumt
    if t2 < tumt:
        emove(p,par,t1,t2)
    else:
        emove(p,par,t1,tumt)
        tumble(p,par)
        evolve(p,par,tumt,t2)
    return

def particle_create(par):
    '''initialize a particle with all its params with no values'''
    p = particle_c(
        pos=-137*np.ones((2)),
        box=-137*np.ones((2)),
        theta=-137,
        ctheta=-137,
        stheta=-137,
        vec=-137*np.ones((2)),
        mode=mode_c,
        tumt=10*par.T
    )
    return p

def particle_init(p,par):
    '''Initializes the state of the particle p'''
    p.pos = np.array([par.L/2,par.L/2])
    p.box = np.zeros((2))
    p.theta = 0
    p.ctheta = np.cos(p.theta)
    p.stheta = np.sin(p.theta)
    p.vec = np.array([p.ctheta,p.stheta])
    p.mode = par.modetab[0]
    if p.mode.tumind:
        p.tumt = generate_run_time(p)
    return

def traj_save(p,par,traj,n):
    '''save the rel and abs position of the particle at step n'''
    traj.traj[0,:,n] = p.pos
    traj.traj[1,:,n] = absolute_position(p,par)
    return

def traj_init(p,par,traj):
    '''initializes the traj array and save first rel and abs position'''
    traj.traj = np.ones((2,2,traj.p+1))
    traj_save(p,par,traj,0)
    return
dim = 2
def msd_compute(msd, traj): # traj = traj.traj[1]
    '''computes msd with many intervals'''
    for lag in range(0,msd.lagmax+1):
        msd.msd[0,lag] = lag * msd.dT
        sum = 0
        for nstart in range(msd.lagmax-lag+1):
            for d in range(dim):
                sum += (traj[d,nstart+lag]-traj[d,nstart])**2
        msd.msd[1,lag] = sum/(msd.lagmax-lag) # dernière loop div par 0 !!
    return

## --------- Main -----------
par,traj,msd = sim_init(filename)
p = particle_create(par)
particle_init(p,par)
#print(p)

traj_init(p,par,traj)

for n in range(traj.p):
    t1, t2 = n*par.dT, (n+1)*par.dT
    evolve(p,par,t1,t2)
    traj_save(p,par,traj,n)

msd_compute(msd,traj.traj[1])

print(msd.msd)
"""
Created on Mon Nov 3 16:15 2025

@author: feyza.beziane
"""
import sys, os
import numpy as np
from src import classes as cl
from src import core as c
from src import aux as x

parfile = os.path.abspath(sys.argv[1])                              # read parfile in run_all.sh
par, traj, msd = x.par_create(), x.traj_create(), x.msd_create()    # creation of instances
x.sim_init(parfile, par, traj, msd)                                 # initialization of the attributes of each instances
p = x.particle_create(par)                                          # creation of the particle
x.particle_init(p,par)                                              # initialization of the attributes of the instance particle
x.traj_save(p,par,traj,0)                                           # save first position
for n in range(traj.p):                                             # main loop
    t1, t2 = n * par.dT, (n+1) * par.dT                             
    c.evolve(p,par,t1,t2)                                           # evolution of particle from t1 to t2
    x.traj_save(p,par,traj,n+1)                                     # save current position
x.traj_write(par, traj)                                             # save relative and absolute trajectories
c.msd_compute(par,msd,traj.traj[1])                                 # computation of msd
x.msd_write(par, msd)                                               # save msd
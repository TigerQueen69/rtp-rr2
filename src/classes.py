"""
Created on Mon Nov 5 17:42 2025

@author: feyza.beziane
"""

import numpy as np
from dataclasses import dataclass, fields

## ------- Definition of variables -------
dim = 2

## ------- Definition of classes -------
@dataclass
class mode_c:
    id:             int         # n° of the mode
    runtype:        int         # = 1 for balistic, 2 for circular
    v:              float       # velocity modulus of the particle
    dr:             float       # rotational diffusion of the particle
    drind:          bool        # indicator of rotational diffusion
    om:             float       # absolute value of the angular velocity
    oms:            int         # gives the sense of rotation during a rotation
    rho:            float       # radius of the circle of rotation in a mode rotate
    tumind:         bool        # indicator of tumble
    tumtype:        int         # 1 for isotropic, 2 for reverse etc.
    runtau:         float       # mean run time
    rundis:         int
    turndis:        int
@dataclass
class particle_c:
    pos:            np.ndarray  # of dim relative coordinates
    box:            np.ndarray  # of dim coordinates of the box
    theta:          float       # angle between the particle's orientation and the x-axis
    ctheta:         float       # cos(theta)
    stheta:         float       # sin(theta)
    vec:            np.ndarray  # unit vector with angle theta with x-axis
    mode:           mode_c      # instance of mode_c for the current mode
    tumt:           float       # instant of next tumble
@dataclass  
class par_c:
    L: 	            float 	    # length of the box
    moden: 	        int	        # number of modes		
    modswitch:      int         # way of changing mode 0 : cyclic ; 1 : rrot       		  
    modetab : 	    list	    # array of all modes instances
    T: 	            float	    # total time of the simulation
    dT: 	        float	    # simulation step		
    seed: 	        int	        # seed line from seed file
    path: 	        str	        # path where results will be written
    rac: 	        str	        # root to construct filenames where the different results will be written
    pathrac:        str         # path + rac
@dataclass
class traj_c:
    traj:           np.ndarray  # array of 2 columns: first for relative position, second for absolute position
    time:           np.ndarray  # array of time
    n:              int         # number of trajectories
    s:              int         # number of intervals of a trajectory
    p:              int         # total number of points of a trajectory
    T:              float       # max time of writing traj in file
    file:           str         # filename of result file for trajectories
@dataclass
class msd_c:
    res:            np.ndarray  # time and values of MSD computed at each time t with time step dTmsd
    Tmax:           float       # maximum time interval for the computation of MSD
    mmax:           int         # maximum step interval for the computation of MSD
    dT:             float       # time step for the computation of MSD
    nT:             int         # index of time total time of simulation in the graduation dTmsd
    qmsddToverdT:   int         # quotient of dTmsd and dT
    p:              int         # total number of points for computation of MSD
    file:           str         # filename of result file for MSD
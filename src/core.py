"""
Created on Mon Nov 3 09:20 2025

@author: feyza.beziane
"""

## ------- Libraries -------

import numpy as np
import time
import logging
from . import aux as x
from . import classes as cl

## ------- Logging setup -------

logger = logging.getLogger(__name__)

## ------- Simulations -------

# logging.basicConfig(
#     level=logging.INFO,  # Niveau minimum affiché
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
# )

# logger = logging.getLogger('core')

# logger.info("Action started")
# logger.warning("Warning")
# logger.error("Error")

## ------- Evolution functions -------

def apply_boundaryconditions(p, par):
    '''Applies boudary condition of a squared box of lenght L'''
    #logger.debug("Applying boundary conditions")
    p.box += np.floor(p.pos/par.L)      # Quotient de la division
    p.pos  = np.mod  (p.pos,par.L)      # Reste de la division
    return

def absolute_position(p, par):
    '''Returns the absolute coordinates of the particle'''
    #logger.debug("Computing absolute position")
    X = p.pos + par.L*p.box
    return X

def update_csthetavec(p): # avoid repeated lines after change of theta
    #logger.debug("Updating theta vector")
    p.ctheta, p.stheta = np.array([np.cos(p.theta), np.sin(p.theta)])
    p.vec              = np.array([p.ctheta, p.stheta])
    return

def runtime_generate(p):
    '''update the instant of next tumble according to the different cases'''
    logger.debug("Generating runtime for tumble")
    pm = p.mode
    if pm.rundis == 1:   # poissonian case
        runtime = np.random.exponential(p.mode.runtau)
    elif pm.rundis == 2: # periodic case
        runtime = p.mode.runtau
    p.tumt += runtime
    return

# def change_mode(p,par):
#     '''Change mode if there exist two'''
#     if par.moden == 2:
#         m = par.moden//p.mode.id    # cycle change of modes 1 to 2 or 2 to 1
#         p.mode = par.modetab[m-1]
#     if p.mode.runtype == 2:
#         if p.mode.oms not in [-1,1]:
#             p.mode.oms = np.random.choice([-1,1])
#     return

def change_mode(p,par):
    logger.debug("Changing mode")
    if   par.moden == 2:
        m = par.moden//p.mode.id    # cyclic change of modes 1 to 2 or 2 to 1
    elif par.moden == 3:
        if par.modswitch == 0:      # cyclic change of modes 1 to 2 to 3 to 1
            m = p.mode.id+1
        elif par.modswitch == 1:    # run-and-rotate : 1 to 2 or 3 and 2 or 3 to 1
            if p.mode.id == 1:
                m = np.random.choice([2,3])
            else:
                m = 1
    p.mode = par.modetab[m-1]

def tumble(p,par):
    '''instantaneous change of orientation, change of mode and update of next instant of tumble'''
    logger.debug("Tumble event triggered")
    match p.mode.tumtype:
        case 1:                     # isotropic
            p.theta  = 2 * np.pi * np.random.uniform()
        case 2:                     # reverse
            p.theta += np.pi
        case 11:                    # test turn of pi/2
            p.theta += np.pi/2
        case 0:                     # no turning
            p.theta += 0
    update_csthetavec(p)
    change_mode(p,par)
    runtime_generate(p)
    return

def emove(p,par,t1,t2):
    logger.debug("Evolving particle from t=%s to t=%s", t1, t2)
    deltat = t2 - t1
    pm = p.mode                                 # saving the current mode for use later
    if pm.runtype == 1:                         # linear
        p.pos += pm.v * deltat * p.vec
    if pm.runtype == 2:                         # circular
        dtheta = pm.om * deltat
        if pm.drind:                            # if there is rotational diffusion
            p.pos += pm.v * deltat * p.vec      # compute new position at each time step
        else:                                   # if no rotational diffusion --> compute just last position in the mode rotate
            posc = p.pos + pm.rho * pm.oms * np.array([-p.stheta, p.ctheta])
            p.pos = posc + np.dot(
                np.array([[np.cos(dtheta), -np.sin(dtheta)],
                          [np.sin(dtheta),  np.cos(dtheta)]]),
                p.pos - posc
            )
        p.theta += pm.oms * dtheta
    if pm.drind:                                # rotational diffusion
        p.theta += np.sqrt(2 * pm.dr * deltat) * np.random.normal()
        update_csthetavec(p)
    apply_boundaryconditions(p,par)
    return

def evolve(p,par,t1,t2):
    '''makes the particle evolve (with possible tumbles) from position from time t1 to time t2'''
    logger.debug("Evolve called from t=%s to t=%s", t1, t2)
    tumt = p.tumt
    if t2 < tumt:
        emove(p,par,t1,t2)
    else:
        emove(p,par,t1,tumt)
        tumble(p,par)
        evolve(p,par,tumt,t2)
    return

## ------- Computations -------

def msd_compute(par,msd,abstraj):
    '''computes msd with many intervals (voir note notes.pdf (p1))'''
    logger.info("Starting MSD computation for %s", par.rac)
    t_start = time.time()
    q = msd.qmsddToverdT
    for m in range(msd.mmax + 1):
        msd.res[0,m] = m * msd.dT
        sum = 0
        for i in range(msd.nT - m + 1):
            for d in range(cl.dim):
                sum += (abstraj[d,q*(i+m)] - abstraj[d,q*i])**2
        msd.res[1,m] = sum/(msd.nT - m + 1)
    t_end = time.time()
    logger.info(
        "Time for computing MSD for %s : %.6fs",
        par.rac,
        t_end - t_start
    )
    return
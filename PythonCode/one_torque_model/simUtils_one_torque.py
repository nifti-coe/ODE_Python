from scipy.integrate import odeint
import numpy as np
import os
import math
from numba import jit


@jit(cache = True)
def FlyTheBug(state0, t, F, alpha, tau0, 
                bhead, ahead, bbutt, abutt, rho_head, rho_butt, 
                rhoA, muA, L1, L2, L3, K, c, g, betaR, nstep, nrun, 
                m1, m2, echead, ecbutt, I1, I2, S_head, S_butt):
    
    # unpack the state vector
    x,xd,y,yd,theta,thetad,phi,phid= state0 # displacement,x and velocity xd  etc...   You got it?'

    #Reynolds number calculation:
    Re_head = rhoA*(np.sqrt((xd**2)+(yd**2)))*(2*bhead)/muA #dimensionless number
    Re_butt = rhoA*(np.sqrt((xd**2)+(yd**2)))*(2*bbutt)/muA #dimensionless number

    #Coefficient of drag stuff:
    Cd_head = 24/np.abs(Re_head) + 6/(1 + np.sqrt(np.abs(Re_head))) + 0.4
    Cd_butt = 24/np.abs(Re_butt) + 6/(1 + np.sqrt(np.abs(Re_butt))) + 0.4
    
    
    h1 = m1 + m2
    h2 = (-1)*L1*m1*np.sin(theta)
    h3 = (-1)*L2*m2*np.sin(phi)
    h4 = L1*m1*np.cos(theta)
    h5 = L2*m2*np.cos(phi)
    h6 = (-1)*F*np.cos(alpha+theta)+(1/2)*Cd_butt*rhoA*S_butt*np.abs(xd)*xd+(1/2)*Cd_head*rhoA*S_head*np.abs(xd)*xd+(-1)*L1*m1*np.cos(theta)*thetad**2+(-1)*L2*m2*np.cos(phi)*phid**2
    h7 = g*(m1+m2)+(1/2)*Cd_butt*rhoA*S_butt*np.abs(yd)*yd+(1/2)*Cd_head*rhoA*S_head*np.abs(yd)*yd+(-1)*L1*m1*thetad**2*np.sin(theta)+(-1)*F*np.sin(alpha+theta)+(-1)*L2*m2*phid**2*np.sin(phi);
    h8 = (-1)*tau0+g*L1*m1*np.cos(theta)+(-1)*K*((-1)*betaR+(-1)*np.pi+(-1)*theta+phi)+(-1)*c*((-1)*thetad+phid)+(-1)*F*L3*np.sin(alpha)
    h9 = tau0+g*L2*m2*np.cos(phi)+K*((-1)*betaR+(-1)*np.pi+(-1)*theta+phi)+c*((-1)*thetad+phid)
    h10 = I1+L1**2*m1
    h11 = I2+L2**2*m2


    xdd = (-1)*(h10*h11*h1**2+(-1)*h11*h1*h2**2+(-1)*h10*h1*h3**2+(-1)*h11*h1*h4**2+h3**2*h4**2+(-2)*h2* 
        h3*h4*h5+(-1)*h10*h1*h5**2+h2**2*h5**2)**(-1)*( 
        h10*h11*h1*h6+(-1)*h11*h4**2*h6+(-1)*h10*h5**2* 
        h6+h11*h2*h4*h7+h10*h3*h5*h7+(-1)*h11*h1*h2* 
        h8+(-1)*h3*h4*h5*h8+h2*h5**2*h8+(-1)*h10*h1* 
        h3*h9+h3*h4**2*h9+(-1)*h2*h4*h5*h9)
  

    ydd = (-1)*((-1)*h10*h11*h1**2+h11*h1*h2**2+h10*h1*
        h3**2+h11*h1*h4**2+(-1)*h3**2*h4**2+2*h2*h3*h4*
        h5+h10*h1*h5**2+(-1)*h2**2*h5**2)**(-1)*((-1)*h11*
        h2*h4*h6+(-1)*h10*h3*h5*h6+(-1)*h10*h11*h1*
        h7+h11*h2**2*h7+h10*h3**2*h7+h11*h1*h4*h8+(-1)*
        h3**2*h4*h8+h2*h3*h5*h8+h2*h3*h4*h9+h10*h1*
        h5*h9+(-1)*h2**2*h5*h9)

    thetadd = (-1)*((-1)*h10*h11*h1**2+h11*h1*h2**2+h10*h1*
        h3**2+h11*h1*h4**2+(-1)*h3**2*h4**2+2*h2*h3*h4*
        h5+h10*h1*h5**2+(-1)*h2**2*h5**2)**(-1)*(h11*h1*
        h2*h6+h3*h4*h5*h6+(-1)*h2*h5**2*h6+h11*h1*
        h4*h7+(-1)*h3**2*h4*h7+h2*h3*h5*h7+(-1)*h11*
        h1**2*h8+h1*h3**2*h8+h1*h5**2*h8+(-1)*h1*h2*
        h3*h9+(-1)*h1*h4*h5*h9)

    phidd = (-1)*((-1)*h10*h11*h1**2+h11*h1*h2**2+h10*h1*
        h3**2+h11*h1*h4**2+(-1)*h3**2*h4**2+2*h2*h3*h4*
        h5+h10*h1*h5**2+(-1)*h2**2*h5**2)**(-1)*(h10*h1*
        h3*h6+(-1)*h3*h4**2*h6+h2*h4*h5*h6+h2*h3*h4*
        h7+h10*h1*h5*h7+(-1)*h2**2*h5*h7+(-1)*h1*h2*
        h3*h8+(-1)*h1*h4*h5*h8+(-1)*h10*h1**2*h9+h1*
        h2**2*h9+h1*h4**2*h9)
    
    return(np.array([xd, xdd,yd,ydd,thetad,thetadd,phid,phidd]))


# this returns the full trajectory
def flyBug_listInput_oneTorque(simNum, t, state0_ICs, FAlphaTau_list, globalList):
    F = FAlphaTau_list[simNum][0]
    alpha = FAlphaTau_list[simNum][1]
    tau0 = FAlphaTau_list[simNum][2]
    state = odeint(FlyTheBug, state0_ICs[simNum], t, args = (F, alpha, tau0, *globalList))
    x, xd = state[:,0], state[:,1]
    y, yd = state[:,2], state[:,3]
    theta, thetad = state[:,4],state[:,5]
    phi, phid = state[:, 6], state[:,7]
    return(np.array([x, xd, y, yd, theta, thetad, phi, phid]))


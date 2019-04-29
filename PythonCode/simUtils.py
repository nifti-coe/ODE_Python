from scipy.integrate import odeint
import numpy as np
import pandas as pd
import os
import math
from numba import jit


@jit
def FlyTheBug(state, t, simNum, dataDict, F, alpha, tau0):
    
    # unpack the state vector
    x,xd,y,yd,theta,thetad,phi,phid = state 


    #Reynolds number calculation:
    Re_head = dataDict["rhoA"]*(np.sqrt((xd**2)+(yd**2)))*(2*dataDict["bhead"])/dataDict["muA"] #dimensionless number
    Re_butt = dataDict["rhoA"]*(np.sqrt((xd**2)+(yd**2)))*(2*dataDict["bbutt"])/dataDict["muA"] #dimensionless number

    #Coefficient of drag stuff:
    Cd_head = 24/np.abs(Re_head) + 6/(1 + np.sqrt(np.abs(Re_head))) + 0.4
    Cd_butt = 24/np.abs(Re_butt) + 6/(1 + np.sqrt(np.abs(Re_butt))) + 0.4
    
    h1 = dataDict["m1"] + dataDict["m2"]
    h2 = (-1)*dataDict["L1"]*dataDict["m1"]*np.sin(theta)
    h3 = (-1)*dataDict["L2"]*dataDict["m2"]*np.sin(phi)
    h4 = dataDict["L1"]*dataDict["m1"]*np.cos(theta)
    h5 = dataDict["L2"]*dataDict["m2"]*np.cos(phi)
    h6 = ((-1)*F*np.cos(alpha+theta)+(1/2)*Cd_butt*dataDict["rhoA"]*dataDict["S_butt"]*np.abs(xd)*xd+
            (1/2)*Cd_head*dataDict["rhoA"]*dataDict["S_head"]*np.abs(xd)*xd+(-1)*dataDict["L1"]*dataDict["m1"]*np.cos(theta)*thetad**2+
            (-1)*dataDict["L2"]*dataDict["m2"]*np.cos(phi)*phid**2)
    h7 = (dataDict["g"]*(dataDict["m1"]+dataDict["m2"])+(1/2)*Cd_butt*dataDict["rhoA"]*dataDict["S_butt"]*np.abs(yd)*yd+
            (1/2)*Cd_head*dataDict["rhoA"]*dataDict["S_head"]*np.abs(yd)*yd+(-1)*dataDict["L1"]*dataDict["m1"]*thetad**2*np.sin(theta)+
            (-1)*F*np.sin(alpha+theta)+(-1)*dataDict["L2"]*dataDict["m2"]*phid**2*np.sin(phi))
    h8 = ((-1)*tau0+dataDict["g"]*dataDict["L1"]*dataDict["m1"]*np.cos(theta)+
            (-1)*dataDict["K"]*((-1)*dataDict["betaR"]+(-1)*np.pi+(-1)*theta+phi)+
            (-1)*dataDict["c"]*((-1)*thetad+phid)+(-1)*F*dataDict["L3"]*np.sin(alpha))
    h9 = (tau0+dataDict["g"]*dataDict["L2"]*dataDict["m2"]*np.cos(phi)+
            dataDict["K"]*((-1)*dataDict["betaR"]+(-1)*np.pi+(-1)*theta+phi)+dataDict["c"]*((-1)*thetad+phid))
    h10 = dataDict["I1"]+dataDict["L1"]**2*dataDict["m1"]
    h11 = dataDict["I2"]+dataDict["L2"]**2*dataDict["m2"]


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
        h3*h9+(-1)*h1*h4*h5*h9);

    phidd = (-1)*((-1)*h10*h11*h1**2+h11*h1*h2**2+h10*h1*
        h3**2+h11*h1*h4**2+(-1)*h3**2*h4**2+2*h2*h3*h4*
        h5+h10*h1*h5**2+(-1)*h2**2*h5**2)**(-1)*(h10*h1*
        h3*h6+(-1)*h3*h4**2*h6+h2*h4*h5*h6+h2*h3*h4*
        h7+h10*h1*h5*h7+(-1)*h2**2*h5*h7+(-1)*h1*h2*
        h3*h8+(-1)*h1*h4*h5*h8+(-1)*h10*h1**2*h9+h1*
        h2**2*h9+h1*h4**2*h9)
    
    return(np.array([xd, xdd,yd,ydd,thetad,thetadd,phid,phidd]))


# this returns the full trajectory
def flyBug_dictInput(simNum, dataDict):
    F = dataDict["randomizedFAlaphTau"][simNum][0]
    alpha = dataDict["randomizedFAlaphTau"][simNum][1]
    tau0 = dataDict["randomizedFAlaphTau"][simNum][2]

    state = odeint(FlyTheBug, dataDict["state0_ICs"], dataDict["t"], args = (simNum, dataDict, F, alpha, tau0), mxstep=5000000)
    x, xd = state[:,0], state[:,1]
    y, yd = state[:,2], state[:,3]
    theta, thetad = state[:,4],state[:,5]
    phi, phid = state[:, 6], state[:,7]
    return(np.array([x, xd, y, yd, theta, thetad, phi, phid]))


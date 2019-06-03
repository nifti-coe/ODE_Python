from scipy.integrate import odeint
import numpy as np
import pandas as pd
import os
import math
from numba import jit


# define directories
baseDir = os.getcwd()
dataDir = r'D:\MothSimulations\11c-AggressiveManeuver\Qstore\hws_am_con'
figDir = r'D:\Dropbox\AcademiaDropbox\mothMachineLearning_dataAndFigs\Figs'
dataOutput = r'D:\Dropbox\AcademiaDropbox\mothMachineLearning_dataAndFigs\DataOutput'
savedModels = r'D:\Dropbox\AcademiaDropbox\mothMachineLearning_dataAndFigs\savedModels'
randomRawData = r'D:/Dropbox/AcademiaDropbox/mothMachineLearning_dataAndFigs/PythonGeneratedData'

if not os.path.exists(dataOutput):
    os.mkdir(dataOutput)
if not os.path.exists(savedModels):
    os.mkdir(savedModels)


# Global variables

# Bunches of parameters ...  these don't vary from run to run 
# masses and moment of inertias in terms of insect density and eccentricity
# of the head/thorax & gaster

bhead = 0.507
ahead = 0.908
bbutt  = 0.1295
abutt  = 1.7475

# cgs  density of insect 
rho = 1 

# cgs density of air
rhoA = 0.00118

# cgs viscosity
muA = 0.000186 

# Length from the thorax-abdomen joint to the center of the 
# head-thorax mass in cm
L1 = 0.908 

# Length from the thorax-abdomen joint to the center of the 
# abdomen mass in cm
L2 = 1.7475  

# Length from the thorax-abdomen joint to the aerodynamic force 
# vector in cm
L3 = 0.75 

# m1 is the mass of the head-thorax
m1 = rho*(4/3)*np.pi*(bhead**2)*ahead

# m2 is the mass of the abdomen 
# (petiole + gaster)
m2 = rho*(4/3)*np.pi*(bbutt**2)*abutt

echead = ahead/bhead; #Eccentricity of head-thorax (unitless)
ecbutt = abutt/bbutt; #Eccentricity of gaster (unitless)
I1 = (1/5)*m1*(bhead**2)*(1 + echead**2); #Moment of inertia of the 
    #head-thorax
I2 = (1/5)*m2*(bbutt**2)*(1 + ecbutt**2); #Moment of inertia of the gaster

#This is the surface area of the object experiencing drag.
S_head = np.pi*bhead**2

#This is the surface area of the object experiencing drag.
S_butt = np.pi*bbutt**2 

# K is the torsional spring constant of the thorax-petiole joint
# in (cm^2)*g/(rad*(s^2))
K = 29.3 

# c is the torsional damping constant of the thorax-petiole joint
# in (cm^2)*g/s
c =  14075.8   

# refref: change. Gravity set to 0 for testing
g =  980.0   #g is the acceleration due to gravity in cm/(s^2)
#g =  0.0   #g is the acceleration due to gravity in cm/(s^2)

# This is the resting configuration of our 
# torsional spring(s) = Initial abdomen angle - initial head angle - pi
betaR =  0.0 
    
nstep = 2 # number of steps in each trajectory
t = np.linspace(0, 0.02, num = nstep, endpoint = True) # time cut into 100 timesteps
nrun = 100000  # (max) number of trajectories.


# initialize the matrix of locations
zeroMatrix = np.zeros([nstep, nrun])
x,      xd,    y,   yd, \
theta, thetad, phi, phid = [zeroMatrix.copy() for ii in 
                                range(len([ "x",     "xd",     "y", "yd", 
                                            "theta", "thetad", "phi", "phid"]))]

# ranges for initial conditions
ranges = np.array([[0, 0], [-1500, 1500], [0, 0], [-1500, 1500],   
                   [0, 2*np.pi], [-25, 25], [0, 2*np.pi], [-25, 25], 
                  [0, 44300], [0, 2*np.pi], [-100000, 100000]])



# generate random initial conditions for state 0
state0 = np.random.uniform(ranges[:, 0], ranges[:, 1], size=(nrun, ranges.shape[0]))

springExponent = 1.0




@jit
def FlyTheBug(state,t, F, alpha, tau0):
    # unpack the state vector
    x,xd,y,yd,theta,thetad,phi,phid= state # displacement,x and velocity xd  etc...   You got it?'
    # compute acceleration xdd = x''
    # Jorge's order .  x,y,theta,phi,xd,yd,thetad,phid
    # .  there is no entry for Q(2) ... which would be y.  I wonder why not?

    #Reynolds number calculation:
    Re_head = rhoA*(np.sqrt((xd**2)+(yd**2)))*(2*bhead)/muA; #dimensionless number
    Re_butt = rhoA*(np.sqrt((xd**2)+(yd**2)))*(2*bbutt)/muA; #dimensionless number

    #Coefficient of drag stuff:
    Cd_head = 24/np.abs(Re_head) + 6/(1 + np.sqrt(np.abs(Re_head))) + 0.4;
    Cd_butt = 24/np.abs(Re_butt) + 6/(1 + np.sqrt(np.abs(Re_butt))) + 0.4;
    
    h1 = m1 + m2;
    h2 = (-1)*L1*m1*np.sin(theta);
    h3 = (-1)*L2*m2*np.sin(phi);
    h4 = L1*m1*np.cos(theta);
    h5 = L2*m2*np.cos(phi);
    h6 = (-1)*F*np.cos(alpha+theta)+(1/2)*Cd_butt*rhoA*S_butt*np.abs(xd)*xd+(1/2)*Cd_head*rhoA*S_head*np.abs(xd)*xd+(-1)*L1*m1*np.cos(theta)*thetad**2+(-1)*L2*m2*np.cos(phi)*phid**2
    h7 = g*(m1+m2)+(1/2)*Cd_butt*rhoA*S_butt*np.abs(yd)*yd+(1/2)*Cd_head*rhoA*S_head*np.abs(yd)*yd+(-1)*L1*m1*thetad**2*np.sin(theta)+(-1)*F*np.sin(alpha+theta)+(-1)*L2*m2*phid**2*np.sin(phi);
    h8 = (-1)*tau0+g*L1*m1*np.cos(theta)+(-1)*K*((-1)*betaR+(-1)*np.pi+(-1)*theta+phi)+(-1)*c*((-1)*thetad+phid)+(-1)*F*L3*np.sin(alpha);
    h9 = tau0+g*L2*m2*np.cos(phi)+K*((-1)*betaR+(-1)*np.pi+(-1)*theta+phi)+c*((-1)*thetad+phid);
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
def flyBug(i):
    F = state0[i,8]
    alpha = state0[i, 9]
    tau0 = state0[i, 10]
    state = odeint(FlyTheBug, state0[i, 0:8], t, args = (F, alpha, tau0), mxstep=5000000)
    x, xd = state[:,0], state[:,1]
    y, yd = state[:,2], state[:,3]
    theta, thetad = state[:,4],state[:,5]
    phi, phid = state[:, 6], state[:,7]
    return(np.array([x, xd, y, yd, theta, thetad, phi, phid]))

# this returns the initial and final states
def flyBug_firstLast(i):
    F = state0[i,8]
    alpha = state0[i, 9]
    tau0 = state0[i, 10]
    state = odeint(FlyTheBug, state0[i, 0:8:], t, args = (F, alpha, tau0), mxstep=5000000 )
    [x0, xf], [xd0, xdf] = state[[0, -1],0], state[[0, -1],1]
    [y0, yf], [yd0, ydf] = state[[0, -1],2], state[[0, -1],3]
    [theta0, thetaf], [thetad0, thetadf] = state[[0, -1],4],state[[0, -1],5]
    [phi0, phif], [phid0, phidf] = state[[0, -1], 6], state[[0, -1],7]
    return(np.array([x0, xf, xd0, xdf, y0, yf, yd0, ydf, theta0, thetaf, thetad0, thetadf, phi0, phif, phid0, phidf, F, alpha, tau0]))


#testDF = pd.read_csv(os.path.join(dataOutput, "NNpreds_RandomICs.csv"))
#testDF["x_0"] = 0 
#testDF["y_0"] = 0
#state00 = np.array(testDF[["x_0", "x_dot_0", "y_0", "y_dot_0", 
#           "theta_0", "theta_dot_0", "phi_0", "phi_dot_0", 
#           "F_pred", "alpha_pred", "tau_pred"]])



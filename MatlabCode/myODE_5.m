%ODE 3d eliminated the use of global variables and now we import a struct
    %with the necessary parameters of the function.
%ODE 3e eliminated the sinusoidal multiplier from tau0.*sin(2.*f.*pi.*t)
    %so that the applied torque is a FIXED value. 
%ODE 4 made MAJOR corrections after re-deriving the model on 7/20/2018. 
    %1/31/19 Corrected error in I1 and I2 terms.

% function dQ = myODE_5(t,Q)
function dQ = myODE_5(t,Q,par,F,alpha,tau0)
                        %Q -> x,y,theta,phi,xdot,ydot,thetadot,phidot
%Unpack the structs
L1 = par.L1; %Length from the thorax-abdomen joint to the center of the 
    %head-thorax mass in cm
L2 = par.L2; %Length from the thorax-abdomen joint to the center of the 
    %abdomen mass in cm
L_petiole = par.L_petiole; %Length of petiole extension as a percentage of 
    %body length.
L3 = par.L3; %Length from the thorax-abdomen joint to the aerodynamic force 
    %vector in cm
rho = par.rho; %The density of the insect in g/(cm^3)
rhoA = par.rhoA; %The density of air in g/(cm^3)
muA = par.muA; %The dynamic viscosity of air at 27C in g/(cm*s)
ahead = par.ahead; %Major axis of head-thorax ellipsoid in cm
abutt = par.abutt; %Major axis of abdomen ellipsoid in cm
bhead = par.bhead; %Minor axis of head-thorax ellipsoid in cm
bbutt = par.bbutt; %Minor axis of abdomen ellipsoid in cm
K = par.K; %K is the torsional spring constant of the thorax-petiole joint
    %in (cm^2)*g/(rad*(s^2))
c = par.c; %c is the torsional damping constant of the thorax-petiole joint
    %in (cm^2)*g/s
g = par.g; %g is the acceleration due to gravity in cm/(s^2)
betaR = par.betaR; %This is the resting configuration of our 
    %torsional spring(s) = Initial abdomen angle - initial head angle - pi

%Where 
%m1 is the mass of the head-thorax
%m2 is the mass of the abdomen (petiole + gaster)
%I1 is the moment of inertia of the head-thorax
%I2 is the moment of inertia of the abdomen (petiole + gaster)
%K is the torsional spring constant of the thorax-petiole joint
%c is the torsional damping constant of the thorax-petiole joint
%F is the F vector (aerodynamics)
%alpha is the angle of the aerodynamic vectory with respect to the
%head-thorax
%tau is the torque applied
%g is the acceleration due to gravity in cm/(s^2)

%masses and moment of inertias in terms of insect density and eccentricity
%of the head/thorax & gaster
m1 = rho*(4/3)*pi*(bhead^2)*ahead; %m1 is the mass of the head-thorax (in cm)
m2 = rho*(4/3)*pi*(bbutt^2)*abutt; %m2 is the mass of the abdomen (in cm)
    %(petiole + gaster)
echead = ahead/bhead; %Eccentricity of head-thorax (unitless)
ecbutt = abutt/bbutt; %Eccentricity of gaster (unitless)
I1 = (1/5)*m1*(bhead^2)*(1 + echead^2); %Moment of inertia of the 
    %head-thorax (in grams*(cm^2))

%The issue corrected on 1/31/19
%Recall the parallel axis theorem: I = I_centerOfMass + m*(d^2)
    %Where m is the mass of the object, and d is the perpendicular distance
    %between the axis of rotation and the object.
I2 = ((1/5)*m2*(bbutt^2)*(1 + ecbutt^2) + (m2*L_petiole^2)); %Moment of 
    %inertia of the gaster (in grams*(cm^2))    


S_head = pi*bhead^2; %This is the surface area of the object experiencing drag.
                %In this case, it is modeled as a sphere (in cm^2).
S_butt = pi*bbutt^2; %This is the surface area of the object experiencing drag.
                %In this case, it is modeled as a sphere (in cm^2).

%Reynolds number calculation:
Re_head = rhoA*(sqrt((Q(5)^2)+(Q(6)^2)))*(2*bhead)/muA; %dimensionless number
Re_butt = rhoA*(sqrt((Q(5)^2)+(Q(6)^2)))*(2*bbutt)/muA; %dimensionless number

%Coefficient of drag stuff:
Cd_head = 24/abs(Re_head) + 6/(1 + sqrt(abs(Re_head))) + 0.4;
Cd_butt = 24/abs(Re_butt) + 6/(1 + sqrt(abs(Re_butt))) + 0.4;

%dQ = zeros(8, 1);

%These are the coefficients for our acceleration equations imported from
%Mathematica. Careful, this'll get messy.

h1 = m1 + m2;
h2 = (-1).*L1.*m1.*sin(Q(3));
h3 = (-1).*L2.*m2.*sin(Q(4));
h4 = L1.*m1.*cos(Q(3));
h5 = L2.*m2.*cos(Q(4));
h6 = (-1).*F.*cos(alpha+Q(3))+(1/2).*Cd_butt.*rhoA.*S_butt.*abs(Q(5)).*Q(...
    5)+(1/2).*Cd_head.*rhoA.*S_head.*abs(Q(5)).*Q(5)+(-1).*L1.*m1.*cos(...
    Q(3)).*Q(7).^2+(-1).*L2.*m2.*cos(Q(4)).*Q(8).^2;
h7 = g.*(m1+m2)+(1/2).*Cd_butt.*rhoA.*S_butt.*abs(Q(6)).*Q(6)+(1/2).*...
    Cd_head.*rhoA.*S_head.*abs(Q(6)).*Q(6)+(-1).*L1.*m1.*Q(7).^2.*sin(Q(...
    3))+(-1).*F.*sin(alpha+Q(3))+(-1).*L2.*m2.*Q(8).^2.*sin(Q(4));
h8 = (-1).*tau0+g.*L1.*m1.*cos(Q(3))+(-1).*K.*((-1).*betaR+(-1).*pi+(...
-1).*Q(3)+Q(4))+(-1).*c.*((-1).*Q(7)+Q(8))+(-1).*F.*L3.*sin(alpha);
h9 = tau0+g.*L2.*m2.*cos(Q(4))+K.*((-1).*betaR+(-1).*pi+(-1).*Q(3)+Q(4)...
    )+c.*((-1).*Q(7)+Q(8));
h10 = I1+L1.^2.*m1;
h11 = I2+L2.^2.*m2;

dQ(1,1) = Q(5); %This is x dot
dQ(2,1) = Q(6); %This is y dot
dQ(3,1) = Q(7); %This is theta dot
dQ(4,1) = Q(8); %This is phi dot
%  everywhere you have a captheta -- that is Q(7), capphi is Q(8)
%  every xdot is Q(5) and ydot is Q(6);
%  every theta is Q(3) and phi is Q(4)
%  there are no x or y alone...   

%x double dot
dQ(5,1) = (-1).*(h10.*h11.*h1.^2+(-1).*h11.*h1.*h2.^2+(-1).*h10.* ...
    h1.*h3.^2+(-1).*h11.*h1.*h4.^2+h3.^2.*h4.^2+(-2).*h2.* ...
    h3.*h4.*h5+(-1).*h10.*h1.*h5.^2+h2.^2.*h5.^2).^(-1).*( ...
    h10.*h11.*h1.*h6+(-1).*h11.*h4.^2.*h6+(-1).*h10.*h5.^2.* ...
    h6+h11.*h2.*h4.*h7+h10.*h3.*h5.*h7+(-1).*h11.*h1.*h2.* ...
    h8+(-1).*h3.*h4.*h5.*h8+h2.*h5.^2.*h8+(-1).*h10.*h1.* ...
    h3.*h9+h3.*h4.^2.*h9+(-1).*h2.*h4.*h5.*h9);
  
%y double dot
dQ(6,1) = (-1).*((-1).*h10.*h11.*h1.^2+h11.*h1.*h2.^2+h10.*h1.* ...
    h3.^2+h11.*h1.*h4.^2+(-1).*h3.^2.*h4.^2+2.*h2.*h3.*h4.* ...
    h5+h10.*h1.*h5.^2+(-1).*h2.^2.*h5.^2).^(-1).*((-1).*h11.* ...
    h2.*h4.*h6+(-1).*h10.*h3.*h5.*h6+(-1).*h10.*h11.*h1.* ...
    h7+h11.*h2.^2.*h7+h10.*h3.^2.*h7+h11.*h1.*h4.*h8+(-1).* ...
    h3.^2.*h4.*h8+h2.*h3.*h5.*h8+h2.*h3.*h4.*h9+h10.*h1.* ...
    h5.*h9+(-1).*h2.^2.*h5.*h9);

%Theta double dot
dQ(7,1) = (-1).*((-1).*h10.*h11.*h1.^2+h11.*h1.*h2.^2+h10.*h1.* ...
    h3.^2+h11.*h1.*h4.^2+(-1).*h3.^2.*h4.^2+2.*h2.*h3.*h4.* ...
    h5+h10.*h1.*h5.^2+(-1).*h2.^2.*h5.^2).^(-1).*(h11.*h1.* ...
    h2.*h6+h3.*h4.*h5.*h6+(-1).*h2.*h5.^2.*h6+h11.*h1.* ...
    h4.*h7+(-1).*h3.^2.*h4.*h7+h2.*h3.*h5.*h7+(-1).*h11.* ...
    h1.^2.*h8+h1.*h3.^2.*h8+h1.*h5.^2.*h8+(-1).*h1.*h2.* ...
    h3.*h9+(-1).*h1.*h4.*h5.*h9);

%Phi double dot
dQ(8,1) = (-1).*((-1).*h10.*h11.*h1.^2+h11.*h1.*h2.^2+h10.*h1.* ...
    h3.^2+h11.*h1.*h4.^2+(-1).*h3.^2.*h4.^2+2.*h2.*h3.*h4.* ...
    h5+h10.*h1.*h5.^2+(-1).*h2.^2.*h5.^2).^(-1).*(h10.*h1.* ...
    h3.*h6+(-1).*h3.*h4.^2.*h6+h2.*h4.*h5.*h6+h2.*h3.*h4.* ...
    h7+h10.*h1.*h5.*h7+(-1).*h2.^2.*h5.*h7+(-1).*h1.*h2.* ...
    h3.*h8+(-1).*h1.*h4.*h5.*h8+(-1).*h10.*h1.^2.*h9+h1.* ...
    h2.^2.*h9+h1.*h4.^2.*h9);

end
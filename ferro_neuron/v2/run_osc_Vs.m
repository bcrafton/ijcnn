clear all

%Run ODE for FET based neuron
%Spikes are generated when discharging is completed (Vs=Vt1, reversed peak). 


%% Parameters

%transconductances of two FETs
gf=1e-5;
gm=1e-5;

%intecept of Id in charging phase, can be view as bias. Id=V0-gf*Vs;
V0=0.4*gf; 

%threshold voltage of discharging FET
VMth=0.25;

%Gate voltage inputs 
%Vgf-inhibition control, also determines bound voltages
%Vgm-excitation, tunes firing rate

%Vgf=0.4 %inhibition on, non-spiking
Vgf=0.3; %inhibition off, spiking
Vgm=0.35; %excitation input, range: (0.25,0.4)

%capacitence
C=8e-8; 

global s; %the flag of charging/discharging (1 - charging; 0 - discharging)
s=1; %initialize system with charging



%% ODE solver
tspan=[0 1]; %simulation time

x0=0.1; %Initialize Vs
F = @(t,x) Vs_dyn(x, t, Vgf, gm, gf, V0, Vgm, VMth, C, s ); %model ODE
opt=odeset('MaxStep',1e-6,'RelTol',1e-5,'AbsTol',1e-5); %solver setting
[t,y] = ode45(F, tspan, x0, opt);
%%

figure
plot(t,y,'LineWidth',2)
% [pks,locs] = findpeaks(-y);
% freq=1/(t(locs(3))-t(locs(2))),






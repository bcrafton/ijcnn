function [ dx ] = Vs_dyn( x, t, Vgf, gm, gf, V0, Vgm, VMth, C, s )
%VS_DYN Summary of this function goes here
%   
global s; %the flag of charging/discharging (1 - charging; 0 - discharging)
Vs=x; %ODE differential varible Vs (Voltage of Capacitor)
dVs=0; %Intialize derivative

% Determine the boundry voltage Vt1, Vt2 for oscillation, based on Vgf(inhibitory input)
% Alt: Piecewise function of Vgf=PW_Vgf(t) can also be implemented here as a control signal

if Vgf==0.3 %(excitation)
   Vt2=0.187786883557357;
   Vt1=0.111106555255706;   
else %Vgf==0.4 (inhibition)
   Vt2=0.320983604156612;
   Vt1=0.219508192947465;
end

% ODE of charging and discharging
if s==1
dVs=(1/C)*((V0-gf*Vs)-gm*(Vgm-VMth)); 
    if Vs>=Vt2 %upper bound of charging reached, flip to discharging
        s=0;
    end
elseif s==0
dVs=-(1/C)*gm*(Vgm-VMth);
    if Vs<=Vt1 %lower bound of discharging reached (spike), flip to charging, 
        s=1;
    end
end

dx=dVs;

end


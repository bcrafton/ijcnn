
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class neuron_group:
    def __init__(self, size, c, vgm, gm, vgf, gf, v0, vmth):
    
        self.size = size
        self.c = c
        self.vgm = vgm
        self.gm = gm
        self.vgf = vgf
        self.gf = gf
        self.v0 = v0
        self.vmth = vmth
        
        self.v = 0.1
        self.vs = []
        self.charge = 1
        
        if self.vgf == 0.3: # vgf==0.3 (excitation)
           self.vt1=0.111106555255706
           self.vt2=0.187786883557357
        else:               # vgf==0.4 (inhibition)
           self.vt1=0.219508192947465
           self.vt2=0.320983604156612
        
    def step(self, t, dt):
        spkd = 0
    
        if self.charge:
            dvdt = (1./self.c) * ((self.v0 - self.gf * self.v) - self.gm * (self.vgm - self.vmth))
            if self.v >= self.vt2: # upper bound of charging reached, flip to discharging
                self.charge = 0
                spkd = 1
        else:
            dvdt = -(1./self.c) * self.gm * (self.vgm - self.vmth)
            if self.v <= self.vt1: # lower bound of discharging reached (spike), flip to charging, 
                self.charge = 1
    
        dv = dvdt * dt
        self.v += dv
        self.vs.append(self.v)
        
        return spkd
        
    def reset(self):
        self.v = 0.1
        self.charge = 1





import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit.pkl', 'rb') as f:
    fit = pickle.load(f)

class Memristor:
    
    def __init__(self):
        self.U = 5e-14
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 1e6
        self.ROFF = 100e6
        self.P = 5            
        self.W = self.W0
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        self.Rs = []
        
    def step(self, V, dt):
        I = (1. / self.R) * V

        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W = np.clip(self.W + dwdt * dt, 1e-9, 9e-9)

        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        self.Rs.append(self.R)

        return I

T = 1e-2
dt = 1e-6
steps = int(T / dt) + 1
Ts = np.linspace(0., T, steps)

m = Memristor()

for t in Ts:
    i = fit((1., 1., 0., m.R))
    v = i * m.R
    print (i, v, m.R)
    m.step(v, dt)
   
plt.ylim(1e6, 100e6)
plt.plot(Ts, m.Rs)
plt.show()

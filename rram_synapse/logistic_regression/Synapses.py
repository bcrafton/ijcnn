
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Synapses:
    
    def __init__(self, shape):
        with open('fit1.pkl', 'rb') as f:
            self.fit1 = pickle.load(f)
        with open('fit2.pkl', 'rb') as f:
            self.fit2 = pickle.load(f)

        self.shape = shape
        self.input_size = self.shape[0]
        self.output_size = self.shape[1]
        self.U = 5e-14
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 1e6
        self.ROFF = 100e6
        self.P = 5            
        self.W = np.ones(shape=self.shape) * self.W0
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
    def step(self, Vd, Vg, Vc, dt, fit=1):
    
        assert(np.shape(Vd) == (self.input_size,))
        assert(np.shape(Vg) == self.shape)
        assert(np.shape(Vc) == (self.output_size,))

        Vd = np.reshape(Vd, (-1, 1))
        Vd = np.repeat(Vd, self.output_size, axis=1)
        Vd = np.reshape(Vd, (-1, 1))

        Vg = np.reshape(Vg, (-1, 1))

        Vc = np.reshape(Vc, (1, -1))
        Vc = np.repeat(Vc, self.input_size, axis=0)
        Vc = np.reshape(Vc, (-1, 1))
        
        #####################################

        r = np.reshape(self.R, (-1, 1))
        
        if fit == 1:
            points = np.concatenate((Vd, Vg), axis=1)
            I = self.fit1(points) / r * 1e6
        else:
            sign = np.sign(Vc)
            Vc = np.absolute(Vc)
            points = np.concatenate((Vc, Vg), axis=1)
            I = self.fit2(points) / r * 1e6
            I = I * sign
            
        I = np.reshape(I, self.shape)

        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W = np.clip(self.W + dwdt * dt, 0., 10e-9)
        
        # make sure to do this at the end.
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
        # I = np.sum(I, axis=0)
        return I
        
        
        
        
        
        
        
        

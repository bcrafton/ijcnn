
import numpy as np
import pylab as plt

class Memristors:
    
    def __init__(self, shape):
        self.shape = shape
        self.input_size = self.shape[0]
        self.output_size = self.shape[1]
        self.U = 5e-14
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 1e3
        self.ROFF = 5e4
        self.P = 5            
        # self.W = np.ones(shape=self.shape) * self.W0
        self.W = np.random.uniform(low=1e-9, high=9e-9, size=shape)
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
    def step(self, V, dt):
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = (1. / self.R) * V
        assert(np.shape(I) == self.shape)
        
        # print (np.average(self.R))
        
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W = np.clip(self.W + dwdt * dt, 1e-9, 9e-9)
        return I


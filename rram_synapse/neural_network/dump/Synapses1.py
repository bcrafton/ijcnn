
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Synapses:
    
    def __init__(self, shape, sign, weights):
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
        # self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        self.R = 1. / weights
        # so the idea with sign is that we flip the memristor terminals so that is gets the correct gradient
        # and we just return the 'negative current' because we are sending it to inh terminal of neuron
        # self.rate = rate
        # self.sign = np.random.choice([-1., 1.], size=(self.shape), replace=True, p=[1.-self.rate, self.rate])
        self.sign = sign

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
            # fit2 is fucked up for Vc, so we just make it positive and flip the sign afterwards.
            sign = np.sign(Vc)
            Vc = np.absolute(Vc)
            points = np.concatenate((Vc, Vg), axis=1)
            I = self.fit2(points) / r * 1e6
            I = I * sign
            
        I = np.reshape(I, self.shape) * self.sign

        '''
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W = np.clip(self.W + dwdt * dt, 0., 10e-9)
        
        # make sure to do this at the end.
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
        # I = np.sum(I, axis=0)
        '''
        return I
        
        
        
        
        
        
        
        

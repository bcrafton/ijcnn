
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Synapses:
    
    def __init__(self, shape, rate):
        with open('../model/forward/forward.pkl', 'rb') as f:
            self.forward = pickle.load(f)
        with open('../model/forward/forward_pwr.pkl', 'rb') as f:
            self.forward_pwr = pickle.load(f)
        with open('../model/backward/backward.pkl', 'rb') as f:
            self.backward = pickle.load(f)
        with open('../model/backward/backward_pwr.pkl', 'rb') as f:
            self.backward_pwr = pickle.load(f)
            
        self.shape = shape
        self.input_size = self.shape[0]
        self.output_size = self.shape[1]
        self.U = 5e-14
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 1e6
        self.ROFF = 100e6
        self.P = 5            
        # self.W = np.ones(shape=self.shape) * self.W0
        self.W = np.random.uniform(low=0., high=10e-9, size=self.shape)
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
        # so the idea with sign is that we flip the memristor terminals so that is gets the correct gradient
        # and we just return the 'negative current' because we are sending it to inh terminal of neuron
        self.rate = rate
        self.sign = np.random.choice([-1., 1.], size=(self.shape), replace=True, p=[1.-self.rate, self.rate])

    def step(self, Vd, Vg, Vc, dt, forward=1):
    
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
        
        if forward == 1:
            # points = np.concatenate((Vg, r), axis=1)
            # p = self.forward(points)
            
            # print (np.sum(p))
            
            points = np.concatenate((Vd, Vg), axis=1)
            I = self.forward(points) / r * 1e6
            I = np.reshape(I, self.shape) * self.sign
            
            return I
            
        else:
            # points = np.concatenate((Vg, r), axis=1)
            # p = self.backward(points)
            
            # print (np.sum(p))
        
            # fit2 is fucked up for Vc, so we just make it positive and flip the sign afterwards.
            sign = np.sign(Vc)
            Vc = np.absolute(Vc)
            points = np.concatenate((Vc, Vg), axis=1)
            # fit2 produces a negative current!!!
            I = self.backward(points) / r * 1e6
            I = I * sign
            I = np.reshape(I, self.shape) * self.sign
            # I = I * self.R
            
            F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
            dwdt = ((self.U * self.RON * I) / self.D) * F
            self.W = np.clip(self.W + dwdt * dt, 0., 10e-9)

            self.W = np.clip(self.W + dwdt * dt, 0., 10e-9)
            pre = np.copy(self.R)
            self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
            post = np.copy(self.R)
            return pre - post
        
        
        
        
        
        
        

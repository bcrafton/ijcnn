
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Synapses:
    
    def __init__(self, shape):
        with open('fit.pkl', 'rb') as f:
            self.fit = pickle.load(f)

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
        # self.W = np.random.uniform(low=1e-9, high=9e-9, size=shape)
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
    def step(self, Vd, Vg, Vc, dt):
        # print (np.shape(Vd), self.input_size)
    
        assert(np.shape(Vd) == (self.input_size,))
        assert(np.shape(Vg) == self.shape)
        assert(np.shape(Vc) == (self.output_size,))

        Vd = np.reshape(Vd, (-1, 1))
        Vd = np.repeat(Vd, self.output_size, axis=1)
        Vd = np.reshape(Vd, (-1, 1))

        '''
        for ii in range(self.input_size):
            start = ii       * self.output_size
            end =   (ii + 1) * self.output_size
            assert( np.all(Vd[start:end] == Vd[start]) )
        '''

        Vg = np.reshape(Vg, (-1, 1))

        Vc = np.reshape(Vc, (1, -1))
        Vc = np.repeat(Vc, self.input_size, axis=0)
        Vc = np.reshape(Vc, (-1, 1))

        # print (np.shape(Vd))
        # print (np.shape(Vg))
        # print (np.shape(Vc))

        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        r = np.reshape(self.R, (-1, 1))
        
        points = np.concatenate((Vd, Vg, Vc, r), axis=1)
        I = self.fit(points)
        I = np.reshape(I, self.shape)

        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W = np.clip(self.W + dwdt * dt, 1e-9, 9e-9)
        
        # problem was we were not recomputing R!!!
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
        # I = np.sum(I, axis=0)
        return I, dwdt
        
        

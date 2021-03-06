
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Synapses:
    
    def __init__(self, shape, rate):
        '''
        with open('fit1.pkl', 'rb') as f:
            self.fit1 = pickle.load(f)
        with open('fit2.pkl', 'rb') as f:
            self.fit2 = pickle.load(f)
        '''
        forward_i = np.genfromtxt('forward_i.csv', delimiter=',')
        forward_vg = np.genfromtxt('forward_vg.csv', delimiter=',')
        backward_i = np.genfromtxt('backward_i.csv', delimiter=',')
        backward_vg  = np.genfromtxt('backward_vg.csv',  delimiter=',')

        self.fit1 = interp1d(forward_vg, forward_i)
        self.fit2 = interp1d(backward_vg, backward_i)
        
        #self.fit1 = CubicSpline(forward_vg, forward_i)
        #self.fit2 = CubicSpline(backward_vg, backward_i)
        
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
        
        # so the idea with sign is that we flip the memristor terminals so that is gets the correct gradient
        # and we just return the 'negative current' because we are sending it to inh terminal of neuron
        self.rate = rate
        self.sign = np.random.choice([-1., 1.], size=(self.shape), replace=True, p=[1.-self.rate, self.rate])

    def step(self, Vd, Vg, Vc, dt, fit=1):
    
        assert(np.shape(Vd) == (self.input_size,))
        assert(np.shape(Vg) == self.shape)
        assert(np.shape(Vc) == (self.output_size,))

        '''
        Vd = np.reshape(Vd, (-1, 1))
        Vd = np.repeat(Vd, self.output_size, axis=1)
        Vd = np.reshape(Vd, (-1, 1))
        '''

        Vg = np.reshape(Vg, (-1))

        Vc = np.reshape(Vc, (1, -1))
        Vc = np.repeat(Vc, self.input_size, axis=0)
        Vc = np.reshape(Vc, (-1))
        
        #####################################

        r = np.reshape(self.R, (-1))
        
        if fit == 1:
            '''
            points = np.concatenate((Vd, Vg), axis=1)
            I = self.fit1(points) / r * 1e6
            '''
            I = self.fit1(Vg) / r * 1e6
        else:
            '''
            # fit2 is fucked up for Vc, so we just make it positive and flip the sign afterwards.
            sign = np.sign(Vc)
            Vc = np.absolute(Vc)
            points = np.concatenate((Vc, Vg), axis=1)
            # fit2 produces a negative current!!!
            I = self.fit2(points) / r * 1e6
            I = I * sign
            '''
            sign = np.sign(Vc)
            I = self.fit2(Vg) / r * 1e6
            I = I * sign
            
        I = np.reshape(I, self.shape) * self.sign

        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W = np.clip(self.W + dwdt * dt, 0., 10e-9)
        
        # make sure to do this at the end.
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
        # I = np.sum(I, axis=0)
        return I
        
        
        
        
        
        
        
        

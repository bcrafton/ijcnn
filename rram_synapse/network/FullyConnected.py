
import numpy as np
from Memristors import Memristors

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class FullyConnected:
    
    def __init__(self, shape):
        self.shape = shape
        self.lr = 0.001
        self.dt = 100e-6
        
        self.input_size = self.shape[0]
        self.output_size = self.shape[1]
        
        self.weights = Memristors(self.shape)
        self.bias = np.zeros(self.output_size)
        
        # V = np.random.uniform(low=-1., high=1., size=shape)
        # for _ in range(100):
        #     self.weights.step(V, self.dt)
        
    def forward(self, X):
        X = np.reshape(X, (-1, 1))
        Y = self.weights.step(X, self.dt)
        Y = np.sum(Y, axis=0) + self.bias
        
        # print (Y)
        Y = softmax(Y)
        # print (Y)
        
        return Y
        
    
    def backward(self, X, E):
        _X = np.reshape(X, (-1, 1))
        _E = np.reshape(E, (1, -1))
        
        # print (np.average(_X))
        # print (_E)
        # print (np.average(DW))
        
        # DW = (-_X - _E)
        DW = -1. * (_X > 0.2) * E
        DB = 1e-5 * E
        
        DO = self.weights.step(DW, self.dt)
        self.bias -= DB 
        
        return DO
        
    '''
    def backward(self, X, E):    
        _X = X
        _X = np.reshape(_X, (-1, 1))
        
        _E = np.reshape(E, (1, -1))
        
        # if E is (-) we were too small
        # if E is (+) we were too large
        # so we subtract
        
        # for memristor
        # if E is (-) we were too small
        # so we want to apply a positive voltage to increase current
        # so we need to subtract E just like before.
        
        DW = _X * _E
        DB = 1e-5 * E
        
        # print (np.std(DB))
        
        _ = self.weights.step(-_X, self.dt / 1000.)
        DO = self.weights.step(-DW, self.dt)
        self.bias -= DB 

        return DO
    '''
    
    
    

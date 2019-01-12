import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit.pkl', 'rb') as f:
    fit = pickle.load(f)

######################################

i1 = fit((1., 1., 0., 1e6))
i2 = fit((1., 1., 0., 100e6))
print (i1 / i2)

i1 = fit((1., .3, 0., 1e6))
i2 = fit((1., .3, 0., 100e6))
print (i1 / i2)

i1 = fit((.3, 1., 0., 1e6))
i2 = fit((.3, 1., 0., 100e6))
print (i1 / i2)

print ('---------------')

######################################

U = 5e-14
D = 10e-9
W0 = 5e-9
RON = 1e3
ROFF = 5e4
P = 5            
W = np.random.uniform(low=1e-9, high=9e-9)
R = RON * (W / D) + ROFF * (1 - (W / D))
print (R)

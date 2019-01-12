import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)

with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)

######################################

i1 = fit1((1., 1.))
i2 = fit1((1., 0.45))
print (i1 / i2)

i1 = fit2((1., 1.))
i2 = fit2((1., 0.45))
print (i1 / i2)

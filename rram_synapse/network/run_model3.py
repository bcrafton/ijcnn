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

Vd = 0.
Vg = 1.
Vc = 1.
r = 10e6

i = fit((Vd, Vg, Vc, r))
print (i)

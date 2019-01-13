import numpy as np
from scipy import interpolate
from scipy import optimize
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

vg = np.linspace(0., 1., 101)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=101) 
vd = np.reshape(vd, (-1, 1))
points = np.concatenate((vd, vg), axis=1)
i1 = fit1(points)

vc = np.ones(shape=101) 
vc = np.reshape(vc, (-1, 1))
points = np.concatenate((vc, vg), axis=1)
i2 = fit2(points)

vc = np.ones(shape=101) * -1.
vc = np.reshape(vc, (-1, 1))
points = np.concatenate((vc, vg), axis=1)
i3 = fit2(points)

# normal_i = (i1 / np.max(i1))
# print (np.concatenate((normal_i, vg), axis=1))

# vg = np.reshape(vg, (-1))
# i1 = np.reshape(i1, (-1))

vd = vd[32:101]
vg = vg[32:101]
# i1 = i1[32:101]

points = np.concatenate((vd, vg), axis=1)
i1 = fit1(points)

# fit = optimize.curve_fit(lambda t, a, b, c: a * np.exp(b*t) + c, vg, i1)
# fit_i1 = 2.24156665e-07*np.exp(1.18705624e+00*vg) + -3.39535853e-07
# print (fit)
# plt.plot(vg, fit_i1, label='fit')

vg = 1. / (2.24156665e-07*np.exp(1.18705624e+00*vg) + -3.39535853e-07)
print (vg)

# plt.plot(vg, i1, label='i1')

# plt.legend()
# plt.show()

######################################


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

############################################################################

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)
    
with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)

vg = np.linspace(0., 1., 101)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=101) 
vd = np.reshape(vd, (-1, 1))

points = np.concatenate((vd, vg), axis=1)
i1 = fit1(points)

############################################################################

i1 = np.reshape(i1, (-1))
vg = np.reshape(vg, (-1))

i1 = i1[32:]
vg = vg[32:]

############################################################################

max_i1 = i1[-1]
val = 0.5 * max_i1

# vg_fit = np.interp(i1, i1, vg)
vg_fit = np.interp(val, i1, vg)
print(vg_fit)

'''
plt.plot(i1, vg, label='i1')
plt.plot(i1, vg_fit, label='fit')
plt.legend()
plt.show()
'''

############################################################################

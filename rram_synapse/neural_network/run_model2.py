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

vc = np.ones(shape=101) 
vc = np.reshape(vc, (-1, 1))

points = np.concatenate((vc, vg), axis=1)
i = fit2(points)

############################################################################

i = np.reshape(i, (-1))
vg = np.reshape(vg, (-1))

############################################################################

min_i = np.min(i)
target = np.linspace(0.25, 0.75) * min_i
print (min_i, target)

vg_fit = np.interp(target, i, vg)
print(vg_fit)

############################################################################

plt.plot(i, vg, label='vg')
plt.plot(target, vg_fit, label='vg fit')
plt.legend()
plt.show()

############################################################################

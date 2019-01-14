import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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
i = fit1(points)
############################################################################
A = np.linspace(0.01, 1., 101)
############################################################################
vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))

f = interp1d(i, vg)

max_i = np.max(i)
target = A * max_i
A1 = f(target)

vg = A1
vg = np.reshape(vg, (-1, 1))
points = np.concatenate((vd, vg), axis=1)
i = fit1(points)

i = np.reshape(i, (-1))

plt.plot(A, i * 2.5)
plt.plot(A, A / 1e6)
plt.legend()
plt.show()
############################################################################


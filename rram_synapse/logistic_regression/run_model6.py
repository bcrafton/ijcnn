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

vg = np.linspace(0., 1., 100)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=100) 
vd = np.reshape(vd, (-1, 1))
points = np.concatenate((vd, vg), axis=1)
i1 = fit1(points)

vc = np.ones(shape=100) 
vc = np.reshape(vc, (-1, 1))
points = np.concatenate((vc, vg), axis=1)
i2 = fit2(points)

vc = np.ones(shape=100) * -1.
vc = np.reshape(vc, (-1, 1))
points = np.concatenate((vc, vg), axis=1)
i3 = fit2(points)

# plt.plot(vg, i1)
# plt.plot(vg, i2)
# plt.plot(vg, i3)
plt.show()

######################################


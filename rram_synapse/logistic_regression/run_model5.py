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

case = 2

# subtract
if case == 0:
    vd = np.zeros(shape=100)
    vg = np.linspace(0., 1., 100)
    vc = np.ones(shape=100)
    r  = np.ones(shape=100) * 10e6


# add
if case == 1:
    vd = np.zeros(shape=100)
    vg = np.linspace(0., 1., 100)
    vc = np.ones(shape=100) * -1.
    r  = np.ones(shape=100) * 10e6


# read 
if case == 2:
    vd = np.ones(shape=100)
    vg = np.linspace(0., 1., 100)
    vc = np.zeros(shape=100)
    r  = np.ones(shape=100) * 10e6


vd = np.reshape(vd, (-1, 1))
vg = np.reshape(vg, (-1, 1))
vc = np.reshape(vc, (-1, 1))
r  = np.reshape(r,  (-1, 1))

points = np.concatenate((vd, vg, vc, r), axis=1)
i = fit(points)
i = np.absolute(i) 
# i = i / np.max(i)

plt.plot(vg, i)
plt.show()

######################################


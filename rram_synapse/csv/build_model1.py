import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# http://www.ece.uci.edu/docs/hspice/hspice_2001_2-87.html
# i guess we can do optimization ? 

try:
    import cPickle as pickle
except ImportError:
    import pickle

print ("loading data")

vd = np.genfromtxt('vd.csv', delimiter=',').reshape(-1, 1)
vg = np.genfromtxt('vg.csv', delimiter=',').reshape(-1, 1)
vc = np.genfromtxt('vc.csv', delimiter=',').reshape(-1, 1)
i  = np.genfromtxt('i.csv',  delimiter=',').reshape(-1, 1)

print ("building interpolator")

# points = np.concatenate((vd, vg), axis=1)
points = np.concatenate((vc, vg), axis=1)
x = points
y = i

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('fit.pkl', 'wb') as f:
    pickle.dump(fit, f)



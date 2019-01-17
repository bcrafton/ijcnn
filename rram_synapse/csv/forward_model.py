import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

print ("loading data")

vg = np.genfromtxt('forward_vg.csv', delimiter=',').reshape(-1, 1)
r  = np.genfromtxt('forward_r.csv', delimiter=',').reshape(-1, 1)
p  = np.genfromtxt('forward_p.csv', delimiter=',').reshape(-1, 1)
pr = np.genfromtxt('forward_pr.csv', delimiter=',').reshape(-1, 1)
pn = np.genfromtxt('forward_pn.csv', delimiter=',').reshape(-1, 1)

print ("building interpolator")

points = np.concatenate((vg, r), axis=1)
x = points
y = p

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('forward.pkl', 'wb') as f:
    pickle.dump(fit, f)



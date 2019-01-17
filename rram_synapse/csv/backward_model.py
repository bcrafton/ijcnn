import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

print ("loading data")

vg = np.genfromtxt('backward_vg.csv', delimiter=',').reshape(-1, 1)
r  = np.genfromtxt('backward_r.csv', delimiter=',').reshape(-1, 1)
p  = np.genfromtxt('backward_p.csv', delimiter=',').reshape(-1, 1)
pr = np.genfromtxt('backward_pr.csv', delimiter=',').reshape(-1, 1)
pn = np.genfromtxt('backward_pn.csv', delimiter=',').reshape(-1, 1)

print ("building interpolator")

points = np.concatenate((vg, r), axis=1)
x = points
y = p

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('backward.pkl', 'wb') as f:
    pickle.dump(fit, f)



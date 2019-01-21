import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

#####################################################################################

vd = np.genfromtxt('backward_vd.csv', delimiter=',').reshape(-1, 1)
vg = np.genfromtxt('backward_vg.csv', delimiter=',').reshape(-1, 1)
vc = np.genfromtxt('backward_vc.csv', delimiter=',').reshape(-1, 1)
i  = np.genfromtxt('backward_i.csv',  delimiter=',').reshape(-1, 1)

points = np.concatenate((vc, vg), axis=1)
x = points
y = i

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('backward.pkl', 'wb') as f:
    pickle.dump(fit, f)

#####################################################################################3

vg = np.genfromtxt('backward_pwr_vg.csv', delimiter=',').reshape(-1, 1)
r  = np.genfromtxt('backward_pwr_r.csv', delimiter=',').reshape(-1, 1)
p  = np.genfromtxt('backward_pwr_p.csv', delimiter=',').reshape(-1, 1)
pr = np.genfromtxt('backward_pwr_pr.csv', delimiter=',').reshape(-1, 1)
pn = np.genfromtxt('backward_pwr_pn.csv', delimiter=',').reshape(-1, 1)

points = np.concatenate((vg, r), axis=1)
x = points
y = p

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('backward_pwr.pkl', 'wb') as f:
    pickle.dump(fit, f)

#####################################################################################

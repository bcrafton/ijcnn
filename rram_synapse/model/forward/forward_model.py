import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

###################################################################################

vd = np.genfromtxt('forward_vd.csv', delimiter=',').reshape(-1, 1)
vg = np.genfromtxt('forward_vg.csv', delimiter=',').reshape(-1, 1)
vc = np.genfromtxt('forward_vc.csv', delimiter=',').reshape(-1, 1)
i  = np.genfromtxt('forward_i.csv',  delimiter=',').reshape(-1, 1)

points = np.concatenate((vd, vg), axis=1)
x = points
y = i

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('forward.pkl', 'wb') as f:
    pickle.dump(fit, f)
    
###################################################################################    

vg = np.genfromtxt('forward_pwr_vg.csv', delimiter=',').reshape(-1, 1)
r  = np.genfromtxt('forward_pwr_r.csv', delimiter=',').reshape(-1, 1)
p  = np.genfromtxt('forward_pwr_p.csv', delimiter=',').reshape(-1, 1)
pr = np.genfromtxt('forward_pwr_pr.csv', delimiter=',').reshape(-1, 1)
pn = np.genfromtxt('forward_pwr_pn.csv', delimiter=',').reshape(-1, 1)

points = np.concatenate((vg, r), axis=1)
x = points
y = p

fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0, rescale=True)

with open('forward_pwr.pkl', 'wb') as f:
    pickle.dump(fit, f)


###################################################################################

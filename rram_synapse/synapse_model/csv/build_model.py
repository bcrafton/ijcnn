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

vd = np.genfromtxt('vd.csv', delimiter=',')
vg = np.genfromtxt('vg.csv', delimiter=',')
vc = np.genfromtxt('vc.csv', delimiter=',')
r  = np.genfromtxt('r.csv',  delimiter=',')
i  = np.genfromtxt('i.csv',  delimiter=',')

print ("building interpolator")

x = np.transpose([vd, vg, vc, r])
y = i
fit = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)

with open('fit.pkl', 'wb') as f:
    pickle.dump(fit, f)



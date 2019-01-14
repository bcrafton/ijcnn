import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

############################################################################
npoints = 20
############################################################################

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)
    
with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)

vg = np.linspace(0., 1., npoints)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=npoints) 
vd = np.reshape(vd, (-1, 1))

############################################################################

def current(X):
    vg = X
    vg = np.reshape(vg, (-1, 1))
    
    vd = np.ones(shape=npoints) 
    vd = np.reshape(vd, (-1, 1))
    
    points = np.concatenate((vd, vg), axis=1)
    i = fit1(points)
    i = np.reshape(i, (-1))
    return i

############################################################################
# X 

X = np.linspace(0.01, 1., npoints)
i_X = current(X)
############################################################################
# A

f = interp1d(i_X, X)

max_i = np.max(i_X)
scale_i = X * max_i
A = f(scale_i)

i_A = current(A)
############################################################################
# mult

mac = X
i_mac = X / 1e6
############################################################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.
f, ax = plt.subplots(2, 1)
f.set_size_inches(3.5, 7.)

ax[0].plot(X, X,   label='X',   marker='.', linestyle = 'None')
ax[0].plot(X, A,   label='A',   marker='.', linestyle = 'None')
# ax[0].plot(X, mac, label='mac', linestyle='--') # '-' for full line

ax[1].plot(X, i_X,   label='i_X',   marker='.', linestyle = 'None')
ax[1].plot(X, i_A,   label='i_A',   marker='.', linestyle = 'None')
ax[1].plot(X, i_mac, label='i_mac', linestyle='--') # '-' for full line
ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[1].ticklabel_format(useOffset=False)
# ax[1].yaxis.set_major_formatter(FormatStrFormatter('10%.0f'))

f.savefig('vscale.png', dpi=1000, bbox_inches='tight')
############################################################################





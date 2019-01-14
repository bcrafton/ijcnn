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

print (vg)

vc = np.ones(shape=101) 
vc = np.reshape(vc, (-1, 1))

points = np.concatenate((vc, vg), axis=1)
i = fit2(points)

############################################################################

vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))
# print(np.shape(vg), np.shape(i))

f = interp1d(i, vg)

min_i = np.min(i)
target = np.linspace(0.0001, 1.0) * min_i
val = f(target)

plt.plot(i, vg, label='vg')
plt.plot(target, val, label='vg fit')
plt.legend()
plt.show()

############################################################################


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

vd = np.ones(shape=101) 
vd = np.reshape(vd, (-1, 1))

points = np.concatenate((vd, vg), axis=1)
i = fit1(points)

############################################################################

i = np.reshape(i, (-1))
vg = np.reshape(vg, (-1))

############################################################################

# ratio = fit1(1., 1.) / fit1(1., 1e-2)
# print (ratio)

############################################################################
'''
vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))

f = interp1d(i, vg)

max_i = np.max(i)
target = np.linspace(1e-2, 1.0, 100) * max_i
val = f(target)

x = np.linspace(1e-2, 1.0)
y = x / 1e6

plt.plot(val, target, label='vg fit')
plt.plot(x, y, label='mult')
plt.legend()
plt.show()
'''
############################################################################
vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))

f = interp1d(i, vg)

max_i = np.max(i)
target = np.linspace(1. / 255., 1.0, 101) * max_i
val = f(target)

vg = val
vg = np.reshape(vg, (-1, 1))
points = np.concatenate((vd, vg), axis=1)
i = fit1(points)

i = np.reshape(i, (-1))
vg = np.reshape(vg, (-1))
############################################################################
# ratio = i[0] / i[-1]
# print (ratio)

print (1e-6 / i[-1])
############################################################################
plt.plot(vg, i, label='transistor', marker='.')
plt.legend()
plt.show()




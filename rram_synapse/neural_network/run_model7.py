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

vg = np.ones(shape=101) 
vg = np.reshape(vg, (-1, 1))

vd = np.linspace(0., 1., 101)
vd = np.reshape(vd, (-1, 1))

points = np.concatenate((vd, vg), axis=1)
i = fit1(points)
############################################################################
vd = np.reshape(vd, (-1))
i = np.reshape(i, (-1))

print (i[0])
print (i[-1])
print (i[-1] / i[0])

plt.plot(vd, i)
plt.show()
############################################################################


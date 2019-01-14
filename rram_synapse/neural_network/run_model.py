import numpy as np
from scipy import interpolate
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)
    
with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)

######################################

vg = np.linspace(0., 1., 101)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=101) 
vd = np.reshape(vd, (-1, 1))
points = np.concatenate((vd, vg), axis=1)
i1 = fit1(points)

vc = np.ones(shape=101) 
vc = np.reshape(vc, (-1, 1))
points = np.concatenate((vc, vg), axis=1)
i2 = fit2(points)

vc = np.ones(shape=101) * -1.
vc = np.reshape(vc, (-1, 1))
points = np.concatenate((vc, vg), axis=1)
i3 = fit2(points)

##############################

vg = vg[30:]
i1 = i1[30:]

vg = np.reshape(vg, (-1))
i1 = np.reshape(i1, (-1))

# fit = optimize.curve_fit(lambda t, a, b, c: a * np.exp(b*t) + c, vg, i1)
# fit = optimize.curve_fit(lambda t, a, b, c: np.log(t / a - c) / b, i1, vg)
# print (fit)
# fit_i1 = fit[0][0] * (fit[0][1] * i1) ** 2 + fit[0][2]

fit = np.polyfit(i1, vg, 4)
val = np.polyval(fit, i1)

plt.plot(i1, vg, label='i1')
plt.plot(i1, val, label='fit')
plt.legend()
plt.show()

######################################


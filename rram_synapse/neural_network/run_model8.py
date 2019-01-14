import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import keras

############################################################################

np.random.seed(0)

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)
    
with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)

############################################################################
vg = np.linspace(0., 1., 256)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=256) 
vd = np.reshape(vd, (-1, 1))

points = np.concatenate((vd, vg), axis=1)
i = fit1(points)
max_i = np.max(i)

vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))

forward_scale = interp1d(i, vg)
############################################################################
x_train = np.random.uniform(low=0.01, high=1., size=(25, 1))

print (x_train)

scaled_x_train = np.copy(x_train)
idx = np.where(scaled_x_train > 0.)
val = forward_scale(scaled_x_train[idx] * max_i)
scaled_x_train[idx] = val

print (scaled_x_train)

############################################################################
print (fit1(1., 0.75491336) * 2.6)
print (0.55332537 * (1. / 1e6))
############################################################################
W = np.ones(shape=(25, 10)) * (1. / 1e6)
############################################################################
A = np.reshape(scaled_x_train, (-1, 1))
############################################################################
vd = np.ones(shape=(25, 1))
vd = np.repeat(vd, 10, axis=1)
vd = np.reshape(vd, (-1, 1))

vg = np.repeat(A, 10, axis=1)
vg = np.reshape(vg, (-1, 1))

points = np.concatenate((vd, vg), axis=1)

i = fit1(points)
i = np.reshape(i, (25, 10))
# i = i * (A > 0)
i = i * 2.6
# i = np.sum(i, axis=0)
############################################################################
mult = W * np.reshape(x_train, (-1, 1))
############################################################################
print (i)
print ()
print (mult)
print ()
############################################################################
i = np.sum(i, axis=0)
mult = np.sum(mult, axis=0)
############################################################################
print (i)
print()
print (mult)





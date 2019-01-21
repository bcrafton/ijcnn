import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import keras

############################################################################

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

ratio = 1e-6 / max_i
print (ratio)
############################################################################
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (60000, 784))
x_train = x_train.astype('float32')
x_train = x_train / np.max(x_train)
x_train = x_train[0]

scaled_x_train = x_train
idx = np.where(scaled_x_train > 0.)
val = forward_scale(scaled_x_train[idx] * max_i)
scaled_x_train[idx] = val
############################################################################
W = np.ones(shape=(784, 100)) * (1. / 1e6)
############################################################################
A = np.reshape(scaled_x_train, (-1, 1))
############################################################################
vd = np.ones(shape=(784, 1))
vd = np.repeat(vd, 100, axis=1)
vd = np.reshape(vd, (-1, 1))

vg = np.repeat(A, 100, axis=1)
vg = np.reshape(vg, (-1, 1))

points = np.concatenate((vd, vg), axis=1)

i = fit1(points)
i = np.reshape(i, (784, 100))
i = i * (A > 0)
i = i * 2.6
# i = np.sum(i, axis=0)

idx = np.where(i > 0.)
print (i[idx][0:100])
# print (np.array(idx)[:, 0:100])

############################################################################
mult = W * np.reshape(x_train, (-1, 1))

idx = np.where(mult > 0.)
print (mult[idx][0:100])
# print (np.array(idx)[:, 0:100])

############################################################################

print (i)
print (mult)

############################################################################



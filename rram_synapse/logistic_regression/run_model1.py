import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit.pkl', 'rb') as f:
    fit = pickle.load(f)

input_size = 784
output_size = 10

Vd = np.random.uniform(low=0., high=1., size=(input_size))
Vg = np.random.uniform(low=0., high=1., size=(input_size, output_size))
Vc = np.random.uniform(low=0., high=1., size=(output_size))
r = np.random.uniform(low=1e-6, high=100e6, size=(input_size, output_size))

###########################################

Vd = np.reshape(Vd, (1, -1))
Vd = np.repeat(Vd, output_size, axis=0)
# print (Vd)
Vd = np.reshape(Vd, (-1, 1))

Vg = np.reshape(Vg, (-1, 1))

Vc = np.reshape(Vc, (-1, 1))
Vc = np.repeat(Vc, input_size, axis=1)
# print (Vc)
Vc = np.reshape(Vc, (-1, 1))

r = np.reshape(r, (-1, 1))


###########################################

print (np.shape(Vd), np.shape(Vg), np.shape(Vc), np.shape(r))

###########################################

points = np.concatenate((Vd, Vg, Vc, r), axis=1)
i = fit(points)

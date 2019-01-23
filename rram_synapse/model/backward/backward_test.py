import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('backward.pkl', 'rb') as f:
    backward = pickle.load(f)

#########################################
vg1 = np.linspace(0., 1., 100).reshape(-1, 1)
vc1 = np.linspace(1., 1., 100).reshape(-1, 1)

x1 = np.concatenate((vc1, vg1), axis=1)
y1 = backward(x1)

plt.plot(vg1, y1)
#########################################
vg2 = np.linspace(0., 1., 100).reshape(-1, 1)
vc2 = np.linspace(-1., -1., 100).reshape(-1, 1)

x2 = np.concatenate((vc2, vg2), axis=1)
y2 = backward(x2)

plt.plot(vg2, y2)
#########################################

plt.show()

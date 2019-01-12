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

'''
for _ in range(10):
    points1 = np.random.uniform(low=0., high=1., size=(100, 3))
    points2 = np.random.uniform(low=1e6, high=100e6, size=(100, 1))
    points = np.concatenate((points1, points2), axis=1)
    i = fit(points)
    print (i)
'''

vd = np.genfromtxt('vd.csv', delimiter=',').reshape(-1, 1)
vg = np.genfromtxt('vg.csv', delimiter=',').reshape(-1, 1)
vc = np.genfromtxt('vc.csv', delimiter=',').reshape(-1, 1)
r  = np.genfromtxt('r.csv',  delimiter=',').reshape(-1, 1)

points = np.concatenate((vd, vg, vc, r), axis=1)

i = fit(points)
l = np.shape(i)[0]
t = np.linspace(0, 1, l)

plt.subplot(6, 1, 1)
plt.plot(t, vd)

plt.subplot(6, 1, 2)
plt.plot(t, vg)

plt.subplot(6, 1, 3)
plt.plot(t, vc)

plt.subplot(6, 1, 4)
plt.plot(t, r)

plt.subplot(6, 1, 5)
plt.plot(t, i)

i  = np.genfromtxt('i.csv',  delimiter=',')
plt.subplot(6, 1, 6)
plt.plot(t, i)


plt.show()



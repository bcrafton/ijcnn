import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

try:
    import cPickle as pickle
except ImportError:
    import pickle

# with open('fit.pkl', 'rb') as f:
#     fit = pickle.load(f)

vd = np.genfromtxt('vd.csv', delimiter=',')
vg = np.genfromtxt('vg.csv', delimiter=',')
r  = np.genfromtxt('r.csv',  delimiter=',')
i  = np.genfromtxt('i.csv',  delimiter=',')

#print (np.shape(vd))
#print (np.shape(vg))
#print (np.shape(r))
#print (np.shape(i))

l = np.shape(i)[0]
t = np.linspace(0, 1, l)

plt.subplot(6, 1, 1)
plt.plot(t, vd)

plt.subplot(6, 1, 2)
plt.plot(t, vg)

plt.subplot(6, 1, 3)
plt.plot(t, r)


i_fit = np.genfromtxt('i.csv',  delimiter=',')
i_fit = np.concatenate((i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200], i_fit[0:200]))
assert(np.all(i_fit[0:10] == i_fit[200:210]))
i_fit = i_fit / r * 1e6
plt.subplot(6, 1, 4)
plt.plot(t, i_fit)

plt.subplot(6, 1, 5)
plt.plot(t, i)

plt.subplot(6, 1, 6)
plt.plot(t, i - i_fit)

plt.show()



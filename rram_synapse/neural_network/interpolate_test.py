
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import keras
import time

np.random.seed(0)

forward_i = np.genfromtxt('forward_i.csv', delimiter=',')
forward_vg = np.genfromtxt('forward_vg.csv', delimiter=',')

fit1 = interp1d(forward_vg, forward_i)
fit2 = np.polyfit(forward_vg, forward_i, deg=3)
W = np.random.uniform(low=-1., high=1., size=(784, 1000))

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train + 1.
x_train = x_train / np.max(x_train)

x_train = np.reshape(x_train, (60000, 784))

_x_train = np.copy(x_train)
_x_train = np.reshape(_x_train, (-1))

start = time.time()
f1 = fit1(_x_train)
print (np.shape(f1))
end = time.time()
print (end - start)

start = time.time()
f2 = np.polyval(fit2, _x_train)
print (np.shape(f2))
end = time.time()
print (end - start)

start = time.time()
f3 = np.dot(x_train, W)
print (np.shape(f3))
end = time.time()
print (end - start)

##################################

'''
forward_i = np.reshape(forward_i, (1, 1, -1))
forward_vg = np.reshape(forward_vg, (1, 1, -1))
_x_train = _x_train.reshape(60000)

import tensorflow as tf

[b, m, k] = tf.contrib.image.interpolate_spline(train_points=forward_vg, train_values=forward_i, query_points=_x_train, order=3) 

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()


_pts = sess.run(pts, feed_dict={})
'''



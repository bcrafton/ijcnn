
import numpy as np

w = np.random.uniform(low=0.0, high=1.0, size=(10, 10))
# w = np.random.uniform(low=1e-8, high=1e-6, size=(100, 100))
x = np.random.uniform(low=0.0, high=1.0, size=(10,))

y1 = np.dot(x, w)
print (y1)

y2 = np.sum(x * w.T, axis=1)
y3 = np.sum(x * w.T, axis=0)
y4 = np.sum(w.T * x, axis=1)
y5 = np.sum(w.T * x, axis=0)
print (y2)
print (y3)
print (y4)
print (y5)

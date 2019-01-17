
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import argparse
from Synapses1 import Synapses
from collections import deque
from scipy.interpolate import interp1d
np.set_printoptions(threshold=np.inf)

np.random.seed(0)

#######################################

try:
    import cPickle as pickle
except ImportError:
    import pickle

with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)
    
with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--dt', type=float, default=1e-5)
parser.add_argument('--dt_scale', type=float, default=8.)
parser.add_argument('--step', type=int, default=1)
args = parser.parse_args()

print (args)

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

vscale_y = 0.3
vscale_x = 1. - vscale_y

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#######################################

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

x_train = np.reshape(x_train, (TRAIN_EXAMPLES, 784))
x_train = x_train.astype('float32')

x_train = x_train / np.max(x_train)

# scale = vscale_x / np.max(x_train)
# x_train = scale * x_train + (x_train > 0) * vscale_y

#######################################

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

x_test = np.reshape(x_test, (TEST_EXAMPLES, 784))
x_test = x_test.astype('float32')

x_test = x_test / np.max(x_test)

# scale = vscale_x / np.max(x_test)
# x_test = scale * x_test + (x_test > 0) * vscale_y

#######################################
vg = np.linspace(0., 1., 101)
vg = np.reshape(vg, (-1, 1))

vd = np.ones(shape=101) 
vd = np.reshape(vd, (-1, 1))

points = np.concatenate((vd, vg), axis=1)
i = fit1(points)
forward_max = np.max(i)
forward_min = np.min(i)

vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))

forward_scale = interp1d(i, vg)
#######################################
x_train = np.reshape(x_train, (-1))

idx = np.where(x_train > 0.)
val = forward_scale(x_train[idx] * forward_max)
x_train[idx] = val

x_train = np.reshape(x_train, (TRAIN_EXAMPLES, 784))
#######################################
x_test = np.reshape(x_test, (-1))

idx = np.where(x_test > 0.)
val = forward_scale(x_test[idx] * forward_max)
x_test[idx] = val

x_test = np.reshape(x_test, (TEST_EXAMPLES, 784))
#######################################
vg = np.linspace(0., 1., 101)
vg = np.reshape(vg, (-1, 1))

vc = np.ones(shape=101) 
vc = np.reshape(vc, (-1, 1))

points = np.concatenate((vc, vg), axis=1)
i = fit2(points)
back_max = np.max(i)
back_min = np.min(i)

assert(np.absolute(back_min) > np.absolute(back_max))

vg = np.reshape(vg, (-1))
i = np.reshape(i, (-1))

backward_scale = interp1d(i, vg)
#######################################

'''
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
'''

def softmax(x):
    # may need scale smaller than this.
    # although this is exactly what we used in the other one.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
    return x * (1. - x)

def relu(x):
    return x * (x > 0)
  
def drelu(x):
    # USE A NOT Z
    return 1.0 * (x > 0)

#######################################

LAYER1 = 784
LAYER2 = 1000
LAYER3 = 10

w1 = np.load('weights1.npy')
s1 = np.load('sign1.npy')

w2 = np.load('weights2.npy')
s2 = np.load('sign2.npy')

weights1 = Synapses(shape=(LAYER1, LAYER2), sign=s1, weights=w1)
bias1 = np.zeros(shape=(LAYER2))
# rate1 = 0.75
# sign1 = np.random.choice([-1., 1.], size=(LAYER1, LAYER2), replace=True, p=[1.-rate1, rate1])

weights2 = Synapses(shape=(LAYER2, LAYER3), sign=s2, weights=w2)
bias2 = np.zeros(shape=(LAYER3))
# rate2 = 0.75
# sign2 = np.random.choice([-1., 1.], size=(LAYER2, LAYER3), replace=True, p=[1.-rate2, rate2])

low = 1e-8
high = 1e-6
b2 = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))
#######################################

count = deque(maxlen=100)

for epoch in range(args.epochs):
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))
    correct = 0
    
    for ex in range(TRAIN_EXAMPLES):
        ##############################
        pre1 = weights1.R
        pre2 = weights2.R
        ##############################
        A1 = x_train[ex]
        ##############################
        Vd = np.ones(shape=(LAYER1))
        Vg = np.repeat(np.reshape(A1, (-1, 1)), LAYER2, axis=1) 
        Vc = np.zeros(shape=(LAYER2))
        
        I = weights1.step(Vd, Vg, Vc, 1e-12, fit=1) # * sign2
        Z2 = np.sum(I, axis=0) * 2.6
        A2 = relu(Z2 / np.max(Z2))
        ##############################

        ##############################
        A2_scaled = np.copy(A2)
        idx = np.where(A2_scaled > 0.)
        x = np.clip(A2_scaled[idx] * forward_max, forward_min, forward_max)
        y = forward_scale(x)
        A2_scaled[idx] = y
        
        Vd = np.ones(shape=(LAYER2))
        Vg = np.repeat(np.reshape(A2_scaled, (-1, 1)), LAYER3, axis=1) 
        Vc = np.zeros(shape=(LAYER3))
        
        I = weights2.step(Vd, Vg, Vc, 1e-12, fit=1) # * sign2
        Z3 = np.sum(I, axis=0) * 2.6
        A3 = softmax(Z3 * 1e6)
        ##############################

        ##############################
        if (np.argmax(A3) == np.argmax(y_train[ex])):
            correct += 1
            count.append(1)
        else:
            count.append(0)

        ##############################
        if ((ex + 1) % 1000 == 0):
            acc = 1.0 * correct / (ex + 1)
            print (acc)
        ##############################
        

    
    
    
    

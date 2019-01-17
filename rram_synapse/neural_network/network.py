
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import argparse
import sys
from Synapses import Synapses
from collections import deque
from scipy.interpolate import interp1d
np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': lambda x: "{0:+0.3f}".format(x)})

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
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--dt', type=float, default=2e-5)
parser.add_argument('--dt_scale', type=float, default=4.)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--hidden', type=int, default=100)
parser.add_argument('--vc_scale', type=float, default=1.)
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
LAYER2 = args.hidden
LAYER3 = 10

weights1 = Synapses(shape=(LAYER1, LAYER2), rate=0.6)
bias1 = np.zeros(shape=(LAYER2))
# rate1 = 0.75
# sign1 = np.random.choice([-1., 1.], size=(LAYER1, LAYER2), replace=True, p=[1.-rate1, rate1])

weights2 = Synapses(shape=(LAYER2, LAYER3), rate=0.6)
bias2 = np.zeros(shape=(LAYER3))
# rate2 = 0.75
# sign2 = np.random.choice([-1., 1.], size=(LAYER2, LAYER3), replace=True, p=[1.-rate2, rate2])

low = 1e-8
high = 1e-6
b2 = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))
#######################################

count = deque(maxlen=1000)

max1 = 0.
max2 = 0.

for epoch in range(args.epochs):
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))
    correct = 0
    miss = 0
    
    for ex in range(TRAIN_EXAMPLES):
        ##############################
        pre1 = 1. / weights1.R
        pre2 = 1. / weights2.R
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
        
        ##############################
        ANS = y_train[ex]
        E = A3 - ANS
        
        D3 = E
        D2 = np.dot(D3, np.transpose(1. / weights2.R * weights2.sign)) * drelu(A2)

        DW2 = np.dot(A2.reshape(LAYER2, 1), D3.reshape(1, LAYER3)) * weights2.sign / 1e6
        DW1 = np.dot(A1.reshape(LAYER1, 1), D2.reshape(1, LAYER2)) * weights1.sign 
        ##############################

        ##############################
        w1 = 1. / np.copy(weights1.R)
        w2 = 1. / np.copy(weights2.R)

        _Z3 = np.dot(A2, w2 * weights2.sign)
        _A3 = softmax(_Z3 * 1e6)
        # print (A3, _A3)
        # print (np.argmax(A3), np.argmax(_A3))
        # assert(np.argmax(A3) == np.argmax(_A3))
        if (np.argmax(A3) != np.argmax(_A3)):
            miss = miss + 1

        pre2 = 1. / np.copy(w2)
        w1 = np.clip(w1 - 0.001 * DW1, 1e-8, 1e-6)
        w2 = np.clip(w2 - 0.001 * DW2, 1e-8, 1e-6)
        post2 = 1. / np.copy(w2)
        dw2 = pre2 - post2
        
        ##############################
        Vd = np.zeros(shape=(LAYER2))
        
        Vg = np.absolute(DW2)
        max2 = max(max2, np.max(Vg))
        Vg = Vg / max2
        idx = np.where(Vg > 0.)
        x = np.clip(Vg[idx] * back_min, back_min, back_max)
        y = backward_scale(x)
        Vg[idx] = y
        
        Vc = 1.0 * (D3 > 0.) + -1.0 * (D3 < 0.)
        
        I = weights2.step(Vd, Vg, Vc, args.dt, fit=2)
        ##############################
        I = -I
        # print (I / np.max(np.absolute(I)))
        # print (DW2 / np.max(np.absolute(DW2)))
        # print (np.average(np.absolute(I)), np.average(np.absolute(dw2)))
        # print ()
        ##############################
        Vd = np.zeros(shape=(LAYER1))
        
        Vg = np.absolute(DW1)
        max1 = max(max1, np.max(Vg))
        Vg = Vg / max1
        idx = np.where(Vg > 0.)
        x = np.clip(Vg[idx] * back_min, back_min, back_max)
        y = backward_scale(x)
        Vg[idx] = y
        
        Vc = 1.0 * (D2 > 0.) + -1.0 * (D2 < 0.)
        
        I = weights1.step(Vd, Vg, Vc, args.dt, fit=2)
        ##############################
        # weights1.R = 1. / w1
        # weights2.R = 1. / w2
        ##############################

        if ((ex + 1) % 1000 == 0):
            acc = 1.0 * correct / (ex + 1)
            miss_rate = 1.0 * miss / (ex + 1)
            print ("%d: accuracy: %f last 1000: %f miss %f" % (ex, acc, np.average(count), miss_rate))
            
        ##############################


    
    
    
    

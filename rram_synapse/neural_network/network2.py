
import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import argparse
from Synapses2 import Synapses
from collections import deque
from scipy.interpolate import interp1d
np.set_printoptions(threshold=np.inf)

np.random.seed(0)

#######################################

try:
    import cPickle as pickle
except ImportError:
    import pickle

'''
with open('fit1.pkl', 'rb') as f:
    fit1 = pickle.load(f)
    
with open('fit2.pkl', 'rb') as f:
    fit2 = pickle.load(f)
'''
#######################################

forward_i = np.genfromtxt('forward_i.csv', delimiter=',')
forward_vg = np.genfromtxt('forward_vg.csv', delimiter=',')
backward_i = np.genfromtxt('backward_i.csv', delimiter=',')
backward_vg  = np.genfromtxt('backward_vg.csv',  delimiter=',')

fit1 = interp1d(forward_vg, forward_i)
fit2 = interp1d(backward_vg, backward_i)

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--dt', type=float, default=1e-4)
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

i = fit1(vg)
forward_max = np.max(i)
forward_min = np.min(i)

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

i = fit2(vg)
back_max = np.max(i)
back_min = np.min(i)

assert(np.absolute(back_min) > np.absolute(back_max))

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

weights1 = Synapses(shape=(LAYER1, LAYER2), rate=0.75)
bias1 = np.zeros(shape=(LAYER2))
# rate1 = 0.75
# sign1 = np.random.choice([-1., 1.], size=(LAYER1, LAYER2), replace=True, p=[1.-rate1, rate1])

weights2 = Synapses(shape=(LAYER2, LAYER3), rate=0.75)
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
        # print (Z2)
        # print (A2)
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
        
        ANS = y_train[ex]
        E = A3 - ANS
        
        D3 = E
        # this does actually evaluate to zero when all weights are the same AND error is 9 0.1 and 1 0.9!
        D2 = np.dot(D3, np.transpose(1. / weights2.R * weights2.sign)) * drelu(A2)

        
        # this wont do anything because we scale it to be [0.45, 1] anyways.
        DW2 = np.dot(A2.reshape(LAYER2, 1), D3.reshape(1, LAYER3)) * weights2.sign # / 1e6
        DW1 = np.dot(A1.reshape(LAYER1, 1), D2.reshape(1, LAYER2)) * weights1.sign 
        ##############################
        
        ##############################
        Vd = np.zeros(shape=(LAYER2))
        
        Vg = np.absolute(DW2)
        # scale = vscale_x / np.max(Vg)
        # Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vg = Vg / np.max(Vg)
        idx = np.where(Vg > 0.)
        x = np.clip(Vg[idx] * back_min, back_min, back_max)
        y = backward_scale(x)
        Vg[idx] = y
        
        Vc = 1.0 * (D3 > 0.) + -0.25 * (D3 < 0.)
        
        for step in range(args.step):
            I = weights2.step(Vd, Vg, Vc, args.dt * args.dt_scale * (1. / float(args.step)), fit=2)
        
        # DO = np.sum(I, axis=0)
        # print (DO)
        ##############################

        ##############################
        Vd = np.zeros(shape=(LAYER1))
        
        Vg = np.absolute(DW1)
        
        # scale = vscale_x / np.max(Vg)
        # Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vg = Vg / np.max(Vg)
        idx = np.where(Vg > 0.)
        x = np.clip(Vg[idx] * back_min, back_min, back_max)
        y = backward_scale(x)
        Vg[idx] = y
        
        Vc = 1.0 * (D2 > 0.) + -1.0 * (D2 < 0.)
        
        for step in range(args.step):
            I = weights1.step(Vd, Vg, Vc, args.dt * (1. / float(args.step)), fit=2)
        
        # DO = np.sum(I, axis=0)
        # print (DO)
        ##############################

        ##############################
        post1 = weights1.R
        post2 = weights2.R

        DR2 = pre2 - post2
        sign_DR2 = np.sign(-DR2)
        sign_DW2 = np.sign(DW2)
        num2 = np.sum(np.absolute(DW2) > 0)
        ratio2 = np.sum(sign_DR2 == sign_DW2) / num2
        # print (ratio2)
        # if its happening bc the memristor rails out then that shudnt count
        # assert(ratio2 >= 0.99)

        DR1 = pre1 - post1
        sign_DR1 = np.sign(-DR1)
        sign_DW1 = np.sign(DW1)
        num1 = np.sum(np.absolute(DW1) > 0)
        ratio1 = np.sum(sign_DR1 == sign_DW1) / num1
        # print (ratio1)
        # if its happening bc the memristor rails out then that shudnt count
        # assert(ratio1 >= 0.99)

        # print (ex)

        ##############################
        if ((ex + 1) % 1000 == 0):
            acc = 1.0 * correct / (ex + 1)
            print ("%d: accuracy: %f last 100: %f" % (ex, acc, np.average(count)))
            print ("Z2  %0.10f %0.10f %0.10f" % (np.std(Z2),  np.min(Z2),  np.max(Z2)))
            print ("A2  %0.10f %0.10f %0.10f" % (np.std(A2),  np.min(A2),  np.max(A2)))
            print ("Z3  %0.10f %0.10f %0.10f" % (np.std(Z3),  np.min(Z3),  np.max(Z3)))
            print ("A3  %0.10f %0.10f %0.10f" % (np.std(A3),  np.min(A3),  np.max(A3)))
            print (np.min(post2 / 1e6), np.max(post2 / 1e6), np.average(post2 / 1e6), np.std(post2 / 1e6))
            print (np.min(post1 / 1e6), np.max(post1 / 1e6), np.average(post1 / 1e6), np.std(post1 / 1e6))
            print (np.min(DR2 / 1e6), np.max(DR2 / 1e6), np.average(DR2 / 1e6), np.std(DR2 / 1e6))
            print (np.min(DR1 / 1e6), np.max(DR1 / 1e6), np.average(DR1 / 1e6), np.std(DR1 / 1e6))
            
            print (np.average(DR2 * (DR2 < 0) / 1e6), np.std(DR2 * (DR2 < 0) / 1e6))
            print (np.average(DR1 * (DR1 < 0) / 1e6), np.std(DR1 * (DR1 < 0) / 1e6))
            print (np.average(DR2 * (DR2 > 0) / 1e6), np.std(DR2 * (DR2 > 0) / 1e6))
            print (np.average(DR1 * (DR1 > 0) / 1e6), np.std(DR1 * (DR1 > 0) / 1e6))
            
            print ()
        ##############################


    
    
    
    


import numpy as np
import pickle
import keras
import matplotlib.pyplot as plt
import argparse
from Synapses import Synapses
from collections import deque
np.set_printoptions(threshold=np.nan)

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--alpha', type=float, default=1e-3)
args = parser.parse_args()

print (args)

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

vscale_y = 0.45
vscale_x = 1. - vscale_y

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#######################################

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')

scale = vscale_x / np.max(x_train)
x_train = scale * x_train + (x_train > 0) * vscale_y

#######################################

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')

scale = vscale_x / np.max(x_test)
x_test = scale * x_test + (x_test > 0) * vscale_y

#######################################
'''
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
'''

def softmax(x):
    # may need scale smaller than this.
    # although this is exactly what we used in the other one.
    x = x * (1. / 1e-6)
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
LAYER2 = 400
LAYER3 = 10

weights1 = Synapses(shape=(LAYER1, LAYER2))
bias1 = np.zeros(shape=(LAYER2))

weights2 = Synapses(shape=(LAYER2, LAYER3))
bias2 = np.zeros(shape=(LAYER3))

high = 1. / np.sqrt(LAYER2)
b2 = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))
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
    
        ##############################
        A1 = x_train[ex]
        
        Vd = A1
        Vg = np.ones(shape=(LAYER1, LAYER2))
        Vc = np.zeros(shape=(LAYER2))
        
        I = weights1.step(Vd, Vg, Vc, 1e-12, fit=1) 
        Z2 = np.sum(I, axis=0)
        A2 = sigmoid(Z2) # probably pushing out all 0.5
        ##############################

        ##############################
        Vd = A2
        Vg = np.ones(shape=(LAYER2, LAYER3))
        Vc = np.zeros(shape=(LAYER3))
        
        I = weights2.step(Vd, Vg, Vc, 1e-12, fit=1) 
        Z3 = np.sum(I, axis=0)
        A3 = softmax(Z3)
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
        D2 = np.dot(D3, np.transpose(b2)) * dsigmoid(A2)
        
        DW2 = np.dot(A2.reshape(LAYER2, 1), D3.reshape(1, LAYER3))
        DW1 = np.dot(A1.reshape(LAYER1, 1), D2.reshape(1, LAYER2))
        ##############################
        
        ##############################
        Vd = np.zeros(shape=(LAYER2))
        
        Vg = np.absolute(DW2)
        scale = vscale_x / np.max(Vg)
        Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vc = 1.0 * (E > 0.) + -1.0 * (E < 0.)
        
        for step in range(100):
            I = weights2.step(Vd, Vg, Vc, 1e-9, fit=2)
        
        # DO = np.sum(I, axis=0)
        # print (DO)
        ##############################

        ##############################
        Vd = np.zeros(shape=(LAYER1))
        
        Vg = np.absolute(DW1)
        scale = vscale_x / np.max(Vg)
        Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vc = 1.0 * (D2 > 0.) + -1.0 * (D2 < 0.)
        
        for step in range(100):
            I = weights1.step(Vd, Vg, Vc, 1e-9, fit=2)
        
        # DO = np.sum(I, axis=0)
        # print (DO)
        ##############################

        ##############################
        post1 = weights1.R
        post2 = weights2.R
        '''
        DR = pre - post
        sign_DR = np.sign(-DR)
        sign_DW = np.sign(DW)
        num = np.sum(np.absolute(DW) > 0)
        ratio = np.sum(sign_DR == sign_DW) / num
        assert(ratio == 1.)
        '''
        ##############################
        
        ##############################
        acc = 1.0 * correct / (ex + 1)
        print ("%d: accuracy: %f last 100: %f" % (ex, acc, np.average(count)))
        
        # print (np.min(post / 1e6), np.max(post / 1e6), np.average(post / 1e6), np.std(post / 1e6))
        ##############################


    
    
    
    

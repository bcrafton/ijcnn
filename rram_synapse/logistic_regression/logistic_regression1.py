
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

def softmax(x):
    # may need scale smaller than this.
    # although this is exactly what we used in the other one.
    x = x * (1. / 1e-6)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#######################################

LAYER1 = 784
LAYER2 = 10

weights = Synapses(shape=(LAYER1, LAYER2))
bias = np.zeros(shape=(LAYER2))

#######################################

T = 1.
dt = 1e-3
steps = int(T / dt) + 1
Ts = np.linspace(0., T, steps)

#######################################

count = deque(maxlen=100)

for epoch in range(args.epochs):
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))
    correct = 0
    
    #######################################
    
    for ex in range(TRAIN_EXAMPLES):
        ##############################
        pre = weights.R
        print (ex)
        ##############################
    
        ##############################
        A1 = x_train[ex]
        
        Vd = A1
        Vg = np.ones(shape=(LAYER1, LAYER2))
        Vc = np.zeros(shape=(LAYER2))
        
        I = weights.step(Vd, Vg, Vc, 1e-12, fit=1) 
        Z2 = np.sum(I, axis=0)
        A2 = softmax(Z2)
        ##############################
        
        ##############################
        if (np.argmax(A2) == np.argmax(y_train[ex])):
            correct += 1
            count.append(1)
        else:
            count.append(0)
        
        ANS = y_train[ex]
        E = A2 - ANS
        
        DW = np.dot(A1.reshape(LAYER1, 1), E.reshape(1, LAYER2))
        ##############################
        
        ##############################
        Vd = np.zeros(shape=(LAYER1))
        
        Vg = np.absolute( DW )
        scale = vscale_x / np.max(Vg)
        Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vc = 1.0 * (E > 0.) + -1.0 * (E < 0.)
        
        ##############################
        for step in range(100):
            I = weights.step(Vd, Vg, Vc, 1e-9, fit=2)
        
        DO = np.sum(I, axis=0)
        print (DO)
        ##############################
        
        ##############################
        post = weights.R
        DR = pre - post
        sign_DR = np.sign(-DR)
        sign_DW = np.sign(DW)
        num = np.sum(np.absolute(DW) > 0)
        ratio = np.sum(sign_DR == sign_DW) / num
        assert(ratio == 1.)
        ##############################
        
        ##############################
        acc = 1.0 * correct / (ex + 1)
        print ("accuracy: %f last 100: %f" % (acc, np.average(count)))
        
        print (np.min(post / 1e6), np.max(post / 1e6), np.average(post / 1e6), np.std(post / 1e6))
        # print (np.min(post), np.max(post), np.average(post), np.std(post))
        # print (np.min(1. / post), np.max(1. / post), np.average(1. / post), np.std(1. / post))
        ##############################


    
    
    
    

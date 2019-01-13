
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

vscale_y = 0.3
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
LAYER2 = 100
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
    
        ##############################
        A1 = x_train[ex]
        
        Vd = A1
        Vg = np.ones(shape=(LAYER1, LAYER2))
        Vc = np.zeros(shape=(LAYER2))
        
        I = weights1.step(Vd, Vg, Vc, 1e-12, fit=1) # * sign1
        Z2 = np.sum(I, axis=0)
        # this is def way to small.
        A2 = relu(Z2 * 1e6)
        # print (A2)
        ##############################

        ##############################
        Vd = A2
        Vg = np.ones(shape=(LAYER2, LAYER3))
        Vc = np.zeros(shape=(LAYER3))
        
        I = weights2.step(Vd, Vg, Vc, 1e-12, fit=1) # * sign2
        Z3 = np.sum(I, axis=0)
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
        D2 = np.dot(D3, np.transpose(b2)) * drelu(A2)
        
        # this wont do anything because we scale it to be [0.45, 1] anyways.
        DW2 = np.dot(A2.reshape(LAYER2, 1), D3.reshape(1, LAYER3)) * weights2.sign / 1e6
        DW1 = np.dot(A1.reshape(LAYER1, 1), D2.reshape(1, LAYER2)) * weights1.sign 
        ##############################
        
        ##############################
        Vd = np.zeros(shape=(LAYER2))
        
        Vg = np.absolute(DW2)
        scale = vscale_x / np.max(Vg)
        Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vc = 1.0 * (D3 > 0.) + -1.0 * (D3 < 0.)
        
        for step in range(10):
            I = weights2.step(Vd, Vg, Vc, 1e-4, fit=2)
        
        # DO = np.sum(I, axis=0)
        # print (DO)
        ##############################

        ##############################
        Vd = np.zeros(shape=(LAYER1))
        
        Vg = np.absolute(DW1)
        scale = vscale_x / np.max(Vg)
        Vg = scale * Vg + (Vg > 0) * vscale_y
        
        Vc = 1.0 * (D2 > 0.) + -1.0 * (D2 < 0.)
        
        for step in range(10):
            I = weights1.step(Vd, Vg, Vc, 1e-4, fit=2)
        
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
        # assert(ratio2 >= 0.95)

        DR1 = pre1 - post1
        sign_DR1 = np.sign(-DR1)
        sign_DW1 = np.sign(DW1)
        num1 = np.sum(np.absolute(DW1) > 0)
        ratio1 = np.sum(sign_DR1 == sign_DW1) / num1
        # print (ratio1)
        # if its happening bc the memristor rails out then that shudnt count
        # assert(ratio1 >= 0.95)

        ##############################
        
        ##############################
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
        print ()
        ##############################


    
    
    
    

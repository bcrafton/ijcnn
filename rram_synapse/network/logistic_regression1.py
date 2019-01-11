
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

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#######################################

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')

scale = 0.7 / np.max(x_train)
x_train = scale * x_train + (x_train > 0) * 0.3

#######################################

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')

scale = 0.7 / np.max(x_test)
x_test = scale * x_test + (x_test > 0) * 0.3

#######################################

def softmax(x):
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
        # print (np.std(pre), np.average(pre))
        ##############################
    
        ##############################
        A1 = x_train[ex]
        
        Vd = A1
        Vg = np.ones(shape=(LAYER1, LAYER2))
        Vc = np.zeros(shape=(LAYER2))
        
        I, _ = weights.step(Vd, Vg, Vc, 1e-9) 
        Z2 = np.sum(I, axis=0)
        # print (Z2)
        
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
        DB = E
        ##############################
        
        ##############################
        Vd = np.zeros(shape=(LAYER1))
        
        Vg = np.absolute( DW )
        scale = 0.7 / np.max(Vg)
        Vg = scale * Vg + (Vg > 0) * 0.3
        
        Vc = 1.0 * (E > 0.) + -1.0 * (E < 0.)
        
        ##############################
        
        for step in range(10):
            # print (step)
            I, dwdt = weights.step(Vd, Vg, Vc, 1e-7)
        
        DO = np.sum(I, axis=0)
        # print (I)
        # print (Vc)
        # print (Vd)
        # print (Vg[:, np.argmin(E)])
        # print (E)
        # print (DO)
        
        bias -= args.alpha * DB
        ##############################
        
        ##############################
        post = weights.R
        # pre - DW = post
        # pre - post = DW
        DR = pre - post
        # smaller R = larger W
        # DR = -DR

        idx = np.where(np.absolute(DW) > 0)
        num = np.sum(np.absolute(DW) > 0)
        
        # print (np.sum(dwdt[idx] > 0), np.sum(dwdt[idx] < 0))
        # print (np.sum(DR[idx] > 0), np.sum(DR[idx] < 0))
        # print (np.sum(DW[idx] > 0), np.sum(DW[idx] < 0))
        
        sign_dwdt = np.sign(-dwdt)
        sign_DR = np.sign(-DR)
        sign_DW = np.sign(DW)
        
        # print (np.sum(sign_dwdt == sign_DW) / num)
        # print (np.sum(sign_DR == sign_DW) / num)
        
        # print (E)
        # print (DO)
        # print (sign_dwdt)
        # print (sign_DR)
        ##############################
        
        ##############################
        # print (np.sum(np.absolute(1. / DR)), np.sum(np.absolute(1. / post)))
        # print ( np.sum(np.absolute(DR)), np.sum(np.absolute(post)), np.sum(np.absolute(DR)) / np.sum(np.absolute(post)) )
        print (np.min(post / 1e6), np.max(post / 1e6), np.average(post / 1e6), np.std(post / 1e6))
        print (np.min(post), np.max(post), np.average(post), np.std(post))
        print (np.min(1. / post), np.max(1. / post), np.average(1. / post), np.std(1. / post))
        ##############################
        
        ##############################
        acc = 1.0 * correct / (ex + 1)
        print ("accuracy: %f last 100: %f" % (acc, np.average(count)))
        # print (y_train[ex])
        ##############################


    
    
    
    

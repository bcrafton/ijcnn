
import numpy as np
import math
import gzip
import time
import pickle
import argparse
import keras
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.00001)
parser.add_argument('--scale', type=float, default=50.0)
parser.add_argument('--low', type=float, default=1./5e4)
args = parser.parse_args()

print (args)

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

#######################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#######################################

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_train /= 255

#######################################

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
x_test /= 255

#######################################

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#######################################

LAYER1 = 784
LAYER2 = 10

low = args.low
high = args.low * args.scale

weights = np.random.uniform(low, high, size=(LAYER1, LAYER2))
bias = np.zeros(shape=(LAYER2))

#######################################

accs = []

for epoch in range(args.epochs):
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))
    
    correct = 0
    for ex in range(TRAIN_EXAMPLES):
        A1 = x_train[ex]
        Z2 = np.dot(A1, weights) + bias
        A2 = softmax(Z2)
        
        if (np.argmax(A2) == np.argmax(y_train[ex])):
            correct += 1
        
        ANS = y_train[ex]
        E = A2 - ANS
        
        DW = np.dot(A1.reshape(LAYER1, 1), E.reshape(1, LAYER2))
        DB = E
        
        # print (np.std(args.alpha * DW))
        
        weights -= args.alpha * DW
        bias -= args.alpha * DB
        
        weights = np.clip(weights, low, high)
     
    acc = 1.0 * correct / TRAIN_EXAMPLES
    accs.append(acc)
    
    print ("accuracy: " + str(acc))
    print (np.min(weights), np.max(weights), np.average(weights), np.std(weights))
    
# plt.hist(np.reshape(weights, (-1)))
# plt.show()


    
    
    
    

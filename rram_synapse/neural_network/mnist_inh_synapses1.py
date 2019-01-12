
import numpy as np
import argparse
import keras

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--examples', type=int, default=60000)
parser.add_argument('--scale', type=float, default=100.)
parser.add_argument('--low', type=float, default=1./100e6)
args = parser.parse_args()

LAYER1 = 784
LAYER2 = 400
LAYER3 = 10

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

#######################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

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

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_train = x_train / np.max(x_train)

y_test = keras.utils.to_categorical(y_test, 10)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
x_test = x_test / np.max(x_test)

#######################################

low = args.low
high = args.low * args.scale

#######################################

weights1 = np.ones(shape=(LAYER1, LAYER2)) * (low / 2.)
bias1 = np.zeros(shape=LAYER2)
rate1 = 0.75
sign1 = np.random.choice([-1., 1.], size=(LAYER1, LAYER2), replace=True, p=[1.-rate1, rate1])

weights2 = np.ones(shape=(LAYER2, LAYER3)) * (low / 2.)
bias2 = np.zeros(shape=LAYER3)
rate2 = 0.75
sign2 = np.random.choice([-1., 1.], size=(LAYER2, LAYER3), replace=True, p=[1.-rate2, rate2])

# hi = 1. / np.sqrt(LAYER2)
# lo = -hi
# b2 = np.random.uniform(low=lo, high=hi, size=(LAYER2, LAYER3))

# thinking b2 should be positive since w2 is positive.
# problem is we include the sign when we send back with w2.
b2 = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))

#######################################

for epoch in range(args.epochs):
    print ("epoch: %d/%d" % (epoch, args.epochs))
    
    ######################################################
    correct = 0
    for ex in range(0, args.examples, args.batch_size):
        start = ex 
        stop = ex + args.batch_size
    
        A1 = x_train[start:stop]
        Z2 = np.dot(A1, weights1 * sign1) # + bias1
        A2 = relu(Z2)
        # A2 = (1. / np.max(A2)) * A2
        # print (A2)
        Z3 = np.dot(A2, weights2 * sign2) # + bias2
        A3 = softmax(Z3)
        # print (A3)
        
        labels = y_train[start:stop]
        
        correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
        
        D3 = A3 - labels
        D2 = np.dot(D3, np.transpose(weights2 * sign2)) * drelu(A2)
        
        DW2 = np.dot(np.transpose(A2), D3) * sign2 # is this correct ? 
        DB2 = np.sum(D3, axis=0) 
        
        DW1 = np.dot(np.transpose(A1), D2) * sign1 # is this correct ? 
        DB1 = np.sum(D2, axis=0)
        
        # weights2 = np.clip(weights2 - args.lr * DW2, low, high)
        # weights1 = np.clip(weights1 - args.lr * DW1, low, high)

        weights2 = np.clip(weights2 - args.lr * DW2, .01, 1.)
        weights1 = np.clip(weights1 - args.lr * DW1, .01, 1.)

        bias2 = bias2 - args.lr * DB2
        bias1 = bias1 - args.lr * DB1
        
    train_acc = 1. * correct / ex
    ######################################################
    correct = 0

    A1 = x_test
    Z2 = np.dot(A1, weights1 * sign1) # + bias1
    A2 = relu(Z2)
    # A2 = (1. / np.max(A2)) * A2
    Z3 = np.dot(A2, weights2 * sign2) # + bias2
    A3 = softmax(Z3)
    
    labels = y_test
    
    correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
            
    test_acc = 1. * correct / TEST_EXAMPLES
    ######################################################
    
    print ("lr: %0.12f | train: %f | test: %f" % (args.lr, train_acc, test_acc))
    
    
    
    
    
    

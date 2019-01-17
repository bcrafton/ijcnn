
import numpy as np
import argparse
import keras
from scipy.interpolate import interp1d
np.set_printoptions(formatter={'float': lambda x: "{0:+0.3f}".format(x)})
 
#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--examples', type=int, default=60000)
parser.add_argument('--scale', type=float, default=100.)
parser.add_argument('--low', type=float, default=1e-8)
args = parser.parse_args()

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

LAYER1 = 784
LAYER2 = 100
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

weights1 = np.ones(shape=(LAYER1, LAYER2)) * (1. / 0.505e8)
bias1 = np.zeros(shape=LAYER2)
rate1 = 0.6
sign1 = np.random.choice([-1., 1.], size=(LAYER1, LAYER2), replace=True, p=[1.-rate1, rate1])

weights2 = np.ones(shape=(LAYER2, LAYER3)) * (1. / 0.505e8)
bias2 = np.zeros(shape=LAYER3)
rate2 = 0.6
sign2 = np.random.choice([-1., 1.], size=(LAYER2, LAYER3), replace=True, p=[1.-rate2, rate2])

# hi = 1. / np.sqrt(LAYER2)
# lo = -hi
# b2 = np.random.uniform(low=lo, high=hi, size=(LAYER2, LAYER3))

# thinking b2 should be positive since w2 is positive.
# problem is we include the sign when we send back with w2.
# b2 = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))

# print (np.min(1. / weights1 / 1e6), np.max(1. / weights1 / 1e6), np.average(1. / weights1 / 1e6), np.std(1./ weights1 / 1e6))
# print (np.min(1. / weights2 / 1e6), np.max(1. / weights2 / 1e6), np.average(1. / weights2 / 1e6), np.std(1./ weights2 / 1e6))

#######################################

for epoch in range(args.epochs):
    print ("epoch: %d/%d" % (epoch, args.epochs))
    
    ######################################################
    correct = 0
    for ex in range(0, args.examples, args.batch_size):
        pre1 = 1. / weights1
        pre2 = 1. / weights2
    
        start = ex 
        stop = ex + args.batch_size
    
        A1 = x_train[start:stop]
        Z2 = np.dot(A1, weights1 * sign1) # + bias1
        A2 = relu(Z2 / np.max(Z2))
        Z3 = np.dot(A2, weights2 * sign2) # + bias2
        A3 = softmax(Z3 / high)
        
        labels = y_train[start:stop]
        
        correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
        
        D3 = A3 - labels
        D2 = np.dot(D3, np.transpose(weights2 * sign2)) * drelu(A2)
        
        DW2 = np.dot(np.transpose(A2), D3) * sign2 / 1e6
        DB2 = np.sum(D3, axis=0) 
        
        DW1 = np.dot(np.transpose(A1), D2) * sign1  
        DB1 = np.sum(D2, axis=0)

        weights2 = np.clip(weights2 - args.lr * DW2, low, high)
        weights1 = np.clip(weights1 - args.lr * DW1, low, high)
        bias2 = bias2 - args.lr * DB2
        bias1 = bias1 - args.lr * DB1

        post1 = 1. / weights1 
        post2 = 1. / weights2

        DR1 = pre1 - post1
        DR2 = pre2 - post2
        
        #####################################
        if ((ex + 1) % 1000 == 0):
        
            Vg = np.absolute(DW1)   
            Vg = Vg / np.max(Vg)
            idx = np.where(Vg > 0.)
            x = np.clip(Vg[idx] * back_min, back_min, back_max)
            y = backward_scale(x)
            Vg[idx] = y
            Vg = np.reshape(Vg, (-1, 1))
            
            Vc = 1.0 * (D2 > 0.) + -1.0 * (D2 < 0.)
            Vc = np.reshape(Vc, (1, -1))
            Vc = np.repeat(Vc, LAYER1, axis=0)
            Vc = np.reshape(Vc, (-1, 1))

            r = np.reshape(1. / weights1, (-1, 1))
            
            sign = np.sign(Vc)
            Vc = np.absolute(Vc)
            points = np.concatenate((Vc, Vg), axis=1)
            I = fit2(points) # / r * 1e6
            I = I * sign
            I = np.reshape(I, (LAYER1, LAYER2)) * sign1
            
            #####################################
            
            mask = (pre1 > 2e6) * (pre1 < 95e6) * (post1 > 2e6) * (post1 < 95e6)
            I = -I
            DW1 = DW1 * mask
            DW1 = (DW1 / np.max(DW1))
            I = I * mask
            I =  (I / np.max(I)) 
            diff = np.sum(np.absolute(I - DW1)) / np.sum(np.absolute(I))
            if (diff > 1e-3):
                print (diff)

            # print (diff)
            # tmp = np.random.uniform(0., 1., size=np.shape(DW1))
            # diff = np.sum(np.absolute(I - tmp)) / np.sum(np.absolute(I))
            # print (diff)
            
            if not np.all(np.sign(I) == np.sign(DW1)):
                # print (np.sign(I) == np.sign(DW2))
                # print (D3)
                # assert( np.all(np.sign(I) == np.sign(DW2)) )
                pass
            
            #####################################
            
            Vg = np.absolute(DW2)   
            Vg = Vg / np.max(Vg)
            idx = np.where(Vg > 0.)
            x = np.clip(Vg[idx] * back_min, back_min, back_max)
            y = backward_scale(x)
            Vg[idx] = y
            Vg = np.reshape(Vg, (-1, 1))
            
            Vc = 1.0 * (D3 > 0.) + -1.0 * (D3 < 0.)
            Vc = np.reshape(Vc, (1, -1))
            Vc = np.repeat(Vc, LAYER2, axis=0)
            Vc = np.reshape(Vc, (-1, 1))

            r = np.reshape(1. / weights2, (-1, 1))
            
            sign = np.sign(Vc)
            Vc = np.absolute(Vc)
            points = np.concatenate((Vc, Vg), axis=1)
            I = fit2(points) # / r * 1e6
            I = I * sign
            I = np.reshape(I, (LAYER2, LAYER3)) * sign2
            
            #####################################
        
            mask = (pre2 > 2e6) * (pre2 < 95e6) * (post2 > 2e6) * (post2 < 95e6)
            I = -I
            DW2 = DW2 * mask
            DW2 = (DW2 / np.max(DW2))
            I = I * mask
            I =  (I / np.max(I)) 
            diff = np.sum(np.absolute(I - DW2)) / np.sum(np.absolute(I))
            if (diff > 1e-3):
                print (diff)
            
            if not np.all(np.sign(I) == np.sign(DW2)):
                # print (np.sign(I) == np.sign(DW2))
                # print (D3)
                # assert( np.all(np.sign(I) == np.sign(DW2)) )
                pass
                
            #####################################

            print (ex)
            print ("Z2  %0.10f %0.10f %0.10f" % (np.std(Z2),  np.min(Z2),  np.max(Z2)))
            print ("A2  %0.10f %0.10f %0.10f" % (np.std(A2),  np.min(A2),  np.max(A2)))
            print ("Z3  %0.10f %0.10f %0.10f" % (np.std(Z3),  np.min(Z3),  np.max(Z3)))
            print ("A3  %0.10f %0.10f %0.10f" % (np.std(A3),  np.min(A3),  np.max(A3)))
            print (np.min(post2 / 1e6), np.max(post2 / 1e6), np.average(post2 / 1e6), np.std(post2 / 1e6))
            print (np.min(post1 / 1e6), np.max(post1 / 1e6), np.average(post1 / 1e6), np.std(post1 / 1e6))
            print (np.min(DR2 / 1e6), np.max(DR2 / 1e6), np.average(DR2 / 1e6), np.std(DR2 / 1e6))
            print (np.min(DR1 / 1e6), np.max(DR1 / 1e6), np.average(DR1 / 1e6), np.std(DR1 / 1e6))
            print ()
            
            np.save('weights1', weights1) 
            np.save('weights2', weights2)

            np.save('sign1', sign1) 
            np.save('sign2', sign2)

            train_acc = 1. * correct / ex
        ######################################################
    correct = 0

    A1 = x_test
    Z2 = np.dot(A1, weights1 * sign1) # + bias1
    # A2 = sigmoid(Z2 / np.max(Z2))
    A2 = relu(Z2 / np.max(Z2))
    # A2 = relu(Z2)
    Z3 = np.dot(A2, weights2 * sign2) # + bias2
    A3 = softmax(Z3 / high)
    
    labels = y_test
    
    correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
            
    test_acc = 1. * correct / TEST_EXAMPLES
    ######################################################
    
    print ("lr: %0.12f | train: %f | test: %f" % (args.lr, train_acc, test_acc))
    



    

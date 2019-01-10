
import numpy as np
import matplotlib.pyplot as plt
import keras
from neuron import neuron_group

np.random.seed(0)

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.
x_train = x_train.reshape(TRAIN_EXAMPLES, 28 * 28)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test / 255.
x_test = x_test.reshape(TEST_EXAMPLES, 28 * 28)
y_test = keras.utils.to_categorical(y_test, 10)

#############

size = 400

#transconductances of two FETs
gf=1e-8
gm=1e-8

#intecept of Id in charging phase, can be view as bias. Id=V0-gf*Vs;
v0=0.4 * gf 

#threshold voltage of discharging FET
vmth=0.25

#Gate voltage inputs 
#Vgf-inhibition control, also determines bound voltages
#Vgm-excitation, tunes firing rate

#Vgf=0.4 #inhibition on, non-spiking
vgf=0.3 #inhibition off, spiking
vgm=0.35 #excitation input, range: (0.25,0.4)

#capacitence
c=1e-9

# neuron
n = neuron_group(size=size, c=c, vgm=vgm, gm=gm, vgf=vgf, gf=gf, v0=v0, vmth=vmth)

#############

# time
T = 1.
dt = 1e-6
steps = int(T / dt) + 1
Ts = np.linspace(0., T, steps)

for t in Ts:
    n.step(t, dt)

print(np.shape(Ts))
print(np.shape(n.vs))

plt.plot(Ts, n.vs)
plt.show()

#############









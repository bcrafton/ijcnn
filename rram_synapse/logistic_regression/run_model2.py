import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from Synapses import Synapses

input_size = 784
output_size = 10

Vd = np.random.uniform(low=0., high=1., size=input_size)
Vg = np.random.uniform(low=0., high=1., size=(input_size, output_size))
Vc = np.random.uniform(low=0., high=1., size=output_size)

###########################################

s = Synapses(shape=(input_size, output_size))

###########################################

T = 1.
dt = 1e-6
steps = int(T / dt) + 1
Ts = np.linspace(0., T, steps)

for t in Ts:
    i = s.step(Vd, Vg, Vc, dt)


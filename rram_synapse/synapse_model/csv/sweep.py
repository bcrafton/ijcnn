
import numpy as np

head = '.data data1\n'
vals = '+ d g c r\n'
data = ''
foot = '.enddata\n'

d_sweep = np.linspace(0.0, 1.0, 15)
g_sweep = np.linspace(0.0, 1.0, 15)
c_sweep = np.linspace(-1.0, 1.0, 25)
r_sweep = np.linspace(1e6, 100e6, 15)

for d in d_sweep:
    for g in g_sweep:
        for c in c_sweep:
            for r in r_sweep:
                point = '+ %0.2f %0.2f %0.2f %0.0f\n' % (d, g, c, r)
                data = data + point

content = head + vals + data + foot
f = open('data', 'wb')
f.write(content)
f.close()

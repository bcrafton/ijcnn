
import numpy as np

head = '.data data1\n'
vals = '+ d g c r\n'
data = ''
foot = '.enddata\n'

v_sweep = np.linspace(0.0, 1.0, 15)
r_sweep = np.linspace(1e6, 100e6, 15)

for a in v_sweep:
    for b in v_sweep:
        for c in v_sweep:
            for d in r_sweep:
                point = '+ %0.2f %0.2f %0.2f %0.0f\n' % (a, b, c, d)
                data = data + point

content = head + vals + data + foot
f = open('data', 'wb')
f.write(content)
f.close()

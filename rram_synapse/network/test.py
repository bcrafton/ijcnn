
import numpy as np

Vc = range(5)

Vc = np.reshape(Vc, (1, -1))
Vc = np.repeat(Vc, 10, axis=0)
Vc = np.reshape(Vc, (-1, 1))

print (Vc)

Vd = range(10)

Vd = np.reshape(Vd, (-1, 1))
Vd = np.repeat(Vd, 5, axis=1)
Vd = np.reshape(Vd, (-1, 1))

print (Vd)

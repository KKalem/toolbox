import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import geometry as G

import time

a = np.random.random((100,3))
b = np.random.random((100,))
stacked_b = np.vstack([b,b,b]).T


t0 = time.time()
for t in range(1000):
    for i in range(3):
        a[:,i] *= b
print( time.time() - t0)
print(a.shape)

a = np.random.random((100,3))
b = np.random.random((100,))
stacked_b = np.vstack([b,b,b]).T

t0 = time.time()
for t in range(1000):
    stacked_b = np.vstack([b,b,b]).T
    a *= stacked_b
print( time.time() - t0)
print(a.shape)

a = np.random.random((100,3))
b = np.random.random((100,))

t0 = time.time()
for t in range(1000):
    b = np.atleast_2d(b)
    a = a*b.T
print( time.time() - t0)
print(a.shape)


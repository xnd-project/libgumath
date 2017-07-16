from signature import float, int
from dispatch import stencil
from dispatch import gufunc
import numpy as np

@stencil
def diff(a:float*2) -> float:
    """Perform a simple difference operation using a point and its left neighbour."""
    return a[1]-a[0]

@stencil
def smooth(a:float*3) -> float:
    """Perform a simple smoothing operation using a point and its two next neighbours."""
    return (a[0]+2*a[1]+a[2])/4

@gufunc
def pixelize(a:int*2*2) -> int:
    """Pixelize an array by replacing a 2x2 array by a single value computed as the mean value."""
    return (a[0][0] + a[0][1] + a[1][0] + a[1][1]) / 4


a = np.arange(16, dtype=np.float64)
d = diff(a)
print('diff({})={}'.format(a, d))
s = smooth(a)
print('smooth({})={}'.format(a, s))

#a = np.arange(256, dtype=np.float64).reshape(16,16)
#p = pixelize(a)


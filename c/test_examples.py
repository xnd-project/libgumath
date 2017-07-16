import examples
import numpy as np

def test_guf():
    a = np.arange(125, dtype=np.float32).reshape(5,5,5)

    out = examples.example_gufunc(a)
    print(out)

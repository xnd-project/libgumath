def compute_output_shape(stencil, input_size):
    assert len(stencil) == 1 # for now 1D stencils only
    stencil_size = stencil[0]
    return input_size - stencil_size + 1

def stencil(kernel):
    # 1) retrieve the stencil's shape
    signature = kernel.__annotations__
    stencil_shape = signature['a'].dims
    # 2) combine the shape with the input shape into an iterator
    def wrapper(input):
        output_shape = compute_output_shape(stencil_shape, len(input))
        output = [None] * output_shape
        for i in range(output_shape):
            output[i] = kernel(input[i:i+stencil_shape[0]])
        return output
    return wrapper


def gufunc(func):
    import numpy as np
    signature = func.__annotations__
    dims = signature['a'].dims
    def iterate(input):
        # from the shape of arg and the shape of the signature,
        # construct an iterator, and loop over it
        shape = input.shape
        shape = [a/b for a, b in zip(shape, dims)]
        output = np.zeros(shape)
        for o, i in np.nditer((output, input), flags=['multi_index']):
            o = func(i)
    return iterate

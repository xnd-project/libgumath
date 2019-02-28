#
# BSD 3-Clause License
#
# Copyright (c) 2017-2018, plures
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from ndtypes import ndt
from xnd import xnd
from ._gumath import *
from . import functions as _fn

try:
    from . import cuda as _cd
except ImportError:
    _cd = None


# ==============================================================================
#                              Init identity elements
# ==============================================================================

# This is done here now, perhaps it should be on the C level.
_fn.add.identity = 0
_fn.multiply.identity = 1


# ==============================================================================
#                             General fold function
# ==============================================================================

def fold(f, acc, x):
    return vfold(x, f=f, acc=acc)


# ==============================================================================
#                        NumPy's reduce in terms of fold
# ==============================================================================

def _get_axes(axes, ndim):
    type_err = "'axes' must be None, a single integer or a tuple of integers"
    value_err = "axis with value %d out of range"
    duplicate_err = "'axes' argument contains duplicate values"
    if axes is None:
        axes = tuple(range(ndim))
    elif isinstance(axes, int):
        axes = (axes,)
    elif not isinstance(axes, tuple) or \
         any(not isinstance(v, int) for v in axes):
        raise TypeError(type_err)

    if any(n >= ndim for n in axes):
        raise ValueError(value_err % n)

    if len(set(axes)) != len(axes):
        raise ValueError(duplicate_err)

    return list(axes)

def _copyto(dest, value):
    x = xnd(value, dtype=dest.dtype)
    _fn.copy(x, out=dest)

def reduce_cpu(f, x, axes, dtype):
    """NumPy's reduce in terms of fold."""
    axes = _get_axes(axes, x.ndim)
    if not axes:
        return x

    permute = [n for n in range(x.ndim) if n not in axes]
    permute = axes + permute

    T = x.transpose(permute=permute)

    N = len(axes)
    t = T.type.at(N, dtype=dtype)
    acc = xnd.empty(t, device=x.device)

    if f.identity is not None:
        _copyto(acc, f.identity)
        tl = T
    elif N == 1 and T.type.shape[0] > 0:
        hd, tl = T[0], T[1:]
        acc[()] = hd
    else:
        raise ValueError(
            "reduction not possible for function without an identity element")

    return fold(f, acc, tl)

def reduce_cuda(g, x, axes, dtype):
    """Reductions in CUDA use the thrust library for speed and have limited
       functionality."""
    if axes != 0:
        raise NotImplementedError("'axes' keyword is not implemented for CUDA")

    return g(x, dtype=dtype)

def get_cuda_reduction_func(f):
    if f == _cd.add:
        return _cd.reduce_add
    elif f == _cd.multiply:
        return _cd.reduce_multiply
    else:
        return None

def reduce(f, x, axes=0, dtype=None):
    if dtype is None:
        dtype = maxcast[x.dtype]

    g = get_cuda_reduction_func(f)
    if g is not None:
        return reduce_cuda(g, x, axes, dtype)

    return reduce_cpu(f, x, axes, dtype)


maxcast = {
  ndt("int8"): ndt("int64"),
  ndt("int16"): ndt("int64"),
  ndt("int32"): ndt("int64"),
  ndt("int64"): ndt("int64"),
  ndt("uint8"): ndt("uint64"),
  ndt("uint16"): ndt("uint64"),
  ndt("uint32"): ndt("uint64"),
  ndt("uint64"): ndt("uint64"),
  ndt("bfloat16"): ndt("float64"),
  ndt("float16"): ndt("float64"),
  ndt("float32"): ndt("float64"),
  ndt("float64"): ndt("float64"),
  ndt("complex32"): ndt("complex128"),
  ndt("complex64"): ndt("complex128"),
  ndt("complex128"): ndt("complex128"),

  ndt("?int8"): ndt("?int64"),
  ndt("?int16"): ndt("?int64"),
  ndt("?int32"): ndt("?int64"),
  ndt("?int64"): ndt("?int64"),
  ndt("?uint8"): ndt("?uint64"),
  ndt("?uint16"): ndt("?uint64"),
  ndt("?uint32"): ndt("?uint64"),
  ndt("?uint64"): ndt("?uint64"),
  ndt("?bfloat16"): ndt("?float64"),
  ndt("?float16"): ndt("?float64"),
  ndt("?float32"): ndt("?float64"),
  ndt("?float64"): ndt("?float64"),
  ndt("?complex32"): ndt("?complex128"),
  ndt("?complex64"): ndt("?complex128"),
  ndt("?complex128"): ndt("?complex128"),
}


# ==============================================================================
#                         Numba's GUVectorize on xnd arrays
# ==============================================================================

try:
    import numpy as np
    from numba.npyufunc import GUVectorize

    def xndvectorize(v):
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            lst = [ndt(s).to_nbformat() for s in v]
            sigs = [x[0] for x in lst]
            coretypes = [x[1] for x in lst]
            if (len(set(sigs))) != 1:
                raise ValueError(
                         "empty list or different signatures in multimethod")
            sig = sigs[0]
        else:
            raise TypeError("unsupported input type %s" % type(v))

        def wrap(func):
            guvec = GUVectorize(func, sig, nopython=True)
            for t in coretypes:
                guvec.add(t)
            g = guvec.build_ufunc()

            def h(*args, **kwargs):
                out = g(*args, **kwargs)
                view = xnd.from_buffer(out)
                ret = xnd.empty(view.type)
                np.copyto(np.array(ret, copy=False), out)
                return ret

            return h

        return wrap

except ImportError:
    xndvectorize = None

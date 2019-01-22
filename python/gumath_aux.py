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

# Python NDarray and functions for generating test cases.

from itertools import accumulate, count, product
from random import randrange
from collections import namedtuple
import math
import struct
import unittest
from randdec import all_unary, all_binary
from randfloat import un_randfloat, bin_randfloat
import numpy as np


def skip_if(condition, reason):
    if condition:
        raise unittest.SkipTest(reason)


# ======================================================================
#                          Minimal test cases
# ======================================================================

TEST_CASES = [
  ([float(i)/100.0 for i in range(2000)], "2000 * float64", "float64"),

  ([[float(i)/100.0 for i in range(1000)], [float(i+1) for i in range(1000)]],
   "2 * 1000 * float64", "float64"),

  (1000 * [[float(i+1) for i in range(2)]], "1000 * 2 * float64", "float64"),

  ([float(i)/10.0 for i in range(2000)], "2000 * float32", "float32"),

  ([[float(i)/10.0 for i in range(1000)], [float(i+1) for i in range(1000)]],
  "2 * 1000 * float32", "float32"),

  (1000 * [[float(i+1) for i in range(2)]], "1000 * 2 * float32", "float32"),
]


# ======================================================================
#            Definition of generalized slicing and indexing
# ======================================================================

def have_none(lst):
    if isinstance(lst, (list, tuple)):
        return any(have_none(item) for item in lst)
    if isinstance(lst, dict):
        return any(have_none(item) for item in lst.values())
    return lst is None

def sinrec(lst):
    if isinstance(lst, list):
        return [sinrec(item) for item in lst]
    elif isinstance(lst, (int, type(None))):
        return None if lst is None else math.sin(lst)
    else:
        raise TypeError("unexpected operand type '%s'" % type(lst))

def mulrec(lst1, lst2):
    if isinstance(lst1, list) and isinstance(lst2, list):
        return [mulrec(*pair) for pair in zip(lst1, lst2)]
    elif isinstance(lst1, (int, type(None))) and isinstance(lst2, (int, type(None))):
        return None if lst1 is None or lst2 is None else lst1 * lst2
    else:
        raise TypeError("unexpected operand types '%s', '%s'" %
                        (type(lst1), type(lst2)))


def maxlevel(lst):
    """Return maximum nesting depth"""
    maxlev = 0
    def f(lst, level):
        nonlocal maxlev
        if isinstance(lst, list):
            level += 1
            maxlev = max(level, maxlev)
            for item in lst:
                f(item, level)
    f(lst, 0)
    return maxlev

def getitem(lst, indices):
    """Definition for multidimensional slicing and indexing on arbitrarily
       shaped nested lists.
    """
    if not indices:
        return lst

    i, indices = indices[0], indices[1:]
    item = list.__getitem__(lst, i)

    if isinstance(i, int):
        return getitem(item, indices)

    # Empty slice: check if all subsequent indices are in range for the
    # full slice, raise IndexError otherwise. This is NumPy's behavior.
    if not item:
        if lst:
           _ = getitem(lst, (slice(None),) + indices)
        elif any(isinstance(k, int) for k in indices):
           raise IndexError
        return []

    return [getitem(x, indices) for x in item]

class NDArray(list):
    """A simple wrapper for using generalized slicing/indexing on a list."""
    def __init__(self, value, dtype=None):
        list.__init__(self, value)
        self.maxlevel = maxlevel(value)

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) > self.maxlevel: # NumPy
            raise IndexError("too many indices")

        if not all(isinstance(i, (int, slice)) for i in indices):
            raise TypeError(
                "index must be int or slice or a tuple of integers and slices")

        result = getitem(self, indices)
        return NDArray(result) if isinstance(result, list) else result

    def sin(self):
        return NDArray(sinrec(self))

    def __mul__(self, other):
        return NDArray(mulrec(self, other))



# ======================================================================
#                          Generate test cases 
# ======================================================================

SUBSCRIPT_FIXED_TEST_CASES = [
  [],
  [[]],
  [[], []],
  [[0], [1]],
  [[0], [1], [2]],
  [[0, 1], [1, 2], [2 ,3]],
  [[[]]],
  [[[0]]],
  [[[], []]],
  [[[0], [1]]],
  [[[0, 1], [2, 3]]],
  [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
  [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
]

SUBSCRIPT_VAR_TEST_CASES = [
  [[[0, 1], [2, 3]], [[4, 5, 6], [7]], [[8, 9]]],
  [[[0, 1], [2, 3]], [[4, 5, None], [None], [7]], [[], [None, 8]], [[9, 10]]],
  [[[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]], [[11, 12, 13, 14], [15, 16, 17], [18, 19]]],
  [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9]], [[10, 11]]]
]

def single_fixed(max_ndim=4, min_shape=1, max_shape=10):
    nat = count()
    shape = [randrange(min_shape, max_shape+1) for _ in range(max_ndim)]

    def f(ndim):
        if ndim == 0:
            return next(nat)
        return [f(ndim-1) for _ in range(shape[ndim-1])]

    return f(max_ndim)

def gen_fixed(max_ndim=4, min_shape=1, max_shape=10):
    assert max_ndim >=0 and min_shape >=0 and min_shape <= max_shape

    for _ in range(30):
        yield single_fixed(max_ndim, min_shape, max_shape)

def single_var(max_ndim=4, min_shape=1, max_shape=10):
    nat = count()

    def f(ndim):
        if ndim == 0:
            return next(nat)
        if ndim == 1:
            shape = randrange(min_shape, max_shape+1)
        else:
            n = 1 if min_shape == 0 else min_shape
            shape = randrange(n, max_shape+1)
        return [f(ndim-1) for _ in range(shape)]

    return f(max_ndim)

def gen_var(max_ndim=4, min_shape=1, max_shape=10):
    assert max_ndim >=0 and min_shape >=0 and min_shape <= max_shape

    for _ in range(30):
        yield single_var(max_ndim, min_shape, max_shape)


def genindices():
    for i in range(4):
        yield (i,)
    for i in range(4):
        for j in range(4):
            yield (i, j)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                yield (i, j, k)

def rslice(ndim):
    start = randrange(0, ndim+1)
    stop = randrange(0, ndim+1)
    step = 0
    while step == 0:
        step = randrange(-ndim-1, ndim+1)
    start = None if randrange(5) == 4 else start
    stop = None if randrange(5) == 4 else stop
    step = None if randrange(5) == 4 else step
    return slice(start, stop, step)

def rslice_neg(ndim):
    start = randrange(-ndim-1, ndim+1)
    stop = randrange(-ndim-1, ndim+1)
    step = 0
    while step == 0:
        step = randrange(-ndim-1, ndim+1)
    return slice(start, stop, step)

def multislice(ndim):
    return tuple(rslice(ndim) for _ in range(randrange(1, ndim+1)))

def randslices(ndim):
    for i in range(5):
        yield multislice(ndim)

def gen_indices_or_slices():
    for i in range(5):
        if randrange(2):
            yield (randrange(4), randrange(4), randrange(4))
        else:
            yield multislice(3)

def genslices(n):
    """Generate all possible slices for a single dimension."""
    def range_with_none():
        yield None
        yield from range(-n, n+1)

    for t in product(range_with_none(), range_with_none(), range_with_none()):
        s = slice(*t)
        if s.step != 0:
            yield s

def genslices_ndim(ndim, shape):
    """Generate all possible slice tuples for 'shape'."""
    iterables = [genslices(shape[n]) for n in range(ndim)]
    yield from product(*iterables)

def mixed_index(max_ndim):
    ndim = randrange(1, max_ndim+1)
    indices = []
    for i in range(1, ndim+1):
        if randrange(2):
            indices.append(randrange(-max_ndim, max_ndim))
        else:
            indices.append(rslice(ndim))
    return tuple(indices)

def mixed_index_neg(max_ndim):
    ndim = randrange(1, max_ndim+1)
    indices = []
    for i in range(1, ndim+1):
        if randrange(2):
            indices.append(randrange(-max_ndim, max_ndim))
        else:
            indices.append(rslice_neg(ndim))
    return tuple(indices)

def mixed_indices(max_ndim):
    for i in range(5):
        yield mixed_index(max_ndim)
    for i in range(5):
        yield mixed_index_neg(max_ndim)

def itos(indices):
    return ", ".join(str(i) if isinstance(i, int) else "%s:%s:%s" %
                     (i.start, i.stop, i.step) for i in indices)


# ======================================================================
#                Split a shape into N almost equal slices
# ======================================================================

def start(i, r, q):
    return i*(q+1) if i < r else r+i*q

def stop(i, r, q):
    return (i+1)*(q+1) if i < r else r+(i+1)*q

def step(i, r, q):
    return q+1 if i < r else q

def sl(i, r, q):
    return slice(start(i, r, q), stop(i, r, q))

def prepend(x, xs):
    return [(x,) + t for t in xs]

def last_column(i, r, q, n):
    return [(sl(i, r, q),) for i in range(n)]

def schedule(n, shape):
    assert isinstance(n, int) and isinstance(shape, list)
    if (n <= 0):
        raise ValueError("n must be greater than zero")
    if shape == []:
        return [()]
    m, ms = shape[0], shape[1:]
    if (m <= 0):
        raise ValueError("shape must be greater than zero")
    if n <= m:
        q, r = divmod(m, n)
        return last_column(0, r, q, n)
    else:
        q, r = divmod(n, m)
        return column(0, r, q, m, ms)

def column(i, r, q, m, ms):
    if i == m: return []
    return prepend(slice(i, i+1),
                   schedule(step(i, r, q), ms)) + \
           column(i+1, r, q, m, ms)

# ======================================================================
#                   Split an xnd object into N subtrees
# ======================================================================

def zero_in_shape(shape):
    for i in shape:
        if i == 0:
            return True
    return False

def split_xnd(x, n, max_outer=None):
    shape = list(x.type.shape)
    if zero_in_shape(shape):
        raise ValueError("split does not support zeros in shape")
    if max_outer is not None:
        shape = shape[:max_outer]
    indices_list = schedule(n, shape)
    return [x[i] for i in indices_list]


# ======================================================================
#                           Generate test cases
# ======================================================================

functions = {
  "unary": {
    "default": ["copy"],
    "arith": ["negative"],
    "complex_math_with_half": ["exp", "log", "log10", "sqrt", "sin", "cos"],
    "complex_math": ["tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                     "asinh", "acosh", "atanh"],
    "real_math_with_half": ["fabs", "exp2", "log2"],
    "real_math": ["expm1", "log1p", "logb", "cbrt", "erf", "erfc", "lgamma",
                  "tgamma", "ceil", "floor", "trunc", "round", "nearbyint"],
    "bitwise": ["invert"],
  },
  "binary": {
    "default": ["add", "subtract", "multiply", "floor_divide", "remainder"],
    "float_result": ["divide"],
    "bool_result": ["less_equal", "less", "greater_equal", "greater"],
    "bitwise": ["bitwise_and", "bitwise_or", "bitwise_xor"]
  },
  "binary_mv": {
    "default": ["divmod"],
  }
}

def complex_noimpl(name):
    return name in functions["unary"]["real_math"] or \
           name in functions["unary"]["real_math_with_half"]

def half_noimpl(name):
    return name in functions["unary"]["real_math"] or \
           name in functions["unary"]["complex_math"] or \
           name in ("floor_divide", "remainder")

tunsigned = ["bool", "uint8", "uint16", "uint32", "uint64"]
tsigned = ["int8", "int16", "int32", "int64"]
tfloat = ["bfloat16", "float16", "float32", "float64"]
tcomplex = ["complex32", "complex64", "complex128"]

tinfo = {
  "bool": (0, 1, 0),
  "uint8": (0, 2**8-1, 0),
  "uint16": (0, 2**16-1, 0),
  "uint32": (0, 2**32-1, 0),
  "uint64": (0, 2**64-1, 0),
  "int8": (-2**7,  2**7-1, 0),
  "int16": (-2**15, 2**15-1, 0),
  "int32": (-2**31, 2**31-1, 0),
  "int64": (-2**63, 2**63-1, 0),
  "float16": (-2**11, 2**11, 15),
  "bfloat16": (-2**8, 2**8, 127),
  "float32": (-2**24, 2**24, 127),
  "float64": (-2**53, 2**53, 1023),
  "complex32": (-2**11, 2**11, 15),
  "complex64": (-2**24, 2**24, 127),
  "complex128": (-2**53, 2**53, 1023)
}

class Tint(object):
    def __init__(self, type):
        if type not in tunsigned + tsigned:
            raise ValueError("not an integer type: '%s'" % type)
        self.type = type
        self.min, self.max, self.exp = tinfo[type]
        self.all = (self.type, self.min, self.max, self.exp)
    def __repr__(self):
        return self.type
    def __eq__(self, other):
        return isinstance(Tint, other) and self.all == other.all
    def __hash__(self):
        return hash(self.all)
    def testcases(self):
        yield 0
        yield self.min
        yield self.max
        for i in range(10):
            yield randrange(self.min, self.max+1)
    def cpu_noimpl(self, f=None):
        return False
    def cpu_nokern(self, f=None):
        return False
    def cuda_noimpl(self, f=None):
        return False
    def cuda_nokern(self, f=None):
        return False

class Tfloat(object):
    def __init__(self, type):
        if type not in tfloat:
            raise ValueError("not a float type: '%s'" % type)
        self.type = type
        self.min, self.max, self.exp = tinfo[type]
        self.all = (self.type, self.min, self.max, self.exp)
    def __repr__(self):
        return self.type
    def __eq__(self, other):
        return isinstance(Tint, other) and self.all == other.all
    def __hash__(self):
        return hash(self.all)
    def testcases(self):
        yield 0
        yield 0.5
        yield -0.5
        yield self.min
        yield self.max
        prec = randrange(1, 10)
        for v in all_unary(prec, self.exp, 1):
            yield float(v)
        for v in un_randfloat():
            yield float(v)
    def cpu_noimpl(self, f=None):
        return self.type == "float16"
    def cpu_nokern(self, f=None):
        return False
    def cuda_noimpl(self, f=None):
        if self.type == "float16":
            return half_noimpl(f)
    def cuda_nokern(self, f=None):
        return False

class Tcomplex(object):
    def __init__(self, type):
        if type not in tcomplex:
            raise ValueError("not a complex type: '%s'" % type)
        self.type = type
        self.min, self.max, self.exp = tinfo[type]
        self.all = (self.type, self.min, self.max, self.exp)
    def __repr__(self):
        return self.type
    def __eq__(self, other):
        return isinstance(Tint, other) and self.all == other.all
    def __hash__(self):
        return hash(self.all)
    def testcases(self):
        yield 0
        yield 0.5
        yield -0.5
        yield 0.5j
        yield -0.5j
        yield self.min
        yield self.max
        prec = randrange(1, 10)
        for v, w in all_binary(prec, self.exp, 1):
            yield complex(float(v), float(w))
        for v, w in bin_randfloat():
            yield complex(float(v), float(w))
    def cpu_noimpl(self, f=None):
        if self.type == "complex32":
            return True
        return complex_noimpl(f)
    def cpu_nokern(self, f=None):
        return f in ("floor_divide", "remainder")
    def cuda_noimpl(self, f=None):
        if self.type == "complex32":
            return True
        return complex_noimpl(f)
    def cuda_nokern(self, f=None):
        return f in ("floor_divide", "remainder")


tinfo_default = [
  Tint("uint8"),
  Tint("uint16"),
  Tint("uint32"),
  Tint("uint64"),
  Tint("int8"),
  Tint("int16"),
  Tint("int32"),
  Tint("int64"),
  Tfloat("float16"),
  Tfloat("bfloat16"),
  Tfloat("float32"),
  Tfloat("float64"),
  Tcomplex("complex32"),
  Tcomplex("complex64"),
  Tcomplex("complex128")
]

tinfo_bitwise = [
  Tint("bool"),
  Tint("uint8"),
  Tint("uint16"),
  Tint("uint32"),
  Tint("uint64"),
  Tint("int8"),
  Tint("int16"),
  Tint("int32"),
  Tint("int64")
]

implemented_sigs = {
  "unary": {
    "default": {}, "float_result": {}
  },
  "binary": {
    "default": {}, "float_result": {}, "bool_result": {}, "bitwise": {}
  },
  "binary_mv": {
    "default": {
       (Tint("uint8"), Tint("uint8")): (Tint("uint8"), Tint("uint8")),
       (Tint("uint16"), Tint("uint16")): (Tint("uint16"), Tint("uint16")),
       (Tint("uint32"), Tint("uint32")): (Tint("uint32"), Tint("uint32")),
       (Tint("uint64"), Tint("uint64")): (Tint("uint64"), Tint("uint64")),
       (Tint("int8"), Tint("int8")): (Tint("int8"), Tint("int8")),
       (Tint("int16"), Tint("int16")): (Tint("int16"), Tint("int16")),
       (Tint("int32"), Tint("int32")): (Tint("int32"), Tint("int32")),
       (Tint("int64"), Tint("int64")): (Tint("int64"), Tint("int64")),
       (Tfloat("float32"), Tfloat("float32")): (Tfloat("float32"), Tfloat("float32")),
       (Tfloat("float64"), Tfloat("float64")): (Tfloat("float64"), Tfloat("float64"))
    },
  }
}

exact_sigs = {
  "unary": {
    "default": {}, "float_result": {}
  },
  "binary": {
    "default": {}, "float_result": {}, "bool_result": {}, "bitwise": {}
  }
}

inexact_sigs = {
  "unary": {
    "default": {}, "float_result": {}
  },
  "binary": {
    "default": {}, "float_result": {}, "bool_result": {}, "bitwise": {}
  }
}

def init_unary_cast(pattern, tinfo, rank):
    t = tinfo[rank]

    start = max(8, rank) if pattern == "float_result" else rank
    found_cast = False

    for i in range(start, len(tinfo_default)):
        cast = tinfo[i]
        if cast.min <= t.min and t.max <= cast.max:
            if found_cast or (t.type=="bfloat16") != (cast.type=="bfloat16"):
                exact_sigs["unary"][pattern][(t,)] = cast
            else:
                found_cast = True
                implemented_sigs["unary"][pattern][(t,)] = cast
                exact_sigs["unary"][pattern][(t,)] = cast
        else:
            inexact_sigs["unary"][pattern][(t,)] = cast

def init_unary_cast_tbl(pattern):
    if pattern == "default":
        tinfo = [Tint("bool")] + tinfo_default
    elif pattern == "float_result":
        tinfo = tinfo_default
    elif pattern == "bitwise":
        tinfo = tinfo_bitwise
    else:
        raise ValueError("unsupported function type '%s'" % func)

    for rank, _ in enumerate(tinfo):
        init_unary_cast(pattern, tinfo, rank)

def is_binary_common_cast(cast, t, u):
    if cast.min <= t.min and t.max <= cast.max and \
       cast.min <= u.min and u.max <= cast.max:
        if isinstance(cast, Tfloat):
            return t.exp <= cast.exp and u.exp <= cast.exp
        else:
            return True
    return False

def init_binary_cast(pattern, tinfo, rank1, rank2):
    min_rank = min(rank1, rank2)
    max_rank = max(rank1, rank2)

    t = tinfo[min_rank]
    u = tinfo[max_rank]

    start = max(8, max_rank) if pattern == "float_result" else max_rank
    smallest_common_cast = False

    for i in range(start, len(tinfo_default)):
        common_cast = tinfo_default[i]
        w = Tint("bool") if pattern == "bool_result" else common_cast
        if is_binary_common_cast(common_cast, t, u):
           if smallest_common_cast:
               exact_sigs["binary"][pattern][(t, u)] = w
           else:
               smallest_common_cast = True
               implemented_sigs["binary"][pattern][(t, u)] = w
               exact_sigs["binary"][pattern][(t, u)] = w
        else:
            inexact_sigs["binary"][pattern][(t, u)] = w

def init_binary_cast_tbl(pattern):
    if pattern == "default" or pattern == "float_result" or pattern == "bool_result":
        tinfo = tinfo_default
    elif pattern == "bitwise":
        tinfo = tinfo_bitwise
    else:
        raise ValueError("unsupported function type '%s'" % pattern)

    for rank1, _ in enumerate(tinfo):
        for rank2, _ in enumerate(tinfo):
            init_binary_cast(pattern, tinfo, rank1, rank2)

_struct_format = {
  "float16": "e",
  "float32": "f",
  "float64": "d",
  "complex32": "e",
  "complex64": "f",
  "complex128": "d"
}

def roundtrip_ne(v, fmt):
    if fmt == "e":
        try:
            struct.pack(fmt, v)
        except (OverflowError, struct.error):
            return True
        else:
            return False
    else:
        if math.isinf(v):
            return False
        s = struct.unpack(fmt, struct.pack(fmt, v))[0]
        return math.isinf(float(s))

def struct_overflow(v, t):
    try:
        fmt = _struct_format[t.type]
    except KeyError:
        return False

    if isinstance(t, Tcomplex):
        return roundtrip_ne(v.real, fmt) or roundtrip_ne(v.imag, fmt)
    else:
        return roundtrip_ne(v, fmt)


init_unary_cast_tbl("default")
init_unary_cast_tbl("float_result")

init_binary_cast_tbl("default")
init_binary_cast_tbl("float_result")
init_binary_cast_tbl("bool_result")
init_binary_cast_tbl("bitwise")


_np_names = {
  "asin" : "arcsin",
  "acos" : "arccos",
  "atan" : "arctan",
  "asinh" : "arcsinh",
  "acosh" : "arccosh",
  "atanh" : "arctanh",
  "nearbyint" : "round",
}

def np_function(name):
    return _np_names.get(name, name)

def np_noimpl(name):
    if name == "round":
        # np.round == gumath.nearbyint
        return True
    try:
        getattr(np, name)
        return False
    except AttributeError:
        return True

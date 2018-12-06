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
import unittest


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
    def __init__(self, value):
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



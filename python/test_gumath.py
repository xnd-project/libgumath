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

import gumath as gm
import gumath.functions as fn
import gumath.examples as ex
from xnd import xnd
from ndtypes import ndt
from extending import Graph
import sys, time
import math
import unittest
import argparse
from gumath_aux import *

try:
    import numpy as np
except ImportError:
    np = None

SKIP_LONG = True
SKIP_BRUTE_FORCE = True

gm.set_max_threads(1)


class TestCall(unittest.TestCase):

    def test_sin_scalar(self):

        x1 = xnd(1.2, type="float64")
        y1 = fn.sin(x1)

        x2 = xnd(1.23e1, type="float32")
        y2 = fn.sin(x2)

        if np is not None:
            a1 = np.array(1.2, dtype="float64")
            b1 = np.sin(a1)

            a2 = np.array(1.23e1, dtype="float32")
            b2 = np.sin(a2)

            np.testing.assert_equal(y1.value, b1)
            np.testing.assert_equal(y2.value, b2)

    def test_sin(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)
            y = fn.sin(x)

            if np is not None:
                a = np.array(lst, dtype=dtype)
                b = np.sin(a)
                np.testing.assert_equal(y, b)

    def test_sin_strided(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)
            if x.type.ndim < 2:
                continue

            y = x[::-2, ::-2]
            z = fn.sin(y)

            if np is not None:
                a = np.array(lst, dtype=dtype)
                b = a[::-2, ::-2]
                c = np.sin(b)
                np.testing.assert_equal(z, c)

    def test_copy(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)
            y = fn.copy(x)

            if np is not None:
                a = np.array(lst, dtype=dtype)
                b = np.copy(a)
                np.testing.assert_equal(y, b)

    def test_copy_strided(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)
            if x.type.ndim < 2:
                continue

            y = x[::-2, ::-2]
            z = fn.copy(y)

            if np is not None:
                a = np.array(lst, dtype=dtype)
                b = a[::-2, ::-2]
                c = np.copy(b)
                np.testing.assert_equal(y, b)

    @unittest.skipIf(sys.platform == "win32", "missing C99 complex support")
    def test_quaternion(self):
  
        lst = [[[1+2j, 4+3j],
                [-4+3j, 1-2j]],
               [[4+2j, 1+10j],
                [-1+10j, 4-2j]],
               [[-4+2j, 3+10j],
                [-3+10j, -4-2j]]]

        x = xnd(lst, type="3 * quaternion64")
        y = ex.multiply(x, x)

        if np is not None:
            a = np.array(lst, dtype="complex64")
            b = np.einsum("ijk,ikl->ijl", a, a)
            np.testing.assert_equal(y, b)

        x = xnd(lst, type="3 * quaternion128")
        y = ex.multiply(x, x)

        if np is not None:
            a = np.array(lst, dtype="complex128")
            b = np.einsum("ijk,ikl->ijl", a, a)
            np.testing.assert_equal(y, b)

        x = xnd("xyz")
        self.assertRaises(TypeError, ex.multiply, x, x)

    @unittest.skipIf(sys.platform == "win32", "missing C99 complex support")
    def test_quaternion_error(self):
  
        lst = [[[1+2j, 4+3j],
                [-4+3j, 1-2j]],
               [[4+2j, 1+10j],
                [-1+10j, 4-2j]],
               [[-4+2j, 3+10j],
                [-3+10j, -4-2j]]]

        x = xnd(lst, type="3 * Foo(2 * 2 * complex64)")
        self.assertRaises(TypeError, ex.multiply, x, x)

    def test_void(self):

        x = ex.randint()
        self.assertEqual(x.type, ndt("int32"))

    def test_multiple_return(self):

        x, y = ex.randtuple()
        self.assertEqual(x.type, ndt("int32"))
        self.assertEqual(y.type, ndt("int32"))

        x, y = ex.divmod10(xnd(233))
        self.assertEqual(x.value, 23)
        self.assertEqual(y.value, 3)


class TestMissingValues(unittest.TestCase):

    def test_missing_values(self):

        x = [{'index': 0, 'name': 'brazil', 'value': 10},
             {'index': 1, 'name': 'france', 'value': None},
             {'index': 1, 'name': 'russia', 'value': 2}]

        y = [{'index': 0, 'name': 'iceland', 'value': 5},
             {'index': 1, 'name': 'norway', 'value': None},
             {'index': 1, 'name': 'italy', 'value': None}]

        z = xnd([x, y], type="2 * 3 * {index: int64, name: string, value: ?int64}")
        ans = ex.count_valid_missing(z)

        self.assertEqual(ans.value, [{'valid': 2, 'missing': 1}, {'valid': 1, 'missing': 2}])


class TestRaggedArrays(unittest.TestCase):

    def test_sin(self):
        s = math.sin
        lst = [[[1.0],
                [2.0, 3.0],
                [4.0, 5.0, 6.0]],
               [[7.0],
                [8.0, 9.0],
                [10.0, 11.0, 12.0]]]

        ans = [[[s(1.0)],
                [s(2.0), s(3.0)],
                [s(4.0), s(5.0), s(6.0)]],
               [[s(7.0)],
                [s(8.0), s(9.0)],
                [s(10.0), s(11.0), s(12.0)]]]

        x = xnd(lst)
        y = fn.sin(x)
        self.assertEqual(y.value, ans)


class TestGraphs(unittest.TestCase):

    def test_shortest_path(self):
        graphs = [[[(1, 1.2), (2, 4.4)],
                   [(2, 2.2)],
                   [(1, 2.3)]],

                  [[(1, 1.2), (2, 4.4)],
                   [(2, 2.2)],
                   [(1, 2.3)],
                   [(2, 1.1)]]]

        ans = [[[[0], [0, 1], [0, 1, 2]],      # graph1, start 0
                [[], [1], [1, 2]],             # graph1, start 1
                [[], [2, 1], [2]]],            # graph1, start 2

               [[[0], [0, 1], [0, 1, 2], []],  # graph2, start 0
                [[], [1], [1, 2], []],         # graph2, start 1
                [[], [2, 1], [2], []],         # graph2, start 2
                [[], [3, 2, 1], [3, 2], [3]]]] # graph2, start 3


        for i, lst in enumerate(graphs):
            N = len(lst)
            graph = Graph(lst)
            for start in range(N):
                node = xnd(start, type="node")
                x = graph.shortest_paths(node)
                self.assertEqual(x.value, ans[i][start])

    def test_constraint(self):
        lst = [[(0, 1.2)],
               [(2, 2.2), (1, 0.1)]]

        self.assertRaises(ValueError, Graph, lst)


class TestPdist(unittest.TestCase):

    def test_exceptions(self):
        x = xnd([], dtype="float64")
        self.assertRaises(TypeError, ex.euclidian_pdist, x)

        x = xnd([[]], dtype="float64")
        self.assertRaises(TypeError, ex.euclidian_pdist, x)

        x = xnd([[], []], dtype="float64")
        self.assertRaises(TypeError, ex.euclidian_pdist, x)

        x = xnd([[1], [1]], dtype="int64")
        self.assertRaises(TypeError, ex.euclidian_pdist, x)

    def test_pdist(self):
        x = xnd([[1]], dtype="float64")
        y = ex.euclidian_pdist(x)
        self.assertEqual(y.value, [])

        x = xnd([[1, 2, 3]], dtype="float64")
        y = ex.euclidian_pdist(x)
        self.assertEqual(y.value, [])

        x = xnd([[-1.2200, -100.5000,   20.1250,  30.1230],
                 [ 2.2200,    2.2720, -122.8400, 122.3330],
                 [ 2.1000,  -25.0000,  100.2000, -99.5000]], dtype="float64")
        y = ex.euclidian_pdist(x)
        self.assertEqual(y.value, [198.78529349275314, 170.0746899276903, 315.75385646576035])


@unittest.skipIf(gm.xndvectorize is None, "test requires numpy and numba")
class TestNumba(unittest.TestCase):

    def test_numba(self):

        @gm.xndvectorize("... * N * M * float64, ... * M * P * float64 -> ... * N * P * float64")
        def matmul(x, y, res):
            col = np.arange(y.shape[0])
            for j in range(y.shape[1]):
                for k in range(y.shape[0]):
                    col[k] = y[k, j]
                for i in range(x.shape[0]):
                    s = 0
                    for k in range(x.shape[1]):
                        s += x[i, k] * col[k]
                    res[i, j] = s

        a = np.arange(50000.0).reshape(1000, 5, 10)
        b = np.arange(70000.0).reshape(1000, 10, 7)
        c = np.einsum("ijk,ikl->ijl", a, b)

        x = xnd(a.tolist(), type="1000 * 5 * 10 * float64")
        y = xnd(b.tolist(), type="1000 * 10 * 7 * float64")
        z = matmul(x, y)

        np.testing.assert_equal(z, c)

    def test_numba_add_scalar(self):

        import numba as nb

        @nb.guvectorize(["void(int64[:], int64, int64[:])"], '(n),()->(n)')
        def g(x, y, res):
            for i in range(x.shape[0]):
                res[i] = x[i] + y

        a = np.arange(5000).reshape(100, 5, 10)
        b = np.arange(500).reshape(100, 5)
        c = g(a, b)

        x = xnd(a.tolist(), type="100 * 5 * 10 * int64")
        y = xnd(b.tolist(), type="100 * 5 * int64")
        z = ex.add_scalar(x, y)

        np.testing.assert_equal(z, c)

        a = np.arange(500)
        b = np.array(100)
        c = g(a, b)

        x = xnd(a.tolist(), type="500 * int64")
        y = xnd(b.tolist(), type="int64")
        z = ex.add_scalar(x, y)

        np.testing.assert_equal(z, c)


class TestUnary(unittest.TestCase):

    def test_acos(self):
        a = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        b = [math.acos(x) for x in a]

        x = xnd(a, dtype="float64")
        y = fn.acos(x)
        self.assertEqual(y, b)

    def test_acos_opt(self):
        a = [0, 0.1, 0.2, None, 0.4, 0.5, 0.6, None]
        b = [math.acos(x) if x is not None else None for x in a]

        x = xnd(a, dtype="?float64")
        y = fn.acos(x)
        self.assertEqual(y, b)

    def test_inexact_cast(self):
        a = [0, 1, 2, 3, 4, 5, 6, 7]
        x = xnd(a, dtype="int64")
        self.assertRaises(ValueError, fn.sin, x)


tinfo_binary = [
  ("uint8",      (0, 2**8-1)),
  ("uint16",     (0, 2**16-1)),
  ("uint32",     (0, 2**32-1)),
  ("uint64",     (0, 2**64-1)),
  ("int8",       (-2**7,  2**7-1)),
  ("int16",      (-2**15, 2**15-1)),
  ("int32",      (-2**31, 2**31-1)),
  ("int64",      (-2**63, 2**63-1)),
  # ("float16",    (-2**11, 2**11)),
  ("float32",    (-2**24, 2**24)),
  ("float64",    (-2**53, 2**53)),
  # ("complex32",  (-2**11, 2**11)),
  # ("complex64",  (-2**24, 2**24)),
  # ("complex128", (-2**53, 2**53))
]

tinfo_bitwise = [
  ("bool",       (0, 1)),
  ("uint8",      (0, 2**8-1)),
  ("uint16",     (0, 2**16-1)),
  ("uint32",     (0, 2**32-1)),
  ("uint64",     (0, 2**64-1)),
  ("int8",       (-2**7,  2**7-1)),
  ("int16",      (-2**15, 2**15-1)),
  ("int32",      (-2**31, 2**31-1)),
  ("int64",      (-2**63, 2**63-1)),
]

def common_cast_binary(rank1, rank2):
    min_rank = min(rank1, rank2)
    max_rank = max(rank1, rank2)
    t = tinfo_binary[min_rank]
    u = tinfo_binary[max_rank]
    for i in range(max_rank, len(tinfo_binary)):
        w = tinfo_binary[i]
        if w[1][0] <= t[1][0] and t[1][1] <= w[1][1] and \
           w[1][0] <= u[1][0] and u[1][1] <= w[1][1]:
               return w
    return None

def common_cast_bitwise(rank1, rank2):
    min_rank = min(rank1, rank2)
    max_rank = max(rank1, rank2)
    t = tinfo_bitwise[min_rank]
    u = tinfo_bitwise[max_rank]
    for i in range(max_rank, len(tinfo_bitwise)):
        w = tinfo_bitwise[i]
        if w[1][0] <= t[1][0] and t[1][1] <= w[1][1] and \
           w[1][0] <= u[1][0] and u[1][1] <= w[1][1]:
               return w
    return None


class TestBinary(unittest.TestCase):

    def test_add(self):
        for rank1, t in enumerate(tinfo_binary):
            for rank2, u in enumerate(tinfo_binary):
                w = common_cast_binary(rank1, rank2)
                x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.add(x, y)
                    self.assertEqual(z, [1, 3, 5, 7, 9, 11, 13, 15])
                else:
                    self.assertRaises(ValueError, fn.add, x, y)

    def test_add_opt(self):
        for rank1, t in enumerate(tinfo_binary):
            for rank2, u in enumerate(tinfo_binary):
                w = common_cast_binary(rank1, rank2)
                x = xnd([0, 1, None, 3, 4, 5, 6, 7], dtype="?" + t[0])
                y = xnd([1, 2, 3, 4, 5, 6, None, 8], dtype="?" + u[0])

                if w is not None:
                    z = fn.add(x, y)
                    self.assertEqual(z, [1, 3, None, 7, 9, 11, None, 15])
                else:
                    self.assertRaises(ValueError, fn.add, x, y)


                x = xnd([0, 1, 2, 3, 4, 5, None, 7], dtype="?" + t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.add(x, y)
                    self.assertEqual(z, [1, 3, 5, 7, 9, 11, None, 15])
                else:
                    self.assertRaises(ValueError, fn.add, x, y)


                x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, None, 8], dtype="?" + u[0])

                if w is not None:
                    z = fn.add(x, y)
                    self.assertEqual(z, [1, 3, 5, 7, 9, 11, None, 15])
                else:
                    self.assertRaises(ValueError, fn.add, x, y)

    def test_subtract(self):
        for rank1, t in enumerate(tinfo_binary):
            for rank2, u in enumerate(tinfo_binary):
                w = common_cast_binary(rank1, rank2)
                x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.subtract(x, y)
                    self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])
                else:
                    self.assertRaises(ValueError, fn.subtract, x, y)

    def test_multiply(self):
        for rank1, t in enumerate(tinfo_binary):
            for rank2, u in enumerate(tinfo_binary):
                w = common_cast_binary(rank1, rank2)
                x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.subtract(x, y)
                    self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])
                else:
                    self.assertRaises(ValueError, fn.subtract, x, y)


class TestBitwise(unittest.TestCase):

    def test_and(self):
        for rank1, t in enumerate(tinfo_bitwise):
            for rank2, u in enumerate(tinfo_bitwise):
                w = common_cast_bitwise(rank1, rank2)
                x = xnd([0, 1, 0, 1, 1, 1, 1, 0], dtype=t[0])
                y = xnd([1, 0, 0, 0, 1, 1, 1, 1], dtype=u[0])

                if w is not None:
                    z = fn.bitwise_and(x, y)
                    self.assertEqual(z, [0, 0, 0, 0, 1, 1, 1, 0])
                else:
                    self.assertRaises(ValueError, fn.bitwise_and, x, y)

    def test_and_opt(self):
        for rank1, t in enumerate(tinfo_bitwise):
            for rank2, u in enumerate(tinfo_bitwise):
                w = common_cast_bitwise(rank1, rank2)

                a = [0, 1, None, 1, 1, 1, 1, 0]
                b = [1, 1, 1, 1, 1, 1, None, 0]
                c = [0, 1, None, 1, 1, 1, None, 0]

                x = xnd(a, dtype="?" + t[0])
                y = xnd(b, dtype="?" + u[0])

                if w is not None:
                    z = fn.bitwise_and(x, y)
                    self.assertEqual(z, c)
                else:
                    self.assertRaises(ValueError, fn.bitwise_and, x, y)


                a = [0, 1, None, 1, 1, 1, None, 0]
                b = [1, 1, 1, 1, 1, 1, 1, 0]
                c = [0, 1, None, 1, 1, 1, None, 0]

                x = xnd(a, dtype="?" + t[0])
                y = xnd(b, dtype=u[0])

                if w is not None:
                    z = fn.bitwise_and(x, y)
                    self.assertEqual(z, c)
                else:
                    self.assertRaises(ValueError, fn.bitwise_and, x, y)

                x = xnd(b, dtype=t[0])
                y = xnd(a, dtype="?" + u[0])

                if w is not None:
                    z = fn.bitwise_and(x, y)
                    self.assertEqual(z, a)
                else:
                    self.assertRaises(ValueError, fn.bitwise_and, x, y)


class TestSpec(unittest.TestCase):

    def __init__(self, *, constr,
                 values, value_generator,
                 indices_generator, indices_generator_args):
        super().__init__()
        self.constr = constr
        self.values = values
        self.value_generator = value_generator
        self.indices_generator = indices_generator
        self.indices_generator_args = indices_generator_args
        self.indices_stack = [None] * 8

    def log_err(self, value, depth):
        """Dump an error as a Python script for debugging."""
        dtype = "?int32" if have_none(value) else "int32"

        sys.stderr.write("\n\nfrom xnd import *\n")
        sys.stderr.write("import gumath.functions as fn\n")
        sys.stderr.write("from test_gumath import NDArray\n")
        sys.stderr.write("lst = %s\n\n" % value)
        sys.stderr.write("x0 = xnd(lst, dtype=\"%s\")\n" % dtype)
        sys.stderr.write("y0 = NDArray(lst)\n" % value)

        for i in range(depth+1):
            sys.stderr.write("x%d = x%d[%s]\n" % (i+1, i, itos(self.indices_stack[i])))
            sys.stderr.write("y%d = y%d[%s]\n" % (i+1, i, itos(self.indices_stack[i])))

        sys.stderr.write("\n")

    def run_single(self, nd, d, indices):
        """Run a single test case."""

        self.assertEqual(len(nd), len(d))

        nd_exception = None
        try:
            nd_result = nd[indices]
        except Exception as e:
            nd_exception =  e

        def_exception = None
        try:
            def_result = d[indices]
        except Exception as e:
            def_exception = e

        if nd_exception or def_exception:
            if nd_exception is None and def_exception.__class__ is IndexError:
                # Example: type = 0 * 0 * int64
                if len(indices) <= nd.ndim:
                    return None, None

            self.assertIs(nd_exception.__class__, def_exception.__class__)
            return None, None

        assert(isinstance(nd_result, xnd))

        x = fn.sin(nd_result)
        y = fn.multiply(nd_result, nd_result)

        if isinstance(def_result, NDArray):
            a = def_result.sin()
            b = def_result * def_result
        elif isinstance(def_result, int):
            a = math.sin(def_result)
            b = def_result * def_result
        elif def_result is None:
            a = None
            b = None
        else:
            raise TypeError("unexpected def_result")

        self.assertEqual(x, a)
        self.assertEqual(y, b)

        return nd_result, def_result

    def run(self):
        def check(nd, d, value, depth):
            if depth > 3: # adjust for longer tests
                return

            g = self.indices_generator(*self.indices_generator_args)

            for indices in g:
                self.indices_stack[depth] = indices

                try:
                    next_nd, next_d = self.run_single(nd, d, indices)
                except Exception as e:
                    self.log_err(value, depth)
                    raise e

                if isinstance(next_d, list): # possibly None or scalar
                    check(next_nd, next_d, value, depth+1)

        for value in self.values:
            dtype = "?int32" if have_none(value) else "int32"
            nd = self.constr(value, dtype=dtype)
            d = NDArray(value)
            check(nd, d, value, 0)

        for max_ndim in range(1, 5):
            for min_shape in (0, 1):
                for max_shape in range(1, 8):
                    for value in self.value_generator(max_ndim, min_shape, max_shape):
                        dtype = "?int32" if have_none(value) else "int32"
                        nd = self.constr(value, dtype=dtype)
                        d = NDArray(value)
                        check(nd, d, value, 0)


class LongIndexSliceTest(unittest.TestCase):

    def test_subarray(self):
        # Multidimensional indexing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

    def test_slices(self):
        # Multidimensional slicing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

    def test_chained_indices_slices(self):
        # Multidimensional indexing and slicing, chained
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=gen_indices_or_slices,
                     indices_generator_args=())
        t.run()


        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=gen_indices_or_slices,
                     indices_generator_args=())
        t.run()

    def test_fixed_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=mixed_indices,
                     indices_generator_args=(3,))
        t.run()

    def test_var_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=mixed_indices,
                     indices_generator_args=(5,))
        t.run()

    def test_slices_brute_force(self):
        # Test all possible slices for the given ndim and shape
        skip_if(SKIP_BRUTE_FORCE, "use --all argument to enable these tests")

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genslices_ndim,
                     indices_generator_args=(3, [3,3,3]))
        t.run()

        t = TestSpec(constr=xnd,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=genslices_ndim,
                     indices_generator_args=(3, [3,3,3]))
        t.run()


ALL_TESTS = [
  TestCall,
  TestRaggedArrays,
  TestMissingValues,
  TestGraphs,
  TestPdist,
  TestNumba,
  TestUnary,
  TestBinary,
  TestBitwise,
  LongIndexSliceTest,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--failfast", action="store_true",
                        help="stop the test run on first error")
    parser.add_argument('--long', action="store_true", help="run long slice tests")
    parser.add_argument('--all', action="store_true", help="run brute force tests")
    args = parser.parse_args()
    SKIP_LONG = not (args.long or args.all)
    SKIP_BRUTE_FORCE = not args.all

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for case in ALL_TESTS:
        s = loader.loadTestsFromTestCase(case)
        suite.addTest(s)

    runner = unittest.TextTestRunner(failfast=args.failfast, verbosity=2)
    result = runner.run(suite)
    ret = not result.wasSuccessful()

    sys.exit(ret)

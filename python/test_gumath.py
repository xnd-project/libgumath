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
from extending import Graph, bfloat16
import sys, time
import math
import unittest
import argparse

try:
    import numpy as np
except ImportError:
    np = None


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


@unittest.skipIf(sys.platform == "win32", "unresolved external symbols")
class TestBFloat16(unittest.TestCase):

    def test_init(self):
        lst = [1.2e10, 2.1121, -3e20]
        ans = [11945377792.0, 2.109375, -2.997595911977802e+20]

        x = bfloat16(lst)
        self.assertEqual(x.value, ans)


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


tinfo = [
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

def common_cast(rank1, rank2):
    min_rank = min(rank1, rank2)
    max_rank = max(rank1, rank2)
    t = tinfo[min_rank]
    u = tinfo[max_rank]
    for i in range(max_rank, len(tinfo)):
        w = tinfo[i]
        if w[1][0] <= t[1][0] and t[1][1] <= w[1][1] and \
           w[1][0] <= u[1][0] and u[1][1] <= w[1][1]:
               return w
    return None

class TestArithmetic(unittest.TestCase):

    def test_add(self):
        for rank1, t in enumerate(tinfo):
            for rank2, u in enumerate(tinfo):
                w = common_cast(rank1, rank2)
                x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.add(x, y)
                    self.assertEqual(z, [1, 3, 5, 7, 9, 11, 13, 15])
                else:
                    self.assertRaises(ValueError, fn.add, x, y)

    def test_add_opt(self):
        for rank1, t in enumerate(tinfo):
            for rank2, u in enumerate(tinfo):
                w = common_cast(rank1, rank2)
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
        for rank1, t in enumerate(tinfo):
            for rank2, u in enumerate(tinfo):
                w = common_cast(rank1, rank2)
                x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.subtract(x, y)
                    self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])
                else:
                    self.assertRaises(ValueError, fn.subtract, x, y)

    def test_multiply(self):
        for rank1, t in enumerate(tinfo):
            for rank2, u in enumerate(tinfo):
                w = common_cast(rank1, rank2)
                x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t[0])
                y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u[0])

                if w is not None:
                    z = fn.subtract(x, y)
                    self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])
                else:
                    self.assertRaises(ValueError, fn.subtract, x, y)


ALL_TESTS = [
  TestCall,
  TestRaggedArrays,
  TestMissingValues,
  TestGraphs,
  TestBFloat16,
  TestPdist,
  TestNumba,
  TestArithmetic,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--failfast", action="store_true",
                        help="stop the test run on first error")
    args = parser.parse_args()

    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    for case in ALL_TESTS:
        s = loader.loadTestsFromTestCase(case)
        suite.addTest(s)

    runner = unittest.TextTestRunner(failfast=args.failfast, verbosity=2)
    result = runner.run(suite)
    ret = not result.wasSuccessful()

    sys.exit(ret)

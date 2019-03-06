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
import platform
import math
import cmath
import unittest
import argparse
from gumath_aux import *

try:
    import gumath.cuda as cd
except ImportError:
    cd = None

try:
    import numpy as np
    np.warnings.filterwarnings('ignore')
except ImportError:
    np = None

SKIP_LONG = True
SKIP_BRUTE_FORCE = True

ARCH = platform.architecture()[0]


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

    def test_unary(self):
        a = [0, None, 2]
        ans = xnd([math.sin(x) if x is not None else None for x in a])

        x = xnd(a, dtype="?float64")
        y = fn.sin(x)
        self.assertEqual(y.value, ans)

    def test_binary(self):
        a = [3, None, 3]
        b = [100, 1, None]
        ans = xnd([t[0] * t[1] if t[0] is not None and t[1] is not None else None
                   for t in zip(a, b)])

        x = xnd(a)
        y = xnd(b)
        z = fn.multiply(x, y)
        self.assertEqual(z.value, ans)

    def test_reduce(self):
        a = [1, None, 2]
        x = xnd(a)

        y = gm.reduce(fn.add, x)
        self.assertEqual(y, None)

        y = gm.reduce(fn.multiply, x)
        self.assertEqual(y, None)

        y = gm.reduce(fn.subtract, x)
        self.assertEqual(y, None)

        x = xnd([], dtype="?int32")

        y = gm.reduce(fn.add, x)
        self.assertEqual(y, 0)

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_reduce_cuda(self):
        a = [1, None, 2]
        x = xnd(a, device="cuda:managed")

        y = gm.reduce(cd.add, x)
        self.assertEqual(y, None)

        y = gm.reduce(cd.multiply, x)
        self.assertEqual(y, None)

        x = xnd([], dtype="?int32", device="cuda:managed")
        y = gm.reduce(fn.add, x)
        self.assertEqual(y, 0)

    def test_comparisons(self):
        a = [1, None, 3, 5]
        b = [2, None, 3, 4]

        x = xnd(a)
        y = xnd(b)

        ans = fn.equal(x, y)
        self.assertEqual(ans.value, [False, None, True, False])

        ans = fn.not_equal(x, y)
        self.assertEqual(ans.value, [True, None, False, True])

        ans = fn.less(x, y)
        self.assertEqual(ans.value, [True, None, False, False])

        ans = fn.less_equal(x, y)
        self.assertEqual(ans.value, [True, None, True, False])

        ans = fn.greater_equal(x, y)
        self.assertEqual(ans.value, [False, None, True, True])

        ans = fn.greater(x, y)
        self.assertEqual(ans.value, [False, None, False, True])

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_comparisons_cuda(self):
        a = [1, None, 3, 5]
        b = [2, None, 3, 4]

        x = xnd(a, device="cuda:managed")
        y = xnd(b, device="cuda:managed")

        ans = cd.equal(x, y)
        self.assertEqual(ans.value, [False, None, True, False])

        ans = cd.not_equal(x, y)
        self.assertEqual(ans.value, [True, None, False, True])

        ans = cd.less(x, y)
        self.assertEqual(ans.value, [True, None, False, False])

        ans = cd.less_equal(x, y)
        self.assertEqual(ans.value, [True, None, True, False])

        ans = cd.greater_equal(x, y)
        self.assertEqual(ans.value, [False, None, True, True])

        ans = cd.greater(x, y)
        self.assertEqual(ans.value, [False, None, False, True])

    def test_equaln(self):
        a = [1, None, 3, 5]
        b = [2, None, 3, 4]

        x = xnd(a)
        y = xnd(b)
        z = fn.equaln(x, y)
        self.assertEqual(z, [False, True, True, False])
        self.assertEqual(z.dtype, ndt("bool"))

        a = [1, None, 3, 5]
        b = [2, 0, 3, 4]

        x = xnd(a)
        y = xnd(b)
        z = fn.equaln(x, y)
        self.assertEqual(z, [False, False, True, False])
        self.assertEqual(z.dtype, ndt("bool"))

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_equaln_cuda(self):
        a = [1, None, 3, 5]
        b = [2, None, 3, 4]

        x = xnd(a, device="cuda:managed")
        y = xnd(b, device="cuda:managed")
        z = cd.equaln(x, y)
        self.assertEqual(z, [False, True, True, False])
        self.assertEqual(z.dtype, ndt("bool"))

        a = [1, None, 3, 5]
        b = [2, 0, 3, 4]

        x = xnd(a, device="cuda:managed")
        y = xnd(b, device="cuda:managed")
        z = cd.equaln(x, y)
        self.assertEqual(z, [False, False, True, False])
        self.assertEqual(z.dtype, ndt("bool"))


class TestEqualN(unittest.TestCase):

    def test_nan_float(self):
        for dtype in "bfloat16", "float32", "float64":
            x = xnd([0, float("nan"), 2], dtype=dtype)

            y = xnd([0, float("nan"), 2], dtype=dtype)
            z = fn.equaln(x, y)
            self.assertEqual(z, [True, True, True])

            y = xnd([0, 1, 2], dtype=dtype)
            z = fn.equaln(x, y)
            self.assertEqual(z, [True, False, True])

    def test_nan_complex(self):
        for dtype in "complex64", "complex128":
            for a, b, ans in [
                (complex(float("nan"), 1.2), complex(float("nan"), 1.2), True),
                (complex(float("nan"), 1.2), complex(float("nan"), 1), False),
                (complex(float("nan"), float("nan")), complex(float("nan"), 1.2), False),

                (complex(1.2, float("nan")), complex(1.2, float("nan")), True),
                (complex(1.2, float("nan")), complex(1, float("nan")), False),
                (complex(float("nan"), float("nan")), complex(1.2, float("nan")), False),

                (complex(float("nan"), float("nan")), complex(float("nan"), float("nan")), True)]:

                x = xnd([0, a, 2], dtype=dtype)
                y = xnd([0, b, 2], dtype=dtype)
                z = fn.equaln(x, y)
                self.assertEqual(z, [True, ans, True])

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_nan_float_cuda(self):
        for dtype in "bfloat16", "float16", "float32", "float64":
            x = xnd([0, float("nan"), 2], dtype=dtype, device="cuda:managed")

            y = xnd([0, float("nan"), 2], dtype=dtype, device="cuda:managed")
            z = cd.equaln(x, y)
            self.assertEqual(z, [True, True, True])

            y = xnd([0, 1, 2], dtype=dtype, device="cuda:managed")
            z = cd.equaln(x, y)
            self.assertEqual(z, [True, False, True])

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_nan_complex_cuda(self):
        for dtype in "complex64", "complex128":
            for a, b, ans in [
                (complex(float("nan"), 1.2), complex(float("nan"), 1.2), True),
                (complex(float("nan"), 1.2), complex(float("nan"), 1), False),
                (complex(float("nan"), float("nan")), complex(float("nan"), 1.2), False),

                (complex(1.2, float("nan")), complex(1.2, float("nan")), True),
                (complex(1.2, float("nan")), complex(1, float("nan")), False),
                (complex(float("nan"), float("nan")), complex(1.2, float("nan")), False),

                (complex(float("nan"), float("nan")), complex(float("nan"), float("nan")), True)]:

                x = xnd([0, a, 2], dtype=dtype, device="cuda:managed")
                y = xnd([0, b, 2], dtype=dtype, device="cuda:managed")
                z = cd.equaln(x, y)
                self.assertEqual(z, [True, ans, True])


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

    @unittest.skipIf(True, "abstract return types are temporarily disabled")
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


class TestOut(unittest.TestCase):

    def test_api_cpu(self):
        # negative
        x = xnd([1, 2, 3])
        y = xnd.empty("3 * int64")
        z = fn.negative(x, out=y)

        self.assertIs(z, y)
        self.assertEqual(y, xnd([-1, -2, -3]))

        # divmod
        x = xnd([10, 20, 30])
        y = xnd([7, 8, 9])
        a = xnd.empty("3 * int64")
        b = xnd.empty("3 * int64")
        q, r = fn.divmod(x, y, out=(a, b))

        self.assertIs(q, a)
        self.assertIs(r, b)

        self.assertEqual(q, xnd([1, 2, 3]))
        self.assertEqual(r, xnd([3, 4, 3]))

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_api_cuda(self):
        # negative
        x = xnd([1, 2, 3], device="cuda:managed")
        y = xnd.empty("3 * int64", device="cuda:managed")
        z = cd.negative(x, out=y)

        self.assertIs(z, y)
        self.assertEqual(y, xnd([-1, -2, -3]))

        # divmod
        x = xnd([10, 20, 30], device="cuda:managed")
        y = xnd([7, 8, 9], device="cuda:managed")
        a = xnd.empty("3 * int64", device="cuda:managed")
        b = xnd.empty("3 * int64", device="cuda:managed")
        q, r = cd.divmod(x, y, out=(a, b))

        self.assertIs(q, a)
        self.assertIs(r, b)

        self.assertEqual(q, xnd([1, 2, 3]))
        self.assertEqual(r, xnd([3, 4, 3]))

    def test_broadcast_cpu(self):
        # multiply
        x = xnd([1, 2, 3])
        y = xnd([2])
        z = xnd.empty("3 * int64")
        ans = fn.multiply(x, y, out=z)

        self.assertIs(ans, z)
        self.assertEqual(ans, xnd([2, 4, 6]))

        x = xnd([1, 2, 3])
        y = xnd(2)
        z = xnd.empty("3 * int64")
        ans = fn.multiply(x, y, out=z)

        self.assertIs(ans, z)
        self.assertEqual(ans, xnd([2, 4, 6]))

        # divmod
        x = xnd([10, 20, 30])
        y = xnd([3])
        a = xnd.empty("3 * int64")
        b = xnd.empty("3 * int64")
        q, r = fn.divmod(x, y, out=(a, b))

        self.assertIs(q, a)
        self.assertIs(r, b)
        self.assertEqual(q, xnd([3, 6, 10]))
        self.assertEqual(r, xnd([1, 2, 0]))

        x = xnd([10, 20, 30])
        y = xnd(3)
        a = xnd.empty("3 * int64")
        b = xnd.empty("3 * int64")
        q, r = fn.divmod(x, y, out=(a, b))

        self.assertIs(q, a)
        self.assertIs(r, b)
        self.assertEqual(q, xnd([3, 6, 10]))
        self.assertEqual(r, xnd([1, 2, 0]))

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_broadcast_cuda(self):
        # multiply
        x = xnd([1, 2, 3], device="cuda:managed")
        y = xnd([2], device="cuda:managed")
        z = xnd.empty("3 * int64", device="cuda:managed")
        ans = fn.multiply(x, y, out=z)

        self.assertIs(ans, z)
        self.assertEqual(ans, xnd([2, 4, 6]))


class TestUnaryCPU(unittest.TestCase):

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


@unittest.skipIf(cd is None, "test requires cuda")
class TestUnaryCUDA(unittest.TestCase):

    def test_cos(self):
        a = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        b = [math.cos(x) for x in a]

        x = xnd(a, dtype="float64", device="cuda:managed")
        y = cd.cos(x)
        self.assertEqual(y, b)

    def test_cos_opt(self):
        a = [0, 0.1, 0.2, None, 0.4, 0.5, 0.6, None]
        b = [math.cos(x) if x is not None else None for x in a]

        x = xnd(a, dtype="?float64", device="cuda:managed")
        y = cd.cos(x)
        self.assertEqual(y, b)

    def test_inexact_cast(self):
        a = [0, 1, 2, 3, 4, 5, 6, 7]
        x = xnd(a, dtype="int64", device="cuda:managed")
        self.assertRaises(ValueError, cd.sin, x)


class TestBinaryCPU(unittest.TestCase):

    def test_binary(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t.type)
            y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u.type)
            z = fn.add(x, y)
            self.assertEqual(z, [1, 3, 5, 7, 9, 11, 13, 15])

    def test_add_opt(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([0, 1, None, 3, 4, 5, 6, 7], dtype="?" + t.type)
            y = xnd([1, 2, 3, 4, 5, 6, None, 8], dtype="?" + u.type)
            z = fn.add(x, y)
            self.assertEqual(z, [1, 3, None, 7, 9, 11, None, 15])

    def test_subtract(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t.type)
            y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u.type)
            z = fn.subtract(x, y)
            self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])

    def test_multiply(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t.type)
            y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u.type)
            z = fn.subtract(x, y)
            self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])


@unittest.skipIf(cd is None, "test requires cuda")
class TestBinaryCUDA(unittest.TestCase):

    def test_binary(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t.type, device="cuda:managed")
            y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u.type, device="cuda:managed")
            z = cd.add(x, y)
            self.assertEqual(z, [1, 3, 5, 7, 9, 11, 13, 15])

    def test_add_opt(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([0, 1, None, 3, 4, 5, 6, 7], dtype="?" + t.type, device="cuda:managed")
            y = xnd([1, 2, 3, 4, 5, 6, None, 8], dtype="?" + u.type, device="cuda:managed")
            z = cd.add(x, y)
            self.assertEqual(z, [1, 3, None, 7, 9, 11, None, 15])

    def test_subtract(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t.type, device="cuda:managed")
            y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u.type, device="cuda:managed")
            z = cd.subtract(x, y)
            self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])

    def test_multiply(self):
        for t, u in implemented_sigs["binary"]["default"]:
            w = implemented_sigs["binary"]["default"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([2, 3, 4, 5, 6, 7, 8, 9], dtype=t.type, device="cuda:managed")
            y = xnd([1, 2, 3, 4, 5, 6, 7, 8], dtype=u.type, device="cuda:managed")
            z = cd.subtract(x, y)
            self.assertEqual(z, [1, 1, 1, 1, 1, 1, 1, 1])


class TestBitwiseCPU(unittest.TestCase):

    def test_and(self):
        for t, u in implemented_sigs["binary"]["bitwise"]:
            w = implemented_sigs["binary"]["bitwise"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t.type)
            x = xnd([0, 1, 0, 1, 1, 1, 1, 0], dtype=t.type)
            y = xnd([1, 0, 0, 0, 1, 1, 1, 1], dtype=u.type)
            z = fn.bitwise_and(x, y)
            self.assertEqual(z, [0, 0, 0, 0, 1, 1, 1, 0])

    def test_and_opt(self):
        for t, u in implemented_sigs["binary"]["bitwise"]:
            w = implemented_sigs["binary"]["bitwise"][(t, u)]

            if t.cpu_noimpl() or u.cpu_noimpl():
                continue

            a = [0, 1, None, 1, 1, 1, 1, 0]
            b = [1, 1, 1, 1, 1, 1, None, 0]
            c = [0, 1, None, 1, 1, 1, None, 0]

            x = xnd(a, dtype="?" + t.type)
            y = xnd(b, dtype="?" + u.type)
            z = fn.bitwise_and(x, y)
            self.assertEqual(z, c)


@unittest.skipIf(cd is None, "test requires cuda")
class TestBitwiseCUDA(unittest.TestCase):

    def test_and(self):
        for t, u in implemented_sigs["binary"]["bitwise"]:
            w = implemented_sigs["binary"]["bitwise"][(t, u)]

            if t.cuda_noimpl() or u.cuda_noimpl():
                continue

            x = xnd([0, 1, 2, 3, 4, 5, 6, 7], dtype=t.type, device="cuda:managed")
            x = xnd([0, 1, 0, 1, 1, 1, 1, 0], dtype=t.type, device="cuda:managed")
            y = xnd([1, 0, 0, 0, 1, 1, 1, 1], dtype=u.type, device="cuda:managed")
            z = cd.bitwise_and(x, y)
            self.assertEqual(z, [0, 0, 0, 0, 1, 1, 1, 0])

    def test_and_opt(self):
        for t, u in implemented_sigs["binary"]["bitwise"]:
            w = implemented_sigs["binary"]["bitwise"][(t, u)]

            if t.cuda_noimpl() or u.cuda_noimpl():
                continue

            a = [0, 1, None, 1, 1, 1, 1, 0]
            b = [1, 1, 1, 1, 1, 1, None, 0]
            c = [0, 1, None, 1, 1, 1, None, 0]

            x = xnd(a, dtype="?" + t.type, device="cuda:managed")
            y = xnd(b, dtype="?" + u.type, device="cuda:managed")
            z = cd.bitwise_and(x, y)
            self.assertEqual(z, c)


@unittest.skipIf(np is None, "test requires numpy")
class TestFunctions(unittest.TestCase):

    def assertRelErrorLess(self, calc, expected, maxerr, msg):
        if cmath.isnan(calc) or cmath.isnan(expected):
            return
        elif cmath.isinf(calc) or cmath.isinf(expected):
            return
        elif abs(expected) < 1e-6 and abs(calc) < 1e-6:
            return
        else:
            err = abs((calc-expected) / expected)
            self.assertLess(err, maxerr, msg)

    def equal(self, calc, expected, msg):
        if np.isnan(calc) and np.isnan(expected):
            return
        else:
            self.assertEqual(calc, expected, msg)

    def assert_equal(self, f, z1, z2, w, msg):
        if w.type == "bfloat16":
            self.assertRelErrorLess(z1, z2, 1e-2, msg)
        elif f in functions["unary"]["real_math"] or \
             f in functions["unary"]["real_math_with_half"] or \
             f in functions["unary"]["complex_math"] or \
             f in functions["unary"]["complex_math_with_half"]:
            self.assertRelErrorLess(z1, z2, 1e-2, msg)
        elif isinstance(z1, complex):
            if f in ("multiply", "divide"):
                self.assertRelErrorLess(z1.real, z2.real, 1e-2, msg)
                self.assertRelErrorLess(z1.imag, z2.imag, 1e-2, msg)
            else:
                self.equal(z1.real, z2.real, msg) and \
                self.equal(z1.imag, z2.imag, msg)
        elif f == "divide" and w.type in ("float16", "float32"):
            self.assertRelErrorLess(z1, z2, 1e-2, msg)
        else:
            return self.equal(z1, z2, msg)

    def create_xnd(self, a, t, dev=None):

        # Check that struct.pack(a) overflows iff xnd(a) overflows.
        overflow = struct_overflow(a, t)
        xnd_overflow = False
        try:
            x = xnd([a], dtype=t.type, device=dev)
        except OverflowError:
            xnd_overflow = True

        self.assertEqual(xnd_overflow, overflow)

        return None if xnd_overflow else x

    def check_unary_not_implemented(self, f, a, t, mod=fn, dev=None):

        x = self.create_xnd(a, t, dev)
        if x is None:
            return

        self.assertRaises(NotImplementedError, getattr(mod, f), x)

    def check_unary_type_error(self, f, a, t, mod=fn, dev=None):

        x = self.create_xnd(a, t, dev)
        if x is None:
            return

        self.assertRaises(TypeError, getattr(mod, f), x)

    def check_unary(self, f, a, t, u, mod=fn, dev=None):

        x1 = self.create_xnd(a, t, dev)
        if x1 is None:
            return

        y1 = getattr(mod, f)(x1)
        self.assertEqual(str(y1[0].type), u.type)
        v1 = y1[0].value

        value = x1.value if t.type == "bfloat16" else a
        dtype = "float32" if t.type == "bfloat16" else t.type

        x2 = np.array([value], dtype=dtype)
        y2 = getattr(np, np_function(f))(x2)
        v2 = y2[0]

        msg = "%s(%s : %s) -> %s    xnd: %s    np: %s" % (f, a, t, u, y1, y2)
        self.assert_equal(f, v1, v2, u, msg)

    def check_binary_not_implemented(self, f, a, t, b, u, mod=fn, dev=None):

        x1 = self.create_xnd(a, t, dev)
        if x1 is None:
            return

        y1 = self.create_xnd(b, u, dev)
        if y1 is None:
            return

        self.assertRaises(NotImplementedError, getattr(mod, f), x1, y1)

    def check_binary_type_error(self, f, a, t, b, u, mod=fn, dev=None):

        x1 = self.create_xnd(a, t, dev)
        if x1 is None:
            return

        y1 = self.create_xnd(b, u, dev)
        if y1 is None:
            return

        self.assertRaises(TypeError, getattr(mod, f), x1, y1)

    def check_binary(self, f, a, t, b, u, w, mod=fn, dev=None):

        x1 = self.create_xnd(a, t, dev)
        if x1 is None:
            return

        y1 = self.create_xnd(b, u, dev)
        if y1 is None:
            return

        z1 = getattr(mod, f)(x1, y1)
        self.assertEqual(str(z1[0].type), w.type)
        v1 = z1[0].value

        dtype1 = "float32" if t.type == "bfloat16" else t.type
        dtype2 = "float32" if u.type == "bfloat16" else u.type
        value1 = x1.value if t.type == "bfloat16" else a
        value2 = y1.value if u.type == "bfloat16" else b

        x2 = np.array([value1], dtype=dtype1)
        y2 = np.array([value2], dtype=dtype2)
        z2 = getattr(np, f)(x2, y2)
        v2 = z2[0]

        msg = "%s(%s : %s, %s : %s) -> %s    xnd: %s    np: %s" % \
              (f, a, t, b, u, w, z1, z2)
        self.assert_equal(f, v1, v2, w, msg)

    def check_binary_mv(self, f, a, t, b, u, v, w, mod=fn, dev=None):

        x1 = self.create_xnd(a, t, dev)
        if x1 is None:
            return

        y1 = self.create_xnd(b, u, dev)
        if y1 is None:
            return

        c1, d1 = getattr(mod, f)(x1, y1)
        self.assertEqual(str(c1[0].type), v.type)
        self.assertEqual(str(d1[0].type), w.type)
        cv1 = c1[0].value
        dv1 = d1[0].value

        x2 = np.array([a], dtype=t.type)
        y2 = np.array([b], dtype=u.type)
        c2, d2 = getattr(np, f)(x2, y2)
        cv2 = c2[0]
        dv2 = d2[0]

        msg = "%s(%s : %s, %s : %s) -> %s, %s    xnd: %s    np: %s" % \
              (f, a, t, b, u, v, w, (cv1, dv1), (cv2, dv2))
        self.assert_equal(f, cv1, cv2, v, msg)
        self.assert_equal(f, dv2, dv2, v, msg)

    @unittest.skipIf(sys.platform == "darwin", "complex trigonometry errors too large")
    @unittest.skipIf(sys.platform == "win32" and ARCH == "32bit", "complex trigonometry errors too large")
    def test_unary_cpu(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        print("\n", flush=True)

        for pattern, return_type in [
              ("default", "default"),
              ("complex_math", "float_result"),
              ("real_math", "float_result")]:

            for f in functions["unary"][pattern]:
                if np_noimpl(f):
                    continue

                print("testing %s ..." % f, flush=True)

                for t, in implemented_sigs["unary"][return_type]:
                    u = implemented_sigs["unary"][return_type][(t,)]

                    print("    %s -> %s" % (t, u), flush=True)

                    for a in t.testcases():
                        if t.cpu_noimpl(f) or u.cpu_noimpl(f):
                            self.check_unary_not_implemented(f, a, t)
                        else:
                            self.check_unary(f, a, t, u)

    def test_binary_cpu(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        print("\n", flush=True)

        for pattern in "default", "float_result", "bool_result":
            for f in functions["binary"][pattern]:
                print("testing %s ..." % f, flush=True)

                for t, u in implemented_sigs["binary"][pattern]:
                    w = implemented_sigs["binary"][pattern][(t, u)]

                    print("    %s, %s -> %s" % (t, u, w), flush=True)

                    for a in t.testcases():
                        for b in u.testcases():
                            if t.cpu_nokern(f) or u.cpu_nokern(f) or w.cpu_nokern(f):
                                self.check_binary_type_error(f, a, t, b, u)
                            elif t.cpu_noimpl(f) or u.cpu_noimpl(f) or w.cpu_noimpl(f):
                                self.check_binary_not_implemented(f, a, t, b, u)
                            else:
                                self.check_binary(f, a, t, b, u, w)

    def test_binary_mv_cpu(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        print("\n", flush=True)

        for f in functions["binary_mv"]["default"]:
            print("testing %s ..." % f, flush=True)

            for t, u in implemented_sigs["binary_mv"]["default"]:
                v, w = implemented_sigs["binary_mv"]["default"][(t, u)]

                print("    %s, %s -> %s, %s" % (t, u, v, w), flush=True)

                for a in t.testcases():
                    for b in u.testcases():
                        self.check_binary_mv(f, a, t, b, u, v, w)

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_unary_cuda(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        print("\n", flush=True)

        for pattern, return_type in [
              ("default", "default"),
              ("complex_math_with_half", "float_result"),
              ("complex_math", "float_result"),
              ("real_math_with_half", "float_result"),
              ("real_math", "float_result")]:

            for f in functions["unary"][pattern]:
                if np_noimpl(f):
                    continue

                print("testing %s ..." % f, flush=True)

                for t, in implemented_sigs["unary"][return_type]:
                    u = implemented_sigs["unary"][return_type][(t,)]

                    print("    %s -> %s" % (t, u), flush=True)

                    for a in t.testcases():
                        if t.cuda_noimpl(f) or u.cuda_noimpl(f):
                            self.check_unary_not_implemented(
                                f, a, t, mod=cd, dev="cuda:managed")
                        else:
                            self.check_unary(f, a, t, u,
                                mod=cd, dev="cuda:managed")

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_binary_cuda(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        print("\n", flush=True)

        for pattern in "float_result", "bool_result":
            for f in functions["binary"][pattern]:
                print("testing %s ..." % f, flush=True)

                for t, u in implemented_sigs["binary"][pattern]:
                    w = implemented_sigs["binary"][pattern][(t, u)]

                    print("    %s, %s -> %s" % (t, u, w), flush=True)

                    for a in t.testcases():
                        for b in u.testcases():
                            if t.cuda_nokern(f) or u.cuda_nokern(f) or w.cuda_nokern(f):
                                self.check_binary_type_error(f, a, t, b, u,
                                    mod=cd, dev="cuda:managed")
                            elif t.type == "complex32" or u.type == "complex32" or w.cuda_noimpl(f):
                                self.check_binary_not_implemented(f, a, t, b, u,
                                    mod=cd, dev="cuda:managed")
                            else:
                                self.check_binary(f, a, t, b, u, w,
                                    mod=cd, dev="cuda:managed")

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_binary_mv_cuda(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        print("\n", flush=True)

        for f in functions["binary_mv"]["default"]:
            print("testing %s ..." % f, flush=True)

            for t, u in implemented_sigs["binary_mv"]["default"]:
                v, w = implemented_sigs["binary_mv"]["default"][(t, u)]

                print("    %s, %s -> %s, %s" % (t, u, v, w), flush=True)

                for a in t.testcases():
                    for b in u.testcases():
                        self.check_binary_mv(f, a, t, b, u, v, w, mod=cd,
                                             dev="cuda:managed")

    def test_divide_inexact_cpu(self):

        t = Tint("uint8")
        u = Tint("uint64")

        a = next(t.testcases())
        b = next(u.testcases())
        self.check_binary_type_error("divide", a, t, b, u)

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_divide_inexact_cuda(self):

        t = Tint("uint8")
        u = Tint("uint64")

        a = next(t.testcases())
        b = next(u.testcases())
        self.check_binary_type_error("divide", a, t, b, u,
                                     mod=cd, dev="cuda:managed")

    def test_divmod_type_error_cpu(self):

        t = Tint("uint8")
        u = Tint("uint64")

        a = next(t.testcases())
        b = next(u.testcases())
        self.check_binary_type_error("divmod", a, t, b, u)

    @unittest.skipIf(cd is None, "test requires cuda")
    def test_divmod_type_error_cuda(self):

        t = Tint("uint8")
        u = Tint("uint64")

        a = next(t.testcases())
        b = next(u.testcases())
        self.check_binary_type_error("divmod", a, t, b, u)


@unittest.skipIf(cd is None, "test requires cuda")
class TestCudaManaged(unittest.TestCase):

    def test_mixed_functions(self):

        x = xnd([1,2,3])
        y = xnd([1,2,3])

        a = xnd([1,2,3], device="cuda:managed")
        b = xnd([1,2,3], device="cuda:managed")

        z = fn.multiply(x, y)
        c = cd.multiply(a, b)
        self.assertEqual(z, c)

        z = fn.multiply(a, b)
        self.assertEqual(z, c)

        z = fn.multiply(x, b)
        self.assertEqual(z, c)

        z = fn.multiply(a, y)
        self.assertEqual(z, c)

        self.assertRaises(ValueError, cd.multiply, x, y)
        self.assertRaises(ValueError, cd.multiply, x, b)
        self.assertRaises(ValueError, cd.multiply, a, y)


class TestSpec(unittest.TestCase):

    def __init__(self, *, constr, ndarray, mod,
                 values, value_generator,
                 indices_generator, indices_generator_args):
        super().__init__()
        self.constr = constr
        self.ndarray = ndarray
        self.mod = mod
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

    def run_reduce(self, nd, d):
        if not isinstance(nd, xnd) or not isinstance(d, np.ndarray):
            return

        for attr in ["add", "subtract", "multiply"]:
            f = getattr(fn, attr)
            g = getattr(np, attr)

            x = nd_exception = None
            try:
                x = gm.reduce(f, nd, dtype=nd.dtype)
            except Exception as e:
                nd_exception =  e

            y = np_exception = None
            try:
                y = g.reduce(d, dtype=d.dtype)
            except Exception as e:
                np_exception =  e

            if nd_exception or np_exception:
                self.assertIs(nd_exception.__class__, np_exception.__class__,
                              "f: %r nd: %r np: %r x: %r y: %r" % (attr, nd, d, x, y))
            else:
                self.assertEqual(x.value, y.tolist(),
                                 "f: %r nd: %r np: %r x: %r y: %r" % (attr, nd, d, x, y))

            for axes in gen_axes(d.ndim):
                nd_exception = None
                try:
                    x = gm.reduce(f, nd, axes=axes, dtype=nd.dtype)
                except Exception as e:
                    nd_exception =  e

                np_exception = None
                try:
                    y = g.reduce(d, axis=axes, dtype=d.dtype)
                except Exception as e:
                    np_exception =  e

                if nd_exception or np_exception:
                    self.assertIs(nd_exception.__class__, np_exception.__class__,
                                  "f: %r axes: %r nd: %r np: %r x: %r y: %r" % (attr, axes, nd, d, x, y))
                else:
                    self.assertEqual(x.value, y.tolist(),
                                     "f: %r axes: %r nd: %r np: %r x: %r y: %r" % (attr, axes, nd, d, x, y))

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

        x = self.mod.sin(nd_result)
        y = self.mod.multiply(nd_result, nd_result)

        if isinstance(def_result, NDArray):
            aa = a = def_result.sin()
            b = def_result * def_result
        elif isinstance(def_result, int):
            aa = a = math.sin(def_result)
            b = def_result * def_result
        elif def_result is None:
            aa = a = None
            aa = b = None
        elif isinstance(def_result, np.ndarray):
            aa = np.sin(def_result)
            a = aa.tolist()
            bb = np.multiply(def_result, def_result)
            b = bb.tolist()
        elif isinstance(def_result, np.int32):
            aa = np.sin(def_result)
            a = aa.tolist()
            bb = np.multiply(def_result, def_result)
            b = bb.tolist()
        else:
            raise TypeError("unexpected def_result: %s : %s" % (def_result, type(def_result)))

        if self.mod == cd:
            np.testing.assert_allclose(x, aa, 1e-6)
            np.testing.assert_allclose(y, bb, 1e-6)
        else:
            self.assertEqual(x, a)
            self.assertEqual(y, b)

        if self.mod == fn:
            self.run_reduce(nd_result, def_result)

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
            if self.constr == xnd:
                nd = xnd(value, dtype=dtype, device=None if self.mod==fn else "cuda:managed")
            else:
                nd = self.constr(value, dtype=dtype)
            # NumPy does not support "?int32", NDArray does not need the dtype.
            d = self.ndarray(value, dtype="int32")
            check(nd, d, value, 0)

        for max_ndim in range(1, 5):
            for min_shape in (0, 1):
                for max_shape in range(1, 8):
                    for value in self.value_generator(max_ndim, min_shape, max_shape):
                        dtype = "?int32" if have_none(value) else "int32"
                        if self.constr == xnd:
                            nd = xnd(value, dtype=dtype, device=None if self.mod==fn else "cuda:managed")
                        else:
                            nd = self.constr(value, dtype=dtype)
                        # See above.
                        d = self.ndarray(value, dtype="int32")
                        check(nd, d, value, 0)


class LongIndexSliceTest(unittest.TestCase):

    def test_subarray(self):
        # Multidimensional indexing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

    @unittest.skipIf(cd is None or np is None, "cuda or numpy not found")
    def test_subarray_cuda(self):
        # Multidimensional indexing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=np.array,
                     mod=cd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genindices,
                     indices_generator_args=())
        t.run()

    def test_slices(self):
        # Multidimensional slicing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

    @unittest.skipIf(cd is None or np is None, "cuda or numpy not found")
    def test_slices_cuda(self):
        # Multidimensional slicing
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=np.array,
                     mod=cd,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=randslices,
                     indices_generator_args=(3,))
        t.run()

    def test_chained_indices_slices(self):
        # Multidimensional indexing and slicing, chained
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=gen_indices_or_slices,
                     indices_generator_args=())
        t.run()


        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=gen_indices_or_slices,
                     indices_generator_args=())
        t.run()

    def test_fixed_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=mixed_indices,
                     indices_generator_args=(3,))
        t.run()

    def test_var_mixed_indices_slices(self):
        # Multidimensional indexing and slicing, mixed
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=mixed_indices,
                     indices_generator_args=(5,))
        t.run()

    def test_slices_brute_force(self):
        # Test all possible slices for the given ndim and shape
        skip_if(SKIP_BRUTE_FORCE, "use --all argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=genslices_ndim,
                     indices_generator_args=(3, [3,3,3]))
        t.run()

        t = TestSpec(constr=xnd,
                     ndarray=NDArray,
                     mod=fn,
                     values=SUBSCRIPT_VAR_TEST_CASES,
                     value_generator=gen_var,
                     indices_generator=genslices_ndim,
                     indices_generator_args=(3, [3,3,3]))
        t.run()

    @unittest.skipIf(np is None, "numpy not found")
    def test_reduce(self):
        skip_if(SKIP_LONG, "use --long argument to enable these tests")

        t = TestSpec(constr=xnd,
                     ndarray=np.array,
                     mod=fn,
                     values=SUBSCRIPT_FIXED_TEST_CASES,
                     value_generator=gen_fixed,
                     indices_generator=mixed_indices,
                     indices_generator_args=(3,))
        t.run()


ALL_TESTS = [
  TestCall,
  TestRaggedArrays,
  TestMissingValues,
  TestEqualN,
  TestGraphs,
  TestPdist,
  TestNumba,
  TestOut,
  TestUnaryCPU,
  TestUnaryCUDA,
  TestBinaryCPU,
  TestBinaryCUDA,
  TestBitwiseCPU,
  TestBitwiseCUDA,
  TestFunctions,
  TestCudaManaged,
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

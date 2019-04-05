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

import sys, os
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "1"

from xnd import *
import io
import argparse
import unittest
from gumath_aux import gen_fixed
from random import randrange

try:
    import numpy as np
    HAVE_ARRAY_FUNCTION = hasattr(np.ndarray, '__array_function__')
    np.warnings.filterwarnings('ignore')
except ImportError:
    np = None
    HAVE_ARRAY_FUNCTION = False


unary_operators = [
  # '__abs__',
  # '__bool__',
  '__invert__',
  '__neg__',
  '__pos__',
]

binary_operators = [
  '__add__',
  '__and__',
  '__eq__',
  '__floordiv__',
  '__ge__',
  '__gt__',
  '__iadd__',
  '__iand__',
  '__ifloordiv__',
  '__imod__',
  '__imul__',
  '__ior__',
  # '__ipow__',
  '__isub__',
  '__ixor__',
  '__le__',
  '__lt__',
  '__mod__',
  '__mul__',
  '__ne__',
  '__or__',
  '__sub__',
  '__xor__'
]

binary_truediv = [
  '__itruediv__',
  '__truediv__',
]


@unittest.skipIf(np is None, "test requires numpy")
class TestOperators(unittest.TestCase):

    def assertStrictEqual(self, other):
        self.assertTrue(self.strict_equal(other))

    def test_unary(self):

        a = array([20, 30, 40])
        x = np.array([20, 30, 40])

        for attr in unary_operators:
            b = getattr(a, attr)()
            y = getattr(x, attr)()
            self.assertEqual(b.tolist(), y.tolist())

    def test_binary(self):

        x = array([20, 30, 40], dtype="int32")
        y = array([3, 5, 7], dtype="int32")

        a = np.array([20, 30, 40], dtype="int32")
        b = np.array([3, 5, 7], dtype="int32")

        for attr in binary_operators:
            z = getattr(x, attr)(y)
            c = getattr(a, attr)(b)
            self.assertEqual(z.tolist(), c.tolist())

        x = array([20, 30, 40], dtype="float64")
        a = np.array([20, 30, 40], dtype="float64")

        for attr in binary_truediv:
            z = getattr(x, attr)(y)
            c = getattr(a, attr)(b)
            self.assertEqual(z.tolist(), c.tolist())



@unittest.skipIf(np is None, "test requires numpy")
class TestArrayUfunc(unittest.TestCase):

    ufuncs =[v for v in np.__dict__.values() if isinstance(v, np.ufunc)]
    # funcs = [v for v in np.__dict__.values() if callable(v) and hasattr(v, '__wrapped__')]
    binary_plus_axis = { "tensordot": (np.ndarray, np.ndarray, int), }
    binary = { "dot": (np.ndarray, np.ndarray), }

    def test_ufuncs(self):

        for x in gen_fixed(3, 1, 5):
            for y in gen_fixed(3, 1, 5):
                xnd_x = array(x, dtype="float32")
                xnd_y = array(y, dtype="float32")
                np_x = np.array(x, dtype="float32")
                np_y = np.array(y, dtype="float32")
                for f in self.ufuncs:
                    arity = 1
                    try:
                        a = f(np_x)
                    except:
                        arity = 2
                        try:
                            a = f(np_x, np_y)
                        except:
                            continue

                    if arity == 1:
                        b = f(xnd_x)
                    else:
                        b = f(xnd_x, xnd_y)

                    np.testing.assert_equal(a, b)

    @unittest.skipIf(not HAVE_ARRAY_FUNCTION,
                     "test requires numpy with __array_function__ support")
    def test_tensordot(self):

        def f(x):
            y = np.tensordot(x, x.T)
            return np.mean(np.exp(y))

        x = array([[1, 2], [3, 4]], dtype="float64")
        ans = f(x)
        self.assertEqual(ans, 3931334297144.042)

    def test_einsum(self):
        # Use the examples from the numpy docs.
        npa = np.arange(25).reshape(5,5)
        npb = np.arange(5)
        npc = np.arange(6).reshape(2,3)

        a = array.from_buffer(npa)
        b = array.from_buffer(npb)
        c = array.from_buffer(npc)

        ans = np.einsum('ii', a)
        self.assertEqual(ans, 60)

        ans = np.einsum('ii->i', a)
        self.assertTrue(np.all(ans == (array([0, 6, 12, 18, 24]))))

        ans = np.einsum(a, [0,0], [0])
        self.assertTrue(np.all(ans == (array([0, 6, 12, 18, 24]))))

        ans = np.diag(a)
        self.assertTrue(np.all(ans == (array([0, 6, 12, 18, 24]))))

        ans = np.einsum('ij,j', a, b)
        self.assertTrue(np.all(ans == (array([ 30,  80, 130, 180, 230]))))

        ans = np.einsum(a, [0,1], b, [1])
        self.assertTrue(np.all(ans == (array([ 30,  80, 130, 180, 230]))))

        ans = np.dot(a, b)
        self.assertTrue(np.all(ans == (array([ 30,  80, 130, 180, 230]))))

        ans = np.einsum('...j,j', a, b)
        self.assertTrue(np.all(ans == (array([ 30,  80, 130, 180, 230]))))

        ans = np.einsum('ji', c)
        self.assertTrue(np.all(ans == (array([[0, 3], [1, 4], [2, 5]]))))

        ans = np.einsum(c, [1,0])
        self.assertTrue(np.all(ans == (array([[0, 3], [1, 4], [2, 5]]))))

        ans = c.T
        self.assertTrue(np.all(ans == (array([[0, 3], [1, 4], [2, 5]]))))

        ans = np.einsum('..., ...', 3, c)
        self.assertTrue(np.all(ans == (array([[ 0,  3,  6], [ 9, 12, 15]]))))

        ans = np.einsum(',ij', 3, c)
        self.assertTrue(np.all(ans == (array([[ 0,  3,  6], [ 9, 12, 15]]))))

        ans = np.einsum(3, [Ellipsis], c, [Ellipsis])
        self.assertTrue(np.all(ans == (array([[ 0,  3,  6], [ 9, 12, 15]]))))

        ans = 3 * c
        self.assertTrue(np.all(ans == (array([[ 0,  3,  6], [ 9, 12, 15]]))))

        ans = np.einsum('i,i', b, b)
        self.assertEqual(ans, 30)

        ans = np.einsum(b, [0], b, [0])
        self.assertEqual(ans, 30)

        ans = np.inner(b, b)
        self.assertEqual(ans, 30)

        ans = np.einsum('i,j', np.arange(2)+1, b)
        self.assertTrue(np.all(ans == (array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]))))

        ans = np.einsum(np.arange(2)+1, [0], b, [1])
        self.assertTrue(np.all(ans == (array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]))))

        ans = np.outer(np.arange(2)+1, b)
        self.assertTrue(np.all(ans == (array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]))))

        ans = np.einsum('i...->...', a)
        self.assertTrue(np.all(ans == (array([50, 55, 60, 65, 70]))))

        ans = np.einsum(a, [0,Ellipsis], [Ellipsis])
        self.assertTrue(np.all(ans == (array([50, 55, 60, 65, 70]))))

        ans = np.sum(a, axis=0)
        self.assertTrue(np.all(ans == (array([50, 55, 60, 65, 70]))))


        npa = np.arange(60.).reshape(3,4,5)
        npb = b = np.arange(24.).reshape(4,3,2)

        a = array.from_buffer(npa)
        b = array.from_buffer(npb)

        expected = array([[ 4400.,  4730.], [ 4532.,  4874.], [ 4664.,  5018.], [ 4796.,  5162.], [ 4928.,  5306.]])
        ans = np.einsum('ijk,jil->kl', a, b)
        self.assertTrue(np.all(ans == expected))

        ans = np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
        self.assertTrue(np.all(ans == expected))

        ans = np.tensordot(a,b, axes=([1,0],[0,1]))
        self.assertTrue(np.all(ans == expected))


        npa = np.arange(6).reshape((3,2))
        npb = np.arange(12).reshape((4,3))

        a = array.from_buffer(npa)
        b = array.from_buffer(npb)

        expected = array([[10, 28, 46, 64], [13, 40, 67, 94]])
        ans = np.einsum('ki,...k->i...', a, b)
        self.assertTrue(np.all(ans == expected))

        ans = np.einsum('ki,...k->i...', a, b)
        self.assertTrue(np.all(ans == expected))

        ans = np.einsum('k...,jk', a, b)
        self.assertTrue(np.all(ans == expected))


        npa = np.zeros((3, 3))
        a = array.from_buffer(npa)

        # expected = array([[ 1.,  0.,  0.], [ 0.,  1.,  0.], [ 0.,  0.,  1.]])
        # np.einsum('ii->i', a)[:] = 1
        # self.assertTrue(np.all(ans == expected))

    @unittest.skipIf(not HAVE_ARRAY_FUNCTION,
                     "test requires numpy with __array_function__ support")
    def test_array_funcs(self):

        for name in self.binary:
            f = getattr(np, name)
            for lst1 in gen_fixed(3, 1, 5):
                for lst2 in gen_fixed(3, 1, 5):
                    a = np.array(lst1, dtype="float32")
                    b = np.array(lst2, dtype="float32")
                    n = min(a.ndim, b.ndim)

                    np_exc = None
                    try:
                        c = f(a, b)
                    except Exception as e:
                        np_exc = e.__class__

                    x = array(lst1, dtype="float32")
                    y = array(lst2, dtype="float32")

                    xnd_exc = None
                    try:
                        z = f(x, y)
                    except Exception as e:
                        xnd_exc = e.__class__

                    if np_exc or xnd_exc:
                        self.assertEqual(xnd_exc, np_exc)
                        continue

                    self.assertTrue(np.all(z == c))

        for name in self.binary_plus_axis:
            f = getattr(np, name)
            for lst1 in gen_fixed(3, 1, 5):
                for lst2 in gen_fixed(3, 1, 5):
                    a = np.array(lst1, dtype="float32")
                    b = np.array(lst2, dtype="float32")
                    n = min(a.ndim, b.ndim)
                    axis = randrange(n)

                    np_exc = None
                    try:
                        c = f(a, b, axis=axis)
                    except Exception as e:
                        np_exc = e.__class__

                    x = array(lst1, dtype="float32")
                    y = array(lst2, dtype="float32")

                    xnd_exc = None
                    try:
                        z = f(x, y, axis=axis)
                    except Exception as e:
                        xnd_exc = e.__class__

                    if np_exc or xnd_exc:
                        self.assertEqual(xnd_exc, np_exc)
                        continue

                    self.assertTrue(np.all(z == c))


ALL_TESTS = [
  TestOperators,
  TestArrayUfunc,
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

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
import unittest
import argparse
from gumath_aux import gen_fixed

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

    allfuncs = set([v for v in np.__dict__.values() if callable(v)])
    ufuncs = set([v for v in np.__dict__.values() if isinstance(v, np.ufunc)])
    funcs = allfuncs - ufuncs

    def allarray(self, v):
        return isinstance(v, array) or all(isinstance(x, array) for x in v)

    def assertAllXndArray(self, v):
        self.assertTrue(self.allarray(v))

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

                    self.assertAllXndArray(b)
                    np.testing.assert_equal(a, b)

    @unittest.skipIf(not HAVE_ARRAY_FUNCTION,
                     "test requires numpy with __array_function__ support")
    def test_array_func(self):

        def f(x):
            y = np.tensordot(x, x.T)
            return np.mean(np.exp(y))

        x = array([[1, 2], [3, 4]], dtype="float64")
        y = np.array([[1, 2], [3, 4]], dtype="float64")

        a = f(x)
        b = f(y)

        self.assertAllXndArray(a)
        self.assertEqual(a.tolist(), b.tolist())


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

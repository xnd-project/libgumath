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
from xnd import xnd
import sys, time
import unittest
import argparse

try:
    import numpy as np
except ImportError:
    np = None



TEST_CASES = [
  ([float(i) for i in range(2000)], "2000 * float64", "float64"),

  ([[float(i) for i in range(1000)], [float(i+1) for i in range(1000)]],
   "2 * 1000 * float64", "float64"),

  (1000 * [[float(i+1) for i in range(2)]], "1000 * 2 * float64", "float64"),

  ([float(i) for i in range(2000)], "2000 * float32", "float32"),

  ([[float(i) for i in range(1000)], [float(i+1) for i in range(1000)]],
   "2 * 1000 * float32", "float32"),

  (1000 * [[float(i+1) for i in range(2)]], "1000 * 2 * float32", "float32"),
]

class TestCall(unittest.TestCase):

    def test_sin(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)

            start = time.time()
            y = gm.sin(x)
            end = time.time()
            # sys.stderr.write("\ngumath: time=%.3f\n" % (end-start))

            if np is not None:
                a = np.array(lst, dtype=dtype)

                start = time.time()
                b = np.sin(a)
                end = time.time()
                # sys.stderr.write("numpy: time=%.3f\n" % (end-start))

                if dtype == "float64":
                    np.testing.assert_equal(y, b)
                else:
                    np.testing.assert_almost_equal(y, b, 7)

    def test_sin_strided(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)
            if x.type.ndim < 2:
                continue

            y = x[::-2, ::-2]

            start = time.time()
            z = gm.sin(y)
            end = time.time()
            # sys.stderr.write("\ngumath: time=%.3f\n" % (end-start))

            if np is not None:
                a = np.array(lst, dtype=dtype)
                b = a[::-2, ::-2]

                start = time.time()
                c = np.sin(b)
                end = time.time()
                # sys.stderr.write("numpy: time=%.3f\n" % (end-start))

                if dtype == "float64":
                    np.testing.assert_equal(z, c)
                else:
                    np.testing.assert_almost_equal(z, c, 7)

    def test_copy(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)

            start = time.time()
            y = gm.copy(x)
            end = time.time()
            # sys.stderr.write("\ngumath: time=%.3f\n" % (end-start))

            if np is not None:
                a = np.array(lst, dtype=dtype)

                start = time.time()
                b = np.copy(a)
                end = time.time()
                # sys.stderr.write("numpy: time=%.3f\n" % (end-start))

                np.testing.assert_equal(y, b)

    def test_copy_strided(self):

        for lst, t, dtype in TEST_CASES:
            x = xnd(lst, type=t)
            if x.type.ndim < 2:
                continue

            y = x[::-2, ::-2]

            start = time.time()
            z = gm.copy(y)
            end = time.time()
            # sys.stderr.write("\ngumath: time=%.3f\n" % (end-start))

            if np is not None:
                a = np.array(lst, dtype=dtype)
                b = a[::-2, ::-2]

                start = time.time()
                c = np.copy(b)
                end = time.time()
                # sys.stderr.write("numpy: time=%.3f\n" % (end-start))

                np.testing.assert_equal(y, b)

    def test_quaternion(self):
  
        lst = [[[1+2j, 4+3j],
                [-4+3j, 1-2j]],
               [[4+2j, 1+10j],
                [-1+10j, 4-2j]],
               [[-4+2j, 3+10j],
                [-3+10j, -4-2j]]]

        x = xnd(lst, type="3 * Q64(2 * 2 * complex64)")
        y = gm.multiply(x, x)

        if np is not None:
            a = np.array(lst, dtype="complex64")
            b = np.matmul(a, a)
            np.testing.assert_equal(y, b)

    def test_quaternion_error(self):
  
        lst = [[[1+2j, 4+3j],
                [-4+3j, 1-2j]],
               [[4+2j, 1+10j],
                [-1+10j, 4-2j]],
               [[-4+2j, 3+10j],
                [-3+10j, -4-2j]]]

        x = xnd(lst, type="3 * Foo(2 * 2 * complex64)")
        self.assertRaises(TypeError, gm.multiply, x, x)


ALL_TESTS = [
  TestCall
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

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
import math
import unittest
import argparse

try:
    import numpy as np
except ImportError:
    np = None

from hypothesis import given, assume
import hypothesis.strategies as st


class TestCall(unittest.TestCase):
    @given(val=st.floats())
    def test_sin_scalar(self, val):

        x1 = xnd(val, type="float64")
        y1 = gm.sin(x1)

        if np is not None:
            a1 = np.array(val, dtype="float64")
            b1 = np.sin(a1)

            np.testing.assert_equal(y1.value, b1)


    @given(lst=st.lists(st.floats()))
    def test_sin(self, lst):

        t = "{} * float64".format(len(lst))
        x = xnd(lst, type=t)
        y = gm.sin(x)

        if np is not None:
            a = np.array(lst, dtype="float64")
            b = np.sin(a)
            np.testing.assert_equal(y, b)


    @given(lst=st.lists(st.floats()))
    def test_sin_xnd(self, lst):

        t = "{} * float64".format(len(lst))
        x = xnd(lst, type=t)
        y = gm.xnd_sin0d(x)
        z = gm.xnd_sin1d(x)

        if np is not None:
            a = np.array(lst, dtype="float64")
            b = np.sin(a)
            np.testing.assert_equal(y, b)
            np.testing.assert_equal(z, b)


    @given(lst=st.lists(st.floats()))
    def test_sin_strided(self, lst):
        
        t = "3 * {} * float64".format(len(lst))
        lst_ndim = [lst, lst, lst]
        x = xnd(lst_ndim, type=t)

        y = x[::-2, ::-2]
        z = gm.sin(y)

        if np is not None:
            a = np.array(lst_ndim, dtype="float64")
            b = a[::-2, ::-2]
            c = np.sin(b)
            np.testing.assert_equal(z, c)

    
    @given(lst=st.lists(st.floats()))
    def test_sin_xnd_strided(self, lst):

        lst_2d = [lst, lst]
        t = "2 * {} * float64".format(len(lst))

        x = xnd(lst_2d, type=t)

        y = x[::-2, ::-2]
        z1 = gm.xnd_sin0d(y)
        z2 = gm.xnd_sin1d(y)

        if np is not None:
            a = np.array(lst_2d, dtype="float64")
            b = a[::-2, ::-2]
            c = np.sin(b)
            np.testing.assert_equal(z1, c)
            np.testing.assert_equal(z2, c)


    @given(lst=st.lists(st.floats()))
    def test_copy(self, lst):

        t = "{} * float64".format(len(lst))
        x = xnd(lst, type=t)
        y = gm.copy(x)

        if np is not None:
            a = np.array(lst, dtype="float64")
            b = np.copy(a)
            np.testing.assert_equal(y, b)


    @given(lst=st.lists(st.floats()))
    def test_copy_strided(self, lst):

        t = "3 * {} * float64".format(len(lst))
        lst_ndim = [lst, lst, lst]
        x = xnd(lst_ndim, type=t)
 
        y = x[::-2, ::-2]
        z = gm.copy(y)

        if np is not None:
            a = np.array(lst_ndim, dtype="float64")
            b = a[::-2, ::-2]
            c = np.copy(b)
            np.testing.assert_equal(y, b)




ALL_TESTS = [
  TestCall,
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

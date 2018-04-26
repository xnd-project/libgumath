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

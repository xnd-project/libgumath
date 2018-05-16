/*
* BSD 3-Clause License
*
* Copyright (c) 2017-2018, plures
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"


int
gm_xnd_map(const gm_xnd_kernel_t f, xnd_t stack[], const int nargs,
           const int outer_dims, ndt_context_t *ctx)
{
    ALLOCA(xnd_t, next, nargs);
    const ndt_t *t;

    if (outer_dims == 0 || nargs == 0) {
        return f(stack, ctx);
    }

    t = stack[0].type;

    switch (t->tag) {
    case FixedDim: {
        const int64_t shape = t->FixedDim.shape;

        for (int k = 1; k < nargs; k++) {
            const ndt_t *u = stack[k].type;

            if (u->tag != FixedDim || u->FixedDim.shape != shape) {
                ndt_err_format(ctx, NDT_RuntimeError,
                    "type or shape mismatch in outer dimensions");
                return -1;
            }
        }

        for (int64_t i = 0; i < shape; i++) {
            for (int k = 0; k < nargs; k++) {
                next[k] = xnd_fixed_dim_next(&stack[k], i);
            }

            if (gm_xnd_map(f, next, nargs, outer_dims-1, ctx) < 0) {
                return -1;
            }
        }

        return 0;
    }

    case VarDim: {
        ALLOCA(int64_t, start, nargs);
        ALLOCA(int64_t, step, nargs);
        const int64_t shape = ndt_var_indices(&start[0], &step[0], t,
                                              stack[0].index, ctx);
        if (shape < 0) {
            return -1;
        }

        for (int k = 1; k < nargs; k++) {
            const ndt_t *u = stack[k].type;

            if (u->tag != VarDim) {
                ndt_err_format(ctx, NDT_RuntimeError,
                    "type mismatch in outer dimensions");
                return -1;
            }

            int64_t n = ndt_var_indices(&start[k], &step[k], u, stack[k].index, ctx);
            if (n < 0) {
                return -1;
            }

            if (n != shape) {
                ndt_err_format(ctx, NDT_RuntimeError,
                    "shape mismatch in outer dimensions");
                return -1;
            }
        }

        for (int64_t i = 0; i < shape; i++) {
            for (int k = 0; k < nargs; k++) {
                next[k] = xnd_var_dim_next(&stack[k], start[k], step[k], i);
            }

            if (gm_xnd_map(f, next, nargs, outer_dims-1, ctx) < 0) {
                return -1;
            }
        }

        return 0;
    }

    default: 
        ndt_err_format(ctx, NDT_NotImplementedError, "unsupported type");
        return -1;
    }
}

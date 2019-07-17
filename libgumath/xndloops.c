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
#include "overflow.h"


static int _gm_xnd_map(const gm_xnd_kernel_t f, xnd_t stack[], const int nargs,
                       const int outer_dims, ndt_context_t *ctx);

int
array_shape_check(xnd_t *x, const int64_t shape, ndt_context_t *ctx)
{
    const ndt_t *t = x->type;

    if (t->tag != Array) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "type mismatch in outer dimensions");
        return -1;
    }

    if (XND_ARRAY_DATA(x->ptr) == NULL) {
        bool overflow = false;
        const int64_t size = MULi64(shape, t->Array.itemsize, &overflow);
        if (overflow) {
            ndt_err_format(ctx, NDT_ValueError,
                "datasize of flexible array is too large");
            return -1;
        }

        char *data = ndt_aligned_calloc(t->align, size);
        if (data == NULL) {
            ndt_err_format(ctx, NDT_MemoryError, "out of memory");
            return -1;
        }

        XND_ARRAY_SHAPE(x->ptr) = shape;
        XND_ARRAY_DATA(x->ptr) = data;

        return 0;
    }
    else if (XND_ARRAY_SHAPE(x->ptr) != shape) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "shape mismatch in outer dimensions");
        return -1;
    }
    else {
        return 0;
    }
}

static inline bool
any_stored_index(xnd_t stack[], const int nargs)
{
    for (int i = 0; i < nargs; i++) {
        if (stack[i].ptr == NULL) {
            continue;
        }

        const ndt_t *t = stack[i].type;
        if (have_stored_index(t)) {
            return true;
        }
    }

    return false;
}

int
gm_xnd_map(const gm_xnd_kernel_t f, xnd_t stack[], const int nargs,
           const int outer_dims, ndt_context_t *ctx)
{
    if (any_stored_index(stack, nargs)) {
        ALLOCA(xnd_t, next, nargs);

        for (int i = 0; i < nargs; i++) {
            const ndt_t *t = stack[i].type;
            if (have_stored_index(t)) {
                next[i] = apply_stored_indices(&stack[i], ctx);
                if (xnd_err_occurred(&next[i])) {
                    return -1;
                }
            }
            else {
                next[i] = stack[i];
            }
        }

        return _gm_xnd_map(f, next, nargs, outer_dims, ctx);
    }

    return _gm_xnd_map(f, stack, nargs, outer_dims, ctx);
}

static int
_gm_xnd_map(const gm_xnd_kernel_t f, xnd_t stack[], const int nargs,
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

    case Array: {
        const int64_t shape = XND_ARRAY_SHAPE(stack[0].ptr);

        for (int k = 1; k < nargs; k++) {
            if (array_shape_check(&stack[k], shape, ctx) < 0) {
                return -1;
            }
        }

        for (int64_t i = 0; i < shape; i++) {
            for (int k = 0; k < nargs; k++) {
                next[k] = xnd_array_next(&stack[k], i);
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

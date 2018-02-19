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


/*********************************************************************/
/*                        Kernel application                         */
/*********************************************************************/

static bool
all_c_contiguous(xnd_t stack[], int n)
{
    for (int i = 0; i < n; i++) {
        if (!ndt_is_c_contiguous(stack[i].type)) {
            return false;
        }
    }

    return true;
}

static bool
all_f_contiguous(xnd_t stack[], int n)
{
    for (int i = 0; i < n; i++) {
        if (!ndt_is_f_contiguous(stack[i].type)) {
            return false;
        }
    }

    return true;
}

static inline bool
all_ndarray(xnd_t stack[], int n)
{
    for (int i = 0; i < n; i++) {
        if (!ndt_is_ndarray(stack[i].type)) {
            return false;
        }
    }

    return true;
}

static int
as_ndarray(xnd_ndarray_t op[], const xnd_t stack[], int n, ndt_context_t *ctx)
{
    for (int i = 0; i < n; i++) {
        if (xnd_as_ndarray(&op[i], &stack[i], ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

static int
apply_kernel(const gm_kernel_t *f, xnd_t stack[], ndt_context_t *ctx)
{
    xnd_ndarray_t op[NDT_MAX_ARGS];
    int in = f->sig->Function.in;
    int out = f->sig->Function.out;
    int n = in + out;

    if (f->C && all_c_contiguous(stack, n)) {
        if (as_ndarray(op, stack, n, ctx) < 0) {
            return -1;
        }
        f->C(op, ctx);
    }
    else if (f->Fortran && all_f_contiguous(stack, n)) {
        if (as_ndarray(op, stack, n, ctx) < 0) {
            return -1;
        }
        f->Fortran(op, ctx);
    }
    else if (f->Strided && all_ndarray(stack, n)) {
        if (as_ndarray(op, stack, n, ctx) < 0) {
            return -1;
        }
        f->Strided(op, ctx);
    }
    else if (f->Xnd) {
        f->Xnd(stack, ctx);
    }
    else {
        ndt_err_format(ctx, NDT_RuntimeError, "could not find kernel");
        return -1;
    }

    return 0;
}

static int
map_rec(const gm_kernel_t *f, xnd_t stack[], int outer_dims, ndt_context_t *ctx)
{
    xnd_t next[NDT_MAX_ARGS];
    const ndt_t *sig = f->sig;
    const ndt_t *t;
    int i, k;

    assert(sig->tag == Function);

    if (outer_dims == 0) {
        return apply_kernel(f, stack, ctx);
    }

    t = stack[0].type;

    switch (t->tag) {
    case FixedDim: {
        int nargs = sig->Function.shape;
        int64_t shape = t->FixedDim.shape;

        for (i = 0; i < shape; i++) {
            for (k = 0; k < nargs; k++) {
                const ndt_t *u = stack[k].type;

                if (u->tag != FixedDim) {
                    ndt_err_format(ctx, NDT_RuntimeError,
                        "expected fixed dimension");
                    return -1;
                }

                if (u->FixedDim.shape != shape) {
                    ndt_err_format(ctx, NDT_ValueError, "shape mismatch in gufunc");
                    return -1;
                }

                next[k] = stack[k];
                next[k].type = u->FixedDim.type;
                next[k].index = stack[k].index + i * u->Concrete.FixedDim.step;
            }

            if (map_rec(f, next, outer_dims-1, ctx) < 0) {
                return -1;
            }
        }
    }

    default: 
        ndt_err_format(ctx, NDT_NotImplementedError, "unsupported type");
        return -1;
   }

    return 0;
}

/* Select a kernel from a multimethod. */
static inline const gm_kernel_t *
select(ndt_t *out[], int *nout,
       int *outer_dims,
       const gm_func_t *f,
       xnd_t stack[], int sp, int nin,
       ndt_context_t *ctx)
{
    ndt_t *in[NDT_MAX_ARGS];
    int i;

    for (i = 0; i < nin; i++) {
        in[i] = (ndt_t *)stack[sp+i].type;
    }

    for (i = 0; i < f->size; i++) {
        const gm_kernel_t *kernel = &f->kernels[i];
        *nout = ndt_typecheck(out, outer_dims, kernel->sig, in, nin, ctx);
        if (*nout >= 0) {
            return kernel;
        }
    }

    return NULL;
}

static inline int
map(const gm_func_t *f, xnd_t stack[], int sp, int nin, ndt_context_t *ctx)
{
    const gm_kernel_t *kernel;
    ndt_t *out[NDT_MAX_ARGS];
    int nout;
    int outer_dims;
    int i, k;

    kernel = select(out, &nout, &outer_dims, f, stack, sp, nin, ctx);
    if (kernel == NULL) {
        return -1;
    }

    for (i = 0; i < nout; i++) {
        xnd_master_t *x = xnd_empty_from_type(out[i], XND_OWN_EMBEDDED, ctx);
        if (x == NULL) {
            for (k = 0; k < i; k++) {
                ;
            }
            return -1;
        }

        stack[sp+nin+i] = x->master;
    }

    map_rec(kernel, stack, outer_dims, ctx);

    return 0;
}

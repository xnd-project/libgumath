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
        f->C(op);
    }
    else if (f->Fortran && all_f_contiguous(stack, n)) {
        if (as_ndarray(op, stack, n, ctx) < 0) {
            return -1;
        }
        f->Fortran(op);
    }
    else if (f->Strided && all_ndarray(stack, n)) {
        if (as_ndarray(op, stack, n, ctx) < 0) {
            return -1;
        }
        f->Strided(op);
    }
    else if (f->Xnd) {
        f->Xnd(stack);
    }
    else {
        ndt_err_format(ctx, NDT_RuntimeError, "could not find kernel");
        return -1;
    }

    return 0;
}

int
gm_map(const gm_kernel_t *f, xnd_t stack[], int outer_dims, ndt_context_t *ctx)
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

            if (gm_map(f, next, outer_dims-1, ctx) < 0) {
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

/* Look up a multimethod by name and select a kernel. */
const gm_kernel_t *
gm_select(ndt_t *out_types[],
          int *outer_dims,
          const char *name,
          ndt_t *in_types[], int nin,
          ndt_context_t *ctx)
{
    const gm_func_t *f;
    int i;

    f = gm_func_find(name, ctx);
    if (f == NULL) {
        return NULL;
    }

    for (i = 0; i < f->nkernels; i++) {
        const gm_kernel_t *kernel = &f->kernels[i];
        if (ndt_typecheck(out_types, outer_dims, kernel->sig, in_types, nin, ctx) >= 0) {
            return kernel;
        }
    }

    return NULL;
}

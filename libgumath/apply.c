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


/*
 * Flatten an xnd container into a 1D representation for direct elementwise
 * kernel application.
 */
static int
flatten(char *args[NDT_MAX_ARGS],
        int64_t dimensions[NDT_MAX_ARGS],
        int64_t steps[NDT_MAX_ARGS],
        xnd_t stack[], int n,
        ndt_context_t *ctx)
{
    xnd_ndarray_t nd;
    int i;

    for (i=0; i < n; i++) {
        if (xnd_as_ndarray(&nd, &stack[i], ctx) < 0) {
            return -1;
        }
        args[i] = nd.ptr;
        dimensions[i] = nd.nelem;
        steps[i] = nd.itemsize;
    }

    return 0;
}

#if 0
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
        const int nargs = sig->Function.nargs;
        const int64_t shape = t->FixedDim.shape;

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

        return 0;
    }

    default: 
        ndt_err_format(ctx, NDT_NotImplementedError, "unsupported type");
        return -1;
   }
}
#endif

int
gm_apply(const gm_kernel_t *kernel, xnd_t stack[], int outer_dims GM_UNUSED,
         ndt_context_t *ctx)
{
    const int nargs = kernel->set->sig->Function.nargs;

    switch (kernel->tag) {
    case Elementwise: {
        char *args[NDT_MAX_ARGS];
        int64_t dimensions[NDT_MAX_ARGS];
        int64_t steps[NDT_MAX_ARGS];

        if (flatten(args, dimensions, steps, stack, nargs, ctx) < 0) {
            return -1;
        }

        return kernel->set->Elementwise(args, dimensions, steps, NULL);
    }
    default: {
        ndt_err_format(ctx, NDT_NotImplementedError, "apply not implemented");
        return -1;
      }
    }
}

static gm_kernel_t
select_kernel(const ndt_apply_spec_t *spec, const gm_kernel_set_t *set,
              ndt_context_t *ctx)
{
    gm_kernel_t kernel = {Xnd, NULL};

    kernel.set = set;

    switch (spec->tag) {
    case Elementwise:
        if (set->Elementwise != NULL) {
            kernel.tag = Elementwise;
            return kernel;
        }
        goto TryStrided;

    case C:
        if (set->C != NULL) {
            kernel.tag = C;
            return kernel;
        }
        goto TryStrided;

    case Fortran:
        if (set->Fortran != NULL) {
            kernel.tag = Fortran;
            return kernel;
        }
        /* fall through */

    case Strided: TryStrided:
        if (set->Strided != NULL) {
            kernel.tag = Strided;
            return kernel;
        }
        /* fall through */

    case Xnd:
        if (set->Xnd != NULL) {
            kernel.tag = Xnd;
            return kernel;
        }
    }

    kernel.set = NULL;
    ndt_err_format(ctx, NDT_RuntimeError, "could not find specialized kernel");
    return kernel;
}

/* Look up a multimethod by name and select a kernel. */
gm_kernel_t
gm_select(ndt_apply_spec_t *spec,
          const char *name,
          const ndt_t *in_types[], int nin,
          ndt_context_t *ctx)
{
    gm_kernel_t empty_kernel = {Xnd, NULL};
    const gm_func_t *f;
    int i;

    f = gm_tbl_find(name, ctx);
    if (f == NULL) {
        return empty_kernel;
    }

    for (i = 0; i < f->nkernels; i++) {
        const gm_kernel_set_t *set = &f->kernels[i];
        if (ndt_typecheck(spec, set->sig, in_types, nin, ctx) < 0) {
            ndt_err_clear(ctx);
            continue;
        }
        return select_kernel(spec, set, ctx);
    }

    ndt_err_format(ctx, NDT_RuntimeError, "could not find kernel");
    return empty_kernel;
}

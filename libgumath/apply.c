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


/* flags that apply to all arguments */
#define OPT_Z (NDT_EXT_ZERO|NDT_INNER_C)
#define OPT_C (NDT_EXT_C|NDT_INNER_C)
#define OPT_SC (NDT_EXT_STRIDED|NDT_INNER_C)
#define OPT_SF (NDT_EXT_STRIDED|NDT_INNER_F)
#define OPT_SS (NDT_EXT_STRIDED|NDT_INNER_STRIDED)

#define INNER_C (NDT_INNER_C)
#define INNER_F (NDT_INNER_F)
#define INNER_S (NDT_INNER_STRIDED)
#define INNER_X (NDT_INNER_XND)

/* kernel requests */
#define REQ_LOOP_C(flags) ((flags&OPT_C) == OPT_C)
#define REQ_LOOP_Z(flags) ((flags&OPT_C) == OPT_C || (flags&OPT_Z) == OPT_Z)
#define REQ_LOOP_SC(flags) ((flags&OPT_SC) == OPT_SC)

#define REQ_INNER_C(flags) ((flags&INNER_C) == INNER_C)
#define REQ_INNER_F(flags) ((flags&INNER_F) == INNER_F)
#define REQ_INNER_S(flags) ((flags&INNER_S) == INNER_S)
#define REQ_INNER_X(flags) ((flags&INNER_X) == INNER_X)


static int
sum_inner_dimensions(const xnd_t stack[], int nargs, int outer_dims)
{
    int sum = 0, n;
    int i;

    for (i = 0; i < nargs; i++) {
        const ndt_t *t = stack[i].type;
        n = t->ndim - outer_dims;
        sum += n == 0 ? 1 : n;
    }

    return sum;
}

static inline bool
opt_safe(int outer, ndt_context_t *ctx)
{
    if (outer == 0) {
        ndt_err_format(ctx, NDT_RuntimeError,
            "internal error: optimized kernel called with outer_dims==0");
        return false;
    }

    return true;
}

int
gm_apply(const gm_kernel_t *kernel, xnd_t stack[], int outer_dims,
         ndt_context_t *ctx)
{
    const int nargs = (int)kernel->set->sig->Function.nargs;

    switch (kernel->flag) {
    case OPT_C: {
        if (!opt_safe(outer_dims, ctx)) {
            return -1;
        }

        return gm_xnd_map(kernel->set->OptC, stack, nargs, outer_dims-1, ctx);
    }

    case OPT_Z: {
        if (!opt_safe(outer_dims, ctx)) {
            return -1;
        }

        return gm_xnd_map(kernel->set->OptZ, stack, nargs, outer_dims-1, ctx);
    }

    case OPT_SC: {
        if (!opt_safe(outer_dims, ctx)) {
            return -1;
        }

        return gm_xnd_map(kernel->set->OptSC, stack, nargs, outer_dims-1, ctx);
    }

    case INNER_C: {
        return gm_xnd_map(kernel->set->C, stack, nargs, outer_dims, ctx);
    }

    case INNER_F: {
        return gm_xnd_map(kernel->set->Fortran, stack, nargs, outer_dims, ctx);
    }

    case INNER_X: {
        return gm_xnd_map(kernel->set->Xnd, stack, nargs, outer_dims, ctx);
    }

    case INNER_S: {
        const int sum_inner = sum_inner_dimensions(stack, nargs, outer_dims);
        const int dims_size = outer_dims + sum_inner;
        const int steps_size = nargs * outer_dims + sum_inner;
        ALLOCA(char *, args, nargs);
        ALLOCA(intptr_t, dimensions, dims_size);
        ALLOCA(intptr_t, steps, steps_size);

        if (gm_np_convert_xnd(args, nargs,
                              dimensions, dims_size,
                              steps, steps_size,
                              stack, outer_dims, ctx) < 0) {
            return -1;
        }

        return gm_np_map(kernel->set->Strided, args, nargs,
                         dimensions, steps, NULL, outer_dims);
      }
    }

    /* NOT REACHED: tags should be exhaustive. */
    ndt_internal_error("invalid tag");
}

static gm_kernel_t
select_kernel(const ndt_apply_spec_t *spec, const gm_kernel_set_t *set,
              ndt_context_t *ctx)
{
    gm_kernel_t kernel = {0U, NULL};

    kernel.set = set;

    if (REQ_LOOP_C(spec->flags) && set->OptC != NULL) {
        kernel.flag = OPT_C;
        return kernel;
    }

    if (REQ_LOOP_Z(spec->flags) && set->OptZ != NULL) {
        kernel.flag = OPT_Z;
        return kernel;
    }

    if (REQ_LOOP_SC(spec->flags) && set->OptSC != NULL) {
        kernel.flag = OPT_SC;
        return kernel;
    }

    if (REQ_INNER_C(spec->flags) && set->C != NULL) {
        kernel.flag = INNER_C;
        return kernel;
    }

    if (REQ_INNER_F(spec->flags) && set->Fortran != NULL) {
        kernel.flag = INNER_F;
        return kernel;
    }

    if (REQ_INNER_S(spec->flags) && set->Strided != NULL) {
        kernel.flag = INNER_S;
        return kernel;
    }

    if (REQ_INNER_X(spec->flags) && set->Xnd != NULL) {
        kernel.flag = INNER_X;
        return kernel;
    }

    kernel.set = NULL;
    ndt_err_format(ctx, NDT_RuntimeError,
        "could not find specialized kernel for '%s' input (available: %s, %s, %s, %s, %s, %s, %s)",
        ndt_apply_flags_as_string(spec),
        set->OptC ? "OptC" : "_",
        set->OptZ ? "OptZ" : "_",
        set->OptSC ? "OptSC" : "_",
        set->C ? "C" : "_",
        set->Fortran ? "Fortran" : "_",
        set->Xnd ? "Xnd" : "_",
        set->Strided ? "Strided" : "_");

    return kernel;
}

/* Look up a multimethod by name and select a kernel. */
gm_kernel_t
gm_select(ndt_apply_spec_t *spec, const gm_tbl_t *tbl, const char *name,
          const ndt_t *types[], const int64_t li[], int nin, int nout,
          bool check_broadcast, const xnd_t args[], ndt_context_t *ctx)
{
    gm_kernel_t empty_kernel = {0U, NULL};
    const gm_func_t *f;
    char *s;
    int i;

    f = gm_tbl_find(tbl, name, ctx);
    if (f == NULL) {
        return empty_kernel;
    }

    if (f->typecheck != NULL) {
        const gm_kernel_set_t *set = f->typecheck(spec, f, types, li, nin, nout,
                                                  check_broadcast, ctx);
        if (set == NULL) {
            return empty_kernel;
        }
        return select_kernel(spec, set, ctx);
    }

    for (i = 0; i < f->nkernels; i++) {
        const gm_kernel_set_t *set = &f->kernels[i];
        if (ndt_typecheck(spec, set->sig, types, li, nin, nout,
                          check_broadcast, set->constraint, args,
                          ctx) < 0) {
            ndt_err_clear(ctx);
            continue;
        }
        return select_kernel(spec, set, ctx);
    }

    s = ndt_list_as_string(types, nin, ctx);
    if (s == NULL) {
        return empty_kernel;
    }

    ndt_err_format(ctx, NDT_TypeError,
        "could not find '%s' kernel for input types '%s'", name, s);
    ndt_free(s);

    return empty_kernel;
}

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
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdbool.h>
#include <complex.h>
#include <assert.h>
#include "ndtypes.h"
#include "gumath.h"


/******************************************************************************/
/*                         Type allocation/deallocation                       */
/******************************************************************************/

gm_func_t *
gm_func_new(const char *name, ndt_context_t *ctx)
{
    gm_func_t *f;

    f = ndt_alloc_size(sizeof *f);
    if (f == NULL) {
        return ndt_memory_error(ctx);
    }

    f->name = ndt_strdup(name, ctx);
    if (f->name == NULL) {
        ndt_free(f);
        return NULL;
    }
    f->typecheck = NULL;
    f->nkernels = 0;

    return f;
}

void
gm_func_del(gm_func_t *f)
{
    ndt_free(f->name);

    for (int i = 0; i < f->nkernels; i++) {
        ndt_decref(f->kernels[i].sig);
    }

    ndt_free(f);
}

gm_func_t *
gm_add_func(gm_tbl_t *tbl, const char *name, ndt_context_t *ctx)
{
    gm_func_t *f = gm_func_new(name, ctx);

    if (f == NULL) {
        return NULL;
    }

    if (gm_tbl_add(tbl, name, f, ctx) < 0) {
        gm_func_del(f);
        return NULL;
    }

    return f;
}

int
gm_add_kernel(gm_tbl_t *tbl, const gm_kernel_init_t *k, ndt_context_t *ctx)
{
    gm_func_t *f = gm_tbl_find(tbl, k->name, ctx);
    gm_kernel_set_t kernel;
    const ndt_t *t;

    if (f == NULL) {
        ndt_err_clear(ctx);
        f = gm_add_func(tbl, k->name, ctx);
        if (f == NULL) {
            return -1;
        }
    }

    t = ndt_from_string_v(k->sig, ctx);
    if (t == NULL) {
        return -1;
    }

    if (f->nkernels == GM_MAX_KERNELS) {
        ndt_decref(t);
        ndt_err_format(ctx, NDT_RuntimeError,
            "%s: maximum number of kernels reached for", f->name);
        return -1;
    }

    kernel.sig = t;
    kernel.constraint = k->constraint;
    kernel.OptC = k->OptC;
    kernel.OptZ = k->OptZ;
    kernel.OptSC = k->OptSC;
    kernel.C = k->C;
    kernel.Fortran = k->Fortran;
    kernel.Xnd = k->Xnd;
    kernel.Strided = k->Strided;

    f->kernels[f->nkernels++] = kernel;
    return 0;
}

int
gm_add_kernel_typecheck(gm_tbl_t *tbl, const gm_kernel_init_t *k, ndt_context_t *ctx,
                        gm_typecheck_t typecheck)
{
    gm_func_t *f = gm_tbl_find(tbl, k->name, ctx);
    gm_kernel_set_t kernel;
    const ndt_t *t;

    if (f == NULL) {
        ndt_err_clear(ctx);
        f = gm_add_func(tbl, k->name, ctx);
        if (f == NULL) {
            return -1;
        }
        f->typecheck = typecheck;
    }

    t = ndt_from_string_v(k->sig, ctx);
    if (t == NULL) {
        return -1;
    }

    if (f->nkernels == GM_MAX_KERNELS) {
        ndt_decref(t);
        ndt_err_format(ctx, NDT_RuntimeError,
            "%s: maximum number of kernels reached for", f->name);
        return -1;
    }

    kernel.sig = t;
    kernel.constraint = k->constraint;
    kernel.OptC = k->OptC;
    kernel.OptZ = k->OptZ;
    kernel.OptSC = k->OptSC;
    kernel.C = k->C;
    kernel.Fortran = k->Fortran;
    kernel.Xnd = k->Xnd;
    kernel.Strided = k->Strided;

    f->kernels[f->nkernels++] = kernel;
    return 0;
}

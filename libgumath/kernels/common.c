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
#include <math.h>
#include <complex.h>
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"
#include "common.h"


/****************************************************************************/
/*                           Unary bitmap kernels                           */
/****************************************************************************/

void
unary_update_bitmap1D(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t liout = stack[1].index;
    const uint8_t *b0 = get_bitmap1D(&stack[0]);
    uint8_t *bout = get_bitmap1D(&stack[1]);
    int64_t N = xnd_fixed_shape(&stack[0]);
    int64_t i;

    assert(b0 != NULL);
    assert(bout != NULL);

    for (i = 0; i < N; i++) {
        if (is_valid(b0, li0+i)) {
            set_valid(bout, liout+i);
        }
    }
}

void
unary_update_bitmap(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t liout = stack[1].index;
    const uint8_t *b0 = get_bitmap(&stack[0]);
    uint8_t *bout = get_bitmap(&stack[1]);

    assert(b0 != NULL);
    assert(bout != NULL);

    if (is_valid(b0, li0)) {
        set_valid(bout, liout);
    }
}


/****************************************************************************/
/*                           Binary bitmap kernels                          */
/****************************************************************************/

void
binary_update_bitmap1D(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t liout = stack[2].index;
    const uint8_t *b0 = get_bitmap1D(&stack[0]);
    const uint8_t *b1 = get_bitmap1D(&stack[1]);
    uint8_t *bout = get_bitmap1D(&stack[2]);
    int64_t N = xnd_fixed_shape(&stack[0]);
    int64_t i;

    if (b0 && b1) {
        for (i = 0; i < N; i++) {
            if (is_valid(b0, li0+i) && is_valid(b1, li1+i)) {
                set_valid(bout, liout+i);
            }
        }
    }
    else if (b0) {
        for (i = 0; i < N; i++) {
            if (is_valid(b0, li0+i)) {
                set_valid(bout, liout+i);
            }
        }
    }
    else if (b1) {
        for (i = 0; i < N; i++) {
            if (is_valid(b1, li1+i)) {
                set_valid(bout, liout+i);
            }
        }
    }
}

void
binary_update_bitmap(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t liout = stack[2].index;
    const uint8_t *b0 = get_bitmap(&stack[0]);
    const uint8_t *b1 = get_bitmap(&stack[1]);
    uint8_t *bout = get_bitmap(&stack[2]);

    assert(bout != NULL);

    if (b0 && b1) {
        if (is_valid(b0, li0) && is_valid(b1, li1)) {
            set_valid(bout, liout);
        }
    }
    else if (b0) {
        if (is_valid(b0, li0)) {
            set_valid(bout, liout);
        }
    }
    else if (b1) {
        if (is_valid(b1, li1)) {
            set_valid(bout, liout);
        }
    }
}


/****************************************************************************/
/*                        Optimized unary typecheck                        */
/****************************************************************************/

const gm_kernel_set_t *
cpu_unary_typecheck(int (*kernel_location)(const ndt_t *, ndt_context_t *),
                    ndt_apply_spec_t *spec, const gm_func_t *f,
                    const ndt_t *types[], const int64_t li[], int nin, int nout,
                    ndt_context_t *ctx)
{
    const gm_kernel_set_t *set;
    const ndt_t *t;
    const ndt_t *dtype;
    int n;

    if (nin != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x): expected 1, got %d",
            f->name, nin);
        return NULL;

    }

    if (nout && nout != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x, out=y): expected 1, got %d",
            f->name, nout);
        return NULL;
    }

    t = types[0];
    assert(ndt_is_concrete(t));

    n = kernel_location(t, ctx);
    if (n < 0) {
        return NULL;
    }
    if (ndt_is_optional(ndt_dtype(t))) {
        n++;
    }

    switch (t->tag) {
    case FixedDim:
        spec->flags = NDT_C|NDT_STRIDED;
        spec->outer_dims = t->ndim;
        if (ndt_is_c_contiguous(ndt_logical_dim_at(t, t->ndim-1))) {
            spec->flags |= NDT_ELEMWISE_1D;
        }
        break;
    case VarDim: case VarDimElem:
        spec->flags = NDT_C;
        spec->outer_dims = ndt_logical_ndim(t);
        n += 2;
        break;
    default:
        assert(t->ndim == 0);
        spec->flags = NDT_C|NDT_STRIDED;
        spec->outer_dims = 0;
        break;
    }

    set = &f->kernels[n];

    dtype = ndt_dtype(set->sig->Function.types[1]);
    dtype = ndt_copy_contiguous_dtype(t, dtype, li[0], ctx);
    if (dtype == NULL) {
        ndt_apply_spec_clear(spec);
        return NULL;
    }

    if (nout) {
        if (!ndt_equal(types[1], dtype)) {
            ndt_err_format(ctx, NDT_ValueError, "invalid type for 'out' argument");
            ndt_apply_spec_clear(spec);
            return NULL;
        }
        ndt_decref(dtype);
    }
    else {
        spec->types[1] = dtype;
    }

    ndt_incref(types[0]);
    spec->types[0] = types[0];
    spec->nin = 1;
    spec->nout = 1;
    spec->nargs = 2;

    return set;
}

const gm_kernel_set_t *
cuda_unary_typecheck(int (*kernel_location)(const ndt_t *, ndt_context_t *),
                     ndt_apply_spec_t *spec, const gm_func_t *f,
                     const ndt_t *types[], const int64_t li[], int nin, int nout,
                     ndt_context_t *ctx)
{
    const gm_kernel_set_t *set;
    const ndt_t *t;
    const ndt_t *dtype;
    int n;

    if (nin != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x): expected 1, got %d",
            f->name, nin);
        return NULL;
    }

    if (nout && nout != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x, out=y): expected 1, got %d",
            f->name, nout);
        return NULL;
    }

    t = types[0];
    assert(ndt_is_concrete(t));

    n = kernel_location(t, ctx);
    if (n < 0) {
        return NULL;
    }
    if (ndt_is_optional(ndt_dtype(t))) {
        n++;
    }

    if (!ndt_is_c_contiguous(ndt_logical_dim_at(t, t->ndim-1))) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "cuda %s kernel: input must be contiguous in the last dimension",
            f->name);
        return NULL;
    }
    spec->flags = NDT_C|NDT_ELEMWISE_1D;
    spec->outer_dims = t->ndim;

    set = &f->kernels[n];

    dtype = ndt_dtype(set->sig->Function.types[1]);
    dtype = ndt_copy_contiguous_dtype(t, dtype, li[0], ctx);
    if (dtype == NULL) {
        ndt_apply_spec_clear(spec);
        return NULL;
    }

    if (nout) {
        if (!ndt_equal(types[1], dtype)) {
            ndt_err_format(ctx, NDT_ValueError, "invalid type for 'out' argument");
            ndt_apply_spec_clear(spec);
            return NULL;
        }
        ndt_decref(dtype);
    }
    else {
        spec->types[1] = dtype;
    }

    ndt_incref(types[0]);
    spec->types[0] = types[0];
    spec->nin = 1;
    spec->nout = 1;
    spec->nargs = 2;

    return set;
}


/****************************************************************************/
/*                        Optimized binary typecheck                        */
/****************************************************************************/

const gm_kernel_set_t *
cpu_binary_typecheck(int (* kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                     ndt_apply_spec_t *spec, const gm_func_t *f,
                     const ndt_t *types[], const int64_t li[], int nin, int nout,
                     ndt_context_t *ctx)
{
    const ndt_t *t0;
    const ndt_t *t1;
    int n;

    assert(spec->flags == 0);
    assert(spec->outer_dims == 0);
    assert(spec->nin == 0);
    assert(spec->nout == 0);
    assert(spec->nargs == 0);

    if (nin != 2) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x, y): expected 2, got %d",
            f->name, nin);
        return NULL;
    }

    t0 = types[0];
    t1 = types[1];
    assert(ndt_is_concrete(t0));
    assert(ndt_is_concrete(t1));

    n = kernel_location(t0, t1, ctx);
    if (n < 0) {
        return NULL;
    }
    if (ndt_is_optional(ndt_dtype(t0))) {
        n = ndt_is_optional(ndt_dtype(t1)) ? n+3 : n+1;
    }
    else if (ndt_is_optional(ndt_dtype(t1))) {
        n = n+2;
    }

    if (t0->tag == VarDim || t0->tag == VarDimElem ||
        t1->tag == VarDim || t1->tag == VarDimElem) {
        const gm_kernel_set_t *set = &f->kernels[n+4];
        if (ndt_typecheck(spec, set->sig, types, li, nin, nout, NULL, NULL,
                          ctx) < 0) {
            return NULL;
        }
        return set;
    }

    const gm_kernel_set_t *set = &f->kernels[n];

    if (ndt_fast_binary_fixed_typecheck(spec, set->sig, types, nin, nout, ctx) < 0) {
        return NULL;
    }

    return set;
}

const gm_kernel_set_t *
cuda_binary_typecheck(int (* kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                      ndt_apply_spec_t *spec, const gm_func_t *f,
                      const ndt_t *in[], const int64_t li[], int nin, int nout,
                      ndt_context_t *ctx)
{
    const ndt_t *t0;
    const ndt_t *t1;
    int n;
    (void)li;

    if (nin != 2) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x, y): expected 2, got %d",
            f->name, nin);
        return NULL;
    }
    t0 = in[0];
    t1 = in[1];
    assert(ndt_is_concrete(t0));
    assert(ndt_is_concrete(t1));

    n = kernel_location(t0, t1, ctx);
    if (n < 0) {
        return NULL;
    }
    if (ndt_is_optional(ndt_dtype(t0))) {
        n = ndt_is_optional(ndt_dtype(t1)) ? n+3 : n+1;
    }
    else if (ndt_is_optional(ndt_dtype(t1))) {
        n = n+2;
    }

    if (!ndt_is_c_contiguous(ndt_logical_dim_at(t0, t0->ndim-1))) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "cuda %s kernel: input must be contiguous in the last dimension",
            f->name);
        return NULL;
    }

    if (!ndt_is_c_contiguous(ndt_logical_dim_at(t1, t1->ndim-1))) {
        ndt_err_format(ctx, NDT_NotImplementedError,
            "cuda %s kernel: input must be contiguous in the last dimension",
            f->name);
        return NULL;
    }

    const gm_kernel_set_t *set = &f->kernels[n];

    if (ndt_fast_binary_fixed_typecheck(spec, set->sig, in, nin, nout, ctx) < 0) {
        return NULL;
    }

    return set;
}

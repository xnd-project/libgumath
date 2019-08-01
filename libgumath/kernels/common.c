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
unary_update_bitmap_1D_S(xnd_t stack[])
{
    const int64_t N = xnd_fixed_shape(&stack[0]);
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t s0 = xnd_fixed_step(&stack[0]);
    const int64_t s1 = xnd_fixed_step(&stack[1]);
    const uint8_t *b0 = get_bitmap1D(&stack[0]);
    uint8_t *b1 = get_bitmap1D(&stack[1]);
    int64_t i, k0, k1;

    assert(b0 != NULL);
    assert(b1 != NULL);

    for (i=0, k0=li0, k1=li1; i<N; i++, k0+=s0, k1+=s1) {
        bool x = is_valid(b0, k0);
        set_bit(b1, k1, x);
    }
}

void
unary_reduce_bitmap_1D_S(xnd_t stack[])
{
    const int64_t N = xnd_fixed_shape(&stack[0]);
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t s0 = xnd_fixed_step(&stack[0]);
    const uint8_t *b0 = get_bitmap1D(&stack[0]);
    uint8_t *b1 = get_bitmap(&stack[1]);
    int64_t i, k0;

    assert(b0 != NULL);
    assert(b1 != NULL);

    for (i=0, k0=li0; i<N; i++, k0+=s0) {
        bool x = is_valid(b0, k0) && is_valid(b1, li1);
        set_bit(b1, li1, x);
    }
}

void
unary_update_bitmap_0D(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const uint8_t *b0 = get_bitmap(&stack[0]);
    uint8_t *b1 = get_bitmap(&stack[1]);

    assert(b0 != NULL);
    assert(b1 != NULL);

    bool x = is_valid(b0, li0);
    set_bit(b1, li1, x);
}


/****************************************************************************/
/*                           Binary bitmap kernels                          */
/****************************************************************************/

void
binary_update_bitmap_1D_S(xnd_t stack[])
{
    const int64_t N = xnd_fixed_shape(&stack[0]);
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t li2 = stack[2].index;
    const int64_t s0 = xnd_fixed_step(&stack[0]);
    const int64_t s1 = xnd_fixed_step(&stack[1]);
    const int64_t s2 = xnd_fixed_step(&stack[2]);
    const uint8_t *b0 = get_bitmap1D(&stack[0]);
    const uint8_t *b1 = get_bitmap1D(&stack[1]);
    uint8_t *b2 = get_bitmap1D(&stack[2]);
    int64_t i, k0, k1, k2;

    if (b0 && b1) {
        for (i=0, k0=li0, k1=li1, k2=li2; i<N; i++, k0+=s0, k1+=s1, k2+=s2) {
            bool x = is_valid(b0, k0) && is_valid(b1, k1);
            set_bit(b2, k2, x);
        }
    }
    else if (b0) {
        for (i=0, k0=li0, k2=li2; i<N; i++, k0+=s0, k2+=s2) {
            bool x = is_valid(b0, k0);
            set_bit(b2, k2, x);
        }
    }
    else if (b1) {
        for (i=0, k1=li1, k2=li2; i<N; i++, k1+=s1, k2+=s2) {
            bool x = is_valid(b1, k1);
            set_bit(b2, k2, x);
        }
    }
}

void
binary_update_bitmap_0D(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t li2 = stack[2].index;
    const uint8_t *b0 = get_bitmap(&stack[0]);
    const uint8_t *b1 = get_bitmap(&stack[1]);
    uint8_t *b2 = get_bitmap(&stack[2]);

    assert(b2 != NULL);

    if (b0 && b1) {
        bool x = is_valid(b0, li0) && is_valid(b1, li1);
        set_bit(b2, li2, x);
    }
    else if (b0) {
        bool x = is_valid(b0, li0);
        set_bit(b2, li2, x);
    }
    else if (b1) {
        bool x = is_valid(b1, li1);
        set_bit(b2, li2, x);
    }
}

void
binary_update_bitmap_1D_S_bool(xnd_t stack[])
{
    const int64_t N = xnd_fixed_shape(&stack[0]);
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t li2 = stack[2].index;
    const int64_t s0 = xnd_fixed_step(&stack[0]);
    const int64_t s1 = xnd_fixed_step(&stack[1]);
    const int64_t s2 = xnd_fixed_step(&stack[2]);
    const uint8_t *b0 = get_bitmap1D(&stack[0]);
    const uint8_t *b1 = get_bitmap1D(&stack[1]);
    bool *x2 = (bool *)apply_index(&stack[2]);
    int64_t i, k0, k1, k2;

    assert(!ndt_is_optional(stack[2].type));

    if (b0 && b1) {
        for (i=0, k0=li0, k1=li1, k2=li2; i<N; i++, k0+=s0, k1+=s1, k2+=s2) {
            bool x = is_valid(b0, k0);
            bool y = is_valid(b1, k1);
            bool z = x2[k2];
            z = x && y ? z : !x && !y;
            x2[k2] = z;
        }
    }
    else if (b0) {
        for (i=0, k0=li0, k2=li2; i<N; i++, k0+=s0, k2+=s2) {
            bool x = is_valid(b0, k0);
            bool z = x2[k2];
            z = x ? z : x;
            x2[k2] = z;
        }
    }
    else if (b1) {
        for (i=0, k1=li1, k2=li2; i<N; i++, k1+=s1, k2+=s2) {
            bool x = is_valid(b1, k1);
            bool z = x2[k2];
            z = x ? z : x;
            x2[k2] = z;
        }
    }
}

void
binary_update_bitmap_0D_bool(xnd_t stack[])
{
    const int64_t li0 = stack[0].index;
    const int64_t li1 = stack[1].index;
    const int64_t li2 = stack[2].index;
    const uint8_t *b0 = get_bitmap(&stack[0]);
    const uint8_t *b1 = get_bitmap(&stack[1]);
    bool *x2 = (bool *)stack[2].ptr;

    assert(!ndt_is_optional(stack[2].type));

    if (b0 && b1) {
        bool x = is_valid(b0, li0);
        bool y = is_valid(b1, li1);
        bool z = x2[li2];
        z = x && y ? z : !x && !y;
        x2[li2] = z;
    }
    else if (b0) {
        bool x = is_valid(b0, li0);
        bool z = x2[li2];
        z = x ? z : x;
        x2[li2] = z;
    }
    else if (b1) {
        bool x = is_valid(b1, li1);
        bool z = x2[li2];
        z = x ? z : x;
        x2[li2] = z;
    }
}


/****************************************************************************/
/*                        Optimized unary typecheck                        */
/****************************************************************************/

const gm_kernel_set_t *
cpu_unary_typecheck(int (*kernel_location)(const ndt_t *, const ndt_t *, ndt_context_t *),
                    ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                    const int64_t li[], int nin, int nout, bool check_broadcast,
                    ndt_context_t *ctx)
{
    const ndt_t *t;
    const ndt_t *u;
    int n;

    assert(spec->flags == 0);
    assert(spec->outer_dims == 0);
    assert(spec->nin == 0);
    assert(spec->nout == 0);
    assert(spec->nargs == 0);

    if (nin != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x): expected 1, got %d",
            f->name, nin);
        return NULL;

    }

    t = types[0];

    if (nout) {
        if (nout != 1) {
            ndt_err_format(ctx, NDT_ValueError,
                "%s(x) expects at most one 'out' argument, got %d",
                f->name, nout);
            return NULL;
        }
        u = types[1];
    }
    else {
        u = types[0];
    }

    assert(ndt_is_concrete(t));
    assert(ndt_is_concrete(u));

    n = kernel_location(t, u, ctx);
    if (n < 0) {
        return NULL;
    }
    if (ndt_is_optional(ndt_dtype(t))) {
        n++;
    }

    if (t->tag == VarDim || t->tag == VarDimElem) {
        const gm_kernel_set_t *set = &f->kernels[n+2];
        if (ndt_typecheck(spec, set->sig, types, li, nin, nout,
                          check_broadcast, NULL, NULL, ctx) < 0) {
            return NULL;
        }
        return set;
    }

    if (t->tag == Array) {
        const gm_kernel_set_t *set = &f->kernels[n+4];
        if (ndt_typecheck(spec, set->sig, types, li, nin, nout,
                          check_broadcast, NULL, NULL, ctx) < 0) {
            return NULL;
        }
        return set;
    }

    const gm_kernel_set_t *set = &f->kernels[n];

    if (ndt_fast_unary_fixed_typecheck(spec, set->sig, types, nin, nout,
                                       check_broadcast, ctx) < 0) {
        return NULL;
    }

    return set;
}

const gm_kernel_set_t *
cuda_unary_typecheck(int (*kernel_location)(const ndt_t *, const ndt_t *, ndt_context_t *),
                     ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                     const int64_t li[], int nin, int nout, bool check_broadcast,
                     ndt_context_t *ctx)
{
    const ndt_t *t;
    const ndt_t *u;
    int n;
    (void)li;

    assert(spec->flags == 0);
    assert(spec->outer_dims == 0);
    assert(spec->nin == 0);
    assert(spec->nout == 0);
    assert(spec->nargs == 0);

    if (nin != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x): expected 1, got %d",
            f->name, nin);
        return NULL;
    }

    t = types[0];

    if (nout) {
        if (nout != 1) {
            ndt_err_format(ctx, NDT_ValueError,
                "%s(x) expects at most one 'out' argument, got %d",
                f->name, nout);
            return NULL;
        }
        u = types[1];
    }
    else {
        u = types[0];
    }

    assert(ndt_is_concrete(t));
    assert(ndt_is_concrete(u));

    n = kernel_location(t, u, ctx);
    if (n < 0) {
        return NULL;
    }
    if (ndt_is_optional(ndt_dtype(t))) {
        n++;
    }

    const gm_kernel_set_t *set = &f->kernels[n];

    if (ndt_fast_unary_fixed_typecheck(spec, set->sig, types, nin, nout,
                                       check_broadcast, ctx) < 0) {
        return NULL;
    }

    return set;
}


/****************************************************************************/
/*                        Optimized binary typecheck                        */
/****************************************************************************/

const gm_kernel_set_t *
cpu_binary_typecheck(int (* kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                     ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                     const int64_t li[], int nin, int nout, bool check_broadcast,
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
        if (ndt_typecheck(spec, set->sig, types, li, nin, nout,
                          check_broadcast, NULL, NULL, ctx) < 0) {
            return NULL;
        }
        return set;
    }

    if (t0->tag == Array || t1->tag == Array) {
        const gm_kernel_set_t *set = &f->kernels[n+8];
        if (ndt_typecheck(spec, set->sig, types, li, nin, nout,
                          check_broadcast, NULL, NULL, ctx) < 0) {
            return NULL;
        }
        return set;
    }

    const gm_kernel_set_t *set = &f->kernels[n];

    if (ndt_fast_binary_fixed_typecheck(spec, set->sig, types, nin, nout,
                                        check_broadcast, ctx) < 0) {
        return NULL;
    }

    return set;
}

const gm_kernel_set_t *
cuda_binary_typecheck(int (* kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                      ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                      const int64_t li[], int nin, int nout, bool check_broadcast,
                      ndt_context_t *ctx)
{
    const ndt_t *t0;
    const ndt_t *t1;
    int n;
    (void)li;

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

    const gm_kernel_set_t *set = &f->kernels[n];

    if (ndt_fast_binary_fixed_typecheck(spec, set->sig, types, nin, nout,
                                        check_broadcast, ctx) < 0) {
        return NULL;
    }

    return set;
}

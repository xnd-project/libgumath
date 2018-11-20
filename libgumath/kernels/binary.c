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


/****************************************************************************/
/*                     Optimized dispatch (exact casting)                   */
/****************************************************************************/

/* Structured kernel locations for fast lookup. */
static int
kernel_location(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx)
{
    const ndt_t *t0 = ndt_dtype(in0);
    const ndt_t *t1 = ndt_dtype(in1);

    switch (t0->tag) {
    case Uint8: {
        switch (t1->tag) {
        case Uint8: return 0;
        case Uint16: return 8;
        case Uint32: return 16;
        case Uint64: return 24;

        case Int8: return 32;
        case Int16: return 40;
        case Int32: return 48;
        case Int64: return 56;

        case Float32: return 64;
        case Float64: return 72;
        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Uint8: return 80;
        case Uint16: return 88;
        case Uint32: return 96;
        case Uint64: return 104;

        case Int8: return 112;
        case Int16: return 120;
        case Int32: return 128;
        case Int64: return 136;

        case Float32: return 144;
        case Float64: return 152;
        default: goto invalid_combination;
        }
    }
    case Uint32: {
        switch (t1->tag) {
        case Uint8: return 160;
        case Uint16: return 168;
        case Uint32: return 176;
        case Uint64: return 184;

        case Int8: return 192;
        case Int16: return 200;
        case Int32: return 208;
        case Int64: return 216;

        case Float32: return 224;
        case Float64: return 232;
        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Uint8: return 240;
        case Uint16: return 248;
        case Uint32: return 256;
        case Uint64: return 264;
        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Uint8: return 272;
        case Uint16: return 280;
        case Uint32: return 288;

        case Int8: return 296;
        case Int16: return 304;
        case Int32: return 312;
        case Int64: return 320;

        case Float32: return 328;
        case Float64: return 336;
        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Uint8: return 344;
        case Uint16: return 352;
        case Uint32: return 360;

        case Int8: return 368;
        case Int16: return 376;
        case Int32: return 384;
        case Int64: return 392;

        case Float32: return 400;
        case Float64: return 408;
        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Uint8: return 416;
        case Uint16: return 424;
        case Uint32: return 432;

        case Int8: return 440;
        case Int16: return 448;
        case Int32: return 456;
        case Int64: return 464;

        case Float32: return 472;
        case Float64: return 480;
        default: goto invalid_combination;
        }
    }
    case Int64: {
        switch (t1->tag) {
        case Uint8: return 488;
        case Uint16: return 496;
        case Uint32: return 504;

        case Int8: return 512;
        case Int16: return 520;
        case Int32: return 528;
        case Int64: return 536;

        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (t1->tag) {
        case Uint8: return 544;
        case Uint16: return 552;
        case Uint32: return 560;

        case Int8: return 568;
        case Int16: return 576;
        case Int32: return 584;

        case Float32: return 592;
        case Float64: return 600;
        default: goto invalid_combination;
        }
    }
    case Float64: {
        switch (t1->tag) {
        case Uint8: return 608;
        case Uint16: return 616;
        case Uint32: return 624;

        case Int8: return 632;
        case Int16: return 640;
        case Int32: return 648;

        case Float32: return 656;
        case Float64: return 664;
        default: goto invalid_combination;
        }
    }

    default:
        goto invalid_combination;
    }

invalid_combination:
    ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
    return -1;
}


/****************************************************************************/
/*                             Optimized typecheck                          */
/****************************************************************************/

static const gm_kernel_set_t *
binary_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                 const ndt_t *in[], int nin,
                 ndt_context_t *ctx)
{
    const ndt_t *t0;
    const ndt_t *t1;
    const ndt_t *dtype;
    int n;

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

    if (t0->tag == VarDim || t1->tag == VarDim) {
        const gm_kernel_set_t *set = &f->kernels[n+4];
        if (ndt_typecheck(spec, set->sig, in, nin, NULL, NULL, ctx) < 0) {
            return NULL;
        }
        return set;
    }

    const gm_kernel_set_t *set = &f->kernels[n];

    dtype = ndt_dtype(set->sig->Function.types[2]);
    if (ndt_fast_binary_fixed_typecheck(spec, set->sig, in, nin, dtype, ctx) < 0) {
        return NULL;
    }

    return set;
}


/****************************************************************************/
/*                           Generated Xnd kernels                          */
/****************************************************************************/

#define XSTRINGIZE(v) #v
#define STRINGIZE(v) XSTRINGIZE(v)

static inline char *
apply_index(const xnd_t *x)
{
    return xnd_fixed_apply_index(x);
}

static inline uint8_t *
get_bitmap(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->ndim == 0);
    return ndt_is_optional(t) ? x->bitmap.data : NULL;
}

static inline uint8_t *
get_bitmap1D(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->ndim == 1 && t->tag == FixedDim);
    return ndt_is_optional(ndt_dtype(t)) ? x->bitmap.data : NULL;
}

static inline int
is_valid(const uint8_t *data, int64_t n)
{
    return data[n / 8] & ((uint8_t)1 << (n % 8));
}

static inline void
set_valid(uint8_t *data, int64_t n)
{
    data[n / 8] |= ((uint8_t)1 << (n % 8));
}

static inline int64_t
linear_index1D(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    const int64_t step = i * t->Concrete.FixedDim.step;
    return x->index + step;
}


static void
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

static void
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

#define XND_BINARY(func, t0, t1, t2, cast) \
static int                                                                     \
gm_fixed_##func##_1D_C_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)   \
{                                                                              \
    const t0##_t *in0 = (const t0##_t *)apply_index(&stack[0]);                \
    const t1##_t *in1 = (const t1##_t *)apply_index(&stack[1]);                \
    t2##_t *out = (t2##_t *)apply_index(&stack[2]);                            \
    int64_t N = xnd_fixed_shape(&stack[0]);                                    \
    (void)ctx;                                                                 \
    int64_t i;                                                                 \
                                                                               \
    for (i = 0; i < N-7; i += 8) {                                             \
        out[i] = func((cast##_t)in0[i], (cast##_t)in1[i]);                     \
        out[i+1] = func((cast##_t)in0[i+1], (cast##_t)in1[i+1]);               \
        out[i+2] = func((cast##_t)in0[i+2], (cast##_t)in1[i+2]);               \
        out[i+3] = func((cast##_t)in0[i+3], (cast##_t)in1[i+3]);               \
        out[i+4] = func((cast##_t)in0[i+4], (cast##_t)in1[i+4]);               \
        out[i+5] = func((cast##_t)in0[i+5], (cast##_t)in1[i+5]);               \
        out[i+6] = func((cast##_t)in0[i+6], (cast##_t)in1[i+6]);               \
        out[i+7] = func((cast##_t)in0[i+7], (cast##_t)in1[i+7]);               \
    }                                                                          \
    for (; i < N; i++) {                                                       \
        out[i] = func((cast##_t)in0[i], (cast##_t)in1[i]);                     \
    }                                                                          \
                                                                               \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                           \
        binary_update_bitmap1D(stack);                                         \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_##func##_0D_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)           \
{                                                                              \
    const xnd_t *in0 = &stack[0];                                              \
    const xnd_t *in1 = &stack[1];                                              \
    xnd_t *out = &stack[2];                                                    \
    (void)ctx;                                                                 \
                                                                               \
    const t0##_t x = *(const t0##_t *)in0->ptr;                                \
    const t1##_t y = *(const t1##_t *)in1->ptr;                                \
    *(t2##_t *)out->ptr = func((cast##_t)x, (cast##_t)y);                      \
                                                                               \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                           \
        binary_update_bitmap(stack);                                           \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}

#define XND_BINARY_INIT(func, t0, t1, t2) \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),             \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1##_##t2,                                              \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
   { .name = STRINGIZE(func),                                                                      \
    .sig = "... * ?" STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),           \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1##_##t2,                                              \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
   { .name = STRINGIZE(func),                                                                      \
    .sig = "... * " STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),           \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1##_##t2,                                              \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
   { .name = STRINGIZE(func),                                                                      \
    .sig = "... * ?" STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),          \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1##_##t2,                                              \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * " STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * " STRINGIZE(t2),    \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),  \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * " STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),  \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 },                                                      \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2), \
    .C = gm_##func##_0D_##t0##_##t1##_##t2 }


/*************
 * Arithmetic
 *************/

#define XND_ALL_BINARY(name) \
    XND_BINARY(name, uint8, uint8, uint8, uint8)         \
    XND_BINARY(name, uint8, uint16, uint16, uint16)      \
    XND_BINARY(name, uint8, uint32, uint32, uint32)      \
    XND_BINARY(name, uint8, uint64, uint64, uint64)      \
    XND_BINARY(name, uint8, int8, int16, int16)          \
    XND_BINARY(name, uint8, int16, int16, int16)         \
    XND_BINARY(name, uint8, int32, int32, int32)         \
    XND_BINARY(name, uint8, int64, int64, int64)         \
    XND_BINARY(name, uint8, float32, float32, float32)   \
    XND_BINARY(name, uint8, float64, float64, float64)   \
                                                         \
    XND_BINARY(name, uint16, uint8, uint16, uint16)      \
    XND_BINARY(name, uint16, uint16, uint16, uint16)     \
    XND_BINARY(name, uint16, uint32, uint32, uint32)     \
    XND_BINARY(name, uint16, uint64, uint64, uint64)     \
    XND_BINARY(name, uint16, int8, int32, int32)         \
    XND_BINARY(name, uint16, int16, int32, int32)        \
    XND_BINARY(name, uint16, int32, int32, int32)        \
    XND_BINARY(name, uint16, int64, int64, int64)        \
    XND_BINARY(name, uint16, float32, float32, float32)  \
    XND_BINARY(name, uint16, float64, float64, float64)  \
                                                         \
    XND_BINARY(name, uint32, uint8, uint32, uint32)      \
    XND_BINARY(name, uint32, uint16, uint32, uint32)     \
    XND_BINARY(name, uint32, uint32, uint32, uint32)     \
    XND_BINARY(name, uint32, uint64, uint64, uint64)     \
    XND_BINARY(name, uint32, int8, int64, int64)         \
    XND_BINARY(name, uint32, int16, int64, int64)        \
    XND_BINARY(name, uint32, int32, int64, int64)        \
    XND_BINARY(name, uint32, int64, int64, int64)        \
    XND_BINARY(name, uint32, float32, float64, float64)  \
    XND_BINARY(name, uint32, float64, float64, float64)  \
                                                         \
    XND_BINARY(name, uint64, uint8, uint64, uint64)      \
    XND_BINARY(name, uint64, uint16, uint64, uint64)     \
    XND_BINARY(name, uint64, uint32, uint64, uint64)     \
    XND_BINARY(name, uint64, uint64, uint64, uint64)     \
                                                         \
    XND_BINARY(name, int8, uint8, int16, int16)          \
    XND_BINARY(name, int8, uint16, int32, int32)         \
    XND_BINARY(name, int8, uint32, int64, int64)         \
    XND_BINARY(name, int8, int8, int8, int8)             \
    XND_BINARY(name, int8, int16, int16, int16)          \
    XND_BINARY(name, int8, int32, int32, int32)          \
    XND_BINARY(name, int8, int64, int64, int64)          \
    XND_BINARY(name, int8, float32, float32, float32)    \
    XND_BINARY(name, int8, float64, float64, float64)    \
                                                         \
    XND_BINARY(name, int16, uint8, int16, int16)         \
    XND_BINARY(name, int16, uint16, int32, int32)        \
    XND_BINARY(name, int16, uint32, int64, int64)        \
    XND_BINARY(name, int16, int8, int16, int16)          \
    XND_BINARY(name, int16, int16, int16, int16)         \
    XND_BINARY(name, int16, int32, int32, int32)         \
    XND_BINARY(name, int16, int64, int64, int64)         \
    XND_BINARY(name, int16, float32, float32, float32)   \
    XND_BINARY(name, int16, float64, float64, float64)   \
                                                         \
    XND_BINARY(name, int32, uint8, int32, int32)         \
    XND_BINARY(name, int32, uint16, int32, int32)        \
    XND_BINARY(name, int32, uint32, int64, int64)        \
    XND_BINARY(name, int32, int8, int32, int32)          \
    XND_BINARY(name, int32, int16, int32, int32)         \
    XND_BINARY(name, int32, int32, int32, int32)         \
    XND_BINARY(name, int32, int64, int64, int64)         \
    XND_BINARY(name, int32, float32, float64, float64)   \
    XND_BINARY(name, int32, float64, float64, float64)   \
                                                         \
    XND_BINARY(name, int64, uint8, int64, int64)         \
    XND_BINARY(name, int64, uint16, int64, int64)        \
    XND_BINARY(name, int64, uint32, int64, int64)        \
    XND_BINARY(name, int64, int8, int64, int64)          \
    XND_BINARY(name, int64, int16, int64, int64)         \
    XND_BINARY(name, int64, int32, int64, int64)         \
    XND_BINARY(name, int64, int64, int64, int64)         \
                                                         \
    XND_BINARY(name, float32, uint8, float32, float32)   \
    XND_BINARY(name, float32, uint16, float32, float32)  \
    XND_BINARY(name, float32, uint32, float64, float64)  \
    XND_BINARY(name, float32, int8, float32, float32)    \
    XND_BINARY(name, float32, int16, float32, float32)   \
    XND_BINARY(name, float32, int32, float64, float64)   \
    XND_BINARY(name, float32, float32, float32, float32) \
    XND_BINARY(name, float32, float64, float64, float64) \
                                                         \
    XND_BINARY(name, float64, uint8, float64, float64)   \
    XND_BINARY(name, float64, uint16, float64, float64)  \
    XND_BINARY(name, float64, uint32, float64, float64)  \
    XND_BINARY(name, float64, int8, float64, float64)    \
    XND_BINARY(name, float64, int16, float64, float64)   \
    XND_BINARY(name, float64, int32, float64, float64)   \
    XND_BINARY(name, float64, float32, float64, float64) \
    XND_BINARY(name, float64, float64, float64, float64)

#define XND_ALL_BINARY_INIT(name) \
    XND_BINARY_INIT(name, uint8, uint8, uint8),       \
    XND_BINARY_INIT(name, uint8, uint16, uint16),     \
    XND_BINARY_INIT(name, uint8, uint32, uint32),     \
    XND_BINARY_INIT(name, uint8, uint64, uint64),     \
    XND_BINARY_INIT(name, uint8, int8, int16),        \
    XND_BINARY_INIT(name, uint8, int16, int16),       \
    XND_BINARY_INIT(name, uint8, int32, int32),       \
    XND_BINARY_INIT(name, uint8, int64, int64),       \
    XND_BINARY_INIT(name, uint8, float32, float32),   \
    XND_BINARY_INIT(name, uint8, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, uint16, uint8, uint16),     \
    XND_BINARY_INIT(name, uint16, uint16, uint16),    \
    XND_BINARY_INIT(name, uint16, uint32, uint32),    \
    XND_BINARY_INIT(name, uint16, uint64, uint64),    \
    XND_BINARY_INIT(name, uint16, int8, int32),       \
    XND_BINARY_INIT(name, uint16, int16, int32),      \
    XND_BINARY_INIT(name, uint16, int32, int32),      \
    XND_BINARY_INIT(name, uint16, int64, int64),      \
    XND_BINARY_INIT(name, uint16, float32, float32),  \
    XND_BINARY_INIT(name, uint16, float64, float64),  \
                                                      \
    XND_BINARY_INIT(name, uint32, uint8, uint32),     \
    XND_BINARY_INIT(name, uint32, uint16, uint32),    \
    XND_BINARY_INIT(name, uint32, uint32, uint32),    \
    XND_BINARY_INIT(name, uint32, uint64, uint64),    \
    XND_BINARY_INIT(name, uint32, int8, int64),       \
    XND_BINARY_INIT(name, uint32, int16, int64),      \
    XND_BINARY_INIT(name, uint32, int32, int64),      \
    XND_BINARY_INIT(name, uint32, int64, int64),      \
    XND_BINARY_INIT(name, uint32, float32, float64),  \
    XND_BINARY_INIT(name, uint32, float64, float64),  \
                                                      \
    XND_BINARY_INIT(name, uint64, uint8, uint64),     \
    XND_BINARY_INIT(name, uint64, uint16, uint64),    \
    XND_BINARY_INIT(name, uint64, uint32, uint64),    \
    XND_BINARY_INIT(name, uint64, uint64, uint64),    \
                                                      \
    XND_BINARY_INIT(name, int8, uint8, int16),        \
    XND_BINARY_INIT(name, int8, uint16, int32),       \
    XND_BINARY_INIT(name, int8, uint32, int64),       \
    XND_BINARY_INIT(name, int8, int8, int8),          \
    XND_BINARY_INIT(name, int8, int16, int16),        \
    XND_BINARY_INIT(name, int8, int32, int32),        \
    XND_BINARY_INIT(name, int8, int64, int64),        \
    XND_BINARY_INIT(name, int8, float32, float32),    \
    XND_BINARY_INIT(name, int8, float64, float64),    \
                                                      \
    XND_BINARY_INIT(name, int16, uint8, int16),       \
    XND_BINARY_INIT(name, int16, uint16, int32),      \
    XND_BINARY_INIT(name, int16, uint32, int64),      \
    XND_BINARY_INIT(name, int16, int8, int16),        \
    XND_BINARY_INIT(name, int16, int16, int16),       \
    XND_BINARY_INIT(name, int16, int32, int32),       \
    XND_BINARY_INIT(name, int16, int64, int64),       \
    XND_BINARY_INIT(name, int16, float32, float32),   \
    XND_BINARY_INIT(name, int16, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, int32, uint8, int32),       \
    XND_BINARY_INIT(name, int32, uint16, int32),      \
    XND_BINARY_INIT(name, int32, uint32, int64),      \
    XND_BINARY_INIT(name, int32, int8, int32),        \
    XND_BINARY_INIT(name, int32, int16, int32),       \
    XND_BINARY_INIT(name, int32, int32, int32),       \
    XND_BINARY_INIT(name, int32, int64, int64),       \
    XND_BINARY_INIT(name, int32, float32, float64),   \
    XND_BINARY_INIT(name, int32, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, int64, uint8, int64),       \
    XND_BINARY_INIT(name, int64, uint16, int64),      \
    XND_BINARY_INIT(name, int64, uint32, int64),      \
    XND_BINARY_INIT(name, int64, int8, int64),        \
    XND_BINARY_INIT(name, int64, int16, int64),       \
    XND_BINARY_INIT(name, int64, int32, int64),       \
    XND_BINARY_INIT(name, int64, int64, int64),       \
                                                      \
    XND_BINARY_INIT(name, float32, uint8, float32),   \
    XND_BINARY_INIT(name, float32, uint16, float32),  \
    XND_BINARY_INIT(name, float32, uint32, float64),  \
    XND_BINARY_INIT(name, float32, int8, float32),    \
    XND_BINARY_INIT(name, float32, int16, float32),   \
    XND_BINARY_INIT(name, float32, int32, float64),   \
    XND_BINARY_INIT(name, float32, float32, float32), \
    XND_BINARY_INIT(name, float32, float64, float64), \
                                                      \
    XND_BINARY_INIT(name, float64, uint8, float64),   \
    XND_BINARY_INIT(name, float64, uint16, float64),  \
    XND_BINARY_INIT(name, float64, uint32, float64),  \
    XND_BINARY_INIT(name, float64, int8, float64),    \
    XND_BINARY_INIT(name, float64, int16, float64),   \
    XND_BINARY_INIT(name, float64, int32, float64),   \
    XND_BINARY_INIT(name, float64, float32, float64), \
    XND_BINARY_INIT(name, float64, float64, float64)


#define add(x, y) x + y
XND_ALL_BINARY(add)

#define subtract(x, y) x - y
XND_ALL_BINARY(subtract)

#define multiply(x, y) x * y
XND_ALL_BINARY(multiply)

#define divide(x, y) x / y
XND_ALL_BINARY(divide)


/*************
 * Comparison
 *************/

#define XND_ALL_COMPARISON(name) \
    XND_BINARY(name, uint8, uint8, bool, uint8)       \
    XND_BINARY(name, uint8, uint16, bool, uint16)     \
    XND_BINARY(name, uint8, uint32, bool, uint32)     \
    XND_BINARY(name, uint8, uint64, bool, uint64)     \
    XND_BINARY(name, uint8, int8, bool, int16)        \
    XND_BINARY(name, uint8, int16, bool, int16)       \
    XND_BINARY(name, uint8, int32, bool, int32)       \
    XND_BINARY(name, uint8, int64, bool, int64)       \
    XND_BINARY(name, uint8, float32, bool, float32)   \
    XND_BINARY(name, uint8, float64, bool, float64)   \
                                                      \
    XND_BINARY(name, uint16, uint8, bool, uint16)     \
    XND_BINARY(name, uint16, uint16, bool, uint16)    \
    XND_BINARY(name, uint16, uint32, bool, uint32)    \
    XND_BINARY(name, uint16, uint64, bool, uint64)    \
    XND_BINARY(name, uint16, int8, bool, int32)       \
    XND_BINARY(name, uint16, int16, bool, int32)      \
    XND_BINARY(name, uint16, int32, bool, int32)      \
    XND_BINARY(name, uint16, int64, bool, int64)      \
    XND_BINARY(name, uint16, float32, bool, float32)  \
    XND_BINARY(name, uint16, float64, bool, float64)  \
                                                      \
    XND_BINARY(name, uint32, uint8, bool, uint32)     \
    XND_BINARY(name, uint32, uint16, bool, uint32)    \
    XND_BINARY(name, uint32, uint32, bool, uint32)    \
    XND_BINARY(name, uint32, uint64, bool, uint64)    \
    XND_BINARY(name, uint32, int8, bool, int64)       \
    XND_BINARY(name, uint32, int16, bool, int64)      \
    XND_BINARY(name, uint32, int32, bool, int64)      \
    XND_BINARY(name, uint32, int64, bool, int64)      \
    XND_BINARY(name, uint32, float32, bool, float64)  \
    XND_BINARY(name, uint32, float64, bool, float64)  \
                                                      \
    XND_BINARY(name, uint64, uint8, bool, uint64)     \
    XND_BINARY(name, uint64, uint16, bool, uint64)    \
    XND_BINARY(name, uint64, uint32, bool, uint64)    \
    XND_BINARY(name, uint64, uint64, bool, uint64)    \
                                                      \
    XND_BINARY(name, int8, uint8, bool, int16)        \
    XND_BINARY(name, int8, uint16, bool, int32)       \
    XND_BINARY(name, int8, uint32, bool, int64)       \
    XND_BINARY(name, int8, int8, bool, int8)          \
    XND_BINARY(name, int8, int16, bool, int16)        \
    XND_BINARY(name, int8, int32, bool, int32)        \
    XND_BINARY(name, int8, int64, bool, int64)        \
    XND_BINARY(name, int8, float32, bool, float32)    \
    XND_BINARY(name, int8, float64, bool, float64)    \
                                                      \
    XND_BINARY(name, int16, uint8, bool, int16)       \
    XND_BINARY(name, int16, uint16, bool, int32)      \
    XND_BINARY(name, int16, uint32, bool, int64)      \
    XND_BINARY(name, int16, int8, bool, int16)        \
    XND_BINARY(name, int16, int16, bool, int16)       \
    XND_BINARY(name, int16, int32, bool, int32)       \
    XND_BINARY(name, int16, int64, bool, int64)       \
    XND_BINARY(name, int16, float32, bool, float32)   \
    XND_BINARY(name, int16, float64, bool, float64)   \
                                                      \
    XND_BINARY(name, int32, uint8, bool, int32)       \
    XND_BINARY(name, int32, uint16, bool, int32)      \
    XND_BINARY(name, int32, uint32, bool, int64)      \
    XND_BINARY(name, int32, int8, bool, int32)        \
    XND_BINARY(name, int32, int16, bool, int32)       \
    XND_BINARY(name, int32, int32, bool, int32)       \
    XND_BINARY(name, int32, int64, bool, int64)       \
    XND_BINARY(name, int32, float32, bool, float64)   \
    XND_BINARY(name, int32, float64, bool, float64)   \
                                                      \
    XND_BINARY(name, int64, uint8, bool, int64)       \
    XND_BINARY(name, int64, uint16, bool, int64)      \
    XND_BINARY(name, int64, uint32, bool, int64)      \
    XND_BINARY(name, int64, int8, bool, int64)        \
    XND_BINARY(name, int64, int16, bool, int64)       \
    XND_BINARY(name, int64, int32, bool, int64)       \
    XND_BINARY(name, int64, int64, bool, int64)       \
                                                      \
    XND_BINARY(name, float32, uint8, bool, float32)   \
    XND_BINARY(name, float32, uint16, bool, float32)  \
    XND_BINARY(name, float32, uint32, bool, float64)  \
    XND_BINARY(name, float32, int8, bool, float32)    \
    XND_BINARY(name, float32, int16, bool, float32)   \
    XND_BINARY(name, float32, int32, bool, float64)   \
    XND_BINARY(name, float32, float32, bool, float32) \
    XND_BINARY(name, float32, float64, bool, float64) \
                                                      \
    XND_BINARY(name, float64, uint8, bool, float64)   \
    XND_BINARY(name, float64, uint16, bool, float64)  \
    XND_BINARY(name, float64, uint32, bool, float64)  \
    XND_BINARY(name, float64, int8, bool, float64)    \
    XND_BINARY(name, float64, int16, bool, float64)   \
    XND_BINARY(name, float64, int32, bool, float64)   \
    XND_BINARY(name, float64, float32, bool, float64) \
    XND_BINARY(name, float64, float64, bool, float64)

#define XND_ALL_COMPARISON_INIT(name) \
    XND_BINARY_INIT(name, uint8, uint8, bool),     \
    XND_BINARY_INIT(name, uint8, uint16, bool),    \
    XND_BINARY_INIT(name, uint8, uint32, bool),    \
    XND_BINARY_INIT(name, uint8, uint64, bool),    \
    XND_BINARY_INIT(name, uint8, int8, bool),      \
    XND_BINARY_INIT(name, uint8, int16, bool),     \
    XND_BINARY_INIT(name, uint8, int32, bool),     \
    XND_BINARY_INIT(name, uint8, int64, bool),     \
    XND_BINARY_INIT(name, uint8, float32, bool),   \
    XND_BINARY_INIT(name, uint8, float64, bool),   \
                                                   \
    XND_BINARY_INIT(name, uint16, uint8, bool),    \
    XND_BINARY_INIT(name, uint16, uint16, bool),   \
    XND_BINARY_INIT(name, uint16, uint32, bool),   \
    XND_BINARY_INIT(name, uint16, uint64, bool),   \
    XND_BINARY_INIT(name, uint16, int8, bool),     \
    XND_BINARY_INIT(name, uint16, int16, bool),    \
    XND_BINARY_INIT(name, uint16, int32, bool),    \
    XND_BINARY_INIT(name, uint16, int64, bool),    \
    XND_BINARY_INIT(name, uint16, float32, bool),  \
    XND_BINARY_INIT(name, uint16, float64, bool),  \
                                                   \
    XND_BINARY_INIT(name, uint32, uint8, bool),    \
    XND_BINARY_INIT(name, uint32, uint16, bool),   \
    XND_BINARY_INIT(name, uint32, uint32, bool),   \
    XND_BINARY_INIT(name, uint32, uint64, bool),   \
    XND_BINARY_INIT(name, uint32, int8, bool),     \
    XND_BINARY_INIT(name, uint32, int16, bool),    \
    XND_BINARY_INIT(name, uint32, int32, bool),    \
    XND_BINARY_INIT(name, uint32, int64, bool),    \
    XND_BINARY_INIT(name, uint32, float32, bool),  \
    XND_BINARY_INIT(name, uint32, float64, bool),  \
                                                   \
    XND_BINARY_INIT(name, uint64, uint8, bool),    \
    XND_BINARY_INIT(name, uint64, uint16, bool),   \
    XND_BINARY_INIT(name, uint64, uint32, bool),   \
    XND_BINARY_INIT(name, uint64, uint64, bool),   \
                                                   \
    XND_BINARY_INIT(name, int8, uint8, bool),      \
    XND_BINARY_INIT(name, int8, uint16, bool),     \
    XND_BINARY_INIT(name, int8, uint32, bool),     \
    XND_BINARY_INIT(name, int8, int8, bool),       \
    XND_BINARY_INIT(name, int8, int16, bool),      \
    XND_BINARY_INIT(name, int8, int32, bool),      \
    XND_BINARY_INIT(name, int8, int64, bool),      \
    XND_BINARY_INIT(name, int8, float32, bool),    \
    XND_BINARY_INIT(name, int8, float64, bool),    \
                                                   \
    XND_BINARY_INIT(name, int16, uint8, bool),     \
    XND_BINARY_INIT(name, int16, uint16, bool),    \
    XND_BINARY_INIT(name, int16, uint32, bool),    \
    XND_BINARY_INIT(name, int16, int8, bool),      \
    XND_BINARY_INIT(name, int16, int16, bool),     \
    XND_BINARY_INIT(name, int16, int32, bool),     \
    XND_BINARY_INIT(name, int16, int64, bool),     \
    XND_BINARY_INIT(name, int16, float32, bool),   \
    XND_BINARY_INIT(name, int16, float64, bool),   \
                                                   \
    XND_BINARY_INIT(name, int32, uint8, bool),     \
    XND_BINARY_INIT(name, int32, uint16, bool),    \
    XND_BINARY_INIT(name, int32, uint32, bool),    \
    XND_BINARY_INIT(name, int32, int8, bool),      \
    XND_BINARY_INIT(name, int32, int16, bool),     \
    XND_BINARY_INIT(name, int32, int32, bool),     \
    XND_BINARY_INIT(name, int32, int64, bool),     \
    XND_BINARY_INIT(name, int32, float32, bool),   \
    XND_BINARY_INIT(name, int32, float64, bool),   \
                                                   \
    XND_BINARY_INIT(name, int64, uint8, bool),     \
    XND_BINARY_INIT(name, int64, uint16, bool),    \
    XND_BINARY_INIT(name, int64, uint32, bool),    \
    XND_BINARY_INIT(name, int64, int8, bool),      \
    XND_BINARY_INIT(name, int64, int16, bool),     \
    XND_BINARY_INIT(name, int64, int32, bool),     \
    XND_BINARY_INIT(name, int64, int64, bool),     \
                                                   \
    XND_BINARY_INIT(name, float32, uint8, bool),   \
    XND_BINARY_INIT(name, float32, uint16, bool),  \
    XND_BINARY_INIT(name, float32, uint32, bool),  \
    XND_BINARY_INIT(name, float32, int8, bool),    \
    XND_BINARY_INIT(name, float32, int16, bool),   \
    XND_BINARY_INIT(name, float32, int32, bool),   \
    XND_BINARY_INIT(name, float32, float32, bool), \
    XND_BINARY_INIT(name, float32, float64, bool), \
                                                   \
    XND_BINARY_INIT(name, float64, uint8, bool),   \
    XND_BINARY_INIT(name, float64, uint16, bool),  \
    XND_BINARY_INIT(name, float64, uint32, bool),  \
    XND_BINARY_INIT(name, float64, int8, bool),    \
    XND_BINARY_INIT(name, float64, int16, bool),   \
    XND_BINARY_INIT(name, float64, int32, bool),   \
    XND_BINARY_INIT(name, float64, float32, bool), \
    XND_BINARY_INIT(name, float64, float64, bool)


#undef bool
#define bool_t _Bool

#define greater(x, y) x > y
XND_ALL_COMPARISON(greater)

#define greater_equal(x, y) x >= y
XND_ALL_COMPARISON(greater_equal)

#define less(x, y) x < y
XND_ALL_COMPARISON(less)

#define less_equal(x, y) x <= y
XND_ALL_COMPARISON(less_equal)


static const gm_kernel_init_t kernels[] = {
  XND_ALL_BINARY_INIT(add),
  XND_ALL_BINARY_INIT(subtract),
  XND_ALL_BINARY_INIT(multiply),
  XND_ALL_BINARY_INIT(divide),
  XND_ALL_COMPARISON_INIT(greater),
  XND_ALL_COMPARISON_INIT(greater_equal),
  XND_ALL_COMPARISON_INIT(less),
  XND_ALL_COMPARISON_INIT(less_equal),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &binary_typecheck) < 0) {
             return -1;
        }
    }

    return 0;
}

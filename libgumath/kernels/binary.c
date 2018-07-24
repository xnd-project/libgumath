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
static ndt_t *
infer_return_type(int *base, const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx)
{
    const ndt_t *t0 = ndt_dtype(in0);
    const ndt_t *t1 = ndt_dtype(in1);
    enum ndt tag;

    switch (t0->tag) {
    case Int8: {
        switch (t1->tag) {
        case Int8: *base = 0; tag = Int8; break;
        case Int16: *base = 3; tag = Int16; break;
        case Int32: *base = 6; tag = Int32; break;
        case Int64: *base = 9; tag = Int64; break;
        case Uint8: *base = 12; tag = Int16; break;
        case Uint16: *base = 15; tag = Int32; break;
        case Uint32: *base = 18; tag = Int64; break;
        case Float32: *base = 21; tag = Float32; break;
        case Float64: *base = 24; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Int16: {
        switch (t1->tag) {
        case Int8: *base = 27; tag = Int16; break;
        case Int16: *base = 30; tag = Int16; break;
        case Int32: *base = 33; tag = Int32; break;
        case Int64: *base = 36; tag = Int64; break;
        case Uint8: *base = 39; tag = Int16; break;
        case Uint16: *base = 42; tag = Int32; break;
        case Uint32: *base = 45; tag = Int64; break;
        case Float32: *base = 48; tag = Float32; break;
        case Float64: *base = 51; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Int32: {
        switch (t1->tag) {
        case Int8: *base = 54; tag = Int32; break;
        case Int16: *base = 57; tag = Int32; break;
        case Int32: *base = 60; tag = Int32; break;
        case Int64: *base = 63; tag = Int64; break;
        case Uint8: *base = 66; tag = Int32; break;
        case Uint16: *base = 69; tag = Int32; break;
        case Uint32: *base = 72; tag = Int64; break;
        case Float64: *base = 75; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Int64: {
        switch (t1->tag) {
        case Int8: *base = 78; tag = Int64; break;
        case Int16: *base = 81; tag = Int64; break;
        case Int32: *base = 83; tag = Int64; break;
        case Int64: *base = 87; tag = Int64; break;
        case Uint8: *base = 90; tag = Int64; break;
        case Uint16: *base = 93; tag = Int64; break;
        case Uint32: *base = 96; tag = Int64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Uint8: {
        switch (t1->tag) {
        case Int8: *base = 99; tag = Int16; break;
        case Int16: *base = 102; tag = Int16; break;
        case Int32: *base = 105; tag = Int32; break;
        case Int64: *base = 108; tag = Int64; break;
        case Uint8: *base = 111; tag = Uint8; break;
        case Uint16: *base = 114; tag = Uint16; break;
        case Uint32: *base = 117; tag = Uint32; break;
        case Uint64: *base = 120; tag = Uint64; break;
        case Float32: *base = 123; tag = Float32; break;
        case Float64: *base = 126; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Uint16: {
        switch (t1->tag) {
        case Int8: *base = 129; tag = Int32; break;
        case Int16: *base = 132; tag = Int32; break;
        case Int32: *base = 135; tag = Int32; break;
        case Int64: *base = 138; tag = Int64; break;
        case Uint8: *base = 141; tag = Uint16; break;
        case Uint16: *base = 144; tag = Uint32; break;
        case Uint32: *base = 147; tag = Uint64; break;
        case Uint64: *base = 150; tag = Uint64; break;
        case Float32: *base = 153; tag = Float32; break;
        case Float64: *base = 156; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Uint32: {
        switch (t1->tag) {
        case Int8: *base = 159; tag = Int64; break;
        case Int16: *base = 162; tag = Int64; break;
        case Int32: *base = 165; tag = Int64; break;
        case Int64: *base = 168; tag = Int64; break;
        case Uint8: *base = 171; tag = Uint32; break;
        case Uint16: *base = 174; tag = Uint32; break;
        case Uint32: *base = 177; tag = Uint32; break;
        case Uint64: *base = 180; tag = Uint64; break;
        case Float64: *base = 183; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Uint64: {
        switch (t1->tag) {
        case Uint8: *base = 186; tag = Uint64; break;
        case Uint16: *base = 189; tag = Uint64; break;
        case Uint32: *base = 192; tag = Uint64; break;
        case Uint64: *base = 195; tag = Uint64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Float32: {
        switch (t1->tag) {
        case Int8: *base = 198; tag = Float32; break;
        case Int16: *base = 201; tag = Float32; break;
        case Uint8: *base = 204; tag = Float32; break;
        case Uint16: *base = 207; tag = Float32; break;
        case Float32: *base = 210; tag = Float32; break;
        case Float64: *base = 213; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    case Float64: {
        switch (t1->tag) {
        case Int8: *base = 216; tag = Float64; break;
        case Int16: *base = 219; tag = Float64; break;
        case Int32: *base = 222; tag = Float64; break;
        case Uint8: *base = 225; tag = Float64; break;
        case Uint16: *base = 228; tag = Float64; break;
        case Uint32: *base = 231; tag = Float64; break;
        case Float32: *base = 234; tag = Float64; break;
        case Float64: *base = 237; tag = Float64; break;
        default: goto invalid_combination;
        }
        break;
    }
    default:
        goto invalid_combination;
    }

    return ndt_primitive(tag, 0, ctx);

invalid_combination:
    ndt_err_format(ctx, NDT_RuntimeError, "invalid dtype");
    return NULL;
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
    ndt_t *dtype;
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

    dtype = infer_return_type(&n, t0, t1, ctx);
    if (dtype == NULL) {
        return NULL;
    }

    if (t0->tag == VarDim || t1->tag == VarDim) {
        const gm_kernel_set_t *set = &f->kernels[n+2];
        ndt_del(dtype); /* temporary */
        if (ndt_typecheck(spec, set->sig, in, nin, NULL, NULL, ctx) < 0) {
            return NULL;
        }
        return set;
    }

    if (t0->ndim == 0 && t1->ndim == 0) {
        n += 1;
    }
 
    const gm_kernel_set_t *set = &f->kernels[n];
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


#define XND_BINARY(func, t0, t1, t2) \
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
        out[i] = func(in0[i], in1[i]);                                         \
        out[i+1] = func(in0[i+1], in1[i+1]);                                   \
        out[i+2] = func(in0[i+2], in1[i+2]);                                   \
        out[i+3] = func(in0[i+3], in1[i+3]);                                   \
        out[i+4] = func(in0[i+4], in1[i+4]);                                   \
        out[i+5] = func(in0[i+5], in1[i+5]);                                   \
        out[i+6] = func(in0[i+6], in1[i+6]);                                   \
        out[i+7] = func(in0[i+7], in1[i+7]);                                   \
    }                                                                          \
    for (; i < N; i++) {                                                       \
        out[i] = func(in0[i], in1[i]);                                         \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_fixed_##func##_1D_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)     \
{                                                                              \
    const xnd_t *in0 = &stack[0];                                              \
    const xnd_t *in1 = &stack[1];                                              \
    xnd_t *out = &stack[2];                                                    \
    int64_t N = xnd_fixed_shape(in0);                                          \
    (void)ctx;                                                                 \
                                                                               \
    for (int64_t i = 0; i < N; i++) {                                          \
        const xnd_t v = xnd_fixed_dim_next(in0, i);                            \
        const xnd_t u = xnd_fixed_dim_next(in1, i);                            \
        const xnd_t w = xnd_fixed_dim_next(out, i);                            \
        const t0##_t x = *(const t0##_t *)v.ptr;                               \
        const t1##_t y = *(const t1##_t *)u.ptr ;                              \
        *(t2##_t *)w.ptr = func(x, y);                                         \
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
    *(t2##_t *)out->ptr = func(x, y);                                          \
                                                                               \
    return 0;                                                                  \
}

#define XND_BINARY_INIT(func, t0, t1, t2) \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * N * " STRINGIZE(t0) ", ... * N * " STRINGIZE(t1) " -> ... * N * " STRINGIZE(t2), \
    .C = gm_fixed_##func##_1D_C_##t0##_##t1##_##t2,                                                \
    .Xnd = gm_fixed_##func##_1D_##t0##_##t1##_##t2 },                                              \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),             \
    .Xnd = gm_##func##_0D_##t0##_##t1##_##t2 },                                                    \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * " STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * " STRINGIZE(t2),    \
    .Xnd = gm_##func##_0D_##t0##_##t1##_##t2 }

#define XND_ALL_BINARY(name) \
    XND_BINARY(name, int8, int8, int8)          \
    XND_BINARY(name, int8, int16, int16)        \
    XND_BINARY(name, int8, int32, int32)        \
    XND_BINARY(name, int8, int64, int64)        \
    XND_BINARY(name, int8, uint8, int16)        \
    XND_BINARY(name, int8, uint16, int32)       \
    XND_BINARY(name, int8, uint32, int64)       \
    XND_BINARY(name, int8, float32, float32)    \
    XND_BINARY(name, int8, float64, float64)    \
                                                \
    XND_BINARY(name, int16, int8, int16)        \
    XND_BINARY(name, int16, int16, int16)       \
    XND_BINARY(name, int16, int32, int32)       \
    XND_BINARY(name, int16, int64, int64)       \
    XND_BINARY(name, int16, uint8, int16)       \
    XND_BINARY(name, int16, uint16, int32)      \
    XND_BINARY(name, int16, uint32, int64)      \
    XND_BINARY(name, int16, float32, float32)   \
    XND_BINARY(name, int16, float64, float64)   \
                                                \
    XND_BINARY(name, int32, int8, int32)        \
    XND_BINARY(name, int32, int16, int32)       \
    XND_BINARY(name, int32, int32, int32)       \
    XND_BINARY(name, int32, int64, int64)       \
    XND_BINARY(name, int32, uint8, int32)       \
    XND_BINARY(name, int32, uint16, int32)      \
    XND_BINARY(name, int32, uint32, int64)      \
    XND_BINARY(name, int32, float64, float64)   \
                                                \
    XND_BINARY(name, int64, int8, int64)        \
    XND_BINARY(name, int64, int16, int64)       \
    XND_BINARY(name, int64, int32, int64)       \
    XND_BINARY(name, int64, int64, int64)       \
    XND_BINARY(name, int64, uint8, int64)       \
    XND_BINARY(name, int64, uint16, int64)      \
    XND_BINARY(name, int64, uint32, int64)      \
                                                \
    XND_BINARY(name, uint8, int8, int16)        \
    XND_BINARY(name, uint8, int16, int16)       \
    XND_BINARY(name, uint8, int32, int32)       \
    XND_BINARY(name, uint8, int64, int64)       \
    XND_BINARY(name, uint8, uint8, uint8)       \
    XND_BINARY(name, uint8, uint16, uint16)     \
    XND_BINARY(name, uint8, uint32, uint32)     \
    XND_BINARY(name, uint8, uint64, uint64)     \
    XND_BINARY(name, uint8, float32, float32)   \
    XND_BINARY(name, uint8, float64, float64)   \
                                                \
    XND_BINARY(name, uint16, int8, int32)       \
    XND_BINARY(name, uint16, int16, int32)      \
    XND_BINARY(name, uint16, int32, int32)      \
    XND_BINARY(name, uint16, int64, int64)      \
    XND_BINARY(name, uint16, uint8, uint16)     \
    XND_BINARY(name, uint16, uint16, uint16)    \
    XND_BINARY(name, uint16, uint32, uint32)    \
    XND_BINARY(name, uint16, uint64, uint64)    \
    XND_BINARY(name, uint16, float32, float32)  \
    XND_BINARY(name, uint16, float64, float64)  \
                                                \
    XND_BINARY(name, uint32, int8, int64)       \
    XND_BINARY(name, uint32, int16, int64)      \
    XND_BINARY(name, uint32, int32, int64)      \
    XND_BINARY(name, uint32, int64, int64)      \
    XND_BINARY(name, uint32, uint8, uint32)     \
    XND_BINARY(name, uint32, uint16, uint32)    \
    XND_BINARY(name, uint32, uint32, uint32)    \
    XND_BINARY(name, uint32, uint64, uint64)    \
    XND_BINARY(name, uint32, float64, float64)  \
                                                \
    XND_BINARY(name, uint64, uint8, uint64)     \
    XND_BINARY(name, uint64, uint16, uint64)    \
    XND_BINARY(name, uint64, uint32, uint64)    \
    XND_BINARY(name, uint64, uint64, uint64)    \
                                                \
    XND_BINARY(name, float32, int8, float32)    \
    XND_BINARY(name, float32, int16, float32)   \
    XND_BINARY(name, float32, uint8, float32)   \
    XND_BINARY(name, float32, uint16, float32)  \
    XND_BINARY(name, float32, float32, float32) \
    XND_BINARY(name, float32, float64, float64) \
                                                \
    XND_BINARY(name, float64, int8, float64)    \
    XND_BINARY(name, float64, int16, float64)   \
    XND_BINARY(name, float64, int32, float64)   \
    XND_BINARY(name, float64, uint8, float64)   \
    XND_BINARY(name, float64, uint16, float64)  \
    XND_BINARY(name, float64, uint32, float64)  \
    XND_BINARY(name, float64, float32, float64) \
    XND_BINARY(name, float64, float64, float64)

#define XND_ALL_BINARY_INIT(name) \
    XND_BINARY_INIT(name, int8, int8, int8),          \
    XND_BINARY_INIT(name, int8, int16, int16),        \
    XND_BINARY_INIT(name, int8, int32, int32),        \
    XND_BINARY_INIT(name, int8, int64, int64),        \
    XND_BINARY_INIT(name, int8, uint8, int16),        \
    XND_BINARY_INIT(name, int8, uint16, int32),       \
    XND_BINARY_INIT(name, int8, uint32, int64),       \
    XND_BINARY_INIT(name, int8, float32, float32),    \
    XND_BINARY_INIT(name, int8, float64, float64),    \
                                                      \
    XND_BINARY_INIT(name, int16, int8, int16),        \
    XND_BINARY_INIT(name, int16, int16, int16),       \
    XND_BINARY_INIT(name, int16, int32, int32),       \
    XND_BINARY_INIT(name, int16, int64, int64),       \
    XND_BINARY_INIT(name, int16, uint8, int16),       \
    XND_BINARY_INIT(name, int16, uint16, int32),      \
    XND_BINARY_INIT(name, int16, uint32, int64),      \
    XND_BINARY_INIT(name, int16, float32, float32),   \
    XND_BINARY_INIT(name, int16, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, int32, int8, int32),        \
    XND_BINARY_INIT(name, int32, int16, int32),       \
    XND_BINARY_INIT(name, int32, int32, int32),       \
    XND_BINARY_INIT(name, int32, int64, int64),       \
    XND_BINARY_INIT(name, int32, uint8, int32),       \
    XND_BINARY_INIT(name, int32, uint16, int32),      \
    XND_BINARY_INIT(name, int32, uint32, int64),      \
    XND_BINARY_INIT(name, int32, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, int64, int8, int64),        \
    XND_BINARY_INIT(name, int64, int16, int64),       \
    XND_BINARY_INIT(name, int64, int32, int64),       \
    XND_BINARY_INIT(name, int64, int64, int64),       \
    XND_BINARY_INIT(name, int64, uint8, int64),       \
    XND_BINARY_INIT(name, int64, uint16, int64),      \
    XND_BINARY_INIT(name, int64, uint32, int64),      \
                                                      \
    XND_BINARY_INIT(name, uint8, int8, int16),        \
    XND_BINARY_INIT(name, uint8, int16, int16),       \
    XND_BINARY_INIT(name, uint8, int32, int32),       \
    XND_BINARY_INIT(name, uint8, int64, int64),       \
    XND_BINARY_INIT(name, uint8, uint8, uint8),       \
    XND_BINARY_INIT(name, uint8, uint16, uint16),     \
    XND_BINARY_INIT(name, uint8, uint32, uint32),     \
    XND_BINARY_INIT(name, uint8, uint64, uint64),     \
    XND_BINARY_INIT(name, uint8, float32, float32),   \
    XND_BINARY_INIT(name, uint8, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, uint16, int8, int32),       \
    XND_BINARY_INIT(name, uint16, int16, int32),      \
    XND_BINARY_INIT(name, uint16, int32, int32),      \
    XND_BINARY_INIT(name, uint16, int64, int64),      \
    XND_BINARY_INIT(name, uint16, uint8, uint16),     \
    XND_BINARY_INIT(name, uint16, uint16, uint16),    \
    XND_BINARY_INIT(name, uint16, uint32, uint32),    \
    XND_BINARY_INIT(name, uint16, uint64, uint64),    \
    XND_BINARY_INIT(name, uint16, float32, float32),  \
    XND_BINARY_INIT(name, uint16, float64, float64),  \
                                                      \
    XND_BINARY_INIT(name, uint32, int8, int64),       \
    XND_BINARY_INIT(name, uint32, int16, int64),      \
    XND_BINARY_INIT(name, uint32, int32, int64),      \
    XND_BINARY_INIT(name, uint32, int64, int64),      \
    XND_BINARY_INIT(name, uint32, uint8, uint32),     \
    XND_BINARY_INIT(name, uint32, uint16, uint32),    \
    XND_BINARY_INIT(name, uint32, uint32, uint32),    \
    XND_BINARY_INIT(name, uint32, uint64, uint64),    \
    XND_BINARY_INIT(name, uint32, float64, float64),  \
                                                      \
    XND_BINARY_INIT(name, uint64, uint8, uint64),     \
    XND_BINARY_INIT(name, uint64, uint16, uint64),    \
    XND_BINARY_INIT(name, uint64, uint32, uint64),    \
    XND_BINARY_INIT(name, uint64, uint64, uint64),    \
                                                      \
    XND_BINARY_INIT(name, float32, int8, float32),    \
    XND_BINARY_INIT(name, float32, int16, float32),   \
    XND_BINARY_INIT(name, float32, uint8, float32),   \
    XND_BINARY_INIT(name, float32, uint16, float32),  \
    XND_BINARY_INIT(name, float32, float32, float32), \
    XND_BINARY_INIT(name, float32, float64, float64), \
                                                      \
    XND_BINARY_INIT(name, float64, int8, float64),    \
    XND_BINARY_INIT(name, float64, int16, float64),   \
    XND_BINARY_INIT(name, float64, int32, float64),   \
    XND_BINARY_INIT(name, float64, uint8, float64),   \
    XND_BINARY_INIT(name, float64, uint16, float64),  \
    XND_BINARY_INIT(name, float64, uint32, float64),  \
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


static const gm_kernel_init_t kernels[] = {
  XND_ALL_BINARY_INIT(add),
  XND_ALL_BINARY_INIT(subtract),
  XND_ALL_BINARY_INIT(multiply),
  XND_ALL_BINARY_INIT(divide),

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

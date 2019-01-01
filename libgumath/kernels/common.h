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


#ifndef COMMON_H
#define COMMON_H


#define XSTRINGIZE(v) #v
#define STRINGIZE(v) XSTRINGIZE(v)


/*****************************************************************************/
/*              Apply linear index to the data pointer (1D kernels)          */
/*****************************************************************************/

static inline char *
apply_index(const xnd_t *x)
{
    return xnd_fixed_apply_index(x);
}


/*****************************************************************************/
/*                          Optimized bitmap handling                        */
/*****************************************************************************/

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


/*****************************************************************************/
/*                         Generate unary kernels                            */
/*****************************************************************************/

#define XND_UNARY(func, t0, cast, t1) \
static int                                                            \
gm_fixed_##func##_1D_C_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                     \
    const t0##_t *in0 = (const t0##_t *)apply_index(&stack[0]);       \
    t1##_t *out = (t1##_t *)apply_index(&stack[1]);                   \
    int64_t N = xnd_fixed_shape(&stack[0]);                           \
    (void)ctx;                                                        \
                                                                      \
    for (int64_t i = 0; i < N; i++) {                                 \
        out[i] = func((cast##_t)in0[i]);                              \
    }                                                                 \
                                                                      \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                  \
        unary_update_bitmap1D(stack);                                 \
    }                                                                 \
                                                                      \
    return 0;                                                         \
}                                                                     \
                                                                      \
static int                                                            \
gm_##func##_0D_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                     \
    const xnd_t *in0 = &stack[0];                                     \
    xnd_t *out = &stack[1];                                           \
    (void)ctx;                                                        \
                                                                      \
    const t0##_t x = *(const t0##_t *)in0->ptr;                       \
    *(t1##_t *)out->ptr = func((cast##_t)x);                          \
                                                                      \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                  \
        unary_update_bitmap(stack);                                   \
    }                                                                 \
                                                                      \
    return 0;                                                         \
}

#define XND_UNARY_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "... * " STRINGIZE(t0) " -> ... * " STRINGIZE(t1),         \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1,                        \
    .C = gm_##func##_0D_##t0##_##t1 },                                \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "... * ?" STRINGIZE(t0) " -> ... * ?" STRINGIZE(t1),       \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1,                        \
    .C = gm_##func##_0D_##t0##_##t1 },                                \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "var... * " STRINGIZE(t0) " -> var... * " STRINGIZE(t1),   \
    .C = gm_##func##_0D_##t0##_##t1 },                                \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "var... * ?" STRINGIZE(t0) " -> var... * ?" STRINGIZE(t1), \
    .C = gm_##func##_0D_##t0##_##t1 }


/*****************************************************************************/
/*                         Generate binary kernels                           */
/*****************************************************************************/

#define XND_BINARY(func, t0, t1, t2, cast) \
static int                                                                   \
gm_fixed_##func##_1D_C_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                            \
    const t0##_t *in0 = (const t0##_t *)apply_index(&stack[0]);              \
    const t1##_t *in1 = (const t1##_t *)apply_index(&stack[1]);              \
    t2##_t *out = (t2##_t *)apply_index(&stack[2]);                          \
    int64_t N = xnd_fixed_shape(&stack[0]);                                  \
    (void)ctx;                                                               \
    int64_t i;                                                               \
                                                                             \
    for (i = 0; i < N-7; i += 8) {                                           \
        out[i] = func((cast##_t)in0[i], (cast##_t)in1[i]);                   \
        out[i+1] = func((cast##_t)in0[i+1], (cast##_t)in1[i+1]);             \
        out[i+2] = func((cast##_t)in0[i+2], (cast##_t)in1[i+2]);             \
        out[i+3] = func((cast##_t)in0[i+3], (cast##_t)in1[i+3]);             \
        out[i+4] = func((cast##_t)in0[i+4], (cast##_t)in1[i+4]);             \
        out[i+5] = func((cast##_t)in0[i+5], (cast##_t)in1[i+5]);             \
        out[i+6] = func((cast##_t)in0[i+6], (cast##_t)in1[i+6]);             \
        out[i+7] = func((cast##_t)in0[i+7], (cast##_t)in1[i+7]);             \
    }                                                                        \
    for (; i < N; i++) {                                                     \
        out[i] = func((cast##_t)in0[i], (cast##_t)in1[i]);                   \
    }                                                                        \
                                                                             \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                         \
        binary_update_bitmap1D(stack);                                       \
    }                                                                        \
                                                                             \
    return 0;                                                                \
}                                                                            \
                                                                             \
static int                                                                   \
gm_##func##_0D_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                            \
    const xnd_t *in0 = &stack[0];                                            \
    const xnd_t *in1 = &stack[1];                                            \
    xnd_t *out = &stack[2];                                                  \
    (void)ctx;                                                               \
                                                                             \
    const t0##_t x = *(const t0##_t *)in0->ptr;                              \
    const t1##_t y = *(const t1##_t *)in1->ptr;                              \
    *(t2##_t *)out->ptr = func((cast##_t)x, (cast##_t)y);                    \
                                                                             \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                         \
        binary_update_bitmap(stack);                                         \
    }                                                                        \
                                                                             \
    return 0;                                                                \
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


/*****************************************************************************/
/*                              Binary typecheck                             */
/*****************************************************************************/

/* LOCAL SCOPE */
NDT_PRAGMA(NDT_HIDE_SYMBOLS_START)

void unary_update_bitmap1D(xnd_t stack[]);
void unary_update_bitmap(xnd_t stack[]);

void binary_update_bitmap1D(xnd_t stack[]);
void binary_update_bitmap(xnd_t stack[]);

const gm_kernel_set_t *cpu_unary_typecheck(int (*kernel_location)(const ndt_t *, ndt_context_t *),
                                           ndt_apply_spec_t *spec, const gm_func_t *f,
                                           const ndt_t *types[], const int64_t li[], int nin, int nout,
                                           ndt_context_t *ctx);

const gm_kernel_set_t *cuda_unary_typecheck(int (*kernel_location)(const ndt_t *, ndt_context_t *),
                                            ndt_apply_spec_t *spec, const gm_func_t *f,
                                            const ndt_t *types[], const int64_t li[], int nin, int nout,
                                            ndt_context_t *ctx);

const gm_kernel_set_t *cpu_binary_typecheck(int (*kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                                            ndt_apply_spec_t *spec, const gm_func_t *f,
                                            const ndt_t *types[], const int64_t li[], int nin, int nout,
                                            ndt_context_t *ctx);

const gm_kernel_set_t *cuda_binary_typecheck(int (* kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                                             ndt_apply_spec_t *spec, const gm_func_t *f,
                                             const ndt_t *types[], const int64_t li[], int nin, int nout,
                                             ndt_context_t *ctx);

/* END LOCAL SCOPE */
NDT_PRAGMA(NDT_HIDE_SYMBOLS_END)


#endif /* COMMON_H */

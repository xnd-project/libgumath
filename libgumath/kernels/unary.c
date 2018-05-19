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
/*                           Generated Xnd kernels                          */
/****************************************************************************/

#define XSTRINGIZE(v) #v
#define STRINGIZE(v) XSTRINGIZE(v)

static inline char *
apply_index(const xnd_t *x)
{
    return xnd_fixed_apply_index(x);
}


#define XND_UNARY(func, t0, t1) \
static int                                                                   \
gm_fixed_##func##_0D_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)          \
{                                                                            \
    const xnd_t *in0 = &stack[0];                                            \
    xnd_t *out = &stack[1];                                                  \
    (void)ctx;                                                               \
                                                                             \
    const t0##_t x = *(const t0##_t *)in0->ptr;                              \
    *(t1##_t *)out->ptr = func(x);                                           \
                                                                             \
    return 0;                                                                \
}                                                                            \
                                                                             \
static int                                                                   \
gm_fixed_##func##_1D_C_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)        \
{                                                                            \
    const t0##_t *in0 = (const t0##_t *)apply_index(&stack[0]);              \
    t1##_t *out = (t1##_t *)apply_index(&stack[1]);                          \
    int64_t N = xnd_fixed_shape(&stack[0]);                                  \
    (void)ctx;                                                               \
                                                                             \
    for (int64_t i = 0; i < N; i++) {                                        \
        out[i] = func(in0[i]);                                               \
    }                                                                        \
                                                                             \
    return 0;                                                                \
}                                                                            \
                                                                             \
static int                                                                   \
gm_fixed_##func##_1D_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)          \
{                                                                            \
    const xnd_t *in0 = &stack[0];                                            \
    xnd_t *out = &stack[1];                                                  \
    int64_t N = xnd_fixed_shape(in0);                                        \
    (void)ctx;                                                               \
                                                                             \
    for (int64_t i = 0; i < N; i++) {                                        \
        const xnd_t v = xnd_fixed_dim_next(in0, i);                          \
        const xnd_t u = xnd_fixed_dim_next(out, i);                          \
        const t0##_t x = *(const t0##_t *)v.ptr;                             \
        *(t1##_t *)u.ptr = func(x);                                          \
    }                                                                        \
                                                                             \
    return 0;                                                                \
}                                                                            \
                                                                             \
static int                                                                   \
gm_var_##func##_0D_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)            \
{                                                                            \
    const xnd_t *in0 = &stack[0];                                            \
    xnd_t *out = &stack[1];                                                  \
    (void)ctx;                                                               \
                                                                             \
    const t0##_t x = *(const t0##_t *)in0->ptr;                              \
    *(t1##_t *)out->ptr = func(x);                                           \
                                                                             \
    return 0;                                                                \
}                                                                            \
                                                                             \
static int                                                                   \
gm_var_##func##_1D_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)            \
{                                                                            \
    int64_t start[2], step[2];                                               \
    int64_t shape, n;                                                        \
                                                                             \
    shape = ndt_var_indices(&start[0], &step[0], stack[0].type,              \
                            stack[0].index, ctx);                            \
    if (shape < 0) {                                                         \
        return -1;                                                           \
    }                                                                        \
                                                                             \
    n = ndt_var_indices(&start[1], &step[1], stack[1].type, stack[1].index,  \
                        ctx);                                                \
    if (n < 0) {                                                             \
        return -1;                                                           \
    }                                                                        \
                                                                             \
    if (n != shape) {                                                        \
        ndt_err_format(ctx, NDT_ValueError, "shape mismatch");               \
        return -1;                                                           \
    }                                                                        \
                                                                             \
    for (int64_t i = 0; i < shape; i++) {                                    \
        const xnd_t in0 = xnd_var_dim_next(&stack[0], start[0], step[0], i); \
        xnd_t out = xnd_var_dim_next(&stack[1], start[1], step[1], i) ;      \
        const t0##_t x = *(const t0##_t *)in0.ptr;                           \
        *(t1##_t *)out.ptr = func(x);                                        \
    }                                                                        \
                                                                             \
    return 0;                                                                \
}

#define XND_UNARY_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                               \
    .sig = "... * N * " STRINGIZE(t0) "-> ... * N * " STRINGIZE(t1),           \
    .C = gm_fixed_##func##_1D_C_##t0##_##t1,                                   \
    .Xnd = gm_fixed_##func##_1D_##t0##_##t1 },                                 \
                                                                               \
  { .name = STRINGIZE(funcname),                                               \
    .sig = "... * " STRINGIZE(t0) "-> ... * " STRINGIZE(t1),                   \
    .Xnd = gm_fixed_##func##_0D_##t0##_##t1 },                                 \
                                                                               \
  { .name = STRINGIZE(funcname),                                               \
    .sig = "var... * var * " STRINGIZE(t0) "-> var... * var * " STRINGIZE(t1), \
    .Xnd = gm_var_##func##_1D_##t0##_##t1 },                                   \
                                                                               \
  { .name = STRINGIZE(funcname),                                               \
    .sig = "var... * " STRINGIZE(t0) "-> var... * " STRINGIZE(t1),             \
    .Xnd = gm_var_##func##_0D_##t0##_##t1 }



XND_UNARY(sinf, float32, float32)
XND_UNARY(sinf, int8, float32)
XND_UNARY(sinf, int16, float32)
XND_UNARY(sinf, uint8, float32)
XND_UNARY(sinf, uint16, float32)

XND_UNARY(sin, float64, float64)
XND_UNARY(sin, int32, float64)
XND_UNARY(sin, uint32, float64)

#define copy(x) x
XND_UNARY(copy, int8, int8)
XND_UNARY(copy, int16, int16)
XND_UNARY(copy, int32, int32)
XND_UNARY(copy, int64, int64)
XND_UNARY(copy, uint8, uint8)
XND_UNARY(copy, uint16, uint16)
XND_UNARY(copy, uint32, uint32)
XND_UNARY(copy, uint64, uint64)
XND_UNARY(copy, float32, float32)
XND_UNARY(copy, float64, float64)


static const gm_kernel_init_t kernels[] = {
  /* COPY */
  XND_UNARY_INIT(copy, copy, int8, int8),
  XND_UNARY_INIT(copy, copy, int16, int16),
  XND_UNARY_INIT(copy, copy, int32, int32),
  XND_UNARY_INIT(copy, copy, int64, int64),
  XND_UNARY_INIT(copy, copy, uint8, uint8),
  XND_UNARY_INIT(copy, copy, uint16, uint16),
  XND_UNARY_INIT(copy, copy, uint32, uint32),
  XND_UNARY_INIT(copy, copy, uint64, uint64),
  XND_UNARY_INIT(copy, copy, float32, float32),
  XND_UNARY_INIT(copy, copy, float64, float64),

  /* SIN */
  XND_UNARY_INIT(sin, sinf, float32, float32),
  XND_UNARY_INIT(sin, sinf, uint8, float32),
  XND_UNARY_INIT(sin, sinf, uint16, float32),
  XND_UNARY_INIT(sin, sinf, int8, float32),
  XND_UNARY_INIT(sin, sinf, int16, float32),

  XND_UNARY_INIT(sin, sin, float64, float64),
  XND_UNARY_INIT(sin, sin, uint32, float64),
  XND_UNARY_INIT(sin, sin, int32, float64),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

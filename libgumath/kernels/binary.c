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


#define XND_BINARY(func, t1, t2, t3) \
static int                                                                     \
gm_fixed_##func##_0D_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx)     \
{                                                                              \
    const xnd_t *in1 = &stack[0];                                              \
    const xnd_t *in2 = &stack[1];                                              \
    xnd_t *out = &stack[2];                                                    \
    (void)ctx;                                                                 \
                                                                               \
    const t1##_t x = *(const t1##_t *)in1->ptr;                                \
    const t2##_t y = *(const t2##_t *)in2->ptr;                                \
    *(t3##_t *)out->ptr = func(x, y);                                          \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_fixed_##func##_1D_C_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx)   \
{                                                                              \
    const t1##_t *in1 = (const t1##_t *)apply_index(&stack[0]);                \
    const t2##_t *in2 = (const t2##_t *)apply_index(&stack[1]);                \
    t3##_t *out = (t3##_t *)apply_index(&stack[2]);                            \
    int64_t N = xnd_fixed_shape(&stack[0]);                                    \
    (void)ctx;                                                                 \
                                                                               \
    for (int64_t i = 0; i < N; i++) {                                          \
        out[i] = func(in1[i], in2[i]);                                         \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_fixed_##func##_1D_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx)     \
{                                                                              \
    const xnd_t *in1 = &stack[0];                                              \
    const xnd_t *in2 = &stack[1];                                              \
    xnd_t *out = &stack[2];                                                    \
    int64_t N = xnd_fixed_shape(in1);                                          \
    (void)ctx;                                                                 \
                                                                               \
    for (int64_t i = 0; i < N; i++) {                                          \
        const xnd_t v = xnd_fixed_dim_next(in1, i);                            \
        const xnd_t u = xnd_fixed_dim_next(in2, i);                            \
        const xnd_t w = xnd_fixed_dim_next(out, i);                            \
        const t1##_t x = *(const t1##_t *)v.ptr;                               \
        const t2##_t y = *(const t2##_t *)u.ptr ;                              \
        *(t3##_t *)w.ptr = func(x, y);                                         \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_var_##func##_0D_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx)       \
{                                                                              \
    const xnd_t *in1 = &stack[0];                                              \
    const xnd_t *in2 = &stack[1];                                              \
    xnd_t *out = &stack[2];                                                    \
    (void)ctx;                                                                 \
                                                                               \
    const t1##_t x = *(const t1##_t *)in1->ptr;                                \
    const t2##_t y = *(const t2##_t *)in2->ptr;                                \
    *(t3##_t *)out->ptr = func(x, y);                                          \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_var_##func##_1D_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx)       \
{                                                                              \
    int64_t start[3], step[3];                                                 \
    int64_t shape, n1, n2;                                                     \
                                                                               \
    shape = ndt_var_indices(&start[0], &step[0], stack[0].type,                \
                            stack[0].index, ctx);                              \
    if (shape < 0) {                                                           \
        return -1;                                                             \
    }                                                                          \
                                                                               \
    n1 = ndt_var_indices(&start[1], &step[1], stack[1].type, stack[1].index,   \
                         ctx);                                                 \
    if (n1 < 0) {                                                              \
        return -1;                                                             \
    }                                                                          \
                                                                               \
    n2 = ndt_var_indices(&start[2], &step[2], stack[2].type, stack[2].index,   \
                         ctx);                                                 \
    if (n2 < 0) {                                                              \
        return -1;                                                             \
    }                                                                          \
                                                                               \
    if (n1 != shape || n2 != shape) {                                          \
        ndt_err_format(ctx, NDT_ValueError, "shape mismatch");                 \
        return -1;                                                             \
    }                                                                          \
                                                                               \
    for (int64_t i = 0; i < shape; i++) {                                      \
        const xnd_t in1 = xnd_var_dim_next(&stack[0], start[0], step[0], i);   \
        const xnd_t in2 = xnd_var_dim_next(&stack[1], start[1], step[1], i);   \
        xnd_t out = xnd_var_dim_next(&stack[2], start[2], step[2], i);         \
        const t1##_t x = *(const t1##_t *)in1.ptr;                             \
        const t2##_t y = *(const t2##_t *)in2.ptr;                             \
        *(t3##_t *)out.ptr = func(x, y);                                       \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}

#define XND_BINARY_INIT(func, t1, t2, t3) \
  { .name = STRINGIZE(func),                                                                                     \
    .sig = "... * N * " STRINGIZE(t1) ", ... * N * " STRINGIZE(t2) "-> ... * N * " STRINGIZE(t3),                \
    .C = gm_fixed_##func##_1D_C_##t1##_##t2##_##t3,                                                              \
    .Xnd = gm_fixed_##func##_1D_##t1##_##t2##_##t3 },                                                            \
                                                                                                                 \
  { .name = STRINGIZE(func),                                                                                     \
    .sig = "... * " STRINGIZE(t1) ", ... * N * " STRINGIZE(t2) "-> ... * " STRINGIZE(t3),                        \
    .Xnd = gm_fixed_##func##_0D_##t1##_##t2##_##t3 },                                                            \
                                                                                                                 \
  { .name = STRINGIZE(func),                                                                                     \
    .sig = "var... * var * " STRINGIZE(t1) ", var... * var * " STRINGIZE(t1) "-> var... * var * " STRINGIZE(t3), \
    .Xnd = gm_var_##func##_1D_##t1##_##t2##_##t3 },                                                              \
                                                                                                                 \
  { .name = STRINGIZE(func),                                                                                     \
    .sig = "var... * " STRINGIZE(t1) ", var... * " STRINGIZE(t2) "-> var... * " STRINGIZE(t3),                   \
    .Xnd = gm_var_##func##_0D_##t1##_##t2##_##t3 }


/****************************************************************************/
/*                                   Add                                    */
/****************************************************************************/

#define add(x, y) x + y
/* signed */
XND_BINARY(add, int8, int8, int8)
XND_BINARY(add, int8, int16, int16)
XND_BINARY(add, int8, int32, int32)
XND_BINARY(add, int8, int64, int64)

XND_BINARY(add, int16, int8, int16)
XND_BINARY(add, int32, int8, int32)
XND_BINARY(add, int64, int8, int64)

XND_BINARY(add, int16, int16, int16)
XND_BINARY(add, int16, int32, int32)
XND_BINARY(add, int16, int64, int64)
XND_BINARY(add, int32, int16, int32)
XND_BINARY(add, int64, int16, int64)

XND_BINARY(add, int32, int32, int32)
XND_BINARY(add, int32, int64, int64)
XND_BINARY(add, int64, int32, int64)

XND_BINARY(add, int64, int64, int64)

/* unsigned */
XND_BINARY(add, uint8, uint8, uint8)
XND_BINARY(add, uint8, uint16, uint16)
XND_BINARY(add, uint8, uint32, uint32)
XND_BINARY(add, uint8, uint64, uint64)
XND_BINARY(add, uint16, uint8, uint16)
XND_BINARY(add, uint32, uint8, uint32)
XND_BINARY(add, uint64, uint8, uint64)

XND_BINARY(add, uint16, uint16, uint16)
XND_BINARY(add, uint16, uint32, uint32)
XND_BINARY(add, uint16, uint64, uint64)
XND_BINARY(add, uint32, uint16, uint32)
XND_BINARY(add, uint64, uint16, uint64)

XND_BINARY(add, uint32, uint32, uint32)
XND_BINARY(add, uint32, uint64, uint64)
XND_BINARY(add, uint64, uint32, uint64)

XND_BINARY(add, uint64, uint64, uint64)

/* signed/float */
XND_BINARY(add, int8, float32, float32)
XND_BINARY(add, int8, float64, float64)
XND_BINARY(add, float32, int8, float32)
XND_BINARY(add, float64, int8, float64)

XND_BINARY(add, int16, float32, float32)
XND_BINARY(add, int16, float64, float64)
XND_BINARY(add, float32, int16, float32)
XND_BINARY(add, float64, int16, float64)

XND_BINARY(add, int32, float64, float64)
XND_BINARY(add, float64, int32, float64)

/* unsigned/float */
XND_BINARY(add, uint8, float32, float32)
XND_BINARY(add, uint8, float64, float64)
XND_BINARY(add, float32, uint8, float32)
XND_BINARY(add, float64, uint8, float64)

XND_BINARY(add, uint16, float32, float32)
XND_BINARY(add, uint16, float64, float64)
XND_BINARY(add, float32, uint16, float32)
XND_BINARY(add, float64, uint16, float64)

XND_BINARY(add, uint32, float64, float64)
XND_BINARY(add, float64, uint32, float64)

/* float/float */
XND_BINARY(add, float32, float32, float32)
XND_BINARY(add, float32, float64, float64)
XND_BINARY(add, float64, float32, float64)
XND_BINARY(add, float64, float64, float64)


/****************************************************************************/
/*                                Multiply                                  */
/****************************************************************************/

#define multiply(x, y) x * y
/* signed */
XND_BINARY(multiply, int8, int8, int8)
XND_BINARY(multiply, int8, int16, int16)
XND_BINARY(multiply, int8, int32, int32)
XND_BINARY(multiply, int8, int64, int64)

XND_BINARY(multiply, int16, int8, int16)
XND_BINARY(multiply, int32, int8, int32)
XND_BINARY(multiply, int64, int8, int64)

XND_BINARY(multiply, int16, int16, int16)
XND_BINARY(multiply, int16, int32, int32)
XND_BINARY(multiply, int16, int64, int64)
XND_BINARY(multiply, int32, int16, int32)
XND_BINARY(multiply, int64, int16, int64)

XND_BINARY(multiply, int32, int32, int32)
XND_BINARY(multiply, int32, int64, int64)
XND_BINARY(multiply, int64, int32, int64)

XND_BINARY(multiply, int64, int64, int64)

/* unsigned */
XND_BINARY(multiply, uint8, uint8, uint8)
XND_BINARY(multiply, uint8, uint16, uint16)
XND_BINARY(multiply, uint8, uint32, uint32)
XND_BINARY(multiply, uint8, uint64, uint64)
XND_BINARY(multiply, uint16, uint8, uint16)
XND_BINARY(multiply, uint32, uint8, uint32)
XND_BINARY(multiply, uint64, uint8, uint64)

XND_BINARY(multiply, uint16, uint16, uint16)
XND_BINARY(multiply, uint16, uint32, uint32)
XND_BINARY(multiply, uint16, uint64, uint64)
XND_BINARY(multiply, uint32, uint16, uint32)
XND_BINARY(multiply, uint64, uint16, uint64)

XND_BINARY(multiply, uint32, uint32, uint32)
XND_BINARY(multiply, uint32, uint64, uint64)
XND_BINARY(multiply, uint64, uint32, uint64)

XND_BINARY(multiply, uint64, uint64, uint64)

/* signed/float */
XND_BINARY(multiply, int8, float32, float32)
XND_BINARY(multiply, int8, float64, float64)
XND_BINARY(multiply, float32, int8, float32)
XND_BINARY(multiply, float64, int8, float64)

XND_BINARY(multiply, int16, float32, float32)
XND_BINARY(multiply, int16, float64, float64)
XND_BINARY(multiply, float32, int16, float32)
XND_BINARY(multiply, float64, int16, float64)

XND_BINARY(multiply, int32, float64, float64)
XND_BINARY(multiply, float64, int32, float64)

/* unsigned/float */
XND_BINARY(multiply, uint8, float32, float32)
XND_BINARY(multiply, uint8, float64, float64)
XND_BINARY(multiply, float32, uint8, float32)
XND_BINARY(multiply, float64, uint8, float64)

XND_BINARY(multiply, uint16, float32, float32)
XND_BINARY(multiply, uint16, float64, float64)
XND_BINARY(multiply, float32, uint16, float32)
XND_BINARY(multiply, float64, uint16, float64)

XND_BINARY(multiply, uint32, float64, float64)
XND_BINARY(multiply, float64, uint32, float64)

/* float/float */
XND_BINARY(multiply, float32, float32, float32)
XND_BINARY(multiply, float32, float64, float64)
XND_BINARY(multiply, float64, float32, float64)
XND_BINARY(multiply, float64, float64, float64)


static const gm_kernel_init_t kernels[] = {
  /***** Add *****/
  /* signed */
  XND_BINARY_INIT(add, int8, int8, int8),
  XND_BINARY_INIT(add, int8, int16, int16),
  XND_BINARY_INIT(add, int8, int32, int32),
  XND_BINARY_INIT(add, int8, int64, int64),

  XND_BINARY_INIT(add, int16, int8, int16),
  XND_BINARY_INIT(add, int32, int8, int32),
  XND_BINARY_INIT(add, int64, int8, int64),

  XND_BINARY_INIT(add, int16, int16, int16),
  XND_BINARY_INIT(add, int16, int32, int32),
  XND_BINARY_INIT(add, int16, int64, int64),
  XND_BINARY_INIT(add, int32, int16, int32),
  XND_BINARY_INIT(add, int64, int16, int64),

  XND_BINARY_INIT(add, int32, int32, int32),
  XND_BINARY_INIT(add, int32, int64, int64),
  XND_BINARY_INIT(add, int64, int32, int64),

  XND_BINARY_INIT(add, int64, int64, int64),

  /* unsigned */
  XND_BINARY_INIT(add, uint8, uint8, uint8),
  XND_BINARY_INIT(add, uint8, uint16, uint16),
  XND_BINARY_INIT(add, uint8, uint32, uint32),
  XND_BINARY_INIT(add, uint8, uint64, uint64),
  XND_BINARY_INIT(add, uint16, uint8, uint16),
  XND_BINARY_INIT(add, uint32, uint8, uint32),
  XND_BINARY_INIT(add, uint64, uint8, uint64),

  XND_BINARY_INIT(add, uint16, uint16, uint16),
  XND_BINARY_INIT(add, uint16, uint32, uint32),
  XND_BINARY_INIT(add, uint16, uint64, uint64),
  XND_BINARY_INIT(add, uint32, uint16, uint32),
  XND_BINARY_INIT(add, uint64, uint16, uint64),

  XND_BINARY_INIT(add, uint32, uint32, uint32),
  XND_BINARY_INIT(add, uint32, uint64, uint64),
  XND_BINARY_INIT(add, uint64, uint32, uint64),

  XND_BINARY_INIT(add, uint64, uint64, uint64),

  /* signed/float */
  XND_BINARY_INIT(add, int8, float32, float32),
  XND_BINARY_INIT(add, int8, float64, float64),
  XND_BINARY_INIT(add, float32, int8, float32),
  XND_BINARY_INIT(add, float64, int8, float64),

  XND_BINARY_INIT(add, int16, float32, float32),
  XND_BINARY_INIT(add, int16, float64, float64),
  XND_BINARY_INIT(add, float32, int16, float32),
  XND_BINARY_INIT(add, float64, int16, float64),

  XND_BINARY_INIT(add, int32, float64, float64),
  XND_BINARY_INIT(add, float64, int32, float64),

  /* unsigned/float */
  XND_BINARY_INIT(add, uint8, float32, float32),
  XND_BINARY_INIT(add, uint8, float64, float64),
  XND_BINARY_INIT(add, float32, uint8, float32),
  XND_BINARY_INIT(add, float64, uint8, float64),

  XND_BINARY_INIT(add, uint16, float32, float32),
  XND_BINARY_INIT(add, uint16, float64, float64),
  XND_BINARY_INIT(add, float32, uint16, float32),
  XND_BINARY_INIT(add, float64, uint16, float64),

  XND_BINARY_INIT(add, uint32, float64, float64),
  XND_BINARY_INIT(add, float64, uint32, float64),

  /* float/float */
  XND_BINARY_INIT(add, float32, float32, float32),
  XND_BINARY_INIT(add, float32, float64, float64),
  XND_BINARY_INIT(add, float64, float32, float64),
  XND_BINARY_INIT(add, float64, float64, float64),


  /***** Multiply *****/
  /* signed */
  XND_BINARY_INIT(multiply, int8, int8, int8),
  XND_BINARY_INIT(multiply, int8, int16, int16),
  XND_BINARY_INIT(multiply, int8, int32, int32),
  XND_BINARY_INIT(multiply, int8, int64, int64),

  XND_BINARY_INIT(multiply, int16, int8, int16),
  XND_BINARY_INIT(multiply, int32, int8, int32),
  XND_BINARY_INIT(multiply, int64, int8, int64),

  XND_BINARY_INIT(multiply, int16, int16, int16),
  XND_BINARY_INIT(multiply, int16, int32, int32),
  XND_BINARY_INIT(multiply, int16, int64, int64),
  XND_BINARY_INIT(multiply, int32, int16, int32),
  XND_BINARY_INIT(multiply, int64, int16, int64),

  XND_BINARY_INIT(multiply, int32, int32, int32),
  XND_BINARY_INIT(multiply, int32, int64, int64),
  XND_BINARY_INIT(multiply, int64, int32, int64),

  XND_BINARY_INIT(multiply, int64, int64, int64),

  /* unsigned */
  XND_BINARY_INIT(multiply, uint8, uint8, uint8),
  XND_BINARY_INIT(multiply, uint8, uint16, uint16),
  XND_BINARY_INIT(multiply, uint8, uint32, uint32),
  XND_BINARY_INIT(multiply, uint8, uint64, uint64),
  XND_BINARY_INIT(multiply, uint16, uint8, uint16),
  XND_BINARY_INIT(multiply, uint32, uint8, uint32),
  XND_BINARY_INIT(multiply, uint64, uint8, uint64),

  XND_BINARY_INIT(multiply, uint16, uint16, uint16),
  XND_BINARY_INIT(multiply, uint16, uint32, uint32),
  XND_BINARY_INIT(multiply, uint16, uint64, uint64),
  XND_BINARY_INIT(multiply, uint32, uint16, uint32),
  XND_BINARY_INIT(multiply, uint64, uint16, uint64),

  XND_BINARY_INIT(multiply, uint32, uint32, uint32),
  XND_BINARY_INIT(multiply, uint32, uint64, uint64),
  XND_BINARY_INIT(multiply, uint64, uint32, uint64),

  XND_BINARY_INIT(multiply, uint64, uint64, uint64),

  /* signed/float */
  XND_BINARY_INIT(multiply, int8, float32, float32),
  XND_BINARY_INIT(multiply, int8, float64, float64),
  XND_BINARY_INIT(multiply, float32, int8, float32),
  XND_BINARY_INIT(multiply, float64, int8, float64),

  XND_BINARY_INIT(multiply, int16, float32, float32),
  XND_BINARY_INIT(multiply, int16, float64, float64),
  XND_BINARY_INIT(multiply, float32, int16, float32),
  XND_BINARY_INIT(multiply, float64, int16, float64),

  XND_BINARY_INIT(multiply, int32, float64, float64),
  XND_BINARY_INIT(multiply, float64, int32, float64),

  /* unsigned/float */
  XND_BINARY_INIT(multiply, uint8, float32, float32),
  XND_BINARY_INIT(multiply, uint8, float64, float64),
  XND_BINARY_INIT(multiply, float32, uint8, float32),
  XND_BINARY_INIT(multiply, float64, uint8, float64),

  XND_BINARY_INIT(multiply, uint16, float32, float32),
  XND_BINARY_INIT(multiply, uint16, float64, float64),
  XND_BINARY_INIT(multiply, float32, uint16, float32),
  XND_BINARY_INIT(multiply, float64, uint16, float64),

  XND_BINARY_INIT(multiply, uint32, float64, float64),
  XND_BINARY_INIT(multiply, float64, uint32, float64),

  /* float/float */
  XND_BINARY_INIT(multiply, float32, float32, float32),
  XND_BINARY_INIT(multiply, float32, float64, float64),
  XND_BINARY_INIT(multiply, float64, float32, float64),
  XND_BINARY_INIT(multiply, float64, float64, float64),

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
        if (gm_add_kernel(tbl, k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

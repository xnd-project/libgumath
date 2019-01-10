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
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"
#include "common.h"
#include "cuda_device_unary.h"


/****************************************************************************/
/*                    Kernel locations for optimized lookup                 */
/****************************************************************************/

static int
id_kernel_location(const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);

    switch (t->tag) {
    case Bool: return 0;

    case Uint8: return 2;
    case Uint16: return 4;
    case Uint32: return 6;
    case Uint64: return 8;

    case Int8: return 10;
    case Int16: return 12;
    case Int32: return 14;
    case Int64: return 16;

    case BFloat16: return 18;
    case Float16: return 20;
    case Float32: return 22;
    case Float64: return 24;

    case Complex32: return 26;
    case Complex64: return 28;
    case Complex128: return 30;

    default:
        ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
        return -1;
    }
}

static int
invert_kernel_location(const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);

    switch (t->tag) {
    case Bool: return 0;

    case Uint8: return 2;
    case Uint16: return 4;
    case Uint32: return 6;
    case Uint64: return 8;

    case Int8: return 10;
    case Int16: return 12;
    case Int32: return 14;
    case Int64: return 16;

    default:
        ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
        return -1;
    }
}

static int
negative_kernel_location(const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);

    switch (t->tag) {
    case Uint8: return 0;
    case Uint16: return 2;
    case Uint32: return 4;

    case Int8: return 6;
    case Int16: return 8;
    case Int32: return 10;
    case Int64: return 12;

    case BFloat16: return 14;
    case Float16: return 16;
    case Float32: return 18;
    case Float64: return 20;

    case Complex32: return 22;
    case Complex64: return 24;
    case Complex128: return 26;

    default:
        ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
        return -1;
    }
}

static int
math_kernel_location(const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);

    switch (t->tag) {
    case Uint8: return 0;
    case Int8: return  2;
    case Float16: return 4;

    case BFloat16: return 6;

    case Uint16: return 8;
    case Int16: return 10;
    case Float32: return 12;

    case Uint32: return 14;
    case Int32: return 16;
    case Float64: return 18;

    case Complex32: return 20;
    case Complex64: return 22;
    case Complex128: return 24;

    default:
        ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
        return -1;
    }
}


/*****************************************************************************/
/*                         CUDA-specific unary macros                        */
/*****************************************************************************/

#define CUDA_UNARY_HOST(name, t0, t1) \
static int                                                                      \
gm_cuda_host_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                               \
    const char *in0 = apply_index(&stack[0]);                                   \
    char *out = apply_index(&stack[1]);                                         \
    int64_t N = xnd_fixed_shape(&stack[0]);                                     \
    (void)ctx;                                                                  \
                                                                                \
    gm_cuda_device_fixed_1D_C_##name##_##t0##_##t1(in0, out, N);                \
                                                                                \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                            \
        unary_update_bitmap1D(stack);                                           \
    }                                                                           \
                                                                                \
    return 0;                                                                   \
}

#define CUDA_NOIMPL_HOST(name, t0, t1) \
static int                                                                      \
gm_cuda_host_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                               \
    (void)stack;                                                                \
                                                                                \
    ndt_err_format(ctx, NDT_NotImplementedError,                                \
        "implementation for " STRINGIZE(name) " : "                             \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                      \
        " currently requires double rounding");                                 \
                                                                                \
    return -1;                                                                  \
}

#define CUDA_UNARY_HOST_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                \
    .sig = "... * " STRINGIZE(t0) " -> ... * " STRINGIZE(t1),   \
    .Opt = gm_cuda_host_fixed_1D_C_##func##_##t0##_##t1,        \
    .C = NULL },                                                \
                                                                \
  { .name = STRINGIZE(funcname),                                \
    .sig = "... * ?" STRINGIZE(t0) " -> ... * ?" STRINGIZE(t1), \
    .Opt = gm_cuda_host_fixed_1D_C_##func##_##t0##_##t1,        \
    .C = NULL }


#undef bool
#define bool_t _Bool


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

CUDA_UNARY_HOST(copy, bool, bool)

CUDA_UNARY_HOST(copy, uint8, uint8)
CUDA_UNARY_HOST(copy, uint16, uint16)
CUDA_UNARY_HOST(copy, uint32, uint32)
CUDA_UNARY_HOST(copy, uint64, uint64)

CUDA_UNARY_HOST(copy, int8, int8)
CUDA_UNARY_HOST(copy, int16, int16)
CUDA_UNARY_HOST(copy, int32, int32)
CUDA_UNARY_HOST(copy, int64, int64)

CUDA_UNARY_HOST(copy, bfloat16, bfloat16)
CUDA_UNARY_HOST(copy, float16, float16)
CUDA_UNARY_HOST(copy, float32, float32)
CUDA_UNARY_HOST(copy, float64, float64)

CUDA_NOIMPL_HOST(copy, complex32, complex32)
CUDA_UNARY_HOST(copy, complex64, complex64)
CUDA_UNARY_HOST(copy, complex128, complex128)


static const gm_kernel_init_t unary_id[] = {
  /* COPY */
  CUDA_UNARY_HOST_INIT(copy, copy, bool, bool),

  CUDA_UNARY_HOST_INIT(copy, copy, uint8, uint8),
  CUDA_UNARY_HOST_INIT(copy, copy, uint16, uint16),
  CUDA_UNARY_HOST_INIT(copy, copy, uint32, uint32),
  CUDA_UNARY_HOST_INIT(copy, copy, uint64, uint64),

  CUDA_UNARY_HOST_INIT(copy, copy, int8, int8),
  CUDA_UNARY_HOST_INIT(copy, copy, int16, int16),
  CUDA_UNARY_HOST_INIT(copy, copy, int32, int32),
  CUDA_UNARY_HOST_INIT(copy, copy, int64, int64),

  CUDA_UNARY_HOST_INIT(copy, copy, bfloat16, bfloat16),
  CUDA_UNARY_HOST_INIT(copy, copy, float16, float16),
  CUDA_UNARY_HOST_INIT(copy, copy, float32, float32),
  CUDA_UNARY_HOST_INIT(copy, copy, float64, float64),

  CUDA_UNARY_HOST_INIT(copy, copy, complex32, complex32),
  CUDA_UNARY_HOST_INIT(copy, copy, complex64, complex64),
  CUDA_UNARY_HOST_INIT(copy, copy, complex128, complex128),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                Bitwise NOT                                */
/*****************************************************************************/

CUDA_UNARY_HOST(invert, bool, bool)

CUDA_UNARY_HOST(invert, uint8, uint8)
CUDA_UNARY_HOST(invert, uint16, uint16)
CUDA_UNARY_HOST(invert, uint32, uint32)
CUDA_UNARY_HOST(invert, uint64, uint64)

CUDA_UNARY_HOST(invert, int8, int8)
CUDA_UNARY_HOST(invert, int16, int16)
CUDA_UNARY_HOST(invert, int32, int32)
CUDA_UNARY_HOST(invert, int64, int64)


static const gm_kernel_init_t unary_invert[] = {
  /* INVERT */
  CUDA_UNARY_HOST_INIT(invert, invert, bool, bool),

  CUDA_UNARY_HOST_INIT(invert, invert, uint8, uint8),
  CUDA_UNARY_HOST_INIT(invert, invert, uint16, uint16),
  CUDA_UNARY_HOST_INIT(invert, invert, uint32, uint32),
  CUDA_UNARY_HOST_INIT(invert, invert, uint64, uint64),

  CUDA_UNARY_HOST_INIT(invert, invert, int8, int8),
  CUDA_UNARY_HOST_INIT(invert, invert, int16, int16),
  CUDA_UNARY_HOST_INIT(invert, invert, int32, int32),
  CUDA_UNARY_HOST_INIT(invert, invert, int64, int64),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                  Negative                                 */
/*****************************************************************************/

CUDA_UNARY_HOST(negative, uint8, int16)
CUDA_UNARY_HOST(negative, uint16, int32)
CUDA_UNARY_HOST(negative, uint32, int64)

CUDA_UNARY_HOST(negative, int8, int8)
CUDA_UNARY_HOST(negative, int16, int16)
CUDA_UNARY_HOST(negative, int32, int32)
CUDA_UNARY_HOST(negative, int64, int64)

CUDA_UNARY_HOST(negative, bfloat16, bfloat16)
CUDA_UNARY_HOST(negative, float16, float16)
CUDA_UNARY_HOST(negative, float32, float32)
CUDA_UNARY_HOST(negative, float64, float64)

CUDA_NOIMPL_HOST(negative, complex32, complex32)
CUDA_UNARY_HOST(negative, complex64, complex64)
CUDA_UNARY_HOST(negative, complex128, complex128)


static const gm_kernel_init_t unary_negative[] = {
  /* NEGATIVE */
  CUDA_UNARY_HOST_INIT(negative, negative, uint8, int16),
  CUDA_UNARY_HOST_INIT(negative, negative, uint16, int32),
  CUDA_UNARY_HOST_INIT(negative, negative, uint32, int64),

  CUDA_UNARY_HOST_INIT(negative, negative, int8, int8),
  CUDA_UNARY_HOST_INIT(negative, negative, int16, int16),
  CUDA_UNARY_HOST_INIT(negative, negative, int32, int32),
  CUDA_UNARY_HOST_INIT(negative, negative, int64, int64),

  CUDA_UNARY_HOST_INIT(negative, negative, bfloat16, bfloat16),
  CUDA_UNARY_HOST_INIT(negative, negative, float16, float16),
  CUDA_UNARY_HOST_INIT(negative, negative, float32, float32),
  CUDA_UNARY_HOST_INIT(negative, negative, float64, float64),

  CUDA_UNARY_HOST_INIT(negative, negative, complex32, complex32),
  CUDA_UNARY_HOST_INIT(negative, negative, complex64, complex64),
  CUDA_UNARY_HOST_INIT(negative, negative, complex128, complex128),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Math                                   */
/*****************************************************************************/

#define _CUDA_ALL_HALF_MATH(name) \
    CUDA_UNARY_HOST(name##f16, uint8, float16)   \
    CUDA_UNARY_HOST(name##f16, int8, float16)    \
    CUDA_UNARY_HOST(name##f16, float16, float16)

#define _CUDA_ALL_HALF_MATH_NOIMPL(name) \
    CUDA_NOIMPL_HOST(name##f16, uint8, float16)   \
    CUDA_NOIMPL_HOST(name##f16, int8, float16)    \
    CUDA_NOIMPL_HOST(name##f16, float16, float16)

#define _CUDA_ALL_COMPLEX_MATH(name) \
    CUDA_NOIMPL_HOST(name, complex32, complex32)  \
    CUDA_UNARY_HOST(name, complex64, complex64)   \
    CUDA_UNARY_HOST(name, complex128, complex128)

#define _CUDA_ALL_COMPLEX_MATH_NOIMPL(name) \
    CUDA_NOIMPL_HOST(name, complex32, complex32)   \
    CUDA_NOIMPL_HOST(name, complex64, complex64)   \
    CUDA_NOIMPL_HOST(name, complex128, complex128)

#define _CUDA_ALL_REAL_MATH(name) \
    CUDA_UNARY_HOST(name##b16, bfloat16, bfloat16) \
    CUDA_UNARY_HOST(name##f, uint16, float32)      \
    CUDA_UNARY_HOST(name##f, int16, float32)       \
    CUDA_UNARY_HOST(name##f, float32, float32)     \
    CUDA_UNARY_HOST(name, uint32, float64)         \
    CUDA_UNARY_HOST(name, int32, float64)          \
    CUDA_UNARY_HOST(name, float64, float64)        \

#define CUDA_ALL_REAL_MATH(name) \
    _CUDA_ALL_HALF_MATH_NOIMPL(name)    \
    _CUDA_ALL_REAL_MATH(name)           \
    _CUDA_ALL_COMPLEX_MATH_NOIMPL(name)

#define CUDA_ALL_REAL_MATH_WITH_HALF(name) \
    _CUDA_ALL_HALF_MATH(name)              \
    _CUDA_ALL_REAL_MATH(name)              \
    _CUDA_ALL_COMPLEX_MATH_NOIMPL(name)

#define CUDA_ALL_COMPLEX_MATH(name) \
    _CUDA_ALL_HALF_MATH_NOIMPL(name) \
    _CUDA_ALL_REAL_MATH(name)        \
    _CUDA_ALL_COMPLEX_MATH(name)

#define CUDA_ALL_COMPLEX_MATH_WITH_HALF(name) \
    _CUDA_ALL_HALF_MATH(name)                 \
    _CUDA_ALL_REAL_MATH(name)                 \
    _CUDA_ALL_COMPLEX_MATH(name)              \


#define CUDA_ALL_UNARY_MATH_INIT(name) \
    CUDA_UNARY_HOST_INIT(name, name##f16, uint8, float16),     \
    CUDA_UNARY_HOST_INIT(name, name##f16, int8, float16),      \
    CUDA_UNARY_HOST_INIT(name, name##f16, float16, float16),   \
                                                               \
    CUDA_UNARY_HOST_INIT(name, name##b16, bfloat16, bfloat16), \
                                                               \
    CUDA_UNARY_HOST_INIT(name, name##f, uint16, float32),      \
    CUDA_UNARY_HOST_INIT(name, name##f, int16, float32),       \
    CUDA_UNARY_HOST_INIT(name, name##f, float32, float32),     \
                                                               \
    CUDA_UNARY_HOST_INIT(name, name, uint32, float64),         \
    CUDA_UNARY_HOST_INIT(name, name, int32, float64),          \
    CUDA_UNARY_HOST_INIT(name, name, float64, float64),        \
                                                               \
    CUDA_UNARY_HOST_INIT(name, name, complex32, complex32),    \
    CUDA_UNARY_HOST_INIT(name, name, complex64, complex64),    \
    CUDA_UNARY_HOST_INIT(name, name, complex128, complex128)


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CUDA_ALL_REAL_MATH_WITH_HALF(fabs)


/*****************************************************************************/
/*                           Exponential functions                           */
/*****************************************************************************/

CUDA_ALL_COMPLEX_MATH_WITH_HALF(exp)
CUDA_ALL_REAL_MATH_WITH_HALF(exp2)
CUDA_ALL_REAL_MATH(expm1)


/*****************************************************************************/
/*                             Logarithm functions                           */
/*****************************************************************************/

CUDA_ALL_COMPLEX_MATH_WITH_HALF(log)
CUDA_ALL_COMPLEX_MATH_WITH_HALF(log10)
CUDA_ALL_REAL_MATH_WITH_HALF(log2)
CUDA_ALL_REAL_MATH(log1p)
CUDA_ALL_REAL_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CUDA_ALL_COMPLEX_MATH_WITH_HALF(sqrt)
CUDA_ALL_REAL_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CUDA_ALL_COMPLEX_MATH_WITH_HALF(sin)
CUDA_ALL_COMPLEX_MATH_WITH_HALF(cos)
CUDA_ALL_COMPLEX_MATH(tan)
CUDA_ALL_COMPLEX_MATH(asin)
CUDA_ALL_COMPLEX_MATH(acos)
CUDA_ALL_COMPLEX_MATH(atan)


/*****************************************************************************/
/*                            Hyperbolic functions                           */
/*****************************************************************************/

CUDA_ALL_COMPLEX_MATH(sinh)
CUDA_ALL_COMPLEX_MATH(cosh)
CUDA_ALL_COMPLEX_MATH(tanh)
CUDA_ALL_COMPLEX_MATH(asinh)
CUDA_ALL_COMPLEX_MATH(acosh)
CUDA_ALL_COMPLEX_MATH(atanh)


/*****************************************************************************/
/*                          Error and gamma functions                        */
/*****************************************************************************/

CUDA_ALL_REAL_MATH(erf)
CUDA_ALL_REAL_MATH(erfc)
CUDA_ALL_REAL_MATH(lgamma)
CUDA_ALL_REAL_MATH(tgamma)


/*****************************************************************************/
/*                           Ceiling, floor, trunc                           */
/*****************************************************************************/

CUDA_ALL_REAL_MATH(ceil)
CUDA_ALL_REAL_MATH(floor)
CUDA_ALL_REAL_MATH(trunc)
CUDA_ALL_REAL_MATH(round)
CUDA_ALL_REAL_MATH(nearbyint)


static const gm_kernel_init_t unary_float[] = {
  /* ABS */
  CUDA_ALL_UNARY_MATH_INIT(fabs),

  /* EXPONENTIAL */
  CUDA_ALL_UNARY_MATH_INIT(exp),
  CUDA_ALL_UNARY_MATH_INIT(exp2),
  CUDA_ALL_UNARY_MATH_INIT(expm1),

  /* LOGARITHM */
  CUDA_ALL_UNARY_MATH_INIT(log),
  CUDA_ALL_UNARY_MATH_INIT(log2),
  CUDA_ALL_UNARY_MATH_INIT(log10),
  CUDA_ALL_UNARY_MATH_INIT(log1p),
  CUDA_ALL_UNARY_MATH_INIT(logb),

  /* POWER */
  CUDA_ALL_UNARY_MATH_INIT(sqrt),
  CUDA_ALL_UNARY_MATH_INIT(cbrt),

  /* TRIGONOMETRIC */
  CUDA_ALL_UNARY_MATH_INIT(sin),
  CUDA_ALL_UNARY_MATH_INIT(cos),
  CUDA_ALL_UNARY_MATH_INIT(tan),
  CUDA_ALL_UNARY_MATH_INIT(asin),
  CUDA_ALL_UNARY_MATH_INIT(acos),
  CUDA_ALL_UNARY_MATH_INIT(atan),

  /* HYPERBOLIC */
  CUDA_ALL_UNARY_MATH_INIT(sinh),
  CUDA_ALL_UNARY_MATH_INIT(cosh),
  CUDA_ALL_UNARY_MATH_INIT(tanh),
  CUDA_ALL_UNARY_MATH_INIT(asinh),
  CUDA_ALL_UNARY_MATH_INIT(acosh),
  CUDA_ALL_UNARY_MATH_INIT(atanh),

  /* ERROR AND GAMMA */
  CUDA_ALL_UNARY_MATH_INIT(erf),
  CUDA_ALL_UNARY_MATH_INIT(erfc),
  CUDA_ALL_UNARY_MATH_INIT(lgamma),
  CUDA_ALL_UNARY_MATH_INIT(tgamma),

  /* CEILING, FLOOR, TRUNC */
  CUDA_ALL_UNARY_MATH_INIT(ceil),
  CUDA_ALL_UNARY_MATH_INIT(floor),
  CUDA_ALL_UNARY_MATH_INIT(trunc),
  CUDA_ALL_UNARY_MATH_INIT(round),
  CUDA_ALL_UNARY_MATH_INIT(nearbyint),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                         Initialize kernel table                          */
/****************************************************************************/

static const gm_kernel_set_t *
unary_id_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                   const ndt_t *types[], const int64_t li[], int nin, int nout,
                   ndt_context_t *ctx)
{
    return cuda_unary_typecheck(id_kernel_location, spec, f, types, li, nin, nout, ctx);
}

static const gm_kernel_set_t *
unary_invert_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                       const ndt_t *types[], const int64_t li[], int nin, int nout,
                       ndt_context_t *ctx)
{
    return cuda_unary_typecheck(invert_kernel_location, spec, f, types, li, nin, nout, ctx);
}

static const gm_kernel_set_t *
unary_negative_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                       const ndt_t *types[], const int64_t li[], int nin, int nout,
                       ndt_context_t *ctx)
{
    return cuda_unary_typecheck(negative_kernel_location, spec, f, types, li, nin, nout, ctx);
}

static const gm_kernel_set_t *
unary_math_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                      const ndt_t *types[], const int64_t li[], int nin, int nout,
                      ndt_context_t *ctx)
{
    return cuda_unary_typecheck(math_kernel_location, spec, f, types, li, nin, nout, ctx);
}

int
gm_init_cuda_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = unary_id; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_id_typecheck) < 0) {
             return -1;
        }
    }

    for (k = unary_invert; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_invert_typecheck) < 0) {
             return -1;
        }
    }

    for (k = unary_negative; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_negative_typecheck) < 0) {
             return -1;
        }
    }

    for (k = unary_float; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_math_typecheck) < 0) {
            return -1;
        }
    }

    return 0;
}

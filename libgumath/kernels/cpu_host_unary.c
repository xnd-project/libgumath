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
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"
#include "common.h"
#include "cpu_device_unary.h"


/****************************************************************************/
/*                    Kernel locations for optimized lookup                 */
/****************************************************************************/

static int
id_kernel_location(const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);

    switch (t->tag) {
    case Bool: return 0;

    case Uint8: return 4;
    case Uint16: return 8;
    case Uint32: return 12;
    case Uint64: return 16;

    case Int8: return 20;
    case Int16: return 24;
    case Int32: return 28;
    case Int64: return 32;

    case BFloat16: return 36;
    case Float16: return 40;
    case Float32: return 44;
    case Float64: return 48;

    case Complex32: return 52;
    case Complex64: return 56;
    case Complex128: return 60;

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

    case Uint8: return 4;
    case Uint16: return 8;
    case Uint32: return 12;
    case Uint64: return 16;

    case Int8: return 20;
    case Int16: return 24;
    case Int32: return 28;
    case Int64: return 32;

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
    case Uint16: return 4;
    case Uint32: return 8;

    case Int8: return 12;
    case Int16: return 16;
    case Int32: return 20;
    case Int64: return 24;

    case BFloat16: return 28;
    case Float16: return 32;
    case Float32: return 36;
    case Float64: return 40;

    case Complex32: return 44;
    case Complex64: return 48;
    case Complex128: return 52;

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
    case Int8: return 4;
    case Float16: return 8;

    case BFloat16: return 12;

    case Uint16: return 16;
    case Int16: return 20;
    case Float32: return 24;

    case Uint32: return 28;
    case Int32: return 32;
    case Float64: return 36;

    case Complex32: return 40;
    case Complex64: return 44;
    case Complex128: return 48;

    default:
        ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
        return -1;
    }
}


/*****************************************************************************/
/*                          CPU-specific unary macros                        */
/*****************************************************************************/

#define CPU_UNARY_HOST(name, t0, t1) \
static int                                                                \
gm_cpu_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                         \
    const char *in0 = apply_index(&stack[0]);                             \
    char *out = apply_index(&stack[1]);                                   \
    int64_t N = xnd_fixed_shape(&stack[0]);                               \
    (void)ctx;                                                            \
                                                                          \
    gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1(in0, out, N);           \
                                                                          \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                      \
        unary_update_bitmap1D(stack);                                     \
    }                                                                     \
                                                                          \
    return 0;                                                             \
}                                                                         \
                                                                          \
static int                                                                \
gm_cpu_0D_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                         \
    const char *in0 = stack[0].ptr;                                       \
    char *out = stack[1].ptr;                                             \
    (void)ctx;                                                            \
                                                                          \
    gm_cpu_device_0D_##name##_##t0##_##t1(in0, out);                      \
                                                                          \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                      \
        unary_update_bitmap(stack);                                       \
    }                                                                     \
                                                                          \
    return 0;                                                             \
}

#define CPU_NOIMPL_HOST(name, t0, t1) \
static int                                                                \
gm_cpu_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                         \
    (void)stack;                                                          \
                                                                          \
    ndt_err_format(ctx, NDT_NotImplementedError,                          \
        "implementation for " STRINGIZE(name) " : "                       \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                \
        " currently requires double rounding");                           \
                                                                          \
    return -1;                                                            \
}                                                                         \
                                                                          \
static int                                                                \
gm_cpu_0D_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                         \
    (void)stack;                                                          \
                                                                          \
    ndt_err_format(ctx, NDT_NotImplementedError,                          \
        "implementation for " STRINGIZE(name) " : "                       \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                \
        " currently requires double rounding");                           \
                                                                          \
    return -1;                                                            \
}


#define CPU_UNARY_HOST_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "... * " STRINGIZE(t0) " -> ... * " STRINGIZE(t1),         \
    .Opt = gm_cpu_fixed_1D_C_##func##_##t0##_##t1,                    \
    .C = gm_cpu_0D_##func##_##t0##_##t1 },                            \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "... * ?" STRINGIZE(t0) " -> ... * ?" STRINGIZE(t1),       \
    .Opt = gm_cpu_fixed_1D_C_##func##_##t0##_##t1,                    \
    .C = gm_cpu_0D_##func##_##t0##_##t1 },                            \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "var... * " STRINGIZE(t0) " -> var... * " STRINGIZE(t1),   \
    .C = gm_cpu_0D_##func##_##t0##_##t1 },                            \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "var... * ?" STRINGIZE(t0) " -> var... * ?" STRINGIZE(t1), \
    .C = gm_cpu_0D_##func##_##t0##_##t1 }


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#undef bool
#define bool_t _Bool

#define copy(x) x
CPU_UNARY_HOST(copy, bool, bool)

CPU_UNARY_HOST(copy, uint8, uint8)
CPU_UNARY_HOST(copy, uint16, uint16)
CPU_UNARY_HOST(copy, uint32, uint32)
CPU_UNARY_HOST(copy, uint64, uint64)

CPU_UNARY_HOST(copy, int8, int8)
CPU_UNARY_HOST(copy, int16, int16)
CPU_UNARY_HOST(copy, int32, int32)
CPU_UNARY_HOST(copy, int64, int64)

CPU_UNARY_HOST(copy, bfloat16, bfloat16)
CPU_NOIMPL_HOST(copy, float16, float16)
CPU_UNARY_HOST(copy, float32, float32)
CPU_UNARY_HOST(copy, float64, float64)

CPU_NOIMPL_HOST(copy, complex32, complex32)
CPU_UNARY_HOST(copy, complex64, complex64)
CPU_UNARY_HOST(copy, complex128, complex128)


static const gm_kernel_init_t unary_id[] = {
  /* COPY */
  CPU_UNARY_HOST_INIT(copy, copy, bool, bool),

  CPU_UNARY_HOST_INIT(copy, copy, uint8, uint8),
  CPU_UNARY_HOST_INIT(copy, copy, uint16, uint16),
  CPU_UNARY_HOST_INIT(copy, copy, uint32, uint32),
  CPU_UNARY_HOST_INIT(copy, copy, uint64, uint64),

  CPU_UNARY_HOST_INIT(copy, copy, int8, int8),
  CPU_UNARY_HOST_INIT(copy, copy, int16, int16),
  CPU_UNARY_HOST_INIT(copy, copy, int32, int32),
  CPU_UNARY_HOST_INIT(copy, copy, int64, int64),

  CPU_UNARY_HOST_INIT(copy, copy, bfloat16, bfloat16),
  CPU_UNARY_HOST_INIT(copy, copy, float16, float16),
  CPU_UNARY_HOST_INIT(copy, copy, float32, float32),
  CPU_UNARY_HOST_INIT(copy, copy, float64, float64),

  CPU_UNARY_HOST_INIT(copy, copy, complex32, complex32),
  CPU_UNARY_HOST_INIT(copy, copy, complex64, complex64),
  CPU_UNARY_HOST_INIT(copy, copy, complex128, complex128),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                Bitwise NOT                                */
/*****************************************************************************/

#define invert(x) !x
CPU_UNARY_HOST(invert, bool, bool)

#undef invert
#define invert(x) ~x
CPU_UNARY_HOST(invert, uint8, uint8)
CPU_UNARY_HOST(invert, uint16, uint16)
CPU_UNARY_HOST(invert, uint32, uint32)
CPU_UNARY_HOST(invert, uint64, uint64)

CPU_UNARY_HOST(invert, int8, int8)
CPU_UNARY_HOST(invert, int16, int16)
CPU_UNARY_HOST(invert, int32, int32)
CPU_UNARY_HOST(invert, int64, int64)


static const gm_kernel_init_t unary_invert[] = {
  /* INVERT */
  CPU_UNARY_HOST_INIT(invert, invert, bool, bool),

  CPU_UNARY_HOST_INIT(invert, invert, uint8, uint8),
  CPU_UNARY_HOST_INIT(invert, invert, uint16, uint16),
  CPU_UNARY_HOST_INIT(invert, invert, uint32, uint32),
  CPU_UNARY_HOST_INIT(invert, invert, uint64, uint64),

  CPU_UNARY_HOST_INIT(invert, invert, int8, int8),
  CPU_UNARY_HOST_INIT(invert, invert, int16, int16),
  CPU_UNARY_HOST_INIT(invert, invert, int32, int32),
  CPU_UNARY_HOST_INIT(invert, invert, int64, int64),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                  Negative                                 */
/*****************************************************************************/

#define negative(x) -x
CPU_UNARY_HOST(negative, uint8, int16)
CPU_UNARY_HOST(negative, uint16, int32)
CPU_UNARY_HOST(negative, uint32, int64)

CPU_UNARY_HOST(negative, int8, int8)
CPU_UNARY_HOST(negative, int16, int16)
CPU_UNARY_HOST(negative, int32, int32)
CPU_UNARY_HOST(negative, int64, int64)

CPU_UNARY_HOST(negative, bfloat16, bfloat16)
CPU_NOIMPL_HOST(negative, float16, float16)
CPU_UNARY_HOST(negative, float32, float32)
CPU_UNARY_HOST(negative, float64, float64)

CPU_NOIMPL_HOST(negative, complex32, complex32)
CPU_UNARY_HOST(negative, complex64, complex64)
CPU_UNARY_HOST(negative, complex128, complex128)


static const gm_kernel_init_t unary_negative[] = {
  /* NEGATIVE */
  CPU_UNARY_HOST_INIT(negative, negative, uint8, int16),
  CPU_UNARY_HOST_INIT(negative, negative, uint16, int32),
  CPU_UNARY_HOST_INIT(negative, negative, uint32, int64),

  CPU_UNARY_HOST_INIT(negative, negative, int8, int8),
  CPU_UNARY_HOST_INIT(negative, negative, int16, int16),
  CPU_UNARY_HOST_INIT(negative, negative, int32, int32),
  CPU_UNARY_HOST_INIT(negative, negative, int64, int64),

  CPU_UNARY_HOST_INIT(negative, negative, bfloat16, bfloat16),
  CPU_UNARY_HOST_INIT(negative, negative, float16, float16),
  CPU_UNARY_HOST_INIT(negative, negative, float32, float32),
  CPU_UNARY_HOST_INIT(negative, negative, float64, float64),

  CPU_UNARY_HOST_INIT(negative, negative, complex32, complex32),
  CPU_UNARY_HOST_INIT(negative, negative, complex64, complex64),
  CPU_UNARY_HOST_INIT(negative, negative, complex128, complex128),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Math                                   */
/*****************************************************************************/

#define _CPU_ALL_HALF_MATH(name) \
    CPU_UNARY_HOST(name##f16, uint8, float16)   \
    CPU_UNARY_HOST(name##f16, int8, float16)    \
    CPU_UNARY_HOST(name##f16, float16, float16)

#define _CPU_ALL_HALF_MATH_NOIMPL(name) \
    CPU_NOIMPL_HOST(name##f16, uint8, float16)   \
    CPU_NOIMPL_HOST(name##f16, int8, float16)    \
    CPU_NOIMPL_HOST(name##f16, float16, float16)

#define _CPU_ALL_COMPLEX_MATH(name) \
    CPU_NOIMPL_HOST(name, complex32, complex32)  \
    CPU_UNARY_HOST(name, complex64, complex64)   \
    CPU_UNARY_HOST(name, complex128, complex128)

#define _CPU_ALL_COMPLEX_MATH_NOIMPL(name) \
    CPU_NOIMPL_HOST(name, complex32, complex32)   \
    CPU_NOIMPL_HOST(name, complex64, complex64)   \
    CPU_NOIMPL_HOST(name, complex128, complex128)

#define _CPU_ALL_REAL_MATH(name) \
    CPU_UNARY_HOST(name##b16, bfloat16, bfloat16) \
    CPU_UNARY_HOST(name##f, uint16, float32)      \
    CPU_UNARY_HOST(name##f, int16, float32)       \
    CPU_UNARY_HOST(name##f, float32, float32)     \
    CPU_UNARY_HOST(name, uint32, float64)         \
    CPU_UNARY_HOST(name, int32, float64)          \
    CPU_UNARY_HOST(name, float64, float64)        \

#define CPU_ALL_REAL_MATH(name) \
    _CPU_ALL_HALF_MATH_NOIMPL(name)    \
    _CPU_ALL_REAL_MATH(name)           \
    _CPU_ALL_COMPLEX_MATH_NOIMPL(name)

#define CPU_ALL_REAL_MATH_WITH_HALF(name) \
    _CPU_ALL_HALF_MATH(name)              \
    _CPU_ALL_REAL_MATH(name)              \
    _CPU_ALL_COMPLEX_MATH_NOIMPL(name)

#define CPU_ALL_COMPLEX_MATH(name) \
    _CPU_ALL_HALF_MATH_NOIMPL(name) \
    _CPU_ALL_REAL_MATH(name)        \
    _CPU_ALL_COMPLEX_MATH(name)

#define CPU_ALL_COMPLEX_MATH_WITH_HALF(name) \
    _CPU_ALL_HALF_MATH(name)                 \
    _CPU_ALL_REAL_MATH(name)                 \
    _CPU_ALL_COMPLEX_MATH(name)              \


#define CPU_ALL_UNARY_MATH_INIT(name) \
    CPU_UNARY_HOST_INIT(name, name##f16, uint8, float16),     \
    CPU_UNARY_HOST_INIT(name, name##f16, int8, float16),      \
    CPU_UNARY_HOST_INIT(name, name##f16, float16, float16),   \
                                                              \
    CPU_UNARY_HOST_INIT(name, name##b16, bfloat16, bfloat16), \
                                                              \
    CPU_UNARY_HOST_INIT(name, name##f, uint16, float32),      \
    CPU_UNARY_HOST_INIT(name, name##f, int16, float32),       \
    CPU_UNARY_HOST_INIT(name, name##f, float32, float32),     \
                                                              \
    CPU_UNARY_HOST_INIT(name, name, uint32, float64),         \
    CPU_UNARY_HOST_INIT(name, name, int32, float64),          \
    CPU_UNARY_HOST_INIT(name, name, float64, float64),        \
                                                              \
    CPU_UNARY_HOST_INIT(name, name, complex32, complex32),    \
    CPU_UNARY_HOST_INIT(name, name, complex64, complex64),    \
    CPU_UNARY_HOST_INIT(name, name, complex128, complex128)


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CPU_ALL_REAL_MATH(fabs)


/*****************************************************************************/
/*                           Exponential functions                           */
/*****************************************************************************/

CPU_ALL_COMPLEX_MATH(exp)
CPU_ALL_REAL_MATH(exp2)
CPU_ALL_REAL_MATH(expm1)


/*****************************************************************************/
/*                             Logarithm functions                           */
/*****************************************************************************/

CPU_ALL_COMPLEX_MATH(log)
CPU_ALL_COMPLEX_MATH(log10)
CPU_ALL_REAL_MATH(log2)
CPU_ALL_REAL_MATH(log1p)
CPU_ALL_REAL_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CPU_ALL_COMPLEX_MATH(sqrt)
CPU_ALL_REAL_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CPU_ALL_COMPLEX_MATH(sin)
CPU_ALL_COMPLEX_MATH(cos)
CPU_ALL_COMPLEX_MATH(tan)
CPU_ALL_COMPLEX_MATH(asin)
CPU_ALL_COMPLEX_MATH(acos)
CPU_ALL_COMPLEX_MATH(atan)


/*****************************************************************************/
/*                            Hyperbolic functions                           */
/*****************************************************************************/

CPU_ALL_COMPLEX_MATH(sinh)
CPU_ALL_COMPLEX_MATH(cosh)
CPU_ALL_COMPLEX_MATH(tanh)
CPU_ALL_COMPLEX_MATH(asinh)
CPU_ALL_COMPLEX_MATH(acosh)
CPU_ALL_COMPLEX_MATH(atanh)


/*****************************************************************************/
/*                          Error and gamma functions                        */
/*****************************************************************************/

CPU_ALL_REAL_MATH(erf)
CPU_ALL_REAL_MATH(erfc)
CPU_ALL_REAL_MATH(lgamma)
CPU_ALL_REAL_MATH(tgamma)


/*****************************************************************************/
/*                           Ceiling, floor, trunc                           */
/*****************************************************************************/

CPU_ALL_REAL_MATH(ceil)
CPU_ALL_REAL_MATH(floor)
CPU_ALL_REAL_MATH(trunc)
CPU_ALL_REAL_MATH(round)
CPU_ALL_REAL_MATH(nearbyint)


static const gm_kernel_init_t unary_float[] = {
  /* ABS */
  CPU_ALL_UNARY_MATH_INIT(fabs),

  /* EXPONENTIAL */
  CPU_ALL_UNARY_MATH_INIT(exp),
  CPU_ALL_UNARY_MATH_INIT(exp2),
  CPU_ALL_UNARY_MATH_INIT(expm1),

  /* LOGARITHM */
  CPU_ALL_UNARY_MATH_INIT(log),
  CPU_ALL_UNARY_MATH_INIT(log2),
  CPU_ALL_UNARY_MATH_INIT(log10),
  CPU_ALL_UNARY_MATH_INIT(log1p),
  CPU_ALL_UNARY_MATH_INIT(logb),

  /* POWER */
  CPU_ALL_UNARY_MATH_INIT(sqrt),
  CPU_ALL_UNARY_MATH_INIT(cbrt),

  /* TRIGONOMETRIC */
  CPU_ALL_UNARY_MATH_INIT(sin),
  CPU_ALL_UNARY_MATH_INIT(cos),
  CPU_ALL_UNARY_MATH_INIT(tan),
  CPU_ALL_UNARY_MATH_INIT(asin),
  CPU_ALL_UNARY_MATH_INIT(acos),
  CPU_ALL_UNARY_MATH_INIT(atan),

  /* HYPERBOLIC */
  CPU_ALL_UNARY_MATH_INIT(sinh),
  CPU_ALL_UNARY_MATH_INIT(cosh),
  CPU_ALL_UNARY_MATH_INIT(tanh),
  CPU_ALL_UNARY_MATH_INIT(asinh),
  CPU_ALL_UNARY_MATH_INIT(acosh),
  CPU_ALL_UNARY_MATH_INIT(atanh),

  /* ERROR AND GAMMA */
  CPU_ALL_UNARY_MATH_INIT(erf),
  CPU_ALL_UNARY_MATH_INIT(erfc),
  CPU_ALL_UNARY_MATH_INIT(lgamma),
  CPU_ALL_UNARY_MATH_INIT(tgamma),

  /* CEILING, FLOOR, TRUNC */
  CPU_ALL_UNARY_MATH_INIT(ceil),
  CPU_ALL_UNARY_MATH_INIT(floor),
  CPU_ALL_UNARY_MATH_INIT(trunc),
  CPU_ALL_UNARY_MATH_INIT(round),
  CPU_ALL_UNARY_MATH_INIT(nearbyint),

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
    return cpu_unary_typecheck(id_kernel_location, spec, f, types, li, nin, nout, ctx);
}

static const gm_kernel_set_t *
unary_invert_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                       const ndt_t *types[], const int64_t li[], int nin, int nout,
                       ndt_context_t *ctx)
{
    return cpu_unary_typecheck(invert_kernel_location, spec, f, types, li, nin, nout, ctx);
}

static const gm_kernel_set_t *
unary_negative_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                         const ndt_t *types[], const int64_t li[], int nin, int nout,
                         ndt_context_t *ctx)
{
    return cpu_unary_typecheck(negative_kernel_location, spec, f, types, li, nin, nout, ctx);
}

static const gm_kernel_set_t *
unary_math_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                     const ndt_t *types[], const int64_t li[], int nin, int nout,
                     ndt_context_t *ctx)
{
    return cpu_unary_typecheck(math_kernel_location, spec, f, types, li, nin, nout, ctx);
}

int
gm_init_cpu_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
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

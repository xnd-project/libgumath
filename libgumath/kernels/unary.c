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
/*                    Kernel locations for optimized lookup                 */
/****************************************************************************/

static int
id_kernel_location(const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);

    switch (t->tag) {
    case Bool: return 0;
    case Int8: return 4;
    case Int16: return 8;
    case Int32: return 12;
    case Int64: return 16;
    case Uint8: return 20;
    case Uint16: return 24;
    case Uint32: return 28;
    case Uint64: return 32;
    case Float32: return 36;
    case Float64: return 40;
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
    case Int8: return 4;
    case Int16: return 8;
    case Int32: return 12;
    case Int64: return 16;
    case Uint8: return 20;
    case Uint16: return 24;
    case Uint32: return 28;
    case Uint64: return 32;
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
    case Int8: return 0;
    case Int16: return 4;
    case Int32: return 8;
    case Int64: return 12;
    case Uint8: return 16;
    case Uint16: return 20;
    case Uint32: return 24;
    case Float32: return 28;
    case Float64: return 32;
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
    case Int8: return 0;
    case Int16: return 4;
    case Uint8: return 8;
    case Uint16: return 12;
    case Float32: return 16;
    case Int32: return 20;
    case Uint32: return 24;
    case Float64: return 28;
    default:
        ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
        return -1;
    }
}


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#undef bool
#define bool_t _Bool

#define copy(x) x
XND_UNARY(copy, bool, bool, bool)
XND_UNARY(copy, int8, int8, int8)
XND_UNARY(copy, int16, int16, int16)
XND_UNARY(copy, int32, int32, int32)
XND_UNARY(copy, int64, int64, int64)
XND_UNARY(copy, uint8, uint8, uint8)
XND_UNARY(copy, uint16, uint16, uint16)
XND_UNARY(copy, uint32, uint32, uint32)
XND_UNARY(copy, uint64, uint64, uint64)
XND_UNARY(copy, float32, float32, float32)
XND_UNARY(copy, float64, float64, float64)

static const gm_kernel_init_t unary_id[] = {
  /* COPY */
  XND_UNARY_INIT(copy, copy, bool, bool),
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

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                              Bitwise NOT                                  */
/*****************************************************************************/

#define invert(x) !x
XND_UNARY(invert, bool, bool, bool)
#undef invert
#define invert(x) ~x
XND_UNARY(invert, int8, int8, int8)
XND_UNARY(invert, int16, int16, int16)
XND_UNARY(invert, int32, int32, int32)
XND_UNARY(invert, int64, int64, int64)
XND_UNARY(invert, uint8, uint8, uint8)
XND_UNARY(invert, uint16, uint16, uint16)
XND_UNARY(invert, uint32, uint32, uint32)
XND_UNARY(invert, uint64, uint64, uint64)


static const gm_kernel_init_t unary_invert[] = {
  /* INVERT */
  XND_UNARY_INIT(invert, invert, bool, bool),
  XND_UNARY_INIT(invert, invert, int8, int8),
  XND_UNARY_INIT(invert, invert, int16, int16),
  XND_UNARY_INIT(invert, invert, int32, int32),
  XND_UNARY_INIT(invert, invert, int64, int64),
  XND_UNARY_INIT(invert, invert, uint8, uint8),
  XND_UNARY_INIT(invert, invert, uint16, uint16),
  XND_UNARY_INIT(invert, invert, uint32, uint32),
  XND_UNARY_INIT(invert, invert, uint64, uint64),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                               Negative                                    */
/*****************************************************************************/

#define negative(x) -x
XND_UNARY(negative, int8, int8, int8)
XND_UNARY(negative, int16, int16, int16)
XND_UNARY(negative, int32, int32, int32)
XND_UNARY(negative, int64, int64, int64)
XND_UNARY(negative, uint8, int16, int16)
XND_UNARY(negative, uint16, int32, int32)
XND_UNARY(negative, uint32, int64, int64)
XND_UNARY(negative, float32, float32, float32)
XND_UNARY(negative, float64, float64, float64)



static const gm_kernel_init_t unary_negative[] = {
  /* INVERT */
  XND_UNARY_INIT(negative, negative, int8, int8),
  XND_UNARY_INIT(negative, negative, int16, int16),
  XND_UNARY_INIT(negative, negative, int32, int32),
  XND_UNARY_INIT(negative, negative, int64, int64),
  XND_UNARY_INIT(negative, negative, uint8, int16),
  XND_UNARY_INIT(negative, negative, uint16, int32),
  XND_UNARY_INIT(negative, negative, uint32, int64),
  XND_UNARY_INIT(negative, negative, float32, float32),
  XND_UNARY_INIT(negative, negative, float64, float64),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Math                                   */
/*****************************************************************************/

#define XND_ALL_UNARY_MATH(name) \
    XND_UNARY(name##f, int8, float32, float32)    \
    XND_UNARY(name##f, int16, float32, float32)   \
    XND_UNARY(name##f, uint8, float32, float32)   \
    XND_UNARY(name##f, uint16, float32, float32)  \
    XND_UNARY(name##f, float32, float32, float32) \
    XND_UNARY(name, int32, float64, float64)      \
    XND_UNARY(name, uint32, float64, float64)     \
    XND_UNARY(name, float64, float64, float64)

#define XND_ALL_UNARY_MATH_INIT(name) \
    XND_UNARY_INIT(name, name##f, int8, float32),    \
    XND_UNARY_INIT(name, name##f, int16, float32),   \
    XND_UNARY_INIT(name, name##f, uint8, float32),   \
    XND_UNARY_INIT(name, name##f, uint16, float32),  \
    XND_UNARY_INIT(name, name##f, float32, float32), \
    XND_UNARY_INIT(name, name, uint32, float64),     \
    XND_UNARY_INIT(name, name, int32, float64),      \
    XND_UNARY_INIT(name, name, float64, float64)


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

XND_ALL_UNARY_MATH(fabs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

XND_ALL_UNARY_MATH(exp)
XND_ALL_UNARY_MATH(exp2)
XND_ALL_UNARY_MATH(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

XND_ALL_UNARY_MATH(log)
XND_ALL_UNARY_MATH(log2)
XND_ALL_UNARY_MATH(log10)
XND_ALL_UNARY_MATH(log1p)
XND_ALL_UNARY_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

XND_ALL_UNARY_MATH(sqrt)
XND_ALL_UNARY_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

XND_ALL_UNARY_MATH(sin)
XND_ALL_UNARY_MATH(cos)
XND_ALL_UNARY_MATH(tan)
XND_ALL_UNARY_MATH(asin)
XND_ALL_UNARY_MATH(acos)
XND_ALL_UNARY_MATH(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

XND_ALL_UNARY_MATH(sinh)
XND_ALL_UNARY_MATH(cosh)
XND_ALL_UNARY_MATH(tanh)
XND_ALL_UNARY_MATH(asinh)
XND_ALL_UNARY_MATH(acosh)
XND_ALL_UNARY_MATH(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

XND_ALL_UNARY_MATH(erf)
XND_ALL_UNARY_MATH(erfc)
XND_ALL_UNARY_MATH(lgamma)
XND_ALL_UNARY_MATH(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

XND_ALL_UNARY_MATH(ceil)
XND_ALL_UNARY_MATH(floor)
XND_ALL_UNARY_MATH(trunc)
XND_ALL_UNARY_MATH(round)
XND_ALL_UNARY_MATH(nearbyint)


static const gm_kernel_init_t unary_float[] = {
  /* ABS */
  XND_ALL_UNARY_MATH_INIT(fabs),

  /* EXPONENTIAL */
  XND_ALL_UNARY_MATH_INIT(exp),
  XND_ALL_UNARY_MATH_INIT(exp2),
  XND_ALL_UNARY_MATH_INIT(expm1),

  /* LOGARITHM */
  XND_ALL_UNARY_MATH_INIT(log),
  XND_ALL_UNARY_MATH_INIT(log2),
  XND_ALL_UNARY_MATH_INIT(log10),
  XND_ALL_UNARY_MATH_INIT(log1p),
  XND_ALL_UNARY_MATH_INIT(logb),

  /* POWER */
  XND_ALL_UNARY_MATH_INIT(sqrt),
  XND_ALL_UNARY_MATH_INIT(cbrt),

  /* TRIGONOMETRIC */
  XND_ALL_UNARY_MATH_INIT(sin),
  XND_ALL_UNARY_MATH_INIT(cos),
  XND_ALL_UNARY_MATH_INIT(tan),
  XND_ALL_UNARY_MATH_INIT(asin),
  XND_ALL_UNARY_MATH_INIT(acos),
  XND_ALL_UNARY_MATH_INIT(atan),

  /* HYPERBOLIC */
  XND_ALL_UNARY_MATH_INIT(sinh),
  XND_ALL_UNARY_MATH_INIT(cosh),
  XND_ALL_UNARY_MATH_INIT(tanh),
  XND_ALL_UNARY_MATH_INIT(asinh),
  XND_ALL_UNARY_MATH_INIT(acosh),
  XND_ALL_UNARY_MATH_INIT(atanh),

  /* ERROR AND GAMMA */
  XND_ALL_UNARY_MATH_INIT(erf),
  XND_ALL_UNARY_MATH_INIT(erfc),
  XND_ALL_UNARY_MATH_INIT(lgamma),
  XND_ALL_UNARY_MATH_INIT(tgamma),

  /* CEILING, FLOOR, TRUNC */
  XND_ALL_UNARY_MATH_INIT(ceil),
  XND_ALL_UNARY_MATH_INIT(floor),
  XND_ALL_UNARY_MATH_INIT(trunc),
  XND_ALL_UNARY_MATH_INIT(round),
  XND_ALL_UNARY_MATH_INIT(nearbyint),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

static const gm_kernel_set_t *
unary_id_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                   const ndt_t *in[], const int64_t li[], int nin,
                   ndt_context_t *ctx)
{
    return unary_typecheck(id_kernel_location, spec, f, in, li, nin, ctx);
}

static const gm_kernel_set_t *
unary_invert_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                       const ndt_t *in[], const int64_t li[], int nin,
                       ndt_context_t *ctx)
{
    return unary_typecheck(invert_kernel_location, spec, f, in, li, nin, ctx);
}

static const gm_kernel_set_t *
unary_negative_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                       const ndt_t *in[], const int64_t li[], int nin,
                       ndt_context_t *ctx)
{
    return unary_typecheck(negative_kernel_location, spec, f, in, li, nin, ctx);
}

static const gm_kernel_set_t *
unary_math_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                      const ndt_t *in[], const int64_t li[], int nin,
                      ndt_context_t *ctx)
{
    return unary_typecheck(math_kernel_location, spec, f, in, li, nin, ctx);
}

int
gm_init_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
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

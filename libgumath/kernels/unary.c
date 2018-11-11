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
/*                        Optimized dispatch (T -> T)                       */
/****************************************************************************/

/* Structured kernel locations for fast lookup. */
static const ndt_t *
infer_id_return(int *base, const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t, *dtype;
    enum ndt tag;

    switch (ndt_dtype(in)->tag) {
    case Bool: *base = 0; tag = Bool; break;
    case Int8: *base = 2; tag = Int8; break;
    case Int16: *base = 4; tag = Int16; break;
    case Int32: *base = 6; tag = Int32; break;
    case Int64: *base = 8; tag = Int64; break;
    case Uint8: *base = 10; tag = Uint8; break;
    case Uint16: *base = 12; tag = Uint16; break;
    case Uint32: *base = 14; tag = Uint32; break;
    case Uint64: *base = 16; tag = Uint64; break;
    case Float32: *base = 18; tag = Float32; break;
    case Float64: *base = 20; tag = Float64; break;
    default:
        ndt_err_format(ctx, NDT_RuntimeError, "invalid dtype");
        return NULL;
    }

    dtype = ndt_primitive(tag, 0, ctx);
    if (dtype == NULL) {
        return NULL;
    }

    t = ndt_copy_contiguous_dtype(in, dtype, ctx);
    ndt_decref(dtype);
    return t;
}

/* Structured kernel locations for fast lookup. */
static const ndt_t *
infer_invert_return(int *base, const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t, *dtype;
    enum ndt tag;

    switch (ndt_dtype(in)->tag) {
    case Bool: *base = 0; tag = Bool; break;
    case Int8: *base = 2; tag = Int8; break;
    case Int16: *base = 4; tag = Int16; break;
    case Int32: *base = 6; tag = Int32; break;
    case Int64: *base = 8; tag = Int64; break;
    case Uint8: *base = 10; tag = Uint8; break;
    case Uint16: *base = 12; tag = Uint16; break;
    case Uint32: *base = 14; tag = Uint32; break;
    case Uint64: *base = 16; tag = Uint64; break;
    default:
        ndt_err_format(ctx, NDT_RuntimeError, "invalid dtype");
        return NULL;
    }

    dtype = ndt_primitive(tag, 0, ctx);
    if (dtype == NULL) {
        return NULL;
    }

    t = ndt_copy_contiguous_dtype(in, dtype, ctx);
    ndt_decref(dtype);
    return t;
}


/****************************************************************************/
/*                   Optimized dispatch (float return values)               */
/****************************************************************************/

/* Structured kernel locations for fast lookup. */
static const ndt_t *
infer_float_return(int *base, const ndt_t *in, ndt_context_t *ctx)
{
    const ndt_t *t, *dtype;
    enum ndt tag;

    switch (ndt_dtype(in)->tag) {
    case Int8: *base = 0; tag = Float32; break;
    case Int16: *base = 2; tag = Float32; break;
    case Uint8: *base = 4; tag = Float32; break;
    case Uint16: *base = 6; tag = Float32; break;
    case Float32: *base = 8; tag = Float32; break;
    case Int32: *base = 10; tag = Float64; break;
    case Uint32: *base = 12; tag = Float64; break;
    case Float64: *base = 14; tag = Float64; break;
    default:
        ndt_err_format(ctx, NDT_RuntimeError, "invalid dtype");
        return NULL;
    }

    dtype = ndt_primitive(tag, 0, ctx);
    if (dtype == NULL) {
        return NULL;
    }

    t = ndt_copy_contiguous_dtype(in, dtype, ctx);
    ndt_decref(dtype);
    return t;
}


/****************************************************************************/
/*                             Optimized typecheck                          */
/****************************************************************************/

static const gm_kernel_set_t *
unary_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                const ndt_t *in[], int nin,
                const ndt_t *(*infer)(int *, const ndt_t *, ndt_context_t *),
                ndt_context_t *ctx)
{
    const ndt_t *t;
    int n;

    if (nin != 1) {
        ndt_err_format(ctx, NDT_ValueError,
            "invalid number of arguments for %s(x): expected 1, got %d",
            f->name, nin);
        return NULL;
    }
    t = in[0];
    assert(ndt_is_concrete(t));

    spec->out[0] = infer(&n, t, ctx);
    if (spec->out[0] == NULL) {
        return NULL;
    }
    spec->nout = 1;
    spec->nbroadcast = 0;

    switch (t->tag) {
    case FixedDim:
        spec->flags = NDT_C|NDT_STRIDED;
        spec->outer_dims = t->ndim;
        if (ndt_is_c_contiguous(ndt_dim_at(t, t->ndim-1))) {
            spec->flags |= NDT_ELEMWISE_1D;
        }
        return &f->kernels[n];
    case VarDim:
        spec->flags = NDT_C;
        spec->outer_dims = t->ndim;
        return &f->kernels[n+1];
    default:
        assert(t->ndim == 0);
        spec->flags = NDT_C|NDT_STRIDED;
        spec->outer_dims = 0;
        return &f->kernels[n];
    }
}

static const gm_kernel_set_t *
unary_id_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                   const ndt_t *in[], int nin,
                   ndt_context_t *ctx)
{
    return unary_typecheck(spec, f, in, nin, infer_id_return, ctx);
}

static const gm_kernel_set_t *
unary_invert_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                   const ndt_t *in[], int nin,
                   ndt_context_t *ctx)
{
    return unary_typecheck(spec, f, in, nin, infer_invert_return, ctx);
}

static const gm_kernel_set_t *
unary_float_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f,
                      const ndt_t *in[], int nin,
                      ndt_context_t *ctx)
{
    return unary_typecheck(spec, f, in, nin, infer_float_return, ctx);
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


#define XND_UNARY(func, t0, t1) \
static int                                                                   \
gm_##func##_0D_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)                \
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
}

#define XND_UNARY_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                    \
    .sig = "... * " STRINGIZE(t0) " -> ... * " STRINGIZE(t1),       \
    .Opt = gm_fixed_##func##_1D_C_##t0##_##t1,                      \
    .C = gm_##func##_0D_##t0##_##t1 },                              \
                                                                    \
  { .name = STRINGIZE(funcname),                                    \
    .sig = "var... * " STRINGIZE(t0) " -> var... * " STRINGIZE(t1), \
    .C = gm_##func##_0D_##t0##_##t1 }

#undef bool
#define bool_t _Bool

/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define copy(x) x
XND_UNARY(copy, bool, bool)
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

/*****************************************************************************/
/*                              Bitwise NOT                                  */
/*****************************************************************************/

#define invert(x) !x
XND_UNARY(invert, bool, bool)
#undef invert
#define invert(x) ~x
XND_UNARY(invert, int8, int8)
XND_UNARY(invert, int16, int16)
XND_UNARY(invert, int32, int32)
XND_UNARY(invert, int64, int64)
XND_UNARY(invert, uint8, uint8)
XND_UNARY(invert, uint16, uint16)
XND_UNARY(invert, uint32, uint32)
XND_UNARY(invert, uint64, uint64)


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
/*                                   Math                                   */
/*****************************************************************************/

#define XND_ALL_UNARY_FLOAT(name) \
    XND_UNARY(name##f, int8, float32)    \
    XND_UNARY(name##f, int16, float32)   \
    XND_UNARY(name##f, uint8, float32)   \
    XND_UNARY(name##f, uint16, float32)  \
    XND_UNARY(name##f, float32, float32) \
    XND_UNARY(name, int32, float64)      \
    XND_UNARY(name, uint32, float64)     \
    XND_UNARY(name, float64, float64)

#define XND_ALL_UNARY_FLOAT_INIT(name) \
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

XND_ALL_UNARY_FLOAT(fabs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(exp)
XND_ALL_UNARY_FLOAT(exp2)
XND_ALL_UNARY_FLOAT(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(log)
XND_ALL_UNARY_FLOAT(log2)
XND_ALL_UNARY_FLOAT(log10)
XND_ALL_UNARY_FLOAT(log1p)
XND_ALL_UNARY_FLOAT(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(sqrt)
XND_ALL_UNARY_FLOAT(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(sin)
XND_ALL_UNARY_FLOAT(cos)
XND_ALL_UNARY_FLOAT(tan)
XND_ALL_UNARY_FLOAT(asin)
XND_ALL_UNARY_FLOAT(acos)
XND_ALL_UNARY_FLOAT(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(sinh)
XND_ALL_UNARY_FLOAT(cosh)
XND_ALL_UNARY_FLOAT(tanh)
XND_ALL_UNARY_FLOAT(asinh)
XND_ALL_UNARY_FLOAT(acosh)
XND_ALL_UNARY_FLOAT(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(erf)
XND_ALL_UNARY_FLOAT(erfc)
XND_ALL_UNARY_FLOAT(lgamma)
XND_ALL_UNARY_FLOAT(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

XND_ALL_UNARY_FLOAT(ceil)
XND_ALL_UNARY_FLOAT(floor)
XND_ALL_UNARY_FLOAT(trunc)
XND_ALL_UNARY_FLOAT(round)
XND_ALL_UNARY_FLOAT(nearbyint)


static const gm_kernel_init_t unary_float[] = {
  /* ABS */
  XND_ALL_UNARY_FLOAT_INIT(fabs),

  /* EXPONENTIAL */
  XND_ALL_UNARY_FLOAT_INIT(exp),
  XND_ALL_UNARY_FLOAT_INIT(exp2),
  XND_ALL_UNARY_FLOAT_INIT(expm1),

  /* LOGARITHM */
  XND_ALL_UNARY_FLOAT_INIT(log),
  XND_ALL_UNARY_FLOAT_INIT(log2),
  XND_ALL_UNARY_FLOAT_INIT(log10),
  XND_ALL_UNARY_FLOAT_INIT(log1p),
  XND_ALL_UNARY_FLOAT_INIT(logb),

  /* POWER */
  XND_ALL_UNARY_FLOAT_INIT(sqrt),
  XND_ALL_UNARY_FLOAT_INIT(cbrt),

  /* TRIGONOMETRIC */
  XND_ALL_UNARY_FLOAT_INIT(sin),
  XND_ALL_UNARY_FLOAT_INIT(cos),
  XND_ALL_UNARY_FLOAT_INIT(tan),
  XND_ALL_UNARY_FLOAT_INIT(asin),
  XND_ALL_UNARY_FLOAT_INIT(acos),
  XND_ALL_UNARY_FLOAT_INIT(atan),

  /* HYPERBOLIC */
  XND_ALL_UNARY_FLOAT_INIT(sinh),
  XND_ALL_UNARY_FLOAT_INIT(cosh),
  XND_ALL_UNARY_FLOAT_INIT(tanh),
  XND_ALL_UNARY_FLOAT_INIT(asinh),
  XND_ALL_UNARY_FLOAT_INIT(acosh),
  XND_ALL_UNARY_FLOAT_INIT(atanh),

  /* ERROR AND GAMMA */
  XND_ALL_UNARY_FLOAT_INIT(erf),
  XND_ALL_UNARY_FLOAT_INIT(erfc),
  XND_ALL_UNARY_FLOAT_INIT(lgamma),
  XND_ALL_UNARY_FLOAT_INIT(tgamma),

  /* CEILING, FLOOR, TRUNC */
  XND_ALL_UNARY_FLOAT_INIT(ceil),
  XND_ALL_UNARY_FLOAT_INIT(floor),
  XND_ALL_UNARY_FLOAT_INIT(trunc),
  XND_ALL_UNARY_FLOAT_INIT(round),
  XND_ALL_UNARY_FLOAT_INIT(nearbyint),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

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

    for (k = unary_float; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_float_typecheck) < 0) {
            return -1;
        }
    }

    return 0;
}

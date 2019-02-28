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
copy_kernel_location(const ndt_t *in, const ndt_t *out, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);
    const ndt_t *u = ndt_dtype(out);

    switch (t->tag) {
    case Bool: {
        switch (u->tag) {
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
        default: goto invalid_combination;
        }
    }

    case Uint8: {
        switch (u->tag) {
        case Uint8: return 32;
        case Uint16: return 34;
        case Uint32: return 36;
        case Uint64: return 38;
        case Int16: return 40;
        case Int32: return 42;
        case Int64: return 44;
        case BFloat16: return 46;
        case Float16: return 48;
        case Float32: return 50;
        case Float64: return 52;
        case Complex32: return 54;
        case Complex64: return 56;
        case Complex128: return 58;
        default: goto invalid_combination;
        }
    }

    case Uint16: {
        switch (u->tag) {
        case Uint16: return 60;
        case Uint32: return 62;
        case Uint64: return 64;
        case Int32: return 66;
        case Int64: return 68;
        case Float32: return 70;
        case Float64: return 72;
        case Complex64: return 74;
        case Complex128: return 76;
        default: goto invalid_combination;
        }
    }

    case Uint32: {
        switch (u->tag) {
        case Uint32: return 78;
        case Uint64: return 80;
        case Int64: return 82;
        case Float64: return 84;
        case Complex128: return 86;
        default: goto invalid_combination;
        }
    }

    case Uint64: {
        switch (u->tag) {
        case Uint64: return 88;
        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (u->tag) {
        case Int8: return 90;
        case Int16: return 92;
        case Int32: return 94;
        case Int64: return 96;
        case BFloat16: return 98;
        case Float16: return 100;
        case Float32: return 102;
        case Float64: return 104;
        case Complex32: return 106;
        case Complex64: return 108;
        case Complex128: return 110;
        default: goto invalid_combination;
        }
    }

    case Int16: {
        switch (u->tag) {
        case Int16: return 112;
        case Int32: return 114;
        case Int64: return 116;
        case Float32: return 118;
        case Float64: return 120;
        case Complex64: return 122;
        case Complex128: return 124;
        default: goto invalid_combination;
        }
    }

    case Int32: {
        switch (u->tag) {
        case Int32: return 126;
        case Int64: return 128;
        case Float64: return 130;
        case Complex128: return 132;
        default: goto invalid_combination;
        }
    }

    case Int64: {
        switch (u->tag) {
        case Int64: return 134;
        default: goto invalid_combination;
        }
    }

    case BFloat16: {
        switch (u->tag) {
        case BFloat16: return 136;
        case Float32: return 138;
        case Float64: return 140;
        case Complex64: return 142;
        case Complex128: return 144;
        default: goto invalid_combination;
        }
    }

    case Float16: {
        switch (u->tag) {
        case Float16: return 146;
        case Float32: return 148;
        case Float64: return 150;
        case Complex32: return 152;
        case Complex64: return 154;
        case Complex128: return 156;
        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (u->tag) {
        case Float32: return 158;
        case Float64: return 160;
        case Complex64: return 162;
        case Complex128: return 164;
        default: goto invalid_combination;
        }
    }

    case Float64: {
        switch (u->tag) {
        case Float64: return 166;
        case Complex128: return 168;
        default: goto invalid_combination;
        }
    }

    case Complex32: {
        switch (u->tag) {
        case Complex32: return 170;
        case Complex64: return 172;
        case Complex128: return 174;
        default: goto invalid_combination;
        }
    }

    case Complex64: {
        switch (u->tag) {
        case Complex64: return 176;
        case Complex128: return 178;
        default: goto invalid_combination;
        }
    }

    case Complex128: {
        switch (u->tag) {
        case Complex128: return 180;
        default: goto invalid_combination;
        }
    }

    default: goto invalid_combination;
    }

invalid_combination:
    ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
    return -1;
}

static int
invert_kernel_location(const ndt_t *in, const ndt_t *out, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);
    (void)out;

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
negative_kernel_location(const ndt_t *in, const ndt_t *out, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);
    (void)out;

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
math_kernel_location(const ndt_t *in, const ndt_t *out, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);
    (void)out;

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

#define CUDA_HOST_UNARY(name, t0, t1) \
static int                                                                      \
gm_cuda_host_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                               \
    const char *a0 = apply_index(&stack[0]);                                    \
    char *a1 = apply_index(&stack[1]);                                          \
    const int64_t N = xnd_fixed_shape(&stack[0]);                               \
    (void)ctx;                                                                  \
                                                                                \
    gm_cuda_device_fixed_1D_C_##name##_##t0##_##t1(a0, a1, N);                  \
                                                                                \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                            \
        unary_update_bitmap1D(stack);                                           \
    }                                                                           \
                                                                                \
    return 0;                                                                   \
}                                                                               \
                                                                                \
static int                                                                      \
gm_cuda_host_fixed_1D_S_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                               \
    const char *a0 = apply_index(&stack[0]);                                    \
    char *a1 = apply_index(&stack[1]);                                          \
    const int64_t N = xnd_fixed_shape(&stack[0]);                               \
    const int64_t s0 = xnd_fixed_step(&stack[0]);                               \
    const int64_t s1 = xnd_fixed_step(&stack[1]);                               \
    (void)ctx;                                                                  \
                                                                                \
    gm_cuda_device_fixed_1D_S_##name##_##t0##_##t1(a0, a1, s0, s1, N);          \
                                                                                \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                            \
        unary_update_bitmap1D(stack);                                           \
    }                                                                           \
                                                                                \
    return 0;                                                                   \
}                                                                               \
                                                                                \
static int                                                                      \
gm_cuda_host_0D_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                               \
    const char *a0 = stack[0].ptr;                                              \
    char *a1 = stack[1].ptr;                                                    \
    (void)ctx;                                                                  \
                                                                                \
    gm_cuda_device_0D_##name##_##t0##_##t1(a0, a1);                             \
                                                                                \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                            \
        unary_update_bitmap(stack);                                             \
    }                                                                           \
                                                                                \
    return 0;                                                                   \
}


#define CUDA_HOST_NOIMPL(name, t0, t1) \
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
}                                                                               \
                                                                                \
static int                                                                      \
gm_cuda_host_fixed_1D_S_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                               \
    (void)stack;                                                                \
                                                                                \
    ndt_err_format(ctx, NDT_NotImplementedError,                                \
        "implementation for " STRINGIZE(name) " : "                             \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                      \
        " currently requires double rounding");                                 \
                                                                                \
    return -1;                                                                  \
}                                                                               \
                                                                                \
static int                                                                      \
gm_cuda_host_0D_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
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


#define CUDA_HOST_UNARY_REDUCE(name, t0, t1) \
static int                                                                       \
gm_cuda_host_1D_C_reduce_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                \
    const char *a0 = apply_index(&stack[0]);                                     \
    char *a1 = stack[1].ptr;                                                     \
    const int64_t N = xnd_fixed_shape(&stack[0]);                                \
    (void)ctx;                                                                   \
                                                                                 \
    gm_cuda_device_1D_C_reduce_##name##_##t0##_##t1(a0, a1, N);                  \
                                                                                 \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                             \
        unary_reduce_bitmap1D(stack);                                            \
    }                                                                            \
                                                                                 \
    return 0;                                                                    \
}

#define CUDA_HOST_REDUCE_NOIMPL(name, t0, t1) \
static int                                                                       \
gm_cuda_host_1D_C_reduce_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                \
    (void)stack;                                                                 \
                                                                                 \
    ndt_err_format(ctx, NDT_NotImplementedError,                                 \
        "No cuda thrust implementation for: " STRINGIZE(name) " : "              \
        STRINGIZE(t0) " -> " STRINGIZE(t1));                                     \
                                                                                 \
    return -1;                                                                   \
}


#define CUDA_HOST_UNARY_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                \
    .sig = "... * " STRINGIZE(t0) " -> ... * " STRINGIZE(t1),   \
    .OptC = gm_cuda_host_fixed_1D_C_##func##_##t0##_##t1,       \
    .OptS = gm_cuda_host_fixed_1D_S_##func##_##t0##_##t1,       \
    .C = gm_cuda_host_0D_##func##_##t0##_##t1 },                \
                                                                \
  { .name = STRINGIZE(funcname),                                \
    .sig = "... * ?" STRINGIZE(t0) " -> ... * ?" STRINGIZE(t1), \
    .OptC = gm_cuda_host_fixed_1D_C_##func##_##t0##_##t1,       \
    .OptS = gm_cuda_host_fixed_1D_S_##func##_##t0##_##t1,       \
    .C = gm_cuda_host_0D_##func##_##t0##_##t1 }                 \


#define CUDA_HOST_UNARY_REDUCE_INIT(funcname, func, t0, t1) \
  { .name = "reduce_" STRINGIZE(funcname),                  \
    .sig = "N * " STRINGIZE(t0) " -> " STRINGIZE(t1),       \
    .C = gm_cuda_host_1D_C_reduce_##func##_##t0##_##t1 },   \
                                                            \
  { .name = "reduce_" STRINGIZE(funcname),                  \
    .sig = "N * ?" STRINGIZE(t0) " -> ?" STRINGIZE(t1),     \
    .C = gm_cuda_host_1D_C_reduce_##func##_##t0##_##t1 }


#undef bool
#define bool_t _Bool


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define CUDA_HOST_ALL_UNARY_COPY(name) \
    CUDA_HOST_UNARY(name, bool, bool)             \
    CUDA_HOST_UNARY(name, bool, uint8)            \
    CUDA_HOST_UNARY(name, bool, uint16)           \
    CUDA_HOST_UNARY(name, bool, uint32)           \
    CUDA_HOST_UNARY(name, bool, uint64)           \
    CUDA_HOST_UNARY(name, bool, int8)             \
    CUDA_HOST_UNARY(name, bool, int16)            \
    CUDA_HOST_UNARY(name, bool, int32)            \
    CUDA_HOST_UNARY(name, bool, int64)            \
    CUDA_HOST_UNARY(name, bool, bfloat16)         \
    CUDA_HOST_UNARY(name, bool, float16)          \
    CUDA_HOST_UNARY(name, bool, float32)          \
    CUDA_HOST_UNARY(name, bool, float64)          \
    CUDA_HOST_NOIMPL(name,bool, complex32)        \
    CUDA_HOST_UNARY(name, bool, complex64)        \
    CUDA_HOST_UNARY(name, bool, complex128)       \
                                                  \
    CUDA_HOST_UNARY(name, uint8, uint8)           \
    CUDA_HOST_UNARY(name, uint8, uint16)          \
    CUDA_HOST_UNARY(name, uint8, uint32)          \
    CUDA_HOST_UNARY(name, uint8, uint64)          \
    CUDA_HOST_UNARY(name, uint8, int16)           \
    CUDA_HOST_UNARY(name, uint8, int32)           \
    CUDA_HOST_UNARY(name, uint8, int64)           \
    CUDA_HOST_UNARY(name, uint8, bfloat16)        \
    CUDA_HOST_UNARY(name, uint8, float16)         \
    CUDA_HOST_UNARY(name, uint8, float32)         \
    CUDA_HOST_UNARY(name, uint8, float64)         \
    CUDA_HOST_NOIMPL(name, uint8, complex32)      \
    CUDA_HOST_UNARY(name, uint8, complex64)       \
    CUDA_HOST_UNARY(name, uint8, complex128)      \
                                                  \
    CUDA_HOST_UNARY(name, uint16, uint16)         \
    CUDA_HOST_UNARY(name, uint16, uint32)         \
    CUDA_HOST_UNARY(name, uint16, uint64)         \
    CUDA_HOST_UNARY(name, uint16, int32)          \
    CUDA_HOST_UNARY(name, uint16, int64)          \
    CUDA_HOST_UNARY(name, uint16, float32)        \
    CUDA_HOST_UNARY(name, uint16, float64)        \
    CUDA_HOST_UNARY(name, uint16, complex64)      \
    CUDA_HOST_UNARY(name, uint16, complex128)     \
                                                  \
    CUDA_HOST_UNARY(name, uint32, uint32)         \
    CUDA_HOST_UNARY(name, uint32, uint64)         \
    CUDA_HOST_UNARY(name, uint32, int64)          \
    CUDA_HOST_UNARY(name, uint32, float64)        \
    CUDA_HOST_UNARY(name, uint32, complex128)     \
                                                  \
    CUDA_HOST_UNARY(name, uint64, uint64)         \
                                                  \
    CUDA_HOST_UNARY(name, int8, int8)             \
    CUDA_HOST_UNARY(name, int8, int16)            \
    CUDA_HOST_UNARY(name, int8, int32)            \
    CUDA_HOST_UNARY(name, int8, int64)            \
    CUDA_HOST_UNARY(name, int8, bfloat16)         \
    CUDA_HOST_UNARY(name, int8, float16)          \
    CUDA_HOST_UNARY(name, int8, float32)          \
    CUDA_HOST_UNARY(name, int8, float64)          \
    CUDA_HOST_NOIMPL(name, int8, complex32)       \
    CUDA_HOST_UNARY(name, int8, complex64)        \
    CUDA_HOST_UNARY(name, int8, complex128)       \
                                                  \
    CUDA_HOST_UNARY(name, int16, int16)           \
    CUDA_HOST_UNARY(name, int16, int32)           \
    CUDA_HOST_UNARY(name, int16, int64)           \
    CUDA_HOST_UNARY(name, int16, float32)         \
    CUDA_HOST_UNARY(name, int16, float64)         \
    CUDA_HOST_UNARY(name, int16, complex64)       \
    CUDA_HOST_UNARY(name, int16, complex128)      \
                                                  \
    CUDA_HOST_UNARY(name, int32, int32)           \
    CUDA_HOST_UNARY(name, int32, int64)           \
    CUDA_HOST_UNARY(name, int32, float64)         \
    CUDA_HOST_UNARY(name, int32, complex128)      \
                                                  \
    CUDA_HOST_UNARY(name, int64, int64)           \
                                                  \
    CUDA_HOST_UNARY(name, bfloat16, bfloat16)     \
    CUDA_HOST_UNARY(name, bfloat16, float32)      \
    CUDA_HOST_UNARY(name, bfloat16, float64)      \
    CUDA_HOST_UNARY(name, bfloat16, complex64)    \
    CUDA_HOST_UNARY(name, bfloat16, complex128)   \
                                                  \
    CUDA_HOST_UNARY(name, float16, float16)       \
    CUDA_HOST_UNARY(name, float16, float32)       \
    CUDA_HOST_UNARY(name, float16, float64)       \
    CUDA_HOST_NOIMPL(name, float16, complex32)    \
    CUDA_HOST_UNARY(name, float16, complex64)     \
    CUDA_HOST_UNARY(name, float16, complex128)    \
                                                  \
    CUDA_HOST_UNARY(name, float32, float32)       \
    CUDA_HOST_UNARY(name, float32, float64)       \
    CUDA_HOST_UNARY(name, float32, complex64)     \
    CUDA_HOST_UNARY(name, float32, complex128)    \
                                                  \
    CUDA_HOST_UNARY(name, float64, float64)       \
    CUDA_HOST_UNARY(name, float64, complex128)    \
                                                  \
    CUDA_HOST_NOIMPL(name, complex32, complex32)  \
    CUDA_HOST_NOIMPL(name, complex32, complex64)  \
    CUDA_HOST_NOIMPL(name, complex32, complex128) \
                                                  \
    CUDA_HOST_UNARY(name, complex64, complex64)   \
    CUDA_HOST_UNARY(name, complex64, complex128)  \
                                                  \
    CUDA_HOST_UNARY(name, complex128, complex128)

#define CUDA_HOST_ALL_UNARY_COPY_INIT(name, func, hfunc) \
    CUDA_HOST_UNARY_INIT(name, func, bool, bool),            \
    CUDA_HOST_UNARY_INIT(name, func, bool, uint8),           \
    CUDA_HOST_UNARY_INIT(name, func, bool, uint16),          \
    CUDA_HOST_UNARY_INIT(name, func, bool, uint32),          \
    CUDA_HOST_UNARY_INIT(name, func, bool, uint64),          \
    CUDA_HOST_UNARY_INIT(name, func, bool, int8),            \
    CUDA_HOST_UNARY_INIT(name, func, bool, int16),           \
    CUDA_HOST_UNARY_INIT(name, func, bool, int32),           \
    CUDA_HOST_UNARY_INIT(name, func, bool, int64),           \
    CUDA_HOST_UNARY_INIT(name, func, bool, bfloat16),        \
    CUDA_HOST_UNARY_INIT(name, hfunc, bool, float16),        \
    CUDA_HOST_UNARY_INIT(name, func, bool, float32),         \
    CUDA_HOST_UNARY_INIT(name, func, bool, float64),         \
    CUDA_HOST_UNARY_INIT(name, func, bool, complex32),       \
    CUDA_HOST_UNARY_INIT(name, func, bool, complex64),       \
    CUDA_HOST_UNARY_INIT(name, func, bool, complex128),      \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, uint8, uint8),          \
    CUDA_HOST_UNARY_INIT(name, func, uint8, uint16),         \
    CUDA_HOST_UNARY_INIT(name, func, uint8, uint32),         \
    CUDA_HOST_UNARY_INIT(name, func, uint8, uint64),         \
    CUDA_HOST_UNARY_INIT(name, func, uint8, int16),          \
    CUDA_HOST_UNARY_INIT(name, func, uint8, int32),          \
    CUDA_HOST_UNARY_INIT(name, func, uint8, int64),          \
    CUDA_HOST_UNARY_INIT(name, func, uint8, bfloat16),       \
    CUDA_HOST_UNARY_INIT(name, hfunc, uint8, float16),       \
    CUDA_HOST_UNARY_INIT(name, func, uint8, float32),        \
    CUDA_HOST_UNARY_INIT(name, func, uint8, float64),        \
    CUDA_HOST_UNARY_INIT(name, func, uint8, complex32),      \
    CUDA_HOST_UNARY_INIT(name, func, uint8, complex64),      \
    CUDA_HOST_UNARY_INIT(name, func, uint8, complex128),     \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, uint16, uint16),        \
    CUDA_HOST_UNARY_INIT(name, func, uint16, uint32),        \
    CUDA_HOST_UNARY_INIT(name, func, uint16, uint64),        \
    CUDA_HOST_UNARY_INIT(name, func, uint16, int32),         \
    CUDA_HOST_UNARY_INIT(name, func, uint16, int64),         \
    CUDA_HOST_UNARY_INIT(name, func, uint16, float32),       \
    CUDA_HOST_UNARY_INIT(name, func, uint16, float64),       \
    CUDA_HOST_UNARY_INIT(name, func, uint16, complex64),     \
    CUDA_HOST_UNARY_INIT(name, func, uint16, complex128),    \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, uint32, uint32),        \
    CUDA_HOST_UNARY_INIT(name, func, uint32, uint64),        \
    CUDA_HOST_UNARY_INIT(name, func, uint32, int64),         \
    CUDA_HOST_UNARY_INIT(name, func, uint32, float64),       \
    CUDA_HOST_UNARY_INIT(name, func, uint32, complex128),    \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, uint64, uint64),        \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, int8, int8),            \
    CUDA_HOST_UNARY_INIT(name, func, int8, int16),           \
    CUDA_HOST_UNARY_INIT(name, func, int8, int32),           \
    CUDA_HOST_UNARY_INIT(name, func, int8, int64),           \
    CUDA_HOST_UNARY_INIT(name, func, int8, bfloat16),        \
    CUDA_HOST_UNARY_INIT(name, hfunc, int8, float16),        \
    CUDA_HOST_UNARY_INIT(name, func, int8, float32),         \
    CUDA_HOST_UNARY_INIT(name, func, int8, float64),         \
    CUDA_HOST_UNARY_INIT(name, func, int8, complex32),       \
    CUDA_HOST_UNARY_INIT(name, func, int8, complex64),       \
    CUDA_HOST_UNARY_INIT(name, func, int8, complex128),      \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, int16, int16),          \
    CUDA_HOST_UNARY_INIT(name, func, int16, int32),          \
    CUDA_HOST_UNARY_INIT(name, func, int16, int64),          \
    CUDA_HOST_UNARY_INIT(name, func, int16, float32),        \
    CUDA_HOST_UNARY_INIT(name, func, int16, float64),        \
    CUDA_HOST_UNARY_INIT(name, func, int16, complex64),      \
    CUDA_HOST_UNARY_INIT(name, func, int16, complex128),     \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, int32, int32),          \
    CUDA_HOST_UNARY_INIT(name, func, int32, int64),          \
    CUDA_HOST_UNARY_INIT(name, func, int32, float64),        \
    CUDA_HOST_UNARY_INIT(name, func, int32, complex128),     \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, int64, int64),          \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, bfloat16, bfloat16),    \
    CUDA_HOST_UNARY_INIT(name, func, bfloat16, float32),     \
    CUDA_HOST_UNARY_INIT(name, func, bfloat16, float64),     \
    CUDA_HOST_UNARY_INIT(name, func, bfloat16, complex64),   \
    CUDA_HOST_UNARY_INIT(name, func, bfloat16, complex128),  \
                                                             \
    CUDA_HOST_UNARY_INIT(name, hfunc, float16, float16),     \
    CUDA_HOST_UNARY_INIT(name, func, float16, float32),      \
    CUDA_HOST_UNARY_INIT(name, func, float16, float64),      \
    CUDA_HOST_UNARY_INIT(name, func, float16, complex32),    \
    CUDA_HOST_UNARY_INIT(name, func, float16, complex64),    \
    CUDA_HOST_UNARY_INIT(name, func, float16, complex128),   \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, float32, float32),      \
    CUDA_HOST_UNARY_INIT(name, func, float32, float64),      \
    CUDA_HOST_UNARY_INIT(name, func, float32, complex64),    \
    CUDA_HOST_UNARY_INIT(name, func, float32, complex128),   \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, float64, float64),      \
    CUDA_HOST_UNARY_INIT(name, func, float64, complex128),   \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, complex32, complex32),  \
    CUDA_HOST_UNARY_INIT(name, func, complex32, complex64),  \
    CUDA_HOST_UNARY_INIT(name, func, complex32, complex128), \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, complex64, complex64),  \
    CUDA_HOST_UNARY_INIT(name, func, complex64, complex128), \
                                                             \
    CUDA_HOST_UNARY_INIT(name, func, complex128, complex128)


CUDA_HOST_ALL_UNARY_COPY(copy)


static const gm_kernel_init_t unary_copy[] = {
  /* COPY */
  CUDA_HOST_ALL_UNARY_COPY_INIT(copy, copy, copy),

  { .name = NULL, .sig = NULL }
};

/*****************************************************************************/
/*                                   Reduce                                  */
/*****************************************************************************/

#define CUDA_HOST_ALL_UNARY_REDUCE(name) \
    CUDA_HOST_UNARY_REDUCE(name, bool, bool)              \
    CUDA_HOST_UNARY_REDUCE(name, bool, uint8)             \
    CUDA_HOST_UNARY_REDUCE(name, bool, uint16)            \
    CUDA_HOST_UNARY_REDUCE(name, bool, uint32)            \
    CUDA_HOST_UNARY_REDUCE(name, bool, uint64)            \
    CUDA_HOST_UNARY_REDUCE(name, bool, int8)              \
    CUDA_HOST_UNARY_REDUCE(name, bool, int16)             \
    CUDA_HOST_UNARY_REDUCE(name, bool, int32)             \
    CUDA_HOST_UNARY_REDUCE(name, bool, int64)             \
    CUDA_HOST_REDUCE_NOIMPL(name, bool, bfloat16)         \
    CUDA_HOST_UNARY_REDUCE(name, bool, float16)           \
    CUDA_HOST_UNARY_REDUCE(name, bool, float32)           \
    CUDA_HOST_UNARY_REDUCE(name, bool, float64)           \
    CUDA_HOST_REDUCE_NOIMPL(name,bool, complex32)         \
    CUDA_HOST_REDUCE_NOIMPL(name, bool, complex64)        \
    CUDA_HOST_REDUCE_NOIMPL(name, bool, complex128)       \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, uint8, uint8)            \
    CUDA_HOST_UNARY_REDUCE(name, uint8, uint16)           \
    CUDA_HOST_UNARY_REDUCE(name, uint8, uint32)           \
    CUDA_HOST_UNARY_REDUCE(name, uint8, uint64)           \
    CUDA_HOST_UNARY_REDUCE(name, uint8, int16)            \
    CUDA_HOST_UNARY_REDUCE(name, uint8, int32)            \
    CUDA_HOST_UNARY_REDUCE(name, uint8, int64)            \
    CUDA_HOST_REDUCE_NOIMPL(name, uint8, bfloat16)        \
    CUDA_HOST_UNARY_REDUCE(name, uint8, float16)          \
    CUDA_HOST_UNARY_REDUCE(name, uint8, float32)          \
    CUDA_HOST_UNARY_REDUCE(name, uint8, float64)          \
    CUDA_HOST_REDUCE_NOIMPL(name, uint8, complex32)       \
    CUDA_HOST_REDUCE_NOIMPL(name, uint8, complex64)       \
    CUDA_HOST_REDUCE_NOIMPL(name, uint8, complex128)      \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, uint16, uint16)          \
    CUDA_HOST_UNARY_REDUCE(name, uint16, uint32)          \
    CUDA_HOST_UNARY_REDUCE(name, uint16, uint64)          \
    CUDA_HOST_UNARY_REDUCE(name, uint16, int32)           \
    CUDA_HOST_UNARY_REDUCE(name, uint16, int64)           \
    CUDA_HOST_UNARY_REDUCE(name, uint16, float32)         \
    CUDA_HOST_UNARY_REDUCE(name, uint16, float64)         \
    CUDA_HOST_REDUCE_NOIMPL(name, uint16, complex64)      \
    CUDA_HOST_REDUCE_NOIMPL(name, uint16, complex128)     \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, uint32, uint32)          \
    CUDA_HOST_UNARY_REDUCE(name, uint32, uint64)          \
    CUDA_HOST_UNARY_REDUCE(name, uint32, int64)           \
    CUDA_HOST_UNARY_REDUCE(name, uint32, float64)         \
    CUDA_HOST_REDUCE_NOIMPL(name, uint32, complex128)     \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, uint64, uint64)          \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, int8, int8)              \
    CUDA_HOST_UNARY_REDUCE(name, int8, int16)             \
    CUDA_HOST_UNARY_REDUCE(name, int8, int32)             \
    CUDA_HOST_UNARY_REDUCE(name, int8, int64)             \
    CUDA_HOST_REDUCE_NOIMPL(name, int8, bfloat16)         \
    CUDA_HOST_UNARY_REDUCE(name, int8, float16)           \
    CUDA_HOST_UNARY_REDUCE(name, int8, float32)           \
    CUDA_HOST_UNARY_REDUCE(name, int8, float64)           \
    CUDA_HOST_REDUCE_NOIMPL(name, int8, complex32)        \
    CUDA_HOST_REDUCE_NOIMPL(name, int8, complex64)        \
    CUDA_HOST_REDUCE_NOIMPL(name, int8, complex128)       \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, int16, int16)            \
    CUDA_HOST_UNARY_REDUCE(name, int16, int32)            \
    CUDA_HOST_UNARY_REDUCE(name, int16, int64)            \
    CUDA_HOST_UNARY_REDUCE(name, int16, float32)          \
    CUDA_HOST_UNARY_REDUCE(name, int16, float64)          \
    CUDA_HOST_REDUCE_NOIMPL(name, int16, complex64)       \
    CUDA_HOST_REDUCE_NOIMPL(name, int16, complex128)      \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, int32, int32)            \
    CUDA_HOST_UNARY_REDUCE(name, int32, int64)            \
    CUDA_HOST_UNARY_REDUCE(name, int32, float64)          \
    CUDA_HOST_REDUCE_NOIMPL(name, int32, complex128)      \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, int64, int64)            \
                                                          \
    CUDA_HOST_REDUCE_NOIMPL(name, bfloat16, bfloat16)     \
    CUDA_HOST_REDUCE_NOIMPL(name, bfloat16, float32)      \
    CUDA_HOST_REDUCE_NOIMPL(name, bfloat16, float64)      \
    CUDA_HOST_REDUCE_NOIMPL(name, bfloat16, complex64)    \
    CUDA_HOST_REDUCE_NOIMPL(name, bfloat16, complex128)   \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, float16, float16)        \
    CUDA_HOST_REDUCE_NOIMPL(name, float16, float32)       \
    CUDA_HOST_REDUCE_NOIMPL(name, float16, float64)       \
    CUDA_HOST_REDUCE_NOIMPL(name, float16, complex32)     \
    CUDA_HOST_REDUCE_NOIMPL(name, float16, complex64)     \
    CUDA_HOST_REDUCE_NOIMPL(name, float16, complex128)    \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, float32, float32)        \
    CUDA_HOST_UNARY_REDUCE(name, float32, float64)        \
    CUDA_HOST_REDUCE_NOIMPL(name, float32, complex64)     \
    CUDA_HOST_REDUCE_NOIMPL(name, float32, complex128)    \
                                                          \
    CUDA_HOST_UNARY_REDUCE(name, float64, float64)        \
    CUDA_HOST_REDUCE_NOIMPL(name, float64, complex128)    \
                                                          \
    CUDA_HOST_REDUCE_NOIMPL(name, complex32, complex32)   \
    CUDA_HOST_REDUCE_NOIMPL(name, complex32, complex64)   \
    CUDA_HOST_REDUCE_NOIMPL(name, complex32, complex128)  \
                                                          \
    CUDA_HOST_REDUCE_NOIMPL(name, complex64, complex64)   \
    CUDA_HOST_REDUCE_NOIMPL(name, complex64, complex128)  \
                                                          \
    CUDA_HOST_REDUCE_NOIMPL(name, complex128, complex128)

#define CUDA_HOST_ALL_UNARY_REDUCE_INIT(name, func) \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, bool),            \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, uint8),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, uint16),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, uint32),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, uint64),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, int8),            \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, int16),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, int32),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, int64),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, bfloat16),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, float16),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, float32),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, float64),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, complex32),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, complex64),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bool, complex128),      \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, uint8),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, uint16),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, uint32),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, uint64),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, int16),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, int32),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, int64),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, bfloat16),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, float16),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, float32),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, float64),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, complex32),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, complex64),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint8, complex128),     \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, uint16),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, uint32),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, uint64),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, int32),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, int64),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, float32),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, float64),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, complex64),     \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint16, complex128),    \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint32, uint32),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint32, uint64),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint32, int64),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint32, float64),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint32, complex128),    \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, uint64, uint64),        \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, int8),            \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, int16),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, int32),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, int64),           \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, bfloat16),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, float16),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, float32),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, float64),         \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, complex32),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, complex64),       \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int8, complex128),      \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, int16),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, int32),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, int64),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, float32),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, float64),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, complex64),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int16, complex128),     \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int32, int32),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int32, int64),          \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int32, float64),        \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int32, complex128),     \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, int64, int64),          \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bfloat16, bfloat16),    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bfloat16, float32),     \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bfloat16, float64),     \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bfloat16, complex64),   \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, bfloat16, complex128),  \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float16, float16),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float16, float32),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float16, float64),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float16, complex32),    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float16, complex64),    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float16, complex128),   \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float32, float32),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float32, float64),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float32, complex64),    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float32, complex128),   \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float64, float64),      \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, float64, complex128),   \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, complex32, complex32),  \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, complex32, complex64),  \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, complex32, complex128), \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, complex64, complex64),  \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, complex64, complex128), \
                                                                    \
    CUDA_HOST_UNARY_REDUCE_INIT(name, func, complex128, complex128)


CUDA_HOST_ALL_UNARY_REDUCE(add)
CUDA_HOST_ALL_UNARY_REDUCE(multiply)


static const gm_kernel_init_t unary_reduce[] = {
  /* REDUCE */
  CUDA_HOST_ALL_UNARY_REDUCE_INIT(add, add),
  CUDA_HOST_ALL_UNARY_REDUCE_INIT(multiply, multiply),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                Bitwise NOT                                */
/*****************************************************************************/

CUDA_HOST_UNARY(invert, bool, bool)

CUDA_HOST_UNARY(invert, uint8, uint8)
CUDA_HOST_UNARY(invert, uint16, uint16)
CUDA_HOST_UNARY(invert, uint32, uint32)
CUDA_HOST_UNARY(invert, uint64, uint64)

CUDA_HOST_UNARY(invert, int8, int8)
CUDA_HOST_UNARY(invert, int16, int16)
CUDA_HOST_UNARY(invert, int32, int32)
CUDA_HOST_UNARY(invert, int64, int64)


static const gm_kernel_init_t unary_invert[] = {
  /* INVERT */
  CUDA_HOST_UNARY_INIT(invert, invert, bool, bool),

  CUDA_HOST_UNARY_INIT(invert, invert, uint8, uint8),
  CUDA_HOST_UNARY_INIT(invert, invert, uint16, uint16),
  CUDA_HOST_UNARY_INIT(invert, invert, uint32, uint32),
  CUDA_HOST_UNARY_INIT(invert, invert, uint64, uint64),

  CUDA_HOST_UNARY_INIT(invert, invert, int8, int8),
  CUDA_HOST_UNARY_INIT(invert, invert, int16, int16),
  CUDA_HOST_UNARY_INIT(invert, invert, int32, int32),
  CUDA_HOST_UNARY_INIT(invert, invert, int64, int64),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                  Negative                                 */
/*****************************************************************************/

CUDA_HOST_UNARY(negative, uint8, int16)
CUDA_HOST_UNARY(negative, uint16, int32)
CUDA_HOST_UNARY(negative, uint32, int64)

CUDA_HOST_UNARY(negative, int8, int8)
CUDA_HOST_UNARY(negative, int16, int16)
CUDA_HOST_UNARY(negative, int32, int32)
CUDA_HOST_UNARY(negative, int64, int64)

CUDA_HOST_UNARY(negative, bfloat16, bfloat16)
CUDA_HOST_UNARY(negative, float16, float16)
CUDA_HOST_UNARY(negative, float32, float32)
CUDA_HOST_UNARY(negative, float64, float64)

CUDA_HOST_NOIMPL(negative, complex32, complex32)
CUDA_HOST_UNARY(negative, complex64, complex64)
CUDA_HOST_UNARY(negative, complex128, complex128)


static const gm_kernel_init_t unary_negative[] = {
  /* NEGATIVE */
  CUDA_HOST_UNARY_INIT(negative, negative, uint8, int16),
  CUDA_HOST_UNARY_INIT(negative, negative, uint16, int32),
  CUDA_HOST_UNARY_INIT(negative, negative, uint32, int64),

  CUDA_HOST_UNARY_INIT(negative, negative, int8, int8),
  CUDA_HOST_UNARY_INIT(negative, negative, int16, int16),
  CUDA_HOST_UNARY_INIT(negative, negative, int32, int32),
  CUDA_HOST_UNARY_INIT(negative, negative, int64, int64),

  CUDA_HOST_UNARY_INIT(negative, negative, bfloat16, bfloat16),
  CUDA_HOST_UNARY_INIT(negative, negative, float16, float16),
  CUDA_HOST_UNARY_INIT(negative, negative, float32, float32),
  CUDA_HOST_UNARY_INIT(negative, negative, float64, float64),

  CUDA_HOST_UNARY_INIT(negative, negative, complex32, complex32),
  CUDA_HOST_UNARY_INIT(negative, negative, complex64, complex64),
  CUDA_HOST_UNARY_INIT(negative, negative, complex128, complex128),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Math                                   */
/*****************************************************************************/

#define _CUDA_ALL_HALF_MATH(name) \
    CUDA_HOST_UNARY(name##f16, uint8, float16)   \
    CUDA_HOST_UNARY(name##f16, int8, float16)    \
    CUDA_HOST_UNARY(name##f16, float16, float16)

#define _CUDA_ALL_HALF_MATH_NOIMPL(name) \
    CUDA_HOST_NOIMPL(name##f16, uint8, float16)   \
    CUDA_HOST_NOIMPL(name##f16, int8, float16)    \
    CUDA_HOST_NOIMPL(name##f16, float16, float16)

#define _CUDA_ALL_COMPLEX_MATH(name) \
    CUDA_HOST_NOIMPL(name, complex32, complex32)  \
    CUDA_HOST_UNARY(name, complex64, complex64)   \
    CUDA_HOST_UNARY(name, complex128, complex128)

#define _CUDA_ALL_COMPLEX_MATH_NOIMPL(name) \
    CUDA_HOST_NOIMPL(name, complex32, complex32)   \
    CUDA_HOST_NOIMPL(name, complex64, complex64)   \
    CUDA_HOST_NOIMPL(name, complex128, complex128)

#define _CUDA_ALL_REAL_MATH(name) \
    CUDA_HOST_UNARY(name##b16, bfloat16, bfloat16) \
    CUDA_HOST_UNARY(name##f, uint16, float32)      \
    CUDA_HOST_UNARY(name##f, int16, float32)       \
    CUDA_HOST_UNARY(name##f, float32, float32)     \
    CUDA_HOST_UNARY(name, uint32, float64)         \
    CUDA_HOST_UNARY(name, int32, float64)          \
    CUDA_HOST_UNARY(name, float64, float64)        \

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
    CUDA_HOST_UNARY_INIT(name, name##f16, uint8, float16),     \
    CUDA_HOST_UNARY_INIT(name, name##f16, int8, float16),      \
    CUDA_HOST_UNARY_INIT(name, name##f16, float16, float16),   \
                                                               \
    CUDA_HOST_UNARY_INIT(name, name##b16, bfloat16, bfloat16), \
                                                               \
    CUDA_HOST_UNARY_INIT(name, name##f, uint16, float32),      \
    CUDA_HOST_UNARY_INIT(name, name##f, int16, float32),       \
    CUDA_HOST_UNARY_INIT(name, name##f, float32, float32),     \
                                                               \
    CUDA_HOST_UNARY_INIT(name, name, uint32, float64),         \
    CUDA_HOST_UNARY_INIT(name, name, int32, float64),          \
    CUDA_HOST_UNARY_INIT(name, name, float64, float64),        \
                                                               \
    CUDA_HOST_UNARY_INIT(name, name, complex32, complex32),    \
    CUDA_HOST_UNARY_INIT(name, name, complex64, complex64),    \
    CUDA_HOST_UNARY_INIT(name, name, complex128, complex128)


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

typedef _Bool bool;

static const gm_kernel_set_t *
unary_copy_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                     const int64_t li[], int nin, int nout, bool check_broadcast,
                     ndt_context_t *ctx)
{
    return cuda_unary_typecheck(copy_kernel_location, spec, f, types, li,
                                nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
unary_invert_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                       const int64_t li[], int nin, int nout, bool check_broadcast,
                       ndt_context_t *ctx)
{
    return cuda_unary_typecheck(invert_kernel_location, spec, f, types, li,
                                nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
unary_negative_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                         const int64_t li[], int nin, int nout, bool check_broadcast,
                         ndt_context_t *ctx)
{
    return cuda_unary_typecheck(negative_kernel_location, spec, f, types, li,
                                nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
unary_math_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                      const int64_t li[], int nin, int nout, bool check_broadcast,
                      ndt_context_t *ctx)
{
    return cuda_unary_typecheck(math_kernel_location, spec, f, types, li,
                                nin, nout, check_broadcast, ctx);
}

int
gm_init_cuda_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = unary_copy; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_copy_typecheck) < 0) {
             return -1;
        }
    }

    for (k = unary_reduce; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
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

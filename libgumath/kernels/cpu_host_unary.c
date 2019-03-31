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
#include "cpu_device_unary.h"


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
        default: goto invalid_combination;
        }
    }

    case Uint8: {
        switch (u->tag) {
        case Uint8: return 64;
        case Uint16: return 68;
        case Uint32: return 72;
        case Uint64: return 76;
        case Int16: return 80;
        case Int32: return 84;
        case Int64: return 88;
        case BFloat16: return 92;
        case Float16: return 96;
        case Float32: return 100;
        case Float64: return 104;
        case Complex32: return 108;
        case Complex64: return 112;
        case Complex128: return 116;
        default: goto invalid_combination;
        }
    }

    case Uint16: {
        switch (u->tag) {
        case Uint16: return 120;
        case Uint32: return 124;
        case Uint64: return 128;
        case Int32: return 132;
        case Int64: return 136;
        case Float32: return 140;
        case Float64: return 144;
        case Complex64: return 148;
        case Complex128: return 152;
        default: goto invalid_combination;
        }
    }

    case Uint32: {
        switch (u->tag) {
        case Uint32: return 156;
        case Uint64: return 160;
        case Int64: return 164;
        case Float64: return 168;
        case Complex128: return 172;
        default: goto invalid_combination;
        }
    }

    case Uint64: {
        switch (u->tag) {
        case Uint64: return 176;
        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (u->tag) {
        case Int8: return 180;
        case Int16: return 184;
        case Int32: return 188;
        case Int64: return 192;
        case BFloat16: return 196;
        case Float16: return 200;
        case Float32: return 204;
        case Float64: return 208;
        case Complex32: return 212;
        case Complex64: return 216;
        case Complex128: return 220;
        default: goto invalid_combination;
        }
    }

    case Int16: {
        switch (u->tag) {
        case Int16: return 224;
        case Int32: return 228;
        case Int64: return 232;
        case Float32: return 236;
        case Float64: return 240;
        case Complex64: return 244;
        case Complex128: return 248;
        default: goto invalid_combination;
        }
    }

    case Int32: {
        switch (u->tag) {
        case Int32: return 252;
        case Int64: return 256;
        case Float64: return 260;
        case Complex128: return 264;
        default: goto invalid_combination;
        }
    }

    case Int64: {
        switch (u->tag) {
        case Int64: return 268;
        default: goto invalid_combination;
        }
    }

    case BFloat16: {
        switch (u->tag) {
        case BFloat16: return 272;
        case Float32: return 276;
        case Float64: return 280;
        case Complex64: return 284;
        case Complex128: return 288;
        default: goto invalid_combination;
        }
    }

    case Float16: {
        switch (u->tag) {
        case Float16: return 292;
        case Float32: return 296;
        case Float64: return 300;
        case Complex32: return 304;
        case Complex64: return 308;
        case Complex128: return 312;
        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (u->tag) {
        case Float32: return 316;
        case Float64: return 320;
        case Complex64: return 324;
        case Complex128: return 328;
        default: goto invalid_combination;
        }
    }

    case Float64: {
        switch (u->tag) {
        case Float64: return 332;
        case Complex128: return 336;
        default: goto invalid_combination;
        }
    }

    case Complex32: {
        switch (u->tag) {
        case Complex32: return 340;
        case Complex64: return 344;
        case Complex128: return 348;
        default: goto invalid_combination;
        }
    }

    case Complex64: {
        switch (u->tag) {
        case Complex64: return 352;
        case Complex128: return 356;
        default: goto invalid_combination;
        }
    }

    case Complex128: {
        switch (u->tag) {
        case Complex128: return 360;
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
negative_kernel_location(const ndt_t *in, const ndt_t *out, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);
    (void)out;

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
math_kernel_location(const ndt_t *in, const ndt_t *out, ndt_context_t *ctx)
{
    const ndt_t *t = ndt_dtype(in);
    (void)out;

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

#define CPU_HOST_UNARY(name, t0, t1) \
static int                                                                     \
gm_cpu_host_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                              \
    const char *a0 = apply_index(&stack[0]);                                   \
    char *a1 = apply_index(&stack[1]);                                         \
    const int64_t N = xnd_fixed_shape(&stack[0]);                              \
    (void)ctx;                                                                 \
                                                                               \
    gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1(a0, a1, N);                  \
                                                                               \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                           \
        unary_update_bitmap_1D_S(stack);                                       \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_cpu_host_fixed_1D_S_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                              \
    const char *a0 = apply_index(&stack[0]);                                   \
    char *a1 = apply_index(&stack[1]);                                         \
    const int64_t N = xnd_fixed_shape(&stack[0]);                              \
    const int64_t s0 = xnd_fixed_step(&stack[0]);                              \
    const int64_t s1 = xnd_fixed_step(&stack[1]);                              \
    (void)ctx;                                                                 \
                                                                               \
    gm_cpu_device_fixed_1D_S_##name##_##t0##_##t1(a0, a1, s0, s1, N);          \
                                                                               \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                           \
        unary_update_bitmap_1D_S(stack);                                       \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}                                                                              \
                                                                               \
static int                                                                     \
gm_cpu_host_0D_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                              \
    const char *a0 = stack[0].ptr;                                             \
    char *a1 = stack[1].ptr;                                                   \
    (void)ctx;                                                                 \
                                                                               \
    gm_cpu_device_0D_##name##_##t0##_##t1(a0, a1);                             \
                                                                               \
    if (ndt_is_optional(ndt_dtype(stack[1].type))) {                           \
        unary_update_bitmap_0D(stack);                                         \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}

#define CPU_HOST_NOIMPL(name, t0, t1) \
static int                                                                     \
gm_cpu_host_fixed_1D_C_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                              \
    (void)stack;                                                               \
                                                                               \
    ndt_err_format(ctx, NDT_NotImplementedError,                               \
        "implementation for " STRINGIZE(name) " : "                            \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                     \
        " currently requires double rounding");                                \
                                                                               \
    return -1;                                                                 \
}                                                                              \
                                                                               \
static int                                                                     \
gm_cpu_host_fixed_1D_S_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx) \
{                                                                              \
    (void)stack;                                                               \
                                                                               \
    ndt_err_format(ctx, NDT_NotImplementedError,                               \
        "implementation for " STRINGIZE(name) " : "                            \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                     \
        " currently requires double rounding");                                \
                                                                               \
    return -1;                                                                 \
}                                                                              \
                                                                               \
static int                                                                     \
gm_cpu_host_0D_##name##_##t0##_##t1(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                              \
    (void)stack;                                                               \
                                                                               \
    ndt_err_format(ctx, NDT_NotImplementedError,                               \
        "implementation for " STRINGIZE(name) " : "                            \
        STRINGIZE(t0) " -> " STRINGIZE(t1)                                     \
        " currently requires double rounding");                                \
                                                                               \
    return -1;                                                                 \
}


#define CPU_HOST_UNARY_INIT(funcname, func, t0, t1) \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "... * " STRINGIZE(t0) " -> ... * " STRINGIZE(t1),         \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1,              \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1,              \
    .C = gm_cpu_host_0D_##func##_##t0##_##t1 },                       \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "... * ?" STRINGIZE(t0) " -> ... * ?" STRINGIZE(t1),       \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1,              \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1,              \
    .C = gm_cpu_host_0D_##func##_##t0##_##t1 },                       \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "var... * " STRINGIZE(t0) " -> var... * " STRINGIZE(t1),   \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1 },                     \
                                                                      \
  { .name = STRINGIZE(funcname),                                      \
    .sig = "var... * ?" STRINGIZE(t0) " -> var... * ?" STRINGIZE(t1), \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1 }


#undef bool
#define bool_t _Bool


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define CPU_HOST_ALL_UNARY(name) \
    CPU_HOST_UNARY(name, bool, bool)             \
    CPU_HOST_UNARY(name, bool, uint8)            \
    CPU_HOST_UNARY(name, bool, uint16)           \
    CPU_HOST_UNARY(name, bool, uint32)           \
    CPU_HOST_UNARY(name, bool, uint64)           \
    CPU_HOST_UNARY(name, bool, int8)             \
    CPU_HOST_UNARY(name, bool, int16)            \
    CPU_HOST_UNARY(name, bool, int32)            \
    CPU_HOST_UNARY(name, bool, int64)            \
    CPU_HOST_UNARY(name, bool, bfloat16)         \
    CPU_HOST_NOIMPL(name, bool, float16)         \
    CPU_HOST_UNARY(name, bool, float32)          \
    CPU_HOST_UNARY(name, bool, float64)          \
    CPU_HOST_NOIMPL(name,bool, complex32)        \
    CPU_HOST_UNARY(name, bool, complex64)        \
    CPU_HOST_UNARY(name, bool, complex128)       \
                                                 \
    CPU_HOST_UNARY(name, uint8, uint8)           \
    CPU_HOST_UNARY(name, uint8, uint16)          \
    CPU_HOST_UNARY(name, uint8, uint32)          \
    CPU_HOST_UNARY(name, uint8, uint64)          \
    CPU_HOST_UNARY(name, uint8, int16)           \
    CPU_HOST_UNARY(name, uint8, int32)           \
    CPU_HOST_UNARY(name, uint8, int64)           \
    CPU_HOST_UNARY(name, uint8, bfloat16)        \
    CPU_HOST_NOIMPL(name, uint8, float16)        \
    CPU_HOST_UNARY(name, uint8, float32)         \
    CPU_HOST_UNARY(name, uint8, float64)         \
    CPU_HOST_NOIMPL(name, uint8, complex32)      \
    CPU_HOST_UNARY(name, uint8, complex64)       \
    CPU_HOST_UNARY(name, uint8, complex128)      \
                                                 \
    CPU_HOST_UNARY(name, uint16, uint16)         \
    CPU_HOST_UNARY(name, uint16, uint32)         \
    CPU_HOST_UNARY(name, uint16, uint64)         \
    CPU_HOST_UNARY(name, uint16, int32)          \
    CPU_HOST_UNARY(name, uint16, int64)          \
    CPU_HOST_UNARY(name, uint16, float32)        \
    CPU_HOST_UNARY(name, uint16, float64)        \
    CPU_HOST_UNARY(name, uint16, complex64)      \
    CPU_HOST_UNARY(name, uint16, complex128)     \
                                                 \
    CPU_HOST_UNARY(name, uint32, uint32)         \
    CPU_HOST_UNARY(name, uint32, uint64)         \
    CPU_HOST_UNARY(name, uint32, int64)          \
    CPU_HOST_UNARY(name, uint32, float64)        \
    CPU_HOST_UNARY(name, uint32, complex128)     \
                                                 \
    CPU_HOST_UNARY(name, uint64, uint64)         \
                                                 \
    CPU_HOST_UNARY(name, int8, int8)             \
    CPU_HOST_UNARY(name, int8, int16)            \
    CPU_HOST_UNARY(name, int8, int32)            \
    CPU_HOST_UNARY(name, int8, int64)            \
    CPU_HOST_UNARY(name, int8, bfloat16)         \
    CPU_HOST_NOIMPL(name, int8, float16)         \
    CPU_HOST_UNARY(name, int8, float32)          \
    CPU_HOST_UNARY(name, int8, float64)          \
    CPU_HOST_NOIMPL(name, int8, complex32)       \
    CPU_HOST_UNARY(name, int8, complex64)        \
    CPU_HOST_UNARY(name, int8, complex128)       \
                                                 \
    CPU_HOST_UNARY(name, int16, int16)           \
    CPU_HOST_UNARY(name, int16, int32)           \
    CPU_HOST_UNARY(name, int16, int64)           \
    CPU_HOST_UNARY(name, int16, float32)         \
    CPU_HOST_UNARY(name, int16, float64)         \
    CPU_HOST_UNARY(name, int16, complex64)       \
    CPU_HOST_UNARY(name, int16, complex128)      \
                                                 \
    CPU_HOST_UNARY(name, int32, int32)           \
    CPU_HOST_UNARY(name, int32, int64)           \
    CPU_HOST_UNARY(name, int32, float64)         \
    CPU_HOST_UNARY(name, int32, complex128)      \
                                                 \
    CPU_HOST_UNARY(name, int64, int64)           \
                                                 \
    CPU_HOST_UNARY(name, bfloat16, bfloat16)     \
    CPU_HOST_UNARY(name, bfloat16, float32)      \
    CPU_HOST_UNARY(name, bfloat16, float64)      \
    CPU_HOST_UNARY(name, bfloat16, complex64)    \
    CPU_HOST_UNARY(name, bfloat16, complex128)   \
                                                 \
    CPU_HOST_NOIMPL(name, float16, float16)      \
    CPU_HOST_NOIMPL(name, float16, float32)      \
    CPU_HOST_NOIMPL(name, float16, float64)      \
    CPU_HOST_NOIMPL(name, float16, complex32)    \
    CPU_HOST_NOIMPL(name, float16, complex64)    \
    CPU_HOST_NOIMPL(name, float16, complex128)   \
                                                 \
    CPU_HOST_UNARY(name, float32, float32)       \
    CPU_HOST_UNARY(name, float32, float64)       \
    CPU_HOST_UNARY(name, float32, complex64)     \
    CPU_HOST_UNARY(name, float32, complex128)    \
                                                 \
    CPU_HOST_UNARY(name, float64, float64)       \
    CPU_HOST_UNARY(name, float64, complex128)    \
                                                 \
    CPU_HOST_NOIMPL(name, complex32, complex32)  \
    CPU_HOST_NOIMPL(name, complex32, complex64)  \
    CPU_HOST_NOIMPL(name, complex32, complex128) \
                                                 \
    CPU_HOST_UNARY(name, complex64, complex64)   \
    CPU_HOST_UNARY(name, complex64, complex128)  \
                                                 \
    CPU_HOST_UNARY(name, complex128, complex128)

#define CPU_HOST_ALL_UNARY_INIT(name, func, hfunc) \
    CPU_HOST_UNARY_INIT(name, func, bool, bool),            \
    CPU_HOST_UNARY_INIT(name, func, bool, uint8),           \
    CPU_HOST_UNARY_INIT(name, func, bool, uint16),          \
    CPU_HOST_UNARY_INIT(name, func, bool, uint32),          \
    CPU_HOST_UNARY_INIT(name, func, bool, uint64),          \
    CPU_HOST_UNARY_INIT(name, func, bool, int8),            \
    CPU_HOST_UNARY_INIT(name, func, bool, int16),           \
    CPU_HOST_UNARY_INIT(name, func, bool, int32),           \
    CPU_HOST_UNARY_INIT(name, func, bool, int64),           \
    CPU_HOST_UNARY_INIT(name, func, bool, bfloat16),        \
    CPU_HOST_UNARY_INIT(name, hfunc, bool, float16),        \
    CPU_HOST_UNARY_INIT(name, func, bool, float32),         \
    CPU_HOST_UNARY_INIT(name, func, bool, float64),         \
    CPU_HOST_UNARY_INIT(name, func, bool, complex32),       \
    CPU_HOST_UNARY_INIT(name, func, bool, complex64),       \
    CPU_HOST_UNARY_INIT(name, func, bool, complex128),      \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, uint8, uint8),          \
    CPU_HOST_UNARY_INIT(name, func, uint8, uint16),         \
    CPU_HOST_UNARY_INIT(name, func, uint8, uint32),         \
    CPU_HOST_UNARY_INIT(name, func, uint8, uint64),         \
    CPU_HOST_UNARY_INIT(name, func, uint8, int16),          \
    CPU_HOST_UNARY_INIT(name, func, uint8, int32),          \
    CPU_HOST_UNARY_INIT(name, func, uint8, int64),          \
    CPU_HOST_UNARY_INIT(name, func, uint8, bfloat16),       \
    CPU_HOST_UNARY_INIT(name, hfunc, uint8, float16),       \
    CPU_HOST_UNARY_INIT(name, func, uint8, float32),        \
    CPU_HOST_UNARY_INIT(name, func, uint8, float64),        \
    CPU_HOST_UNARY_INIT(name, func, uint8, complex32),      \
    CPU_HOST_UNARY_INIT(name, func, uint8, complex64),      \
    CPU_HOST_UNARY_INIT(name, func, uint8, complex128),     \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, uint16, uint16),        \
    CPU_HOST_UNARY_INIT(name, func, uint16, uint32),        \
    CPU_HOST_UNARY_INIT(name, func, uint16, uint64),        \
    CPU_HOST_UNARY_INIT(name, func, uint16, int32),         \
    CPU_HOST_UNARY_INIT(name, func, uint16, int64),         \
    CPU_HOST_UNARY_INIT(name, func, uint16, float32),       \
    CPU_HOST_UNARY_INIT(name, func, uint16, float64),       \
    CPU_HOST_UNARY_INIT(name, func, uint16, complex64),     \
    CPU_HOST_UNARY_INIT(name, func, uint16, complex128),    \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, uint32, uint32),        \
    CPU_HOST_UNARY_INIT(name, func, uint32, uint64),        \
    CPU_HOST_UNARY_INIT(name, func, uint32, int64),         \
    CPU_HOST_UNARY_INIT(name, func, uint32, float64),       \
    CPU_HOST_UNARY_INIT(name, func, uint32, complex128),    \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, uint64, uint64),        \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, int8, int8),            \
    CPU_HOST_UNARY_INIT(name, func, int8, int16),           \
    CPU_HOST_UNARY_INIT(name, func, int8, int32),           \
    CPU_HOST_UNARY_INIT(name, func, int8, int64),           \
    CPU_HOST_UNARY_INIT(name, func, int8, bfloat16),        \
    CPU_HOST_UNARY_INIT(name, hfunc, int8, float16),        \
    CPU_HOST_UNARY_INIT(name, func, int8, float32),         \
    CPU_HOST_UNARY_INIT(name, func, int8, float64),         \
    CPU_HOST_UNARY_INIT(name, func, int8, complex32),       \
    CPU_HOST_UNARY_INIT(name, func, int8, complex64),       \
    CPU_HOST_UNARY_INIT(name, func, int8, complex128),      \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, int16, int16),          \
    CPU_HOST_UNARY_INIT(name, func, int16, int32),          \
    CPU_HOST_UNARY_INIT(name, func, int16, int64),          \
    CPU_HOST_UNARY_INIT(name, func, int16, float32),        \
    CPU_HOST_UNARY_INIT(name, func, int16, float64),        \
    CPU_HOST_UNARY_INIT(name, func, int16, complex64),      \
    CPU_HOST_UNARY_INIT(name, func, int16, complex128),     \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, int32, int32),          \
    CPU_HOST_UNARY_INIT(name, func, int32, int64),          \
    CPU_HOST_UNARY_INIT(name, func, int32, float64),        \
    CPU_HOST_UNARY_INIT(name, func, int32, complex128),     \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, int64, int64),          \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, bfloat16, bfloat16),    \
    CPU_HOST_UNARY_INIT(name, func, bfloat16, float32),     \
    CPU_HOST_UNARY_INIT(name, func, bfloat16, float64),     \
    CPU_HOST_UNARY_INIT(name, func, bfloat16, complex64),   \
    CPU_HOST_UNARY_INIT(name, func, bfloat16, complex128),  \
                                                            \
    CPU_HOST_UNARY_INIT(name, hfunc, float16, float16),     \
    CPU_HOST_UNARY_INIT(name, func, float16, float32),      \
    CPU_HOST_UNARY_INIT(name, func, float16, float64),      \
    CPU_HOST_UNARY_INIT(name, func, float16, complex32),    \
    CPU_HOST_UNARY_INIT(name, func, float16, complex64),    \
    CPU_HOST_UNARY_INIT(name, func, float16, complex128),   \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, float32, float32),      \
    CPU_HOST_UNARY_INIT(name, func, float32, float64),      \
    CPU_HOST_UNARY_INIT(name, func, float32, complex64),    \
    CPU_HOST_UNARY_INIT(name, func, float32, complex128),   \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, float64, float64),      \
    CPU_HOST_UNARY_INIT(name, func, float64, complex128),   \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, complex32, complex32),  \
    CPU_HOST_UNARY_INIT(name, func, complex32, complex64),  \
    CPU_HOST_UNARY_INIT(name, func, complex32, complex128), \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, complex64, complex64),  \
    CPU_HOST_UNARY_INIT(name, func, complex64, complex128), \
                                                            \
    CPU_HOST_UNARY_INIT(name, func, complex128, complex128)


CPU_HOST_ALL_UNARY(copy)
CPU_HOST_ALL_UNARY(abs)


static const gm_kernel_init_t unary_copy[] = {
  /* COPY */
  CPU_HOST_ALL_UNARY_INIT(copy, copy, copy),
  CPU_HOST_ALL_UNARY_INIT(abs, abs, abs),

  { .name = NULL, .sig = NULL }
};

/*****************************************************************************/
/*                                Bitwise NOT                                */
/*****************************************************************************/

CPU_HOST_UNARY(invert, bool, bool)

CPU_HOST_UNARY(invert, uint8, uint8)
CPU_HOST_UNARY(invert, uint16, uint16)
CPU_HOST_UNARY(invert, uint32, uint32)
CPU_HOST_UNARY(invert, uint64, uint64)

CPU_HOST_UNARY(invert, int8, int8)
CPU_HOST_UNARY(invert, int16, int16)
CPU_HOST_UNARY(invert, int32, int32)
CPU_HOST_UNARY(invert, int64, int64)


static const gm_kernel_init_t unary_invert[] = {
  /* INVERT */
  CPU_HOST_UNARY_INIT(invert, invert, bool, bool),

  CPU_HOST_UNARY_INIT(invert, invert, uint8, uint8),
  CPU_HOST_UNARY_INIT(invert, invert, uint16, uint16),
  CPU_HOST_UNARY_INIT(invert, invert, uint32, uint32),
  CPU_HOST_UNARY_INIT(invert, invert, uint64, uint64),

  CPU_HOST_UNARY_INIT(invert, invert, int8, int8),
  CPU_HOST_UNARY_INIT(invert, invert, int16, int16),
  CPU_HOST_UNARY_INIT(invert, invert, int32, int32),
  CPU_HOST_UNARY_INIT(invert, invert, int64, int64),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                  Negative                                 */
/*****************************************************************************/

CPU_HOST_UNARY(negative, uint8, int16)
CPU_HOST_UNARY(negative, uint16, int32)
CPU_HOST_UNARY(negative, uint32, int64)

CPU_HOST_UNARY(negative, int8, int8)
CPU_HOST_UNARY(negative, int16, int16)
CPU_HOST_UNARY(negative, int32, int32)
CPU_HOST_UNARY(negative, int64, int64)

CPU_HOST_UNARY(negative, bfloat16, bfloat16)
CPU_HOST_NOIMPL(negative, float16, float16)
CPU_HOST_UNARY(negative, float32, float32)
CPU_HOST_UNARY(negative, float64, float64)

CPU_HOST_NOIMPL(negative, complex32, complex32)
CPU_HOST_UNARY(negative, complex64, complex64)
CPU_HOST_UNARY(negative, complex128, complex128)


static const gm_kernel_init_t unary_negative[] = {
  /* NEGATIVE */
  CPU_HOST_UNARY_INIT(negative, negative, uint8, int16),
  CPU_HOST_UNARY_INIT(negative, negative, uint16, int32),
  CPU_HOST_UNARY_INIT(negative, negative, uint32, int64),

  CPU_HOST_UNARY_INIT(negative, negative, int8, int8),
  CPU_HOST_UNARY_INIT(negative, negative, int16, int16),
  CPU_HOST_UNARY_INIT(negative, negative, int32, int32),
  CPU_HOST_UNARY_INIT(negative, negative, int64, int64),

  CPU_HOST_UNARY_INIT(negative, negative, bfloat16, bfloat16),
  CPU_HOST_UNARY_INIT(negative, negative, float16, float16),
  CPU_HOST_UNARY_INIT(negative, negative, float32, float32),
  CPU_HOST_UNARY_INIT(negative, negative, float64, float64),

  CPU_HOST_UNARY_INIT(negative, negative, complex32, complex32),
  CPU_HOST_UNARY_INIT(negative, negative, complex64, complex64),
  CPU_HOST_UNARY_INIT(negative, negative, complex128, complex128),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Math                                   */
/*****************************************************************************/

#define _CPU_ALL_HALF_MATH(name) \
    CPU_HOST_UNARY(name##f16, uint8, float16)   \
    CPU_HOST_UNARY(name##f16, int8, float16)    \
    CPU_HOST_UNARY(name##f16, float16, float16)

#define _CPU_ALL_HALF_MATH_NOIMPL(name) \
    CPU_HOST_NOIMPL(name##f16, uint8, float16)   \
    CPU_HOST_NOIMPL(name##f16, int8, float16)    \
    CPU_HOST_NOIMPL(name##f16, float16, float16)

#define _CPU_ALL_COMPLEX_MATH(name) \
    CPU_HOST_NOIMPL(name, complex32, complex32)  \
    CPU_HOST_UNARY(name, complex64, complex64)   \
    CPU_HOST_UNARY(name, complex128, complex128)

#define _CPU_ALL_COMPLEX_MATH_NOIMPL(name) \
    CPU_HOST_NOIMPL(name, complex32, complex32)   \
    CPU_HOST_NOIMPL(name, complex64, complex64)   \
    CPU_HOST_NOIMPL(name, complex128, complex128)

#define _CPU_ALL_REAL_MATH(name) \
    CPU_HOST_UNARY(name##b16, bfloat16, bfloat16) \
    CPU_HOST_UNARY(name##f, uint16, float32)      \
    CPU_HOST_UNARY(name##f, int16, float32)       \
    CPU_HOST_UNARY(name##f, float32, float32)     \
    CPU_HOST_UNARY(name, uint32, float64)         \
    CPU_HOST_UNARY(name, int32, float64)          \
    CPU_HOST_UNARY(name, float64, float64)        \

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
    CPU_HOST_UNARY_INIT(name, name##f16, uint8, float16),     \
    CPU_HOST_UNARY_INIT(name, name##f16, int8, float16),      \
    CPU_HOST_UNARY_INIT(name, name##f16, float16, float16),   \
                                                              \
    CPU_HOST_UNARY_INIT(name, name##b16, bfloat16, bfloat16), \
                                                              \
    CPU_HOST_UNARY_INIT(name, name##f, uint16, float32),      \
    CPU_HOST_UNARY_INIT(name, name##f, int16, float32),       \
    CPU_HOST_UNARY_INIT(name, name##f, float32, float32),     \
                                                              \
    CPU_HOST_UNARY_INIT(name, name, uint32, float64),         \
    CPU_HOST_UNARY_INIT(name, name, int32, float64),          \
    CPU_HOST_UNARY_INIT(name, name, float64, float64),        \
                                                              \
    CPU_HOST_UNARY_INIT(name, name, complex32, complex32),    \
    CPU_HOST_UNARY_INIT(name, name, complex64, complex64),    \
    CPU_HOST_UNARY_INIT(name, name, complex128, complex128)


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

typedef _Bool bool;

static const gm_kernel_set_t *
unary_copy_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                     const int64_t li[], int nin, int nout, bool check_broadcast,
                     ndt_context_t *ctx)
{
    return cpu_unary_typecheck(copy_kernel_location, spec, f, types, li,
                               nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
unary_invert_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                       const int64_t li[], int nin, int nout, bool check_broadcast,
                       ndt_context_t *ctx)
{
    return cpu_unary_typecheck(invert_kernel_location, spec, f, types, li,
                               nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
unary_negative_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                         const int64_t li[], int nin, int nout, bool check_broadcast,
                         ndt_context_t *ctx)
{
    return cpu_unary_typecheck(negative_kernel_location, spec, f, types, li,
                               nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
unary_math_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                     const int64_t li[], int nin, int nout, bool check_broadcast,
                     ndt_context_t *ctx)
{
    return cpu_unary_typecheck(math_kernel_location, spec, f, types, li,
                               nin, nout, check_broadcast, ctx);
}

int
gm_init_cpu_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = unary_copy; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &unary_copy_typecheck) < 0) {
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

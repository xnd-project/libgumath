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
#include "cpu_device_binary.h"


/****************************************************************************/
/*                     Optimized dispatch (exact casting)                   */
/****************************************************************************/

/* Structured kernel locations for fast lookup. */
static int
kernel_location(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx)
{
    const ndt_t *t0 = ndt_dtype(in0);
    const ndt_t *t1 = ndt_dtype(in1);

    switch (t0->tag) {
    case Uint8: {
        switch (t1->tag) {
        case Uint8: return 0;
        case Uint16: return 8;
        case Uint32: return 16;
        case Uint64: return 24;

        case Int8: return 32;
        case Int16: return 40;
        case Int32: return 48;
        case Int64: return 56;

        case Float16: return 64;
        case Float32: return 72;
        case Float64: return 80;

        case Complex32: return 88;
        case Complex64: return 96;
        case Complex128: return 104;

        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Uint8: return 112;
        case Uint16: return 120;
        case Uint32: return 128;
        case Uint64: return 136;

        case Int8: return 144;
        case Int16: return 152;
        case Int32: return 160;
        case Int64: return 168;

        case Float16: return 176;
        case Float32: return 184;
        case Float64: return 192;

        case Complex32: return 200;
        case Complex64: return 208;
        case Complex128: return 216;

        default: goto invalid_combination;
        }
    }
    case Uint32: {
        switch (t1->tag) {
        case Uint8: return 224;
        case Uint16: return 232;
        case Uint32: return 240;
        case Uint64: return 248;

        case Int8: return 256;
        case Int16: return 264;
        case Int32: return 272;
        case Int64: return 280;

        case Float16: return 288;
        case Float32: return 296;
        case Float64: return 304;

        case Complex32: return 312;
        case Complex64: return 320;
        case Complex128: return 328;

        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Uint8: return 336;
        case Uint16: return 344;
        case Uint32: return 352;
        case Uint64: return 360;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Uint8: return 368;
        case Uint16: return 376;
        case Uint32: return 384;

        case Int8: return 392;
        case Int16: return 400;
        case Int32: return 408;
        case Int64: return 416;

        case Float16: return 424;
        case Float32: return 432;
        case Float64: return 440;

        case Complex32: return 448;
        case Complex64: return 456;
        case Complex128: return 464;

        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Uint8: return 472;
        case Uint16: return 480;
        case Uint32: return 488;

        case Int8: return 496;
        case Int16: return 504;
        case Int32: return 512;
        case Int64: return 520;

        case Float16: return 528;
        case Float32: return 536;
        case Float64: return 544;

        case Complex32: return 552;
        case Complex64: return 560;
        case Complex128: return 568;

        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Uint8: return 576;
        case Uint16: return 584;
        case Uint32: return 592;

        case Int8: return 600;
        case Int16: return 608;
        case Int32: return 616;
        case Int64: return 624;

        case Float16: return 632;
        case Float32: return 640;
        case Float64: return 648;

        case Complex32: return 656;
        case Complex64: return 664;
        case Complex128: return 672;

        default: goto invalid_combination;
        }
    }
    case Int64: {
        switch (t1->tag) {
        case Uint8: return 680;
        case Uint16: return 688;
        case Uint32: return 696;

        case Int8: return 704;
        case Int16: return 712;
        case Int32: return 720;
        case Int64: return 728;

        default: goto invalid_combination;
        }
    }

    case Float16: {
        switch (t1->tag) {
        case Uint8: return 736;
        case Uint16: return 744;
        case Uint32: return 752;

        case Int8: return 760;
        case Int16: return 768;
        case Int32: return 776;

        case Float16: return 784;
        case Float32: return 792;
        case Float64: return 800;

        case Complex32: return 808;
        case Complex64: return 816;
        case Complex128: return 824;

        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (t1->tag) {
        case Uint8: return 832;
        case Uint16: return 840;
        case Uint32: return 848;

        case Int8: return 856;
        case Int16: return 864;
        case Int32: return 872;

        case Float16: return 880;
        case Float32: return 888;
        case Float64: return 896;

        case Complex32: return 904;
        case Complex64: return 912;
        case Complex128: return 920;

        default: goto invalid_combination;
        }
    }

    case Float64: {
        switch (t1->tag) {
        case Uint8: return 928;
        case Uint16: return 936;
        case Uint32: return 944;

        case Int8: return 952;
        case Int16: return 960;
        case Int32: return 968;

        case Float16: return 976;
        case Float32: return 984;
        case Float64: return 992;

        case Complex32: return 1000;
        case Complex64: return 1008;
        case Complex128: return 1016;

        default: goto invalid_combination;
        }
    }

    case Complex32: {
        switch (t1->tag) {
        case Uint8: return 1024;
        case Uint16: return 1032;
        case Uint32: return 1040;

        case Int8: return 1048;
        case Int16: return 1056;
        case Int32: return 1064;

        case Float16: return 1072;
        case Float32: return 1080;
        case Float64: return 1088;

        case Complex32: return 1096;
        case Complex64: return 1104;
        case Complex128: return 1112;

        default: goto invalid_combination;
        }
    }

    case Complex64: {
        switch (t1->tag) {
        case Uint8: return 1120;
        case Uint16: return 1128;
        case Uint32: return 1136;

        case Int8: return 1144;
        case Int16: return 1152;
        case Int32: return 1160;

        case Float16: return 1168;
        case Float32: return 1176;
        case Float64: return 1184;

        case Complex32: return 1192;
        case Complex64: return 1200;
        case Complex128: return 1208;

        default: goto invalid_combination;
        }
    }

    case Complex128: {
        switch (t1->tag) {
        case Uint8: return 1216;
        case Uint16: return 1224;
        case Uint32: return 1232;

        case Int8: return 1240;
        case Int16: return 1248;
        case Int32: return 1256;

        case Float16: return 1264;
        case Float32: return 1272;
        case Float64: return 1280;

        case Complex32: return 1288;
        case Complex64: return 1296;
        case Complex128: return 1304;

        default: goto invalid_combination;
        }
    }

    default:
        goto invalid_combination;
    }

invalid_combination:
    ndt_err_format(ctx, NDT_ValueError, "invalid dtype");
    return -1;
}

#define CPU_HOST_BINARY(name, t0, t1, t2) \
static int                                                                       \
gm_cpu_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                \
    const char *in0 = apply_index(&stack[0]);                                    \
    const char *in1 = apply_index(&stack[1]);                                    \
    char *out = apply_index(&stack[2]);                                          \
    int64_t N = xnd_fixed_shape(&stack[0]);                                      \
    (void)ctx;                                                                   \
                                                                                 \
    gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2(in0, in1, out, N);      \
                                                                                 \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                             \
        binary_update_bitmap1D(stack);                                           \
    }                                                                            \
                                                                                 \
    return 0;                                                                    \
}                                                                                \
                                                                                 \
static int                                                                       \
gm_cpu_0D_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                                \
    const char *in0 = stack[0].ptr;                                              \
    const char *in1 = stack[1].ptr;                                              \
    char *out = stack[2].ptr;                                                    \
    (void)ctx;                                                                   \
                                                                                 \
    gm_cpu_device_0D_##name##_##t0##_##t1##_##t2(in0, in1, out);                 \
                                                                                 \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                             \
        binary_update_bitmap(stack);                                             \
    }                                                                            \
                                                                                 \
    return 0;                                                                    \
}

#define CPU_HOST_NOIMPL(name, t0, t1, t2) \
static int                                                                       \
gm_cpu_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                \
    (void)stack;                                                                 \
                                                                                 \
    ndt_err_format(ctx, NDT_NotImplementedError,                                 \
        "implementation for " STRINGIZE(name) " : "                              \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                    \
        " currently requires double rounding");                                  \
                                                                                 \
    return -1;                                                                   \
}                                                                                \
                                                                                 \
static int                                                                       \
gm_cpu_0D_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                                \
    (void)stack;                                                                 \
                                                                                 \
    ndt_err_format(ctx, NDT_NotImplementedError,                                 \
        "implementation for " STRINGIZE(name) " : "                              \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                    \
        " currently requires double rounding");                                  \
                                                                                 \
    return -1;                                                                   \
}

#define CPU_HOST_NOKERN(name, t0, t1, t2) \
static int                                                                       \
gm_cpu_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                \
    (void)stack;                                                                 \
                                                                                 \
    ndt_err_format(ctx, NDT_TypeError,                                           \
        "no kernel for " STRINGIZE(name) " : "                                   \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));                  \
                                                                                 \
    return -1;                                                                   \
}                                                                                \
                                                                                 \
static int                                                                       \
gm_cpu_0D_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                                \
    (void)stack;                                                                 \
                                                                                 \
    ndt_err_format(ctx, NDT_TypeError,                                           \
        "no kernel for " STRINGIZE(name) " : "                                   \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));                  \
                                                                                 \
    return -1;                                                                   \
}

#define CPU_HOST_BINARY_INIT(func, t0, t1, t2) \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),             \
    .Opt = gm_cpu_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * ?" STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),           \
    .Opt = gm_cpu_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * " STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),           \
    .Opt = gm_cpu_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "... * ?" STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),          \
    .Opt = gm_cpu_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * " STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * " STRINGIZE(t2),    \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),  \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * " STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),  \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 },                                                  \
                                                                                                   \
  { .name = STRINGIZE(func),                                                                       \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2), \
    .C = gm_cpu_0D_##func##_##t0##_##t1##_##t2 }


/*****************************************************************************/
/*                                 Arithmetic                                */
/*****************************************************************************/

#define CPU_HOST_ALL_ARITHMETIC(name) \
    CPU_HOST_BINARY(name, uint8, uint8, uint8)                \
    CPU_HOST_BINARY(name, uint8, uint16, uint16)              \
    CPU_HOST_BINARY(name, uint8, uint32, uint32)              \
    CPU_HOST_BINARY(name, uint8, uint64, uint64)              \
    CPU_HOST_BINARY(name, uint8, int8, int16)                 \
    CPU_HOST_BINARY(name, uint8, int16, int16)                \
    CPU_HOST_BINARY(name, uint8, int32, int32)                \
    CPU_HOST_BINARY(name, uint8, int64, int64)                \
    CPU_HOST_NOIMPL(name, uint8, float16, float16)            \
    CPU_HOST_BINARY(name, uint8, float32, float32)            \
    CPU_HOST_BINARY(name, uint8, float64, float64)            \
    CPU_HOST_NOIMPL(name, uint8, complex32, complex32)        \
    CPU_HOST_BINARY(name, uint8, complex64, complex64)        \
    CPU_HOST_BINARY(name, uint8, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, uint16, uint8, uint16)              \
    CPU_HOST_BINARY(name, uint16, uint16, uint16)             \
    CPU_HOST_BINARY(name, uint16, uint32, uint32)             \
    CPU_HOST_BINARY(name, uint16, uint64, uint64)             \
    CPU_HOST_BINARY(name, uint16, int8, int32)                \
    CPU_HOST_BINARY(name, uint16, int16, int32)               \
    CPU_HOST_BINARY(name, uint16, int32, int32)               \
    CPU_HOST_BINARY(name, uint16, int64, int64)               \
    CPU_HOST_NOIMPL(name, uint16, float16, float32)           \
    CPU_HOST_BINARY(name, uint16, float32, float32)           \
    CPU_HOST_BINARY(name, uint16, float64, float64)           \
    CPU_HOST_NOIMPL(name, uint16, complex32, complex64)       \
    CPU_HOST_BINARY(name, uint16, complex64, complex64)       \
    CPU_HOST_BINARY(name, uint16, complex128, complex128)     \
                                                              \
    CPU_HOST_BINARY(name, uint32, uint8, uint32)              \
    CPU_HOST_BINARY(name, uint32, uint16, uint32)             \
    CPU_HOST_BINARY(name, uint32, uint32, uint32)             \
    CPU_HOST_BINARY(name, uint32, uint64, uint64)             \
    CPU_HOST_BINARY(name, uint32, int8, int64)                \
    CPU_HOST_BINARY(name, uint32, int16, int64)               \
    CPU_HOST_BINARY(name, uint32, int32, int64)               \
    CPU_HOST_BINARY(name, uint32, int64, int64)               \
    CPU_HOST_NOIMPL(name, uint32, float16, float64)           \
    CPU_HOST_BINARY(name, uint32, float32, float64)           \
    CPU_HOST_BINARY(name, uint32, float64, float64)           \
    CPU_HOST_NOIMPL(name, uint32, complex32, complex128)      \
    CPU_HOST_BINARY(name, uint32, complex64, complex128)      \
    CPU_HOST_BINARY(name, uint32, complex128, complex128)     \
                                                              \
    CPU_HOST_BINARY(name, uint64, uint8, uint64)              \
    CPU_HOST_BINARY(name, uint64, uint16, uint64)             \
    CPU_HOST_BINARY(name, uint64, uint32, uint64)             \
    CPU_HOST_BINARY(name, uint64, uint64, uint64)             \
                                                              \
    CPU_HOST_BINARY(name, int8, uint8, int16)                 \
    CPU_HOST_BINARY(name, int8, uint16, int32)                \
    CPU_HOST_BINARY(name, int8, uint32, int64)                \
    CPU_HOST_BINARY(name, int8, int8, int8)                   \
    CPU_HOST_BINARY(name, int8, int16, int16)                 \
    CPU_HOST_BINARY(name, int8, int32, int32)                 \
    CPU_HOST_BINARY(name, int8, int64, int64)                 \
    CPU_HOST_NOIMPL(name, int8, float16, float16)             \
    CPU_HOST_BINARY(name, int8, float32, float32)             \
    CPU_HOST_BINARY(name, int8, float64, float64)             \
    CPU_HOST_NOIMPL(name, int8, complex32, complex32)         \
    CPU_HOST_BINARY(name, int8, complex64, complex64)         \
    CPU_HOST_BINARY(name, int8, complex128, complex128)       \
                                                              \
    CPU_HOST_BINARY(name, int16, uint8, int16)                \
    CPU_HOST_BINARY(name, int16, uint16, int32)               \
    CPU_HOST_BINARY(name, int16, uint32, int64)               \
    CPU_HOST_BINARY(name, int16, int8, int16)                 \
    CPU_HOST_BINARY(name, int16, int16, int16)                \
    CPU_HOST_BINARY(name, int16, int32, int32)                \
    CPU_HOST_BINARY(name, int16, int64, int64)                \
    CPU_HOST_NOIMPL(name, int16, float16, float32)            \
    CPU_HOST_BINARY(name, int16, float32, float32)            \
    CPU_HOST_BINARY(name, int16, float64, float64)            \
    CPU_HOST_NOIMPL(name, int16, complex32, complex64)        \
    CPU_HOST_BINARY(name, int16, complex64, complex64)        \
    CPU_HOST_BINARY(name, int16, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, int32, uint8, int32)                \
    CPU_HOST_BINARY(name, int32, uint16, int32)               \
    CPU_HOST_BINARY(name, int32, uint32, int64)               \
    CPU_HOST_BINARY(name, int32, int8, int32)                 \
    CPU_HOST_BINARY(name, int32, int16, int32)                \
    CPU_HOST_BINARY(name, int32, int32, int32)                \
    CPU_HOST_BINARY(name, int32, int64, int64)                \
    CPU_HOST_NOIMPL(name, int32, float16, float64)            \
    CPU_HOST_BINARY(name, int32, float32, float64)            \
    CPU_HOST_BINARY(name, int32, float64, float64)            \
    CPU_HOST_NOIMPL(name, int32, complex32, complex128)       \
    CPU_HOST_BINARY(name, int32, complex64, complex128)       \
    CPU_HOST_BINARY(name, int32, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, int64, uint8, int64)                \
    CPU_HOST_BINARY(name, int64, uint16, int64)               \
    CPU_HOST_BINARY(name, int64, uint32, int64)               \
    CPU_HOST_BINARY(name, int64, int8, int64)                 \
    CPU_HOST_BINARY(name, int64, int16, int64)                \
    CPU_HOST_BINARY(name, int64, int32, int64)                \
    CPU_HOST_BINARY(name, int64, int64, int64)                \
                                                              \
    CPU_HOST_NOIMPL(name, float16, uint8, float16)            \
    CPU_HOST_NOIMPL(name, float16, uint16, float32)           \
    CPU_HOST_NOIMPL(name, float16, uint32, float64)           \
    CPU_HOST_NOIMPL(name, float16, int8, float16)             \
    CPU_HOST_NOIMPL(name, float16, int16, float32)            \
    CPU_HOST_NOIMPL(name, float16, int32, float64)            \
    CPU_HOST_NOIMPL(name, float16, float16, float16)          \
    CPU_HOST_NOIMPL(name, float16, float32, float32)          \
    CPU_HOST_NOIMPL(name, float16, float64, float64)          \
    CPU_HOST_NOIMPL(name, float16, complex32, complex32)      \
    CPU_HOST_NOIMPL(name, float16, complex64, complex64)      \
    CPU_HOST_NOIMPL(name, float16, complex128, complex128)    \
                                                              \
    CPU_HOST_BINARY(name, float32, uint8, float32)            \
    CPU_HOST_BINARY(name, float32, uint16, float32)           \
    CPU_HOST_BINARY(name, float32, uint32, float64)           \
    CPU_HOST_BINARY(name, float32, int8, float32)             \
    CPU_HOST_BINARY(name, float32, int16, float32)            \
    CPU_HOST_BINARY(name, float32, int32, float64)            \
    CPU_HOST_NOIMPL(name, float32, float16, float32)          \
    CPU_HOST_BINARY(name, float32, float32, float32)          \
    CPU_HOST_BINARY(name, float32, float64, float64)          \
    CPU_HOST_NOIMPL(name, float32, complex32, complex64)      \
    CPU_HOST_BINARY(name, float32, complex64, complex64)      \
    CPU_HOST_BINARY(name, float32, complex128, complex128)    \
                                                              \
    CPU_HOST_BINARY(name, float64, uint8, float64)            \
    CPU_HOST_BINARY(name, float64, uint16, float64)           \
    CPU_HOST_BINARY(name, float64, uint32, float64)           \
    CPU_HOST_BINARY(name, float64, int8, float64)             \
    CPU_HOST_BINARY(name, float64, int16, float64)            \
    CPU_HOST_BINARY(name, float64, int32, float64)            \
    CPU_HOST_NOIMPL(name, float64, float16, float64)          \
    CPU_HOST_BINARY(name, float64, float32, float64)          \
    CPU_HOST_BINARY(name, float64, float64, float64)          \
    CPU_HOST_NOIMPL(name, float64, complex32, complex128)     \
    CPU_HOST_BINARY(name, float64, complex64, complex128)     \
    CPU_HOST_BINARY(name, float64, complex128, complex128)    \
                                                              \
    CPU_HOST_NOIMPL(name, complex32, uint8, complex32)        \
    CPU_HOST_NOIMPL(name, complex32, uint16, complex64)       \
    CPU_HOST_NOIMPL(name, complex32, uint32, complex128)      \
    CPU_HOST_NOIMPL(name, complex32, int8, complex32)         \
    CPU_HOST_NOIMPL(name, complex32, int16, complex64)        \
    CPU_HOST_NOIMPL(name, complex32, int32, complex128)       \
    CPU_HOST_NOIMPL(name, complex32, float16, complex32)      \
    CPU_HOST_NOIMPL(name, complex32, float32, complex64)      \
    CPU_HOST_NOIMPL(name, complex32, float64, complex128)     \
    CPU_HOST_NOIMPL(name, complex32, complex32, complex32)    \
    CPU_HOST_NOIMPL(name, complex32, complex64, complex64)    \
    CPU_HOST_NOIMPL(name, complex32, complex128, complex128)  \
                                                              \
    CPU_HOST_BINARY(name, complex64, uint8, complex64)        \
    CPU_HOST_BINARY(name, complex64, uint16, complex64)       \
    CPU_HOST_BINARY(name, complex64, uint32, complex128)      \
    CPU_HOST_BINARY(name, complex64, int8, complex64)         \
    CPU_HOST_BINARY(name, complex64, int16, complex64)        \
    CPU_HOST_BINARY(name, complex64, int32, complex128)       \
    CPU_HOST_NOIMPL(name, complex64, float16, complex64)      \
    CPU_HOST_BINARY(name, complex64, float32, complex64)      \
    CPU_HOST_BINARY(name, complex64, float64, complex128)     \
    CPU_HOST_NOIMPL(name, complex64, complex32, complex64)    \
    CPU_HOST_BINARY(name, complex64, complex64, complex64)    \
    CPU_HOST_BINARY(name, complex64, complex128, complex128)  \
                                                              \
    CPU_HOST_BINARY(name, complex128, uint8, complex128)      \
    CPU_HOST_BINARY(name, complex128, uint16, complex128)     \
    CPU_HOST_BINARY(name, complex128, uint32, complex128)     \
    CPU_HOST_BINARY(name, complex128, int8, complex128)       \
    CPU_HOST_BINARY(name, complex128, int16, complex128)      \
    CPU_HOST_BINARY(name, complex128, int32, complex128)      \
    CPU_HOST_NOIMPL(name, complex128, float16, complex128)    \
    CPU_HOST_BINARY(name, complex128, float32, complex128)    \
    CPU_HOST_BINARY(name, complex128, float64, complex128)    \
    CPU_HOST_NOIMPL(name, complex128, complex32, complex128)  \
    CPU_HOST_BINARY(name, complex128, complex64, complex128)  \
    CPU_HOST_BINARY(name, complex128, complex128, complex128)

#define CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN(name) \
    CPU_HOST_NOIMPL(name, uint8, uint8, float16)              \
    CPU_HOST_BINARY(name, uint8, uint16, float32)             \
    CPU_HOST_BINARY(name, uint8, uint32, float64)             \
    CPU_HOST_NOKERN(name, uint8, uint64, uint64)              \
    CPU_HOST_NOIMPL(name, uint8, int8, float16)               \
    CPU_HOST_BINARY(name, uint8, int16, float32)              \
    CPU_HOST_BINARY(name, uint8, int32, float64)              \
    CPU_HOST_NOKERN(name, uint8, int64, int64)                \
    CPU_HOST_NOIMPL(name, uint8, float16, float16)            \
    CPU_HOST_BINARY(name, uint8, float32, float32)            \
    CPU_HOST_BINARY(name, uint8, float64, float64)            \
    CPU_HOST_NOIMPL(name, uint8, complex32, complex32)        \
    CPU_HOST_BINARY(name, uint8, complex64, complex64)        \
    CPU_HOST_BINARY(name, uint8, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, uint16, uint8, float32)             \
    CPU_HOST_BINARY(name, uint16, uint16, float32)            \
    CPU_HOST_BINARY(name, uint16, uint32, float64)            \
    CPU_HOST_NOKERN(name, uint16, uint64, uint64)             \
    CPU_HOST_BINARY(name, uint16, int8, float32)              \
    CPU_HOST_BINARY(name, uint16, int16, float32)             \
    CPU_HOST_BINARY(name, uint16, int32, float64)             \
    CPU_HOST_NOKERN(name, uint16, int64, int64)               \
    CPU_HOST_NOIMPL(name, uint16, float16, float32)           \
    CPU_HOST_BINARY(name, uint16, float32, float32)           \
    CPU_HOST_BINARY(name, uint16, float64, float64)           \
    CPU_HOST_NOIMPL(name, uint16, complex32, complex64)       \
    CPU_HOST_BINARY(name, uint16, complex64, complex64)       \
    CPU_HOST_BINARY(name, uint16, complex128, complex128)     \
                                                              \
    CPU_HOST_BINARY(name, uint32, uint8, float64)             \
    CPU_HOST_BINARY(name, uint32, uint16, float64)            \
    CPU_HOST_BINARY(name, uint32, uint32, float64)            \
    CPU_HOST_NOKERN(name, uint32, uint64, uint64)             \
    CPU_HOST_BINARY(name, uint32, int8, float64)              \
    CPU_HOST_BINARY(name, uint32, int16, float64)             \
    CPU_HOST_BINARY(name, uint32, int32, float64)             \
    CPU_HOST_NOKERN(name, uint32, int64, int64)               \
    CPU_HOST_NOIMPL(name, uint32, float16, float64)           \
    CPU_HOST_BINARY(name, uint32, float32, float64)           \
    CPU_HOST_BINARY(name, uint32, float64, float64)           \
    CPU_HOST_NOIMPL(name, uint32, complex32, complex128)      \
    CPU_HOST_BINARY(name, uint32, complex64, complex128)      \
    CPU_HOST_BINARY(name, uint32, complex128, complex128)     \
                                                              \
    CPU_HOST_NOKERN(name, uint64, uint8, uint64)              \
    CPU_HOST_NOKERN(name, uint64, uint16, uint64)             \
    CPU_HOST_NOKERN(name, uint64, uint32, uint64)             \
    CPU_HOST_NOKERN(name, uint64, uint64, uint64)             \
                                                              \
    CPU_HOST_NOIMPL(name, int8, uint8, float16)               \
    CPU_HOST_BINARY(name, int8, uint16, float32)              \
    CPU_HOST_BINARY(name, int8, uint32, float64)              \
    CPU_HOST_NOIMPL(name, int8, int8, float16)                \
    CPU_HOST_BINARY(name, int8, int16, float32)               \
    CPU_HOST_BINARY(name, int8, int32, float64)               \
    CPU_HOST_NOKERN(name, int8, int64, int64)                 \
    CPU_HOST_NOIMPL(name, int8, float16, float16)             \
    CPU_HOST_BINARY(name, int8, float32, float32)             \
    CPU_HOST_BINARY(name, int8, float64, float64)             \
    CPU_HOST_NOIMPL(name, int8, complex32, complex32)         \
    CPU_HOST_BINARY(name, int8, complex64, complex64)         \
    CPU_HOST_BINARY(name, int8, complex128, complex128)       \
                                                              \
    CPU_HOST_BINARY(name, int16, uint8, float32)              \
    CPU_HOST_BINARY(name, int16, uint16, float32)             \
    CPU_HOST_BINARY(name, int16, uint32, float64)             \
    CPU_HOST_BINARY(name, int16, int8, float32)               \
    CPU_HOST_BINARY(name, int16, int16, float32)              \
    CPU_HOST_BINARY(name, int16, int32, float64)              \
    CPU_HOST_NOKERN(name, int16, int64, int64)                \
    CPU_HOST_NOIMPL(name, int16, float16, float32)            \
    CPU_HOST_BINARY(name, int16, float32, float32)            \
    CPU_HOST_BINARY(name, int16, float64, float64)            \
    CPU_HOST_NOIMPL(name, int16, complex32, complex64)        \
    CPU_HOST_BINARY(name, int16, complex64, complex64)        \
    CPU_HOST_BINARY(name, int16, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, int32, uint8, float64)              \
    CPU_HOST_BINARY(name, int32, uint16, float64)             \
    CPU_HOST_BINARY(name, int32, uint32, float64)             \
    CPU_HOST_BINARY(name, int32, int8, float64)               \
    CPU_HOST_BINARY(name, int32, int16, float64)              \
    CPU_HOST_BINARY(name, int32, int32, float64)              \
    CPU_HOST_NOKERN(name, int32, int64, int64)                \
    CPU_HOST_NOIMPL(name, int32, float16, float64)            \
    CPU_HOST_BINARY(name, int32, float32, float64)            \
    CPU_HOST_BINARY(name, int32, float64, float64)            \
    CPU_HOST_NOIMPL(name, int32, complex32, complex128)       \
    CPU_HOST_BINARY(name, int32, complex64, complex128)       \
    CPU_HOST_BINARY(name, int32, complex128, complex128)      \
                                                              \
    CPU_HOST_NOKERN(name, int64, uint8, int64)                \
    CPU_HOST_NOKERN(name, int64, uint16, int64)               \
    CPU_HOST_NOKERN(name, int64, uint32, int64)               \
    CPU_HOST_NOKERN(name, int64, int8, int64)                 \
    CPU_HOST_NOKERN(name, int64, int16, int64)                \
    CPU_HOST_NOKERN(name, int64, int32, int64)                \
    CPU_HOST_NOKERN(name, int64, int64, int64)                \
                                                              \
    CPU_HOST_NOIMPL(name, float16, uint8, float16)            \
    CPU_HOST_NOIMPL(name, float16, uint16, float32)           \
    CPU_HOST_NOIMPL(name, float16, uint32, float64)           \
    CPU_HOST_NOIMPL(name, float16, int8, float16)             \
    CPU_HOST_NOIMPL(name, float16, int16, float32)            \
    CPU_HOST_NOIMPL(name, float16, int32, float64)            \
    CPU_HOST_NOIMPL(name, float16, float16, float16)          \
    CPU_HOST_NOIMPL(name, float16, float32, float32)          \
    CPU_HOST_NOIMPL(name, float16, float64, float64)          \
    CPU_HOST_NOIMPL(name, float16, complex32, complex32)      \
    CPU_HOST_NOIMPL(name, float16, complex64, complex64)      \
    CPU_HOST_NOIMPL(name, float16, complex128, complex128)    \
                                                              \
    CPU_HOST_BINARY(name, float32, uint8, float32)            \
    CPU_HOST_BINARY(name, float32, uint16, float32)           \
    CPU_HOST_BINARY(name, float32, uint32, float64)           \
    CPU_HOST_BINARY(name, float32, int8, float32)             \
    CPU_HOST_BINARY(name, float32, int16, float32)            \
    CPU_HOST_BINARY(name, float32, int32, float64)            \
    CPU_HOST_NOIMPL(name, float32, float16, float32)          \
    CPU_HOST_BINARY(name, float32, float32, float32)          \
    CPU_HOST_BINARY(name, float32, float64, float64)          \
    CPU_HOST_NOIMPL(name, float32, complex32, complex64)      \
    CPU_HOST_BINARY(name, float32, complex64, complex64)      \
    CPU_HOST_BINARY(name, float32, complex128, complex128)    \
                                                              \
    CPU_HOST_BINARY(name, float64, uint8, float64)            \
    CPU_HOST_BINARY(name, float64, uint16, float64)           \
    CPU_HOST_BINARY(name, float64, uint32, float64)           \
    CPU_HOST_BINARY(name, float64, int8, float64)             \
    CPU_HOST_BINARY(name, float64, int16, float64)            \
    CPU_HOST_BINARY(name, float64, int32, float64)            \
    CPU_HOST_NOIMPL(name, float64, float16, float64)          \
    CPU_HOST_BINARY(name, float64, float32, float64)          \
    CPU_HOST_BINARY(name, float64, float64, float64)          \
    CPU_HOST_NOIMPL(name, float64, complex32, complex128)     \
    CPU_HOST_BINARY(name, float64, complex64, complex128)     \
    CPU_HOST_BINARY(name, float64, complex128, complex128)    \
                                                              \
    CPU_HOST_NOIMPL(name, complex32, uint8, complex32)        \
    CPU_HOST_NOIMPL(name, complex32, uint16, complex64)       \
    CPU_HOST_NOIMPL(name, complex32, uint32, complex128)      \
    CPU_HOST_NOIMPL(name, complex32, int8, complex32)         \
    CPU_HOST_NOIMPL(name, complex32, int16, complex64)        \
    CPU_HOST_NOIMPL(name, complex32, int32, complex128)       \
    CPU_HOST_NOIMPL(name, complex32, float16, complex32)      \
    CPU_HOST_NOIMPL(name, complex32, float32, complex64)      \
    CPU_HOST_NOIMPL(name, complex32, float64, complex128)     \
    CPU_HOST_NOIMPL(name, complex32, complex32, complex32)    \
    CPU_HOST_NOIMPL(name, complex32, complex64, complex64)    \
    CPU_HOST_NOIMPL(name, complex32, complex128, complex128)  \
                                                              \
    CPU_HOST_BINARY(name, complex64, uint8, complex64)        \
    CPU_HOST_BINARY(name, complex64, uint16, complex64)       \
    CPU_HOST_BINARY(name, complex64, uint32, complex128)      \
    CPU_HOST_BINARY(name, complex64, int8, complex64)         \
    CPU_HOST_BINARY(name, complex64, int16, complex64)        \
    CPU_HOST_BINARY(name, complex64, int32, complex128)       \
    CPU_HOST_NOIMPL(name, complex64, float16, complex64)      \
    CPU_HOST_BINARY(name, complex64, float32, complex64)      \
    CPU_HOST_BINARY(name, complex64, float64, complex128)     \
    CPU_HOST_NOIMPL(name, complex64, complex32, complex64)    \
    CPU_HOST_BINARY(name, complex64, complex64, complex64)    \
    CPU_HOST_BINARY(name, complex64, complex128, complex128)  \
                                                              \
    CPU_HOST_BINARY(name, complex128, uint8, complex128)      \
    CPU_HOST_BINARY(name, complex128, uint16, complex128)     \
    CPU_HOST_BINARY(name, complex128, uint32, complex128)     \
    CPU_HOST_BINARY(name, complex128, int8, complex128)       \
    CPU_HOST_BINARY(name, complex128, int16, complex128)      \
    CPU_HOST_BINARY(name, complex128, int32, complex128)      \
    CPU_HOST_NOIMPL(name, complex128, float16, complex128)    \
    CPU_HOST_BINARY(name, complex128, float32, complex128)    \
    CPU_HOST_BINARY(name, complex128, float64, complex128)    \
    CPU_HOST_NOIMPL(name, complex128, complex32, complex128)  \
    CPU_HOST_BINARY(name, complex128, complex64, complex128)  \
    CPU_HOST_BINARY(name, complex128, complex128, complex128)

#define CPU_HOST_ALL_ARITHMETIC_INIT(name) \
    CPU_HOST_BINARY_INIT(name, uint8, uint8, uint8),                \
    CPU_HOST_BINARY_INIT(name, uint8, uint16, uint16),              \
    CPU_HOST_BINARY_INIT(name, uint8, uint32, uint32),              \
    CPU_HOST_BINARY_INIT(name, uint8, uint64, uint64),              \
    CPU_HOST_BINARY_INIT(name, uint8, int8, int16),                 \
    CPU_HOST_BINARY_INIT(name, uint8, int16, int16),                \
    CPU_HOST_BINARY_INIT(name, uint8, int32, int32),                \
    CPU_HOST_BINARY_INIT(name, uint8, int64, int64),                \
    CPU_HOST_BINARY_INIT(name, uint8, float16, float16),            \
    CPU_HOST_BINARY_INIT(name, uint8, float32, float32),            \
    CPU_HOST_BINARY_INIT(name, uint8, float64, float64),            \
    CPU_HOST_BINARY_INIT(name, uint8, complex32, complex32),        \
    CPU_HOST_BINARY_INIT(name, uint8, complex64, complex64),        \
    CPU_HOST_BINARY_INIT(name, uint8, complex128, complex128),      \
                                                                    \
    CPU_HOST_BINARY_INIT(name, uint16, uint8, uint16),              \
    CPU_HOST_BINARY_INIT(name, uint16, uint16, uint16),             \
    CPU_HOST_BINARY_INIT(name, uint16, uint32, uint32),             \
    CPU_HOST_BINARY_INIT(name, uint16, uint64, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint16, int8, int32),                \
    CPU_HOST_BINARY_INIT(name, uint16, int16, int32),               \
    CPU_HOST_BINARY_INIT(name, uint16, int32, int32),               \
    CPU_HOST_BINARY_INIT(name, uint16, int64, int64),               \
    CPU_HOST_BINARY_INIT(name, uint16, float16, float32),           \
    CPU_HOST_BINARY_INIT(name, uint16, float32, float32),           \
    CPU_HOST_BINARY_INIT(name, uint16, float64, float64),           \
    CPU_HOST_BINARY_INIT(name, uint16, complex32, complex64),       \
    CPU_HOST_BINARY_INIT(name, uint16, complex64, complex64),       \
    CPU_HOST_BINARY_INIT(name, uint16, complex128, complex128),     \
                                                                    \
    CPU_HOST_BINARY_INIT(name, uint32, uint8, uint32),              \
    CPU_HOST_BINARY_INIT(name, uint32, uint16, uint32),             \
    CPU_HOST_BINARY_INIT(name, uint32, uint32, uint32),             \
    CPU_HOST_BINARY_INIT(name, uint32, uint64, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint32, int8, int64),                \
    CPU_HOST_BINARY_INIT(name, uint32, int16, int64),               \
    CPU_HOST_BINARY_INIT(name, uint32, int32, int64),               \
    CPU_HOST_BINARY_INIT(name, uint32, int64, int64),               \
    CPU_HOST_BINARY_INIT(name, uint32, float16, float64),           \
    CPU_HOST_BINARY_INIT(name, uint32, float32, float64),           \
    CPU_HOST_BINARY_INIT(name, uint32, float64, float64),           \
    CPU_HOST_BINARY_INIT(name, uint32, complex32, complex128),      \
    CPU_HOST_BINARY_INIT(name, uint32, complex64, complex128),      \
    CPU_HOST_BINARY_INIT(name, uint32, complex128, complex128),     \
                                                                    \
    CPU_HOST_BINARY_INIT(name, uint64, uint8, uint64),              \
    CPU_HOST_BINARY_INIT(name, uint64, uint16, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint64, uint32, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint64, uint64, uint64),             \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int8, uint8, int16),                 \
    CPU_HOST_BINARY_INIT(name, int8, uint16, int32),                \
    CPU_HOST_BINARY_INIT(name, int8, uint32, int64),                \
    CPU_HOST_BINARY_INIT(name, int8, int8, int8),                   \
    CPU_HOST_BINARY_INIT(name, int8, int16, int16),                 \
    CPU_HOST_BINARY_INIT(name, int8, int32, int32),                 \
    CPU_HOST_BINARY_INIT(name, int8, int64, int64),                 \
    CPU_HOST_BINARY_INIT(name, int8, float16, float16),             \
    CPU_HOST_BINARY_INIT(name, int8, float32, float32),             \
    CPU_HOST_BINARY_INIT(name, int8, float64, float64),             \
    CPU_HOST_BINARY_INIT(name, int8, complex32, complex32),         \
    CPU_HOST_BINARY_INIT(name, int8, complex64, complex64),         \
    CPU_HOST_BINARY_INIT(name, int8, complex128, complex128),       \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int16, uint8, int16),                \
    CPU_HOST_BINARY_INIT(name, int16, uint16, int32),               \
    CPU_HOST_BINARY_INIT(name, int16, uint32, int64),               \
    CPU_HOST_BINARY_INIT(name, int16, int8, int16),                 \
    CPU_HOST_BINARY_INIT(name, int16, int16, int16),                \
    CPU_HOST_BINARY_INIT(name, int16, int32, int32),                \
    CPU_HOST_BINARY_INIT(name, int16, int64, int64),                \
    CPU_HOST_BINARY_INIT(name, int16, float16, float32),            \
    CPU_HOST_BINARY_INIT(name, int16, float32, float32),            \
    CPU_HOST_BINARY_INIT(name, int16, float64, float64),            \
    CPU_HOST_BINARY_INIT(name, int16, complex32, complex64),        \
    CPU_HOST_BINARY_INIT(name, int16, complex64, complex64),        \
    CPU_HOST_BINARY_INIT(name, int16, complex128, complex128),      \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int32, uint8, int32),                \
    CPU_HOST_BINARY_INIT(name, int32, uint16, int32),               \
    CPU_HOST_BINARY_INIT(name, int32, uint32, int64),               \
    CPU_HOST_BINARY_INIT(name, int32, int8, int32),                 \
    CPU_HOST_BINARY_INIT(name, int32, int16, int32),                \
    CPU_HOST_BINARY_INIT(name, int32, int32, int32),                \
    CPU_HOST_BINARY_INIT(name, int32, int64, int64),                \
    CPU_HOST_BINARY_INIT(name, int32, float16, float64),            \
    CPU_HOST_BINARY_INIT(name, int32, float32, float64),            \
    CPU_HOST_BINARY_INIT(name, int32, float64, float64),            \
    CPU_HOST_BINARY_INIT(name, int32, complex32, complex128),       \
    CPU_HOST_BINARY_INIT(name, int32, complex64, complex128),       \
    CPU_HOST_BINARY_INIT(name, int32, complex128, complex128),      \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int64, uint8, int64),                \
    CPU_HOST_BINARY_INIT(name, int64, uint16, int64),               \
    CPU_HOST_BINARY_INIT(name, int64, uint32, int64),               \
    CPU_HOST_BINARY_INIT(name, int64, int8, int64),                 \
    CPU_HOST_BINARY_INIT(name, int64, int16, int64),                \
    CPU_HOST_BINARY_INIT(name, int64, int32, int64),                \
    CPU_HOST_BINARY_INIT(name, int64, int64, int64),                \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float16, uint8, float16),            \
    CPU_HOST_BINARY_INIT(name, float16, uint16, float32),           \
    CPU_HOST_BINARY_INIT(name, float16, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float16, int8, float16),             \
    CPU_HOST_BINARY_INIT(name, float16, int16, float32),            \
    CPU_HOST_BINARY_INIT(name, float16, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float16, float16, float16),          \
    CPU_HOST_BINARY_INIT(name, float16, float32, float32),          \
    CPU_HOST_BINARY_INIT(name, float16, float64, float64),          \
    CPU_HOST_BINARY_INIT(name, float16, complex32, complex32),      \
    CPU_HOST_BINARY_INIT(name, float16, complex64, complex64),      \
    CPU_HOST_BINARY_INIT(name, float16, complex128, complex128),    \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float32, uint8, float32),            \
    CPU_HOST_BINARY_INIT(name, float32, uint16, float32),           \
    CPU_HOST_BINARY_INIT(name, float32, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float32, int8, float32),             \
    CPU_HOST_BINARY_INIT(name, float32, int16, float32),            \
    CPU_HOST_BINARY_INIT(name, float32, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float32, float16, float32),          \
    CPU_HOST_BINARY_INIT(name, float32, float32, float32),          \
    CPU_HOST_BINARY_INIT(name, float32, float64, float64),          \
    CPU_HOST_BINARY_INIT(name, float32, complex32, complex64),      \
    CPU_HOST_BINARY_INIT(name, float32, complex64, complex64),      \
    CPU_HOST_BINARY_INIT(name, float32, complex128, complex128),    \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float64, uint8, float64),            \
    CPU_HOST_BINARY_INIT(name, float64, uint16, float64),           \
    CPU_HOST_BINARY_INIT(name, float64, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float64, int8, float64),             \
    CPU_HOST_BINARY_INIT(name, float64, int16, float64),            \
    CPU_HOST_BINARY_INIT(name, float64, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float64, float16, float64),          \
    CPU_HOST_BINARY_INIT(name, float64, float32, float64),          \
    CPU_HOST_BINARY_INIT(name, float64, float64, float64),          \
    CPU_HOST_BINARY_INIT(name, float64, complex32, complex128),     \
    CPU_HOST_BINARY_INIT(name, float64, complex64, complex128),     \
    CPU_HOST_BINARY_INIT(name, float64, complex128, complex128),    \
                                                                    \
    CPU_HOST_BINARY_INIT(name, complex32, uint8, complex32),        \
    CPU_HOST_BINARY_INIT(name, complex32, uint16, complex64),       \
    CPU_HOST_BINARY_INIT(name, complex32, uint32, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex32, int8, complex32),         \
    CPU_HOST_BINARY_INIT(name, complex32, int16, complex64),        \
    CPU_HOST_BINARY_INIT(name, complex32, int32, complex128),       \
    CPU_HOST_BINARY_INIT(name, complex32, float16, complex32),      \
    CPU_HOST_BINARY_INIT(name, complex32, float32, complex64),      \
    CPU_HOST_BINARY_INIT(name, complex32, float64, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex32, complex32, complex32),    \
    CPU_HOST_BINARY_INIT(name, complex32, complex64, complex64),    \
    CPU_HOST_BINARY_INIT(name, complex32, complex128, complex128),  \
                                                                    \
    CPU_HOST_BINARY_INIT(name, complex64, uint8, complex64),        \
    CPU_HOST_BINARY_INIT(name, complex64, uint16, complex64),       \
    CPU_HOST_BINARY_INIT(name, complex64, uint32, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex64, int8, complex64),         \
    CPU_HOST_BINARY_INIT(name, complex64, int16, complex64),        \
    CPU_HOST_BINARY_INIT(name, complex64, int32, complex128),       \
    CPU_HOST_BINARY_INIT(name, complex64, float16, complex64),      \
    CPU_HOST_BINARY_INIT(name, complex64, float32, complex64),      \
    CPU_HOST_BINARY_INIT(name, complex64, float64, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex64, complex32, complex64),    \
    CPU_HOST_BINARY_INIT(name, complex64, complex64, complex64),    \
    CPU_HOST_BINARY_INIT(name, complex64, complex128, complex128),  \
                                                                    \
    CPU_HOST_BINARY_INIT(name, complex128, uint8, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex128, uint16, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex128, uint32, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex128, int8, complex128),       \
    CPU_HOST_BINARY_INIT(name, complex128, int16, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex128, int32, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex128, float16, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, float32, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, float64, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, complex32, complex128),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex64, complex128),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex128, complex128)

#define CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN_INIT(name) \
    CPU_HOST_BINARY_INIT(name, uint8, uint8, float16),              \
    CPU_HOST_BINARY_INIT(name, uint8, uint16, float32),             \
    CPU_HOST_BINARY_INIT(name, uint8, uint32, float64),             \
    CPU_HOST_BINARY_INIT(name, uint8, uint64, uint64),              \
    CPU_HOST_BINARY_INIT(name, uint8, int8, float16),               \
    CPU_HOST_BINARY_INIT(name, uint8, int16, float32),              \
    CPU_HOST_BINARY_INIT(name, uint8, int32, float64),              \
    CPU_HOST_BINARY_INIT(name, uint8, int64, int64),                \
    CPU_HOST_BINARY_INIT(name, uint8, float16, float16),            \
    CPU_HOST_BINARY_INIT(name, uint8, float32, float32),            \
    CPU_HOST_BINARY_INIT(name, uint8, float64, float64),            \
    CPU_HOST_BINARY_INIT(name, uint8, complex32, complex32),        \
    CPU_HOST_BINARY_INIT(name, uint8, complex64, complex64),        \
    CPU_HOST_BINARY_INIT(name, uint8, complex128, complex128),      \
                                                                    \
    CPU_HOST_BINARY_INIT(name, uint16, uint8, float32),             \
    CPU_HOST_BINARY_INIT(name, uint16, uint16, float32),            \
    CPU_HOST_BINARY_INIT(name, uint16, uint32, float64),            \
    CPU_HOST_BINARY_INIT(name, uint16, uint64, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint16, int8, float32),              \
    CPU_HOST_BINARY_INIT(name, uint16, int16, float32),             \
    CPU_HOST_BINARY_INIT(name, uint16, int32, float64),             \
    CPU_HOST_BINARY_INIT(name, uint16, int64, int64),               \
    CPU_HOST_BINARY_INIT(name, uint16, float16, float32),           \
    CPU_HOST_BINARY_INIT(name, uint16, float32, float32),           \
    CPU_HOST_BINARY_INIT(name, uint16, float64, float64),           \
    CPU_HOST_BINARY_INIT(name, uint16, complex32, complex64),       \
    CPU_HOST_BINARY_INIT(name, uint16, complex64, complex64),       \
    CPU_HOST_BINARY_INIT(name, uint16, complex128, complex128),     \
                                                                    \
    CPU_HOST_BINARY_INIT(name, uint32, uint8, float64),             \
    CPU_HOST_BINARY_INIT(name, uint32, uint16, float64),            \
    CPU_HOST_BINARY_INIT(name, uint32, uint32, float64),            \
    CPU_HOST_BINARY_INIT(name, uint32, uint64, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint32, int8, float64),              \
    CPU_HOST_BINARY_INIT(name, uint32, int16, float64),             \
    CPU_HOST_BINARY_INIT(name, uint32, int32, float64),             \
    CPU_HOST_BINARY_INIT(name, uint32, int64, int64),               \
    CPU_HOST_BINARY_INIT(name, uint32, float16, float64),           \
    CPU_HOST_BINARY_INIT(name, uint32, float32, float64),           \
    CPU_HOST_BINARY_INIT(name, uint32, float64, float64),           \
    CPU_HOST_BINARY_INIT(name, uint32, complex32, complex128),      \
    CPU_HOST_BINARY_INIT(name, uint32, complex64, complex128),      \
    CPU_HOST_BINARY_INIT(name, uint32, complex128, complex128),     \
                                                                    \
    CPU_HOST_BINARY_INIT(name, uint64, uint8, uint64),              \
    CPU_HOST_BINARY_INIT(name, uint64, uint16, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint64, uint32, uint64),             \
    CPU_HOST_BINARY_INIT(name, uint64, uint64, uint64),             \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int8, uint8, float16),               \
    CPU_HOST_BINARY_INIT(name, int8, uint16, float32),              \
    CPU_HOST_BINARY_INIT(name, int8, uint32, float64),              \
    CPU_HOST_BINARY_INIT(name, int8, int8, float16),                \
    CPU_HOST_BINARY_INIT(name, int8, int16, float32),               \
    CPU_HOST_BINARY_INIT(name, int8, int32, float64),               \
    CPU_HOST_BINARY_INIT(name, int8, int64, int64),                 \
    CPU_HOST_BINARY_INIT(name, int8, float16, float16),             \
    CPU_HOST_BINARY_INIT(name, int8, float32, float32),             \
    CPU_HOST_BINARY_INIT(name, int8, float64, float64),             \
    CPU_HOST_BINARY_INIT(name, int8, complex32, complex32),         \
    CPU_HOST_BINARY_INIT(name, int8, complex64, complex64),         \
    CPU_HOST_BINARY_INIT(name, int8, complex128, complex128),       \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int16, uint8, float32),              \
    CPU_HOST_BINARY_INIT(name, int16, uint16, float32),             \
    CPU_HOST_BINARY_INIT(name, int16, uint32, float64),             \
    CPU_HOST_BINARY_INIT(name, int16, int8, float32),               \
    CPU_HOST_BINARY_INIT(name, int16, int16, float32),              \
    CPU_HOST_BINARY_INIT(name, int16, int32, float64),              \
    CPU_HOST_BINARY_INIT(name, int16, int64, int64),                \
    CPU_HOST_BINARY_INIT(name, int16, float16, float32),            \
    CPU_HOST_BINARY_INIT(name, int16, float32, float32),            \
    CPU_HOST_BINARY_INIT(name, int16, float64, float64),            \
    CPU_HOST_BINARY_INIT(name, int16, complex32, complex64),        \
    CPU_HOST_BINARY_INIT(name, int16, complex64, complex64),        \
    CPU_HOST_BINARY_INIT(name, int16, complex128, complex128),      \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int32, uint8, float64),              \
    CPU_HOST_BINARY_INIT(name, int32, uint16, float64),             \
    CPU_HOST_BINARY_INIT(name, int32, uint32, float64),             \
    CPU_HOST_BINARY_INIT(name, int32, int8, float64),               \
    CPU_HOST_BINARY_INIT(name, int32, int16, float64),              \
    CPU_HOST_BINARY_INIT(name, int32, int32, float64),              \
    CPU_HOST_BINARY_INIT(name, int32, int64, int64),                \
    CPU_HOST_BINARY_INIT(name, int32, float16, float64),            \
    CPU_HOST_BINARY_INIT(name, int32, float32, float64),            \
    CPU_HOST_BINARY_INIT(name, int32, float64, float64),            \
    CPU_HOST_BINARY_INIT(name, int32, complex32, complex128),       \
    CPU_HOST_BINARY_INIT(name, int32, complex64, complex128),       \
    CPU_HOST_BINARY_INIT(name, int32, complex128, complex128),      \
                                                                    \
    CPU_HOST_BINARY_INIT(name, int64, uint8, int64),                \
    CPU_HOST_BINARY_INIT(name, int64, uint16, int64),               \
    CPU_HOST_BINARY_INIT(name, int64, uint32, int64),               \
    CPU_HOST_BINARY_INIT(name, int64, int8, int64),                 \
    CPU_HOST_BINARY_INIT(name, int64, int16, int64),                \
    CPU_HOST_BINARY_INIT(name, int64, int32, int64),                \
    CPU_HOST_BINARY_INIT(name, int64, int64, int64),                \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float16, uint8, float16),            \
    CPU_HOST_BINARY_INIT(name, float16, uint16, float32),           \
    CPU_HOST_BINARY_INIT(name, float16, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float16, int8, float16),             \
    CPU_HOST_BINARY_INIT(name, float16, int16, float32),            \
    CPU_HOST_BINARY_INIT(name, float16, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float16, float16, float16),          \
    CPU_HOST_BINARY_INIT(name, float16, float32, float32),          \
    CPU_HOST_BINARY_INIT(name, float16, float64, float64),          \
    CPU_HOST_BINARY_INIT(name, float16, complex32, complex32),      \
    CPU_HOST_BINARY_INIT(name, float16, complex64, complex64),      \
    CPU_HOST_BINARY_INIT(name, float16, complex128, complex128),    \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float32, uint8, float32),            \
    CPU_HOST_BINARY_INIT(name, float32, uint16, float32),           \
    CPU_HOST_BINARY_INIT(name, float32, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float32, int8, float32),             \
    CPU_HOST_BINARY_INIT(name, float32, int16, float32),            \
    CPU_HOST_BINARY_INIT(name, float32, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float32, float16, float32),          \
    CPU_HOST_BINARY_INIT(name, float32, float32, float32),          \
    CPU_HOST_BINARY_INIT(name, float32, float64, float64),          \
    CPU_HOST_BINARY_INIT(name, float32, complex32, complex64),      \
    CPU_HOST_BINARY_INIT(name, float32, complex64, complex64),      \
    CPU_HOST_BINARY_INIT(name, float32, complex128, complex128),    \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float64, uint8, float64),            \
    CPU_HOST_BINARY_INIT(name, float64, uint16, float64),           \
    CPU_HOST_BINARY_INIT(name, float64, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float64, int8, float64),             \
    CPU_HOST_BINARY_INIT(name, float64, int16, float64),            \
    CPU_HOST_BINARY_INIT(name, float64, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float64, float16, float64),          \
    CPU_HOST_BINARY_INIT(name, float64, float32, float64),          \
    CPU_HOST_BINARY_INIT(name, float64, float64, float64),          \
    CPU_HOST_BINARY_INIT(name, float64, complex32, complex128),     \
    CPU_HOST_BINARY_INIT(name, float64, complex64, complex128),     \
    CPU_HOST_BINARY_INIT(name, float64, complex128, complex128),    \
                                                                    \
    CPU_HOST_BINARY_INIT(name, complex32, uint8, complex32),        \
    CPU_HOST_BINARY_INIT(name, complex32, uint16, complex64),       \
    CPU_HOST_BINARY_INIT(name, complex32, uint32, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex32, int8, complex32),         \
    CPU_HOST_BINARY_INIT(name, complex32, int16, complex64),        \
    CPU_HOST_BINARY_INIT(name, complex32, int32, complex128),       \
    CPU_HOST_BINARY_INIT(name, complex32, float16, complex32),      \
    CPU_HOST_BINARY_INIT(name, complex32, float32, complex64),      \
    CPU_HOST_BINARY_INIT(name, complex32, float64, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex32, complex32, complex32),    \
    CPU_HOST_BINARY_INIT(name, complex32, complex64, complex64),    \
    CPU_HOST_BINARY_INIT(name, complex32, complex128, complex128),  \
                                                                    \
    CPU_HOST_BINARY_INIT(name, complex64, uint8, complex64),        \
    CPU_HOST_BINARY_INIT(name, complex64, uint16, complex64),       \
    CPU_HOST_BINARY_INIT(name, complex64, uint32, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex64, int8, complex64),         \
    CPU_HOST_BINARY_INIT(name, complex64, int16, complex64),        \
    CPU_HOST_BINARY_INIT(name, complex64, int32, complex128),       \
    CPU_HOST_BINARY_INIT(name, complex64, float16, complex64),      \
    CPU_HOST_BINARY_INIT(name, complex64, float32, complex64),      \
    CPU_HOST_BINARY_INIT(name, complex64, float64, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex64, complex32, complex64),    \
    CPU_HOST_BINARY_INIT(name, complex64, complex64, complex64),    \
    CPU_HOST_BINARY_INIT(name, complex64, complex128, complex128),  \
                                                                    \
    CPU_HOST_BINARY_INIT(name, complex128, uint8, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex128, uint16, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex128, uint32, complex128),     \
    CPU_HOST_BINARY_INIT(name, complex128, int8, complex128),       \
    CPU_HOST_BINARY_INIT(name, complex128, int16, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex128, int32, complex128),      \
    CPU_HOST_BINARY_INIT(name, complex128, float16, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, float32, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, float64, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, complex32, complex128),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex64, complex128),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex128, complex128)


#define add(x, y) x + y
CPU_HOST_ALL_ARITHMETIC(add)

#define subtract(x, y) x - y
CPU_HOST_ALL_ARITHMETIC(subtract)

#define multiply(x, y) x * y
CPU_HOST_ALL_ARITHMETIC(multiply)

#define divide(x, y) x / y
CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN(divide)


/*****************************************************************************/
/*                                 Comparison                                */
/*****************************************************************************/

#define CPU_HOST_ALL_COMPARISON(name) \
    CPU_HOST_BINARY(name, uint8, uint8, bool)           \
    CPU_HOST_BINARY(name, uint8, uint16, bool)          \
    CPU_HOST_BINARY(name, uint8, uint32, bool)          \
    CPU_HOST_BINARY(name, uint8, uint64, bool)          \
    CPU_HOST_BINARY(name, uint8, int8, bool)            \
    CPU_HOST_BINARY(name, uint8, int16, bool)           \
    CPU_HOST_BINARY(name, uint8, int32, bool)           \
    CPU_HOST_BINARY(name, uint8, int64, bool)           \
    CPU_HOST_NOIMPL(name, uint8, float16, bool)         \
    CPU_HOST_BINARY(name, uint8, float32, bool)         \
    CPU_HOST_BINARY(name, uint8, float64, bool)         \
    CPU_HOST_NOIMPL(name, uint8, complex32, bool)       \
    CPU_HOST_BINARY(name, uint8, complex64, bool)       \
    CPU_HOST_BINARY(name, uint8, complex128, bool)      \
                                                        \
    CPU_HOST_BINARY(name, uint16, uint8, bool)          \
    CPU_HOST_BINARY(name, uint16, uint16, bool)         \
    CPU_HOST_BINARY(name, uint16, uint32, bool)         \
    CPU_HOST_BINARY(name, uint16, uint64, bool)         \
    CPU_HOST_BINARY(name, uint16, int8, bool)           \
    CPU_HOST_BINARY(name, uint16, int16, bool)          \
    CPU_HOST_BINARY(name, uint16, int32, bool)          \
    CPU_HOST_BINARY(name, uint16, int64, bool)          \
    CPU_HOST_NOIMPL(name, uint16, float16, bool)        \
    CPU_HOST_BINARY(name, uint16, float32, bool)        \
    CPU_HOST_BINARY(name, uint16, float64, bool)        \
    CPU_HOST_NOIMPL(name, uint16, complex32, bool)      \
    CPU_HOST_BINARY(name, uint16, complex64, bool)      \
    CPU_HOST_BINARY(name, uint16, complex128, bool)     \
                                                        \
    CPU_HOST_BINARY(name, uint32, uint8, bool)          \
    CPU_HOST_BINARY(name, uint32, uint16, bool)         \
    CPU_HOST_BINARY(name, uint32, uint32, bool)         \
    CPU_HOST_BINARY(name, uint32, uint64, bool)         \
    CPU_HOST_BINARY(name, uint32, int8, bool)           \
    CPU_HOST_BINARY(name, uint32, int16, bool)          \
    CPU_HOST_BINARY(name, uint32, int32, bool)          \
    CPU_HOST_BINARY(name, uint32, int64, bool)          \
    CPU_HOST_NOIMPL(name, uint32, float16, bool)        \
    CPU_HOST_BINARY(name, uint32, float32, bool)        \
    CPU_HOST_BINARY(name, uint32, float64, bool)        \
    CPU_HOST_NOIMPL(name, uint32, complex32, bool)      \
    CPU_HOST_BINARY(name, uint32, complex64, bool)      \
    CPU_HOST_BINARY(name, uint32, complex128, bool)     \
                                                        \
    CPU_HOST_BINARY(name, uint64, uint8, bool)          \
    CPU_HOST_BINARY(name, uint64, uint16, bool)         \
    CPU_HOST_BINARY(name, uint64, uint32, bool)         \
    CPU_HOST_BINARY(name, uint64, uint64, bool)         \
                                                        \
    CPU_HOST_BINARY(name, int8, uint8, bool)            \
    CPU_HOST_BINARY(name, int8, uint16, bool)           \
    CPU_HOST_BINARY(name, int8, uint32, bool)           \
    CPU_HOST_BINARY(name, int8, int8, bool)             \
    CPU_HOST_BINARY(name, int8, int16, bool)            \
    CPU_HOST_BINARY(name, int8, int32, bool)            \
    CPU_HOST_BINARY(name, int8, int64, bool)            \
    CPU_HOST_NOIMPL(name, int8, float16, bool)          \
    CPU_HOST_BINARY(name, int8, float32, bool)          \
    CPU_HOST_BINARY(name, int8, float64, bool)          \
    CPU_HOST_NOIMPL(name, int8, complex32, bool)        \
    CPU_HOST_BINARY(name, int8, complex64, bool)        \
    CPU_HOST_BINARY(name, int8, complex128, bool)       \
                                                        \
    CPU_HOST_BINARY(name, int16, uint8, bool)           \
    CPU_HOST_BINARY(name, int16, uint16, bool)          \
    CPU_HOST_BINARY(name, int16, uint32, bool)          \
    CPU_HOST_BINARY(name, int16, int8, bool)            \
    CPU_HOST_BINARY(name, int16, int16, bool)           \
    CPU_HOST_BINARY(name, int16, int32, bool)           \
    CPU_HOST_BINARY(name, int16, int64, bool)           \
    CPU_HOST_NOIMPL(name, int16, float16, bool)         \
    CPU_HOST_BINARY(name, int16, float32, bool)         \
    CPU_HOST_BINARY(name, int16, float64, bool)         \
    CPU_HOST_NOIMPL(name, int16, complex32, bool)       \
    CPU_HOST_BINARY(name, int16, complex64, bool)       \
    CPU_HOST_BINARY(name, int16, complex128, bool)      \
                                                        \
    CPU_HOST_BINARY(name, int32, uint8, bool)           \
    CPU_HOST_BINARY(name, int32, uint16, bool)          \
    CPU_HOST_BINARY(name, int32, uint32, bool)          \
    CPU_HOST_BINARY(name, int32, int8, bool)            \
    CPU_HOST_BINARY(name, int32, int16, bool)           \
    CPU_HOST_BINARY(name, int32, int32, bool)           \
    CPU_HOST_BINARY(name, int32, int64, bool)           \
    CPU_HOST_NOIMPL(name, int32, float16, bool)         \
    CPU_HOST_BINARY(name, int32, float32, bool)         \
    CPU_HOST_BINARY(name, int32, float64, bool)         \
    CPU_HOST_NOIMPL(name, int32, complex32, bool)       \
    CPU_HOST_BINARY(name, int32, complex64, bool)       \
    CPU_HOST_BINARY(name, int32, complex128, bool)      \
                                                        \
    CPU_HOST_BINARY(name, int64, uint8, bool)           \
    CPU_HOST_BINARY(name, int64, uint16, bool)          \
    CPU_HOST_BINARY(name, int64, uint32, bool)          \
    CPU_HOST_BINARY(name, int64, int8, bool)            \
    CPU_HOST_BINARY(name, int64, int16, bool)           \
    CPU_HOST_BINARY(name, int64, int32, bool)           \
    CPU_HOST_BINARY(name, int64, int64, bool)           \
                                                        \
    CPU_HOST_NOIMPL(name, float16, uint8, bool)         \
    CPU_HOST_NOIMPL(name, float16, uint16, bool)        \
    CPU_HOST_NOIMPL(name, float16, uint32, bool)        \
    CPU_HOST_NOIMPL(name, float16, int8, bool)          \
    CPU_HOST_NOIMPL(name, float16, int16, bool)         \
    CPU_HOST_NOIMPL(name, float16, int32, bool)         \
    CPU_HOST_NOIMPL(name, float16, float16, bool)       \
    CPU_HOST_NOIMPL(name, float16, float32, bool)       \
    CPU_HOST_NOIMPL(name, float16, float64, bool)       \
    CPU_HOST_NOIMPL(name, float16, complex32, bool)     \
    CPU_HOST_NOIMPL(name, float16, complex64, bool)     \
    CPU_HOST_NOIMPL(name, float16, complex128, bool)    \
                                                        \
    CPU_HOST_BINARY(name, float32, uint8, bool)         \
    CPU_HOST_BINARY(name, float32, uint16, bool)        \
    CPU_HOST_BINARY(name, float32, uint32, bool)        \
    CPU_HOST_BINARY(name, float32, int8, bool)          \
    CPU_HOST_BINARY(name, float32, int16, bool)         \
    CPU_HOST_BINARY(name, float32, int32, bool)         \
    CPU_HOST_NOIMPL(name, float32, float16, bool)       \
    CPU_HOST_BINARY(name, float32, float32, bool)       \
    CPU_HOST_BINARY(name, float32, float64, bool)       \
    CPU_HOST_NOIMPL(name, float32, complex32, bool)     \
    CPU_HOST_BINARY(name, float32, complex64, bool)     \
    CPU_HOST_BINARY(name, float32, complex128, bool)    \
                                                        \
    CPU_HOST_BINARY(name, float64, uint8, bool)         \
    CPU_HOST_BINARY(name, float64, uint16, bool)        \
    CPU_HOST_BINARY(name, float64, uint32, bool)        \
    CPU_HOST_BINARY(name, float64, int8, bool)          \
    CPU_HOST_BINARY(name, float64, int16, bool)         \
    CPU_HOST_BINARY(name, float64, int32, bool)         \
    CPU_HOST_NOIMPL(name, float64, float16, bool)       \
    CPU_HOST_BINARY(name, float64, float32, bool)       \
    CPU_HOST_BINARY(name, float64, float64, bool)       \
    CPU_HOST_NOIMPL(name, float64, complex32, bool)     \
    CPU_HOST_BINARY(name, float64, complex64, bool)     \
    CPU_HOST_BINARY(name, float64, complex128, bool)    \
                                                        \
    CPU_HOST_NOIMPL(name, complex32, uint8, bool)       \
    CPU_HOST_NOIMPL(name, complex32, uint16, bool)      \
    CPU_HOST_NOIMPL(name, complex32, uint32, bool)      \
    CPU_HOST_NOIMPL(name, complex32, int8, bool)        \
    CPU_HOST_NOIMPL(name, complex32, int16, bool)       \
    CPU_HOST_NOIMPL(name, complex32, int32, bool)       \
    CPU_HOST_NOIMPL(name, complex32, float16, bool)     \
    CPU_HOST_NOIMPL(name, complex32, float32, bool)     \
    CPU_HOST_NOIMPL(name, complex32, float64, bool)     \
    CPU_HOST_NOIMPL(name, complex32, complex32, bool)   \
    CPU_HOST_NOIMPL(name, complex32, complex64, bool)   \
    CPU_HOST_NOIMPL(name, complex32, complex128, bool)  \
                                                        \
    CPU_HOST_BINARY(name, complex64, uint8, bool)       \
    CPU_HOST_BINARY(name, complex64, uint16, bool)      \
    CPU_HOST_BINARY(name, complex64, uint32, bool)      \
    CPU_HOST_BINARY(name, complex64, int8, bool)        \
    CPU_HOST_BINARY(name, complex64, int16, bool)       \
    CPU_HOST_BINARY(name, complex64, int32, bool)       \
    CPU_HOST_NOIMPL(name, complex64, float16, bool)     \
    CPU_HOST_BINARY(name, complex64, float32, bool)     \
    CPU_HOST_BINARY(name, complex64, float64, bool)     \
    CPU_HOST_NOIMPL(name, complex64, complex32, bool)   \
    CPU_HOST_BINARY(name, complex64, complex64, bool)   \
    CPU_HOST_BINARY(name, complex64, complex128, bool)  \
                                                        \
    CPU_HOST_BINARY(name, complex128, uint8, bool)      \
    CPU_HOST_BINARY(name, complex128, uint16, bool)     \
    CPU_HOST_BINARY(name, complex128, uint32, bool)     \
    CPU_HOST_BINARY(name, complex128, int8, bool)       \
    CPU_HOST_BINARY(name, complex128, int16, bool)      \
    CPU_HOST_BINARY(name, complex128, int32, bool)      \
    CPU_HOST_NOIMPL(name, complex128, float16, bool)    \
    CPU_HOST_BINARY(name, complex128, float32, bool)    \
    CPU_HOST_BINARY(name, complex128, float64, bool)    \
    CPU_HOST_NOIMPL(name, complex128, complex32, bool)  \
    CPU_HOST_BINARY(name, complex128, complex64, bool)  \
    CPU_HOST_BINARY(name, complex128, complex128, bool)

#define CPU_HOST_ALL_COMPARISON_INIT(name) \
    CPU_HOST_BINARY_INIT(name, uint8, uint8, bool),           \
    CPU_HOST_BINARY_INIT(name, uint8, uint16, bool),          \
    CPU_HOST_BINARY_INIT(name, uint8, uint32, bool),          \
    CPU_HOST_BINARY_INIT(name, uint8, uint64, bool),          \
    CPU_HOST_BINARY_INIT(name, uint8, int8, bool),            \
    CPU_HOST_BINARY_INIT(name, uint8, int16, bool),           \
    CPU_HOST_BINARY_INIT(name, uint8, int32, bool),           \
    CPU_HOST_BINARY_INIT(name, uint8, int64, bool),           \
    CPU_HOST_BINARY_INIT(name, uint8, float16, bool),         \
    CPU_HOST_BINARY_INIT(name, uint8, float32, bool),         \
    CPU_HOST_BINARY_INIT(name, uint8, float64, bool),         \
    CPU_HOST_BINARY_INIT(name, uint8, complex32, bool),       \
    CPU_HOST_BINARY_INIT(name, uint8, complex64, bool),       \
    CPU_HOST_BINARY_INIT(name, uint8, complex128, bool),      \
                                                              \
    CPU_HOST_BINARY_INIT(name, uint16, uint8, bool),          \
    CPU_HOST_BINARY_INIT(name, uint16, uint16, bool),         \
    CPU_HOST_BINARY_INIT(name, uint16, uint32, bool),         \
    CPU_HOST_BINARY_INIT(name, uint16, uint64, bool),         \
    CPU_HOST_BINARY_INIT(name, uint16, int8, bool),           \
    CPU_HOST_BINARY_INIT(name, uint16, int16, bool),          \
    CPU_HOST_BINARY_INIT(name, uint16, int32, bool),          \
    CPU_HOST_BINARY_INIT(name, uint16, int64, bool),          \
    CPU_HOST_BINARY_INIT(name, uint16, float16, bool),        \
    CPU_HOST_BINARY_INIT(name, uint16, float32, bool),        \
    CPU_HOST_BINARY_INIT(name, uint16, float64, bool),        \
    CPU_HOST_BINARY_INIT(name, uint16, complex32, bool),      \
    CPU_HOST_BINARY_INIT(name, uint16, complex64, bool),      \
    CPU_HOST_BINARY_INIT(name, uint16, complex128, bool),     \
                                                              \
    CPU_HOST_BINARY_INIT(name, uint32, uint8, bool),          \
    CPU_HOST_BINARY_INIT(name, uint32, uint16, bool),         \
    CPU_HOST_BINARY_INIT(name, uint32, uint32, bool),         \
    CPU_HOST_BINARY_INIT(name, uint32, uint64, bool),         \
    CPU_HOST_BINARY_INIT(name, uint32, int8, bool),           \
    CPU_HOST_BINARY_INIT(name, uint32, int16, bool),          \
    CPU_HOST_BINARY_INIT(name, uint32, int32, bool),          \
    CPU_HOST_BINARY_INIT(name, uint32, int64, bool),          \
    CPU_HOST_BINARY_INIT(name, uint32, float16, bool),        \
    CPU_HOST_BINARY_INIT(name, uint32, float32, bool),        \
    CPU_HOST_BINARY_INIT(name, uint32, float64, bool),        \
    CPU_HOST_BINARY_INIT(name, uint32, complex32, bool),      \
    CPU_HOST_BINARY_INIT(name, uint32, complex64, bool),      \
    CPU_HOST_BINARY_INIT(name, uint32, complex128, bool),     \
                                                              \
    CPU_HOST_BINARY_INIT(name, uint64, uint8, bool),          \
    CPU_HOST_BINARY_INIT(name, uint64, uint16, bool),         \
    CPU_HOST_BINARY_INIT(name, uint64, uint32, bool),         \
    CPU_HOST_BINARY_INIT(name, uint64, uint64, bool),         \
                                                              \
    CPU_HOST_BINARY_INIT(name, int8, uint8, bool),            \
    CPU_HOST_BINARY_INIT(name, int8, uint16, bool),           \
    CPU_HOST_BINARY_INIT(name, int8, uint32, bool),           \
    CPU_HOST_BINARY_INIT(name, int8, int8, bool),             \
    CPU_HOST_BINARY_INIT(name, int8, int16, bool),            \
    CPU_HOST_BINARY_INIT(name, int8, int32, bool),            \
    CPU_HOST_BINARY_INIT(name, int8, int64, bool),            \
    CPU_HOST_BINARY_INIT(name, int8, float16, bool),          \
    CPU_HOST_BINARY_INIT(name, int8, float32, bool),          \
    CPU_HOST_BINARY_INIT(name, int8, float64, bool),          \
    CPU_HOST_BINARY_INIT(name, int8, complex32, bool),        \
    CPU_HOST_BINARY_INIT(name, int8, complex64, bool),        \
    CPU_HOST_BINARY_INIT(name, int8, complex128, bool),       \
                                                              \
    CPU_HOST_BINARY_INIT(name, int16, uint8, bool),           \
    CPU_HOST_BINARY_INIT(name, int16, uint16, bool),          \
    CPU_HOST_BINARY_INIT(name, int16, uint32, bool),          \
    CPU_HOST_BINARY_INIT(name, int16, int8, bool),            \
    CPU_HOST_BINARY_INIT(name, int16, int16, bool),           \
    CPU_HOST_BINARY_INIT(name, int16, int32, bool),           \
    CPU_HOST_BINARY_INIT(name, int16, int64, bool),           \
    CPU_HOST_BINARY_INIT(name, int16, float16, bool),         \
    CPU_HOST_BINARY_INIT(name, int16, float32, bool),         \
    CPU_HOST_BINARY_INIT(name, int16, float64, bool),         \
    CPU_HOST_BINARY_INIT(name, int16, complex32, bool),       \
    CPU_HOST_BINARY_INIT(name, int16, complex64, bool),       \
    CPU_HOST_BINARY_INIT(name, int16, complex128, bool),      \
                                                              \
    CPU_HOST_BINARY_INIT(name, int32, uint8, bool),           \
    CPU_HOST_BINARY_INIT(name, int32, uint16, bool),          \
    CPU_HOST_BINARY_INIT(name, int32, uint32, bool),          \
    CPU_HOST_BINARY_INIT(name, int32, int8, bool),            \
    CPU_HOST_BINARY_INIT(name, int32, int16, bool),           \
    CPU_HOST_BINARY_INIT(name, int32, int32, bool),           \
    CPU_HOST_BINARY_INIT(name, int32, int64, bool),           \
    CPU_HOST_BINARY_INIT(name, int32, float16, bool),         \
    CPU_HOST_BINARY_INIT(name, int32, float32, bool),         \
    CPU_HOST_BINARY_INIT(name, int32, float64, bool),         \
    CPU_HOST_BINARY_INIT(name, int32, complex32, bool),       \
    CPU_HOST_BINARY_INIT(name, int32, complex64, bool),       \
    CPU_HOST_BINARY_INIT(name, int32, complex128, bool),      \
                                                              \
    CPU_HOST_BINARY_INIT(name, int64, uint8, bool),           \
    CPU_HOST_BINARY_INIT(name, int64, uint16, bool),          \
    CPU_HOST_BINARY_INIT(name, int64, uint32, bool),          \
    CPU_HOST_BINARY_INIT(name, int64, int8, bool),            \
    CPU_HOST_BINARY_INIT(name, int64, int16, bool),           \
    CPU_HOST_BINARY_INIT(name, int64, int32, bool),           \
    CPU_HOST_BINARY_INIT(name, int64, int64, bool),           \
                                                              \
    CPU_HOST_BINARY_INIT(name, float16, uint8, bool),         \
    CPU_HOST_BINARY_INIT(name, float16, uint16, bool),        \
    CPU_HOST_BINARY_INIT(name, float16, uint32, bool),        \
    CPU_HOST_BINARY_INIT(name, float16, int8, bool),          \
    CPU_HOST_BINARY_INIT(name, float16, int16, bool),         \
    CPU_HOST_BINARY_INIT(name, float16, int32, bool),         \
    CPU_HOST_BINARY_INIT(name, float16, float16, bool),       \
    CPU_HOST_BINARY_INIT(name, float16, float32, bool),       \
    CPU_HOST_BINARY_INIT(name, float16, float64, bool),       \
    CPU_HOST_BINARY_INIT(name, float16, complex32, bool),     \
    CPU_HOST_BINARY_INIT(name, float16, complex64, bool),     \
    CPU_HOST_BINARY_INIT(name, float16, complex128, bool),    \
                                                              \
    CPU_HOST_BINARY_INIT(name, float32, uint8, bool),         \
    CPU_HOST_BINARY_INIT(name, float32, uint16, bool),        \
    CPU_HOST_BINARY_INIT(name, float32, uint32, bool),        \
    CPU_HOST_BINARY_INIT(name, float32, int8, bool),          \
    CPU_HOST_BINARY_INIT(name, float32, int16, bool),         \
    CPU_HOST_BINARY_INIT(name, float32, int32, bool),         \
    CPU_HOST_BINARY_INIT(name, float32, float16, bool),       \
    CPU_HOST_BINARY_INIT(name, float32, float32, bool),       \
    CPU_HOST_BINARY_INIT(name, float32, float64, bool),       \
    CPU_HOST_BINARY_INIT(name, float32, complex32, bool),     \
    CPU_HOST_BINARY_INIT(name, float32, complex64, bool),     \
    CPU_HOST_BINARY_INIT(name, float32, complex128, bool),    \
                                                              \
    CPU_HOST_BINARY_INIT(name, float64, uint8, bool),         \
    CPU_HOST_BINARY_INIT(name, float64, uint16, bool),        \
    CPU_HOST_BINARY_INIT(name, float64, uint32, bool),        \
    CPU_HOST_BINARY_INIT(name, float64, int8, bool),          \
    CPU_HOST_BINARY_INIT(name, float64, int16, bool),         \
    CPU_HOST_BINARY_INIT(name, float64, int32, bool),         \
    CPU_HOST_BINARY_INIT(name, float64, float16, bool),       \
    CPU_HOST_BINARY_INIT(name, float64, float32, bool),       \
    CPU_HOST_BINARY_INIT(name, float64, float64, bool),       \
    CPU_HOST_BINARY_INIT(name, float64, complex32, bool),     \
    CPU_HOST_BINARY_INIT(name, float64, complex64, bool),     \
    CPU_HOST_BINARY_INIT(name, float64, complex128, bool),    \
                                                              \
    CPU_HOST_BINARY_INIT(name, complex32, uint8, bool),       \
    CPU_HOST_BINARY_INIT(name, complex32, uint16, bool),      \
    CPU_HOST_BINARY_INIT(name, complex32, uint32, bool),      \
    CPU_HOST_BINARY_INIT(name, complex32, int8, bool),        \
    CPU_HOST_BINARY_INIT(name, complex32, int16, bool),       \
    CPU_HOST_BINARY_INIT(name, complex32, int32, bool),       \
    CPU_HOST_BINARY_INIT(name, complex32, float16, bool),     \
    CPU_HOST_BINARY_INIT(name, complex32, float32, bool),     \
    CPU_HOST_BINARY_INIT(name, complex32, float64, bool),     \
    CPU_HOST_BINARY_INIT(name, complex32, complex32, bool),   \
    CPU_HOST_BINARY_INIT(name, complex32, complex64, bool),   \
    CPU_HOST_BINARY_INIT(name, complex32, complex128, bool),  \
                                                              \
    CPU_HOST_BINARY_INIT(name, complex64, uint8, bool),       \
    CPU_HOST_BINARY_INIT(name, complex64, uint16, bool),      \
    CPU_HOST_BINARY_INIT(name, complex64, uint32, bool),      \
    CPU_HOST_BINARY_INIT(name, complex64, int8, bool),        \
    CPU_HOST_BINARY_INIT(name, complex64, int16, bool),       \
    CPU_HOST_BINARY_INIT(name, complex64, int32, bool),       \
    CPU_HOST_BINARY_INIT(name, complex64, float16, bool),     \
    CPU_HOST_BINARY_INIT(name, complex64, float32, bool),     \
    CPU_HOST_BINARY_INIT(name, complex64, float64, bool),     \
    CPU_HOST_BINARY_INIT(name, complex64, complex32, bool),   \
    CPU_HOST_BINARY_INIT(name, complex64, complex64, bool),   \
    CPU_HOST_BINARY_INIT(name, complex64, complex128, bool),  \
                                                              \
    CPU_HOST_BINARY_INIT(name, complex128, uint8, bool),      \
    CPU_HOST_BINARY_INIT(name, complex128, uint16, bool),     \
    CPU_HOST_BINARY_INIT(name, complex128, uint32, bool),     \
    CPU_HOST_BINARY_INIT(name, complex128, int8, bool),       \
    CPU_HOST_BINARY_INIT(name, complex128, int16, bool),      \
    CPU_HOST_BINARY_INIT(name, complex128, int32, bool),      \
    CPU_HOST_BINARY_INIT(name, complex128, float16, bool),    \
    CPU_HOST_BINARY_INIT(name, complex128, float32, bool),    \
    CPU_HOST_BINARY_INIT(name, complex128, float64, bool),    \
    CPU_HOST_BINARY_INIT(name, complex128, complex32, bool),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex64, bool),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex128, bool)


#undef bool
#define bool_t _Bool

#define less_equal(x, y) x <= y
CPU_HOST_ALL_COMPARISON(less_equal)

#define less(x, y) x < y
CPU_HOST_ALL_COMPARISON(less)

#define greater_equal(x, y) x >= y
CPU_HOST_ALL_COMPARISON(greater_equal)

#define greater(x, y) x > y
CPU_HOST_ALL_COMPARISON(greater)


static const gm_kernel_init_t kernels[] = {
  CPU_HOST_ALL_ARITHMETIC_INIT(add),
  CPU_HOST_ALL_ARITHMETIC_INIT(subtract),
  CPU_HOST_ALL_ARITHMETIC_INIT(multiply),
  CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN_INIT(divide),
  CPU_HOST_ALL_COMPARISON_INIT(greater),
  CPU_HOST_ALL_COMPARISON_INIT(greater_equal),
  CPU_HOST_ALL_COMPARISON_INIT(less),
  CPU_HOST_ALL_COMPARISON_INIT(less_equal),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

static const gm_kernel_set_t *
typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
          const int64_t li[], int nin, int nout, ndt_context_t *ctx)
{
    return cpu_binary_typecheck(kernel_location, spec, f, types, li, nin, nout, ctx);
}

int
gm_init_cpu_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &typecheck) < 0) {
             return -1;
        }
    }

    return 0;
}

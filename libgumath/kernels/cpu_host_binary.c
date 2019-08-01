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
#include "cpu_device_binary.h"


/****************************************************************************/
/*                     Optimized dispatch (exact casting)                   */
/****************************************************************************/

/* Structured kernel locations for fast lookup. */
static int
binary_kernel_location(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx)
{
    const ndt_t *t0 = ndt_dtype(in0);
    const ndt_t *t1 = ndt_dtype(in1);

    switch (t0->tag) {
    case Uint8: {
        switch (t1->tag) {
        case Uint8: return 0;
        case Uint16: return 12;
        case Uint32: return 24;
        case Uint64: return 36;

        case Int8: return 48;
        case Int16: return 60;
        case Int32: return 72;
        case Int64: return 84;

        case BFloat16: return 96;
        case Float16: return 108;
        case Float32: return 120;
        case Float64: return 132;

        case Complex32: return 144;
        case Complex64: return 156;
        case Complex128: return 168;

        default: goto invalid_combination;
        }
    }

    case Uint16: {
        switch (t1->tag) {
        case Uint8: return 180;
        case Uint16: return 192;
        case Uint32: return 204;
        case Uint64: return 216;

        case Int8: return 228;
        case Int16: return 240;
        case Int32: return 252;
        case Int64: return 264;

        case BFloat16: return 276;
        case Float16: return 288;
        case Float32: return 300;
        case Float64: return 312;

        case Complex32: return 324;
        case Complex64: return 336;
        case Complex128: return 348;

        default: goto invalid_combination;
        }
    }

    case Uint32: {
        switch (t1->tag) {
        case Uint8: return 360;
        case Uint16: return 372;
        case Uint32: return 384;
        case Uint64: return 396;

        case Int8: return 408;
        case Int16: return 420;
        case Int32: return 432;
        case Int64: return 444;

        case BFloat16: return 456;
        case Float16: return 468;
        case Float32: return 480;
        case Float64: return 492;

        case Complex32: return 504;
        case Complex64: return 516;
        case Complex128: return 528;

        default: goto invalid_combination;
        }
    }

    case Uint64: {
        switch (t1->tag) {
        case Uint8: return 540;
        case Uint16: return 552;
        case Uint32: return 564;
        case Uint64: return 576;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Uint8: return 588;
        case Uint16: return 600;
        case Uint32: return 612;

        case Int8: return 624;
        case Int16: return 636;
        case Int32: return 648;
        case Int64: return 660;

        case BFloat16: return 672;
        case Float16: return 684;
        case Float32: return 696;
        case Float64: return 708;

        case Complex32: return 720;
        case Complex64: return 732;
        case Complex128: return 744;

        default: goto invalid_combination;
        }
    }

    case Int16: {
        switch (t1->tag) {
        case Uint8: return 756;
        case Uint16: return 768;
        case Uint32: return 780;

        case Int8: return 792;
        case Int16: return 804;
        case Int32: return 816;
        case Int64: return 828;

        case BFloat16: return 840;
        case Float16: return 852;
        case Float32: return 864;
        case Float64: return 876;

        case Complex32: return 888;
        case Complex64: return 900;
        case Complex128: return 912;

        default: goto invalid_combination;
        }
    }

    case Int32: {
        switch (t1->tag) {
        case Uint8: return 924;
        case Uint16: return 936;
        case Uint32: return 948;

        case Int8: return 960;
        case Int16: return 972;
        case Int32: return 984;
        case Int64: return 996;

        case BFloat16: return 1008;
        case Float16: return 1020;
        case Float32: return 1032;
        case Float64: return 1044;

        case Complex32: return 1056;
        case Complex64: return 1068;
        case Complex128: return 1080;

        default: goto invalid_combination;
        }
    }

    case Int64: {
        switch (t1->tag) {
        case Uint8: return 1092;
        case Uint16: return 1104;
        case Uint32: return 1116;

        case Int8: return 1128;
        case Int16: return 1140;
        case Int32: return 1152;
        case Int64: return 1164;

        default: goto invalid_combination;
        }
    }

    case BFloat16: {
        switch (t1->tag) {
        case Uint8: return 1176;
        case Uint16: return 1188;
        case Uint32: return 1200;

        case Int8: return 1212;
        case Int16: return 1224;
        case Int32: return 1236;

        case BFloat16: return 1248;
        case Float16: return 1260;
        case Float32: return 1272;
        case Float64: return 1284;

        case Complex32: return 1296;
        case Complex64: return 1308;
        case Complex128: return 1320;

        default: goto invalid_combination;
        }
    }

    case Float16: {
        switch (t1->tag) {
        case Uint8: return 1332;
        case Uint16: return 1344;
        case Uint32: return 1356;

        case Int8: return 1368;
        case Int16: return 1380;
        case Int32: return 1392;

        case BFloat16: return 1404;
        case Float16: return 1416;
        case Float32: return 1428;
        case Float64: return 1440;

        case Complex32: return 1452;
        case Complex64: return 1464;
        case Complex128: return 1476;

        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (t1->tag) {
        case Uint8: return 1488;
        case Uint16: return 1500;
        case Uint32: return 1512;

        case Int8: return 1524;
        case Int16: return 1536;
        case Int32: return 1548;

        case BFloat16: return 1560;
        case Float16: return 1572;
        case Float32: return 1584;
        case Float64: return 1596;

        case Complex32: return 1608;
        case Complex64: return 1620;
        case Complex128: return 1632;

        default: goto invalid_combination;
        }
    }

    case Float64: {
        switch (t1->tag) {
        case Uint8: return 1644;
        case Uint16: return 1656;
        case Uint32: return 1668;

        case Int8: return 1680;
        case Int16: return 1692;
        case Int32: return 1704;

        case BFloat16: return 1716;
        case Float16: return 1728;
        case Float32: return 1740;
        case Float64: return 1752;

        case Complex32: return 1764;
        case Complex64: return 1776;
        case Complex128: return 1788;

        default: goto invalid_combination;
        }
    }

    case Complex32: {
        switch (t1->tag) {
        case Uint8: return 1800;
        case Uint16: return 1812;
        case Uint32: return 1824;

        case Int8: return 1836;
        case Int16: return 1848;
        case Int32: return 1860;

        case BFloat16: return 1872;
        case Float16: return 1884;
        case Float32: return 1896;
        case Float64: return 1908;

        case Complex32: return 1920;
        case Complex64: return 1932;
        case Complex128: return 1944;

        default: goto invalid_combination;
        }
    }

    case Complex64: {
        switch (t1->tag) {
        case Uint8: return 1956;
        case Uint16: return 1968;
        case Uint32: return 1980;

        case Int8: return 1992;
        case Int16: return 2004;
        case Int32: return 2016;

        case BFloat16: return 2028;
        case Float16: return 2040;
        case Float32: return 2052;
        case Float64: return 2064;

        case Complex32: return 2076;
        case Complex64: return 2088;
        case Complex128: return 2100;

        default: goto invalid_combination;
        }
    }

    case Complex128: {
        switch (t1->tag) {
        case Uint8: return 2112;
        case Uint16: return 2124;
        case Uint32: return 2136;

        case Int8: return 2148;
        case Int16: return 2160;
        case Int32: return 2172;

        case BFloat16: return 2184;
        case Float16: return 2196;
        case Float32: return 2208;
        case Float64: return 2220;

        case Complex32: return 2232;
        case Complex64: return 2244;
        case Complex128: return 2256;

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

static int
bitwise_kernel_location(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx)
{
    const ndt_t *t0 = ndt_dtype(in0);
    const ndt_t *t1 = ndt_dtype(in1);

    switch (t0->tag) {
    case Bool: {
        switch (t1->tag) {
        case Bool: return 0;

        case Uint8: return 12;
        case Uint16: return 24;
        case Uint32: return 36;
        case Uint64: return 48;

        case Int8: return 60;
        case Int16: return 72;
        case Int32: return 84;
        case Int64: return 96;

        default: goto invalid_combination;
        }
    }

    case Uint8: {
        switch (t1->tag) {
        case Bool: return 108;

        case Uint8: return 120;
        case Uint16: return 132;
        case Uint32: return 144;
        case Uint64: return 156;

        case Int8: return 168;
        case Int16: return 180;
        case Int32: return 192;
        case Int64: return 204;

        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Bool: return 216;

        case Int8: return 228;
        case Int16: return 240;
        case Int32: return 252;
        case Int64: return 264;

        case Uint8: return 276;
        case Uint16: return 288;
        case Uint32: return 300;
        case Uint64: return 312;

        default: goto invalid_combination;
        }
    }
    case Uint32: {
        switch (t1->tag) {
        case Bool: return 324;

        case Uint8: return 336;
        case Uint16: return 348;
        case Uint32: return 360;
        case Uint64: return 372;

        case Int8: return 384;
        case Int16: return 396;
        case Int32: return 408;
        case Int64: return 420;

        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Bool: return 432;

        case Uint8: return 444;
        case Uint16: return 456;
        case Uint32: return 468;
        case Uint64: return 480;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Bool: return 492;

        case Uint8: return 504;
        case Uint16: return 516;
        case Uint32: return 528;

        case Int8: return 540;
        case Int16: return 552;
        case Int32: return 564;
        case Int64: return 576;

        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Bool: return 588;

        case Uint8: return 600;
        case Uint16: return 612;
        case Uint32: return 624;

        case Int8: return 636;
        case Int16: return 648;
        case Int32: return 660;
        case Int64: return 672;

        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Bool: return 684;

        case Uint8: return 696;
        case Uint16: return 708;
        case Uint32: return 720;

        case Int8: return 732;
        case Int16: return 744;
        case Int32: return 756;
        case Int64: return 768;

        default: goto invalid_combination;
        }
    }

    case Int64: {
        switch (t1->tag) {
        case Bool: return 780;

        case Uint8: return 792;
        case Uint16: return 804;
        case Uint32: return 816;

        case Int8: return 828;
        case Int16: return 840;
        case Int32: return 852;
        case Int64: return 864;

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


#define CPU_CHECK_POWER_EXP(t1) \
static inline int                                                     \
check_power_exp_##t1(const char *a1, ndt_context_t *ctx)              \
{                                                                     \
    const t1##_t exp = *(const t1##_t *)a1;                           \
    if (exp < 0) {                                                    \
        ndt_err_format(ctx, NDT_ValueError,                           \
            "negative exponents are not allowed for integer powers"); \
        return -1;                                                    \
    }                                                                 \
                                                                      \
    return 0;                                                         \
}

#define CPU_CHECK_POWER_EXP_SUCCESS(t1) \
static inline int                                        \
check_power_exp_##t1(const char *a1, ndt_context_t *ctx) \
{                                                        \
    (void)a1;                                            \
    (void)ctx;                                           \
                                                         \
    return 0;                                            \
}


CPU_CHECK_POWER_EXP(int8)
CPU_CHECK_POWER_EXP(int16)
CPU_CHECK_POWER_EXP(int32)
CPU_CHECK_POWER_EXP(int64)

CPU_CHECK_POWER_EXP_SUCCESS(bool)

CPU_CHECK_POWER_EXP_SUCCESS(uint8)
CPU_CHECK_POWER_EXP_SUCCESS(uint16)
CPU_CHECK_POWER_EXP_SUCCESS(uint32)
CPU_CHECK_POWER_EXP_SUCCESS(uint64)

CPU_CHECK_POWER_EXP_SUCCESS(bfloat16)
CPU_CHECK_POWER_EXP_SUCCESS(float32)
CPU_CHECK_POWER_EXP_SUCCESS(float64)

CPU_CHECK_POWER_EXP_SUCCESS(complex64)
CPU_CHECK_POWER_EXP_SUCCESS(complex128)


#define CPU_HOST_BINARY(name, t0, t1, t2) \
static int                                                                             \
gm_cpu_host_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)  \
{                                                                                      \
    const char *a0 = apply_index(&stack[0]);                                           \
    const char *a1 = apply_index(&stack[1]);                                           \
    char *a2 = apply_index(&stack[2]);                                                 \
    const int64_t N = xnd_fixed_shape(&stack[0]);                                      \
    (void)ctx;                                                                         \
                                                                                       \
    if (strcmp(STRINGIZE(name), "power") == 0) {                                       \
        if (check_power_exp_##t1(a1, ctx) < 0) {                                       \
            return -1;                                                                 \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2(a0, a1, a2, N);               \
                                                                                       \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                                   \
        binary_update_bitmap_1D_S(stack);                                              \
    }                                                                                  \
    else if (strcmp(STRINGIZE(name), "equaln") == 0) {                                 \
        binary_update_bitmap_1D_S_bool(stack);                                         \
    }                                                                                  \
                                                                                       \
    return 0;                                                                          \
}                                                                                      \
                                                                                       \
static int                                                                             \
gm_cpu_host_fixed_1D_S_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)  \
{                                                                                      \
    const char *a0 = apply_index(&stack[0]);                                           \
    const char *a1 = apply_index(&stack[1]);                                           \
    char *a2 = apply_index(&stack[2]);                                                 \
    const int64_t N = xnd_fixed_shape(&stack[0]);                                      \
    const int64_t s0 = xnd_fixed_step(&stack[0]);                                      \
    const int64_t s1 = xnd_fixed_step(&stack[1]);                                      \
    const int64_t s2 = xnd_fixed_step(&stack[2]);                                      \
    (void)ctx;                                                                         \
                                                                                       \
    if (strcmp(STRINGIZE(name), "power") == 0) {                                       \
        if (check_power_exp_##t1(a1, ctx) < 0) {                                       \
            return -1;                                                                 \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    gm_cpu_device_fixed_1D_S_##name##_##t0##_##t1##_##t2(a0, a1, a2, s0, s1, s2, N);   \
                                                                                       \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                                   \
        binary_update_bitmap_1D_S(stack);                                              \
    }                                                                                  \
    else if (strcmp(STRINGIZE(name), "equaln") == 0) {                                 \
        binary_update_bitmap_1D_S_bool(stack);                                         \
    }                                                                                  \
                                                                                       \
    return 0;                                                                          \
}                                                                                      \
                                                                                       \
static int                                                                             \
gm_cpu_host_array_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)  \
{                                                                                      \
    const char *a0 = XND_ARRAY_DATA(stack[0].ptr);                                     \
    const int64_t N = XND_ARRAY_SHAPE(stack[0].ptr);                                   \
    (void)ctx;                                                                         \
                                                                                       \
    if (array_shape_check(&stack[1], N, ctx) < 0) {                                    \
        return -1;                                                                     \
    }                                                                                  \
    const char *a1 = XND_ARRAY_DATA(stack[1].ptr);                                     \
                                                                                       \
    if (array_shape_check(&stack[2], N, ctx) < 0) {                                    \
        return -1;                                                                     \
    }                                                                                  \
    char *a2 = XND_ARRAY_DATA(stack[2].ptr);                                           \
                                                                                       \
    if (strcmp(STRINGIZE(name), "power") == 0) {                                       \
        if (check_power_exp_##t1(a1, ctx) < 0) {                                       \
            return -1;                                                                 \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2(a0, a1, a2, N);               \
                                                                                       \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                                   \
        binary_update_bitmap_1D_S(stack);                                              \
    }                                                                                  \
    else if (strcmp(STRINGIZE(name), "equaln") == 0) {                                 \
        binary_update_bitmap_1D_S_bool(stack);                                         \
    }                                                                                  \
                                                                                       \
    return 0;                                                                          \
}                                                                                      \
                                                                                       \
static int                                                                             \
gm_cpu_host_0D_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)          \
{                                                                                      \
    const char *a0 = stack[0].ptr;                                                     \
    const char *a1 = stack[1].ptr;                                                     \
    char *a2 = stack[2].ptr;                                                           \
    (void)ctx;                                                                         \
                                                                                       \
    if (strcmp(STRINGIZE(name), "power") == 0) {                                       \
        if (check_power_exp_##t1(a1, ctx) < 0) {                                       \
            return -1;                                                                 \
        }                                                                              \
    }                                                                                  \
                                                                                       \
    gm_cpu_device_0D_##name##_##t0##_##t1##_##t2(a0, a1, a2);                          \
                                                                                       \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                                   \
        binary_update_bitmap_0D(stack);                                                \
    }                                                                                  \
    else if (strcmp(STRINGIZE(name), "equaln") == 0) {                                 \
        binary_update_bitmap_0D_bool(stack);                                           \
    }                                                                                  \
                                                                                       \
    return 0;                                                                          \
}


#define CPU_HOST_NOIMPL(name, t0, t1, t2) \
static int                                                                            \
gm_cpu_host_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_NotImplementedError,                                      \
        "implementation for " STRINGIZE(name) " : "                                   \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                         \
        " currently requires double rounding");                                       \
                                                                                      \
    return -1;                                                                        \
}                                                                                     \
                                                                                      \
static int                                                                            \
gm_cpu_host_fixed_1D_S_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_NotImplementedError,                                      \
        "implementation for " STRINGIZE(name) " : "                                   \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                         \
        " currently requires double rounding");                                       \
                                                                                      \
    return -1;                                                                        \
}                                                                                     \
                                                                                      \
static int                                                                            \
gm_cpu_host_array_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_NotImplementedError,                                      \
        "implementation for " STRINGIZE(name) " : "                                   \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                         \
        " currently requires double rounding");                                       \
                                                                                      \
    return -1;                                                                        \
}                                                                                     \
                                                                                      \
static int                                                                            \
gm_cpu_host_0D_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_NotImplementedError,                                      \
        "implementation for " STRINGIZE(name) " : "                                   \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                         \
        " currently requires double rounding");                                       \
                                                                                      \
    return -1;                                                                        \
}

#define CPU_HOST_NOKERN(name, t0, t1, t2) \
static int                                                                            \
gm_cpu_host_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_TypeError,                                                \
        "no kernel for " STRINGIZE(name) " : "                                        \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));                       \
                                                                                      \
    return -1;                                                                        \
}                                                                                     \
                                                                                      \
static int                                                                            \
gm_cpu_host_fixed_1D_S_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_TypeError,                                                \
        "no kernel for " STRINGIZE(name) " : "                                        \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));                       \
                                                                                      \
    return -1;                                                                        \
}                                                                                     \
                                                                                      \
static int                                                                            \
gm_cpu_host_array_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_TypeError,                                                \
        "no kernel for " STRINGIZE(name) " : "                                        \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));                       \
                                                                                      \
    return -1;                                                                        \
}                                                                                     \
                                                                                      \
static int                                                                            \
gm_cpu_host_0D_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                                     \
    (void)stack;                                                                      \
                                                                                      \
    ndt_err_format(ctx, NDT_TypeError,                                                \
        "no kernel for " STRINGIZE(name) " : "                                        \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));                       \
                                                                                      \
    return -1;                                                                        \
}


#define CPU_HOST_BINARY_INIT(func, t0, t1, t2) \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),                   \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                          \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "... * ?" STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),                 \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                          \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "... * " STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),                 \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                          \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "... * ?" STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),                \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                          \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                          \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "var... * " STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * " STRINGIZE(t2),          \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),        \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "var... * " STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),        \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * ?" STRINGIZE(t2),       \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                 \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "array... * " STRINGIZE(t0) ", array... * " STRINGIZE(t1) " -> array... * " STRINGIZE(t2),    \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 },                                        \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "array... * ?" STRINGIZE(t0) ", array... * " STRINGIZE(t1) " -> array... * ?" STRINGIZE(t2),  \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 },                                        \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "array... * " STRINGIZE(t0) ", array... * ?" STRINGIZE(t1) " -> array... * ?" STRINGIZE(t2),  \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 },                                        \
                                                                                                         \
  { .name = STRINGIZE(func),                                                                             \
    .sig = "array... * ?" STRINGIZE(t0) ", array... * ?" STRINGIZE(t1) " -> array... * ?" STRINGIZE(t2), \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 }


#define CPU_HOST_EQUALN_INIT(func, t0, t1, t2) \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),                  \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                         \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                         \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "... * ?" STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),                 \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                         \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                         \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "... * " STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * " STRINGIZE(t2),                 \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                         \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                         \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "... * ?" STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * " STRINGIZE(t2),                \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                         \
    .OptS = gm_cpu_host_fixed_1D_S_##func##_##t0##_##t1##_##t2,                                         \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "var... * " STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * " STRINGIZE(t2),         \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * " STRINGIZE(t1) " -> var... * " STRINGIZE(t2),        \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "var... * " STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * " STRINGIZE(t2),        \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "var... * ?" STRINGIZE(t0) ", var... * ?" STRINGIZE(t1) " -> var... * " STRINGIZE(t2),       \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2 },                                                \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "array... * " STRINGIZE(t0) ", array... * " STRINGIZE(t1) " -> array... * " STRINGIZE(t2),   \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 },                                       \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "array... * ?" STRINGIZE(t0) ", array... * " STRINGIZE(t1) " -> array... * " STRINGIZE(t2),  \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 },                                       \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "array... * " STRINGIZE(t0) ", array... * ?" STRINGIZE(t1) " -> array... * " STRINGIZE(t2),  \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 },                                       \
                                                                                                        \
  { .name = STRINGIZE(func),                                                                            \
    .sig = "array... * ?" STRINGIZE(t0) ", array... * ?" STRINGIZE(t1) " -> array... * " STRINGIZE(t2), \
    .OptC = gm_cpu_host_array_1D_C_##func##_##t0##_##t1##_##t2 }


#undef bool
#define bool_t _Bool


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
    CPU_HOST_BINARY(name, uint8, bfloat16, bfloat16)          \
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
    CPU_HOST_BINARY(name, uint16, bfloat16, float32)          \
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
    CPU_HOST_BINARY(name, uint32, bfloat16, float64)          \
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
    CPU_HOST_BINARY(name, int8, bfloat16, bfloat16)           \
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
    CPU_HOST_BINARY(name, int16, bfloat16, float32)           \
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
    CPU_HOST_BINARY(name, int32, bfloat16, float64)           \
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
    CPU_HOST_BINARY(name, bfloat16, uint8, bfloat16)          \
    CPU_HOST_BINARY(name, bfloat16, uint16, float32)          \
    CPU_HOST_BINARY(name, bfloat16, uint32, float64)          \
    CPU_HOST_BINARY(name, bfloat16, int8, bfloat16)           \
    CPU_HOST_BINARY(name, bfloat16, int16, float32)           \
    CPU_HOST_BINARY(name, bfloat16, int32, float64)           \
    CPU_HOST_BINARY(name, bfloat16, bfloat16, bfloat16)       \
    CPU_HOST_NOIMPL(name, bfloat16, float16, float32)         \
    CPU_HOST_BINARY(name, bfloat16, float32, float32)         \
    CPU_HOST_BINARY(name, bfloat16, float64, float64)         \
    CPU_HOST_NOIMPL(name, bfloat16, complex32, complex64)     \
    CPU_HOST_BINARY(name, bfloat16, complex64, complex64)     \
    CPU_HOST_BINARY(name, bfloat16, complex128, complex128)   \
                                                              \
    CPU_HOST_NOIMPL(name, float16, uint8, float16)            \
    CPU_HOST_NOIMPL(name, float16, uint16, float32)           \
    CPU_HOST_NOIMPL(name, float16, uint32, float64)           \
    CPU_HOST_NOIMPL(name, float16, int8, float16)             \
    CPU_HOST_NOIMPL(name, float16, int16, float32)            \
    CPU_HOST_NOIMPL(name, float16, int32, float64)            \
    CPU_HOST_NOIMPL(name, float16, bfloat16, float32)         \
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
    CPU_HOST_BINARY(name, float32, bfloat16, float32)         \
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
    CPU_HOST_BINARY(name, float64, bfloat16, float64)         \
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
    CPU_HOST_NOIMPL(name, complex32, bfloat16, complex64)     \
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
    CPU_HOST_BINARY(name, complex64, bfloat16, complex64)     \
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
    CPU_HOST_BINARY(name, complex128, bfloat16, complex128)   \
    CPU_HOST_NOIMPL(name, complex128, float16, complex128)    \
    CPU_HOST_BINARY(name, complex128, float32, complex128)    \
    CPU_HOST_BINARY(name, complex128, float64, complex128)    \
    CPU_HOST_NOIMPL(name, complex128, complex32, complex128)  \
    CPU_HOST_BINARY(name, complex128, complex64, complex128)  \
    CPU_HOST_BINARY(name, complex128, complex128, complex128)

#define CPU_HOST_ALL_ARITHMETIC_NO_COMPLEX(name) \
    CPU_HOST_BINARY(name, uint8, uint8, uint8)                \
    CPU_HOST_BINARY(name, uint8, uint16, uint16)              \
    CPU_HOST_BINARY(name, uint8, uint32, uint32)              \
    CPU_HOST_BINARY(name, uint8, uint64, uint64)              \
    CPU_HOST_BINARY(name, uint8, int8, int16)                 \
    CPU_HOST_BINARY(name, uint8, int16, int16)                \
    CPU_HOST_BINARY(name, uint8, int32, int32)                \
    CPU_HOST_BINARY(name, uint8, int64, int64)                \
    CPU_HOST_BINARY(name, uint8, bfloat16, bfloat16)          \
    CPU_HOST_NOIMPL(name, uint8, float16, float16)            \
    CPU_HOST_BINARY(name, uint8, float32, float32)            \
    CPU_HOST_BINARY(name, uint8, float64, float64)            \
    CPU_HOST_NOKERN(name, uint8, complex32, complex32)        \
    CPU_HOST_NOKERN(name, uint8, complex64, complex64)        \
    CPU_HOST_NOKERN(name, uint8, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, uint16, uint8, uint16)              \
    CPU_HOST_BINARY(name, uint16, uint16, uint16)             \
    CPU_HOST_BINARY(name, uint16, uint32, uint32)             \
    CPU_HOST_BINARY(name, uint16, uint64, uint64)             \
    CPU_HOST_BINARY(name, uint16, int8, int32)                \
    CPU_HOST_BINARY(name, uint16, int16, int32)               \
    CPU_HOST_BINARY(name, uint16, int32, int32)               \
    CPU_HOST_BINARY(name, uint16, int64, int64)               \
    CPU_HOST_BINARY(name, uint16, bfloat16, float32)          \
    CPU_HOST_NOIMPL(name, uint16, float16, float32)           \
    CPU_HOST_BINARY(name, uint16, float32, float32)           \
    CPU_HOST_BINARY(name, uint16, float64, float64)           \
    CPU_HOST_NOKERN(name, uint16, complex32, complex64)       \
    CPU_HOST_NOKERN(name, uint16, complex64, complex64)       \
    CPU_HOST_NOKERN(name, uint16, complex128, complex128)     \
                                                              \
    CPU_HOST_BINARY(name, uint32, uint8, uint32)              \
    CPU_HOST_BINARY(name, uint32, uint16, uint32)             \
    CPU_HOST_BINARY(name, uint32, uint32, uint32)             \
    CPU_HOST_BINARY(name, uint32, uint64, uint64)             \
    CPU_HOST_BINARY(name, uint32, int8, int64)                \
    CPU_HOST_BINARY(name, uint32, int16, int64)               \
    CPU_HOST_BINARY(name, uint32, int32, int64)               \
    CPU_HOST_BINARY(name, uint32, int64, int64)               \
    CPU_HOST_BINARY(name, uint32, bfloat16, float64)          \
    CPU_HOST_NOIMPL(name, uint32, float16, float64)           \
    CPU_HOST_BINARY(name, uint32, float32, float64)           \
    CPU_HOST_BINARY(name, uint32, float64, float64)           \
    CPU_HOST_NOKERN(name, uint32, complex32, complex128)      \
    CPU_HOST_NOKERN(name, uint32, complex64, complex128)      \
    CPU_HOST_NOKERN(name, uint32, complex128, complex128)     \
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
    CPU_HOST_BINARY(name, int8, bfloat16, bfloat16)           \
    CPU_HOST_NOIMPL(name, int8, float16, float16)             \
    CPU_HOST_BINARY(name, int8, float32, float32)             \
    CPU_HOST_BINARY(name, int8, float64, float64)             \
    CPU_HOST_NOKERN(name, int8, complex32, complex32)         \
    CPU_HOST_NOKERN(name, int8, complex64, complex64)         \
    CPU_HOST_NOKERN(name, int8, complex128, complex128)       \
                                                              \
    CPU_HOST_BINARY(name, int16, uint8, int16)                \
    CPU_HOST_BINARY(name, int16, uint16, int32)               \
    CPU_HOST_BINARY(name, int16, uint32, int64)               \
    CPU_HOST_BINARY(name, int16, int8, int16)                 \
    CPU_HOST_BINARY(name, int16, int16, int16)                \
    CPU_HOST_BINARY(name, int16, int32, int32)                \
    CPU_HOST_BINARY(name, int16, int64, int64)                \
    CPU_HOST_BINARY(name, int16, bfloat16, float32)           \
    CPU_HOST_NOIMPL(name, int16, float16, float32)            \
    CPU_HOST_BINARY(name, int16, float32, float32)            \
    CPU_HOST_BINARY(name, int16, float64, float64)            \
    CPU_HOST_NOKERN(name, int16, complex32, complex64)        \
    CPU_HOST_NOKERN(name, int16, complex64, complex64)        \
    CPU_HOST_NOKERN(name, int16, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, int32, uint8, int32)                \
    CPU_HOST_BINARY(name, int32, uint16, int32)               \
    CPU_HOST_BINARY(name, int32, uint32, int64)               \
    CPU_HOST_BINARY(name, int32, int8, int32)                 \
    CPU_HOST_BINARY(name, int32, int16, int32)                \
    CPU_HOST_BINARY(name, int32, int32, int32)                \
    CPU_HOST_BINARY(name, int32, int64, int64)                \
    CPU_HOST_BINARY(name, int32, bfloat16, float64)           \
    CPU_HOST_NOIMPL(name, int32, float16, float64)            \
    CPU_HOST_BINARY(name, int32, float32, float64)            \
    CPU_HOST_BINARY(name, int32, float64, float64)            \
    CPU_HOST_NOKERN(name, int32, complex32, complex128)       \
    CPU_HOST_NOKERN(name, int32, complex64, complex128)       \
    CPU_HOST_NOKERN(name, int32, complex128, complex128)      \
                                                              \
    CPU_HOST_BINARY(name, int64, uint8, int64)                \
    CPU_HOST_BINARY(name, int64, uint16, int64)               \
    CPU_HOST_BINARY(name, int64, uint32, int64)               \
    CPU_HOST_BINARY(name, int64, int8, int64)                 \
    CPU_HOST_BINARY(name, int64, int16, int64)                \
    CPU_HOST_BINARY(name, int64, int32, int64)                \
    CPU_HOST_BINARY(name, int64, int64, int64)                \
                                                              \
    CPU_HOST_BINARY(name, bfloat16, uint8, bfloat16)          \
    CPU_HOST_BINARY(name, bfloat16, uint16, float32)          \
    CPU_HOST_BINARY(name, bfloat16, uint32, float64)          \
    CPU_HOST_BINARY(name, bfloat16, int8, bfloat16)           \
    CPU_HOST_BINARY(name, bfloat16, int16, float32)           \
    CPU_HOST_BINARY(name, bfloat16, int32, float64)           \
    CPU_HOST_BINARY(name, bfloat16, bfloat16, bfloat16)       \
    CPU_HOST_NOIMPL(name, bfloat16, float16, float32)         \
    CPU_HOST_BINARY(name, bfloat16, float32, float32)         \
    CPU_HOST_BINARY(name, bfloat16, float64, float64)         \
    CPU_HOST_NOKERN(name, bfloat16, complex32, complex64)     \
    CPU_HOST_NOKERN(name, bfloat16, complex64, complex64)     \
    CPU_HOST_NOKERN(name, bfloat16, complex128, complex128)   \
                                                              \
    CPU_HOST_NOIMPL(name, float16, uint8, float16)            \
    CPU_HOST_NOIMPL(name, float16, uint16, float32)           \
    CPU_HOST_NOIMPL(name, float16, uint32, float64)           \
    CPU_HOST_NOIMPL(name, float16, int8, float16)             \
    CPU_HOST_NOIMPL(name, float16, int16, float32)            \
    CPU_HOST_NOIMPL(name, float16, int32, float64)            \
    CPU_HOST_NOIMPL(name, float16, bfloat16, float32)         \
    CPU_HOST_NOIMPL(name, float16, float16, float16)          \
    CPU_HOST_NOIMPL(name, float16, float32, float32)          \
    CPU_HOST_NOIMPL(name, float16, float64, float64)          \
    CPU_HOST_NOKERN(name, float16, complex32, complex32)      \
    CPU_HOST_NOKERN(name, float16, complex64, complex64)      \
    CPU_HOST_NOKERN(name, float16, complex128, complex128)    \
                                                              \
    CPU_HOST_BINARY(name, float32, uint8, float32)            \
    CPU_HOST_BINARY(name, float32, uint16, float32)           \
    CPU_HOST_BINARY(name, float32, uint32, float64)           \
    CPU_HOST_BINARY(name, float32, int8, float32)             \
    CPU_HOST_BINARY(name, float32, int16, float32)            \
    CPU_HOST_BINARY(name, float32, int32, float64)            \
    CPU_HOST_BINARY(name, float32, bfloat16, float32)         \
    CPU_HOST_NOIMPL(name, float32, float16, float32)          \
    CPU_HOST_BINARY(name, float32, float32, float32)          \
    CPU_HOST_BINARY(name, float32, float64, float64)          \
    CPU_HOST_NOKERN(name, float32, complex32, complex64)      \
    CPU_HOST_NOKERN(name, float32, complex64, complex64)      \
    CPU_HOST_NOKERN(name, float32, complex128, complex128)    \
                                                              \
    CPU_HOST_BINARY(name, float64, uint8, float64)            \
    CPU_HOST_BINARY(name, float64, uint16, float64)           \
    CPU_HOST_BINARY(name, float64, uint32, float64)           \
    CPU_HOST_BINARY(name, float64, int8, float64)             \
    CPU_HOST_BINARY(name, float64, int16, float64)            \
    CPU_HOST_BINARY(name, float64, int32, float64)            \
    CPU_HOST_BINARY(name, float64, bfloat16, float64)         \
    CPU_HOST_NOIMPL(name, float64, float16, float64)          \
    CPU_HOST_BINARY(name, float64, float32, float64)          \
    CPU_HOST_BINARY(name, float64, float64, float64)          \
    CPU_HOST_NOKERN(name, float64, complex32, complex128)     \
    CPU_HOST_NOKERN(name, float64, complex64, complex128)     \
    CPU_HOST_NOKERN(name, float64, complex128, complex128)    \
                                                              \
    CPU_HOST_NOKERN(name, complex32, uint8, complex32)        \
    CPU_HOST_NOKERN(name, complex32, uint16, complex64)       \
    CPU_HOST_NOKERN(name, complex32, uint32, complex128)      \
    CPU_HOST_NOKERN(name, complex32, int8, complex32)         \
    CPU_HOST_NOKERN(name, complex32, int16, complex64)        \
    CPU_HOST_NOKERN(name, complex32, int32, complex128)       \
    CPU_HOST_NOKERN(name, complex32, bfloat16, complex64)     \
    CPU_HOST_NOKERN(name, complex32, float16, complex32)      \
    CPU_HOST_NOKERN(name, complex32, float32, complex64)      \
    CPU_HOST_NOKERN(name, complex32, float64, complex128)     \
    CPU_HOST_NOKERN(name, complex32, complex32, complex32)    \
    CPU_HOST_NOKERN(name, complex32, complex64, complex64)    \
    CPU_HOST_NOKERN(name, complex32, complex128, complex128)  \
                                                              \
    CPU_HOST_NOKERN(name, complex64, uint8, complex64)        \
    CPU_HOST_NOKERN(name, complex64, uint16, complex64)       \
    CPU_HOST_NOKERN(name, complex64, uint32, complex128)      \
    CPU_HOST_NOKERN(name, complex64, int8, complex64)         \
    CPU_HOST_NOKERN(name, complex64, int16, complex64)        \
    CPU_HOST_NOKERN(name, complex64, int32, complex128)       \
    CPU_HOST_NOKERN(name, complex64, bfloat16, complex64)     \
    CPU_HOST_NOKERN(name, complex64, float16, complex64)      \
    CPU_HOST_NOKERN(name, complex64, float32, complex64)      \
    CPU_HOST_NOKERN(name, complex64, float64, complex128)     \
    CPU_HOST_NOKERN(name, complex64, complex32, complex64)    \
    CPU_HOST_NOKERN(name, complex64, complex64, complex64)    \
    CPU_HOST_NOKERN(name, complex64, complex128, complex128)  \
                                                              \
    CPU_HOST_NOKERN(name, complex128, uint8, complex128)      \
    CPU_HOST_NOKERN(name, complex128, uint16, complex128)     \
    CPU_HOST_NOKERN(name, complex128, uint32, complex128)     \
    CPU_HOST_NOKERN(name, complex128, int8, complex128)       \
    CPU_HOST_NOKERN(name, complex128, int16, complex128)      \
    CPU_HOST_NOKERN(name, complex128, int32, complex128)      \
    CPU_HOST_NOKERN(name, complex128, bfloat16, complex128)   \
    CPU_HOST_NOKERN(name, complex128, float16, complex128)    \
    CPU_HOST_NOKERN(name, complex128, float32, complex128)    \
    CPU_HOST_NOKERN(name, complex128, float64, complex128)    \
    CPU_HOST_NOKERN(name, complex128, complex32, complex128)  \
    CPU_HOST_NOKERN(name, complex128, complex64, complex128)  \
    CPU_HOST_NOKERN(name, complex128, complex128, complex128)

#define CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN(name) \
    CPU_HOST_NOIMPL(name, uint8, uint8, float16)              \
    CPU_HOST_BINARY(name, uint8, uint16, float32)             \
    CPU_HOST_BINARY(name, uint8, uint32, float64)             \
    CPU_HOST_NOKERN(name, uint8, uint64, uint64)              \
    CPU_HOST_NOIMPL(name, uint8, int8, float16)               \
    CPU_HOST_BINARY(name, uint8, int16, float32)              \
    CPU_HOST_BINARY(name, uint8, int32, float64)              \
    CPU_HOST_NOKERN(name, uint8, int64, int64)                \
    CPU_HOST_BINARY(name, uint8, bfloat16, bfloat16)          \
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
    CPU_HOST_BINARY(name, uint16, bfloat16, float32)          \
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
    CPU_HOST_BINARY(name, uint32, bfloat16, float64)          \
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
    CPU_HOST_BINARY(name, int8, bfloat16, bfloat16)           \
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
    CPU_HOST_BINARY(name, int16, bfloat16, float32)           \
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
    CPU_HOST_BINARY(name, int32, bfloat16, float64)           \
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
    CPU_HOST_BINARY(name, bfloat16, uint8, bfloat16)          \
    CPU_HOST_BINARY(name, bfloat16, uint16, float32)          \
    CPU_HOST_BINARY(name, bfloat16, uint32, float64)          \
    CPU_HOST_BINARY(name, bfloat16, int8, bfloat16)           \
    CPU_HOST_BINARY(name, bfloat16, int16, float32)           \
    CPU_HOST_BINARY(name, bfloat16, int32, float64)           \
    CPU_HOST_BINARY(name, bfloat16, bfloat16, bfloat16)       \
    CPU_HOST_NOIMPL(name, bfloat16, float16, float32)         \
    CPU_HOST_BINARY(name, bfloat16, float32, float32)         \
    CPU_HOST_BINARY(name, bfloat16, float64, float64)         \
    CPU_HOST_NOIMPL(name, bfloat16, complex32, complex64)     \
    CPU_HOST_BINARY(name, bfloat16, complex64, complex64)     \
    CPU_HOST_BINARY(name, bfloat16, complex128, complex128)   \
                                                              \
    CPU_HOST_NOIMPL(name, float16, uint8, float16)            \
    CPU_HOST_NOIMPL(name, float16, uint16, float32)           \
    CPU_HOST_NOIMPL(name, float16, uint32, float64)           \
    CPU_HOST_NOIMPL(name, float16, int8, float16)             \
    CPU_HOST_NOIMPL(name, float16, int16, float32)            \
    CPU_HOST_NOIMPL(name, float16, int32, float64)            \
    CPU_HOST_NOIMPL(name, float16, bfloat16, float32)         \
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
    CPU_HOST_BINARY(name, float32, bfloat16, float32)         \
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
    CPU_HOST_BINARY(name, float64, bfloat16, float64)         \
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
    CPU_HOST_NOIMPL(name, complex32, bfloat16, complex64)     \
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
    CPU_HOST_BINARY(name, complex64, bfloat16, complex64)     \
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
    CPU_HOST_BINARY(name, complex128, bfloat16, complex128)   \
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
    CPU_HOST_BINARY_INIT(name, uint8, bfloat16, bfloat16),          \
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
    CPU_HOST_BINARY_INIT(name, uint16, bfloat16, float32),          \
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
    CPU_HOST_BINARY_INIT(name, uint32, bfloat16, float64),          \
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
    CPU_HOST_BINARY_INIT(name, int8, bfloat16, bfloat16),           \
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
    CPU_HOST_BINARY_INIT(name, int16, bfloat16, float32),           \
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
    CPU_HOST_BINARY_INIT(name, int32, bfloat16, float64),           \
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
    CPU_HOST_BINARY_INIT(name, bfloat16, uint8, bfloat16),          \
    CPU_HOST_BINARY_INIT(name, bfloat16, uint16, float32),          \
    CPU_HOST_BINARY_INIT(name, bfloat16, uint32, float64),          \
    CPU_HOST_BINARY_INIT(name, bfloat16, int8, bfloat16),           \
    CPU_HOST_BINARY_INIT(name, bfloat16, int16, float32),           \
    CPU_HOST_BINARY_INIT(name, bfloat16, int32, float64),           \
    CPU_HOST_BINARY_INIT(name, bfloat16, bfloat16, bfloat16),       \
    CPU_HOST_BINARY_INIT(name, bfloat16, float16, float32),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, float32, float32),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, float64, float64),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex32, complex64),     \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex64, complex64),     \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex128, complex128),   \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float16, uint8, float16),            \
    CPU_HOST_BINARY_INIT(name, float16, uint16, float32),           \
    CPU_HOST_BINARY_INIT(name, float16, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float16, int8, float16),             \
    CPU_HOST_BINARY_INIT(name, float16, int16, float32),            \
    CPU_HOST_BINARY_INIT(name, float16, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float16, bfloat16, float32),         \
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
    CPU_HOST_BINARY_INIT(name, float32, bfloat16, float32),         \
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
    CPU_HOST_BINARY_INIT(name, float64, bfloat16, float64),         \
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
    CPU_HOST_BINARY_INIT(name, complex32, bfloat16, complex64),     \
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
    CPU_HOST_BINARY_INIT(name, complex64, bfloat16, complex64),     \
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
    CPU_HOST_BINARY_INIT(name, complex128, bfloat16, complex128),   \
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
    CPU_HOST_BINARY_INIT(name, uint8, bfloat16, bfloat16),          \
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
    CPU_HOST_BINARY_INIT(name, uint16, bfloat16, float32),          \
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
    CPU_HOST_BINARY_INIT(name, uint32, bfloat16, float64),          \
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
    CPU_HOST_BINARY_INIT(name, int8, bfloat16, bfloat16),           \
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
    CPU_HOST_BINARY_INIT(name, int16, bfloat16, float32),           \
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
    CPU_HOST_BINARY_INIT(name, int32, bfloat16, float64),           \
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
    CPU_HOST_BINARY_INIT(name, bfloat16, uint8, bfloat16),          \
    CPU_HOST_BINARY_INIT(name, bfloat16, uint16, float32),          \
    CPU_HOST_BINARY_INIT(name, bfloat16, uint32, float64),          \
    CPU_HOST_BINARY_INIT(name, bfloat16, int8, bfloat16),           \
    CPU_HOST_BINARY_INIT(name, bfloat16, int16, float32),           \
    CPU_HOST_BINARY_INIT(name, bfloat16, int32, float64),           \
    CPU_HOST_BINARY_INIT(name, bfloat16, bfloat16, bfloat16),       \
    CPU_HOST_BINARY_INIT(name, bfloat16, float16, float32),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, float32, float32),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, float64, float64),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex32, complex64),     \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex64, complex64),     \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex128, complex128),   \
                                                                    \
    CPU_HOST_BINARY_INIT(name, float16, uint8, float16),            \
    CPU_HOST_BINARY_INIT(name, float16, uint16, float32),           \
    CPU_HOST_BINARY_INIT(name, float16, uint32, float64),           \
    CPU_HOST_BINARY_INIT(name, float16, int8, float16),             \
    CPU_HOST_BINARY_INIT(name, float16, int16, float32),            \
    CPU_HOST_BINARY_INIT(name, float16, int32, float64),            \
    CPU_HOST_BINARY_INIT(name, float16, bfloat16, float32),         \
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
    CPU_HOST_BINARY_INIT(name, float32, bfloat16, float32),         \
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
    CPU_HOST_BINARY_INIT(name, float64, bfloat16, float64),         \
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
    CPU_HOST_BINARY_INIT(name, complex32, bfloat16, complex64),     \
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
    CPU_HOST_BINARY_INIT(name, complex64, bfloat16, complex64),     \
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
    CPU_HOST_BINARY_INIT(name, complex128, bfloat16, complex128),   \
    CPU_HOST_BINARY_INIT(name, complex128, float16, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, float32, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, float64, complex128),    \
    CPU_HOST_BINARY_INIT(name, complex128, complex32, complex128),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex64, complex128),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex128, complex128)


CPU_HOST_ALL_ARITHMETIC(add)
CPU_HOST_ALL_ARITHMETIC(subtract)
CPU_HOST_ALL_ARITHMETIC(multiply)
CPU_HOST_ALL_ARITHMETIC_NO_COMPLEX(floor_divide)
CPU_HOST_ALL_ARITHMETIC_NO_COMPLEX(remainder)
CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN(divide)
CPU_HOST_ALL_ARITHMETIC(power)


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
    CPU_HOST_BINARY(name, uint8, bfloat16, bool)        \
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
    CPU_HOST_BINARY(name, uint16, bfloat16, bool)       \
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
    CPU_HOST_BINARY(name, uint32, bfloat16, bool)       \
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
    CPU_HOST_BINARY(name, int8, bfloat16, bool)         \
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
    CPU_HOST_BINARY(name, int16, bfloat16, bool)        \
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
    CPU_HOST_BINARY(name, int32, bfloat16, bool)        \
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
    CPU_HOST_BINARY(name, bfloat16, uint8, bool)        \
    CPU_HOST_BINARY(name, bfloat16, uint16, bool)       \
    CPU_HOST_BINARY(name, bfloat16, uint32, bool)       \
    CPU_HOST_BINARY(name, bfloat16, int8, bool)         \
    CPU_HOST_BINARY(name, bfloat16, int16, bool)        \
    CPU_HOST_BINARY(name, bfloat16, int32, bool)        \
    CPU_HOST_BINARY(name, bfloat16, bfloat16, bool)     \
    CPU_HOST_NOIMPL(name, bfloat16, float16, bool)      \
    CPU_HOST_BINARY(name, bfloat16, float32, bool)      \
    CPU_HOST_BINARY(name, bfloat16, float64, bool)      \
    CPU_HOST_NOIMPL(name, bfloat16, complex32, bool)    \
    CPU_HOST_BINARY(name, bfloat16, complex64, bool)    \
    CPU_HOST_BINARY(name, bfloat16, complex128, bool)   \
                                                        \
    CPU_HOST_NOIMPL(name, float16, uint8, bool)         \
    CPU_HOST_NOIMPL(name, float16, uint16, bool)        \
    CPU_HOST_NOIMPL(name, float16, uint32, bool)        \
    CPU_HOST_NOIMPL(name, float16, int8, bool)          \
    CPU_HOST_NOIMPL(name, float16, int16, bool)         \
    CPU_HOST_NOIMPL(name, float16, int32, bool)         \
    CPU_HOST_NOIMPL(name, float16, bfloat16, bool)      \
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
    CPU_HOST_BINARY(name, float32, bfloat16, bool)      \
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
    CPU_HOST_BINARY(name, float64, bfloat16, bool)      \
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
    CPU_HOST_NOIMPL(name, complex32, bfloat16, bool)    \
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
    CPU_HOST_BINARY(name, complex64, bfloat16, bool)    \
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
    CPU_HOST_BINARY(name, complex128, bfloat16, bool)   \
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
    CPU_HOST_BINARY_INIT(name, uint8, bfloat16, bool),        \
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
    CPU_HOST_BINARY_INIT(name, uint16, bfloat16, bool),       \
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
    CPU_HOST_BINARY_INIT(name, uint32, bfloat16, bool),       \
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
    CPU_HOST_BINARY_INIT(name, int8, bfloat16, bool),         \
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
    CPU_HOST_BINARY_INIT(name, int16, bfloat16, bool),        \
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
    CPU_HOST_BINARY_INIT(name, int32, bfloat16, bool),        \
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
    CPU_HOST_BINARY_INIT(name, bfloat16, uint8, bool),        \
    CPU_HOST_BINARY_INIT(name, bfloat16, uint16, bool),       \
    CPU_HOST_BINARY_INIT(name, bfloat16, uint32, bool),       \
    CPU_HOST_BINARY_INIT(name, bfloat16, int8, bool),         \
    CPU_HOST_BINARY_INIT(name, bfloat16, int16, bool),        \
    CPU_HOST_BINARY_INIT(name, bfloat16, int32, bool),        \
    CPU_HOST_BINARY_INIT(name, bfloat16, bfloat16, bool),     \
    CPU_HOST_BINARY_INIT(name, bfloat16, float16, bool),      \
    CPU_HOST_BINARY_INIT(name, bfloat16, float32, bool),      \
    CPU_HOST_BINARY_INIT(name, bfloat16, float64, bool),      \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex32, bool),    \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex64, bool),    \
    CPU_HOST_BINARY_INIT(name, bfloat16, complex128, bool),   \
                                                              \
    CPU_HOST_BINARY_INIT(name, float16, uint8, bool),         \
    CPU_HOST_BINARY_INIT(name, float16, uint16, bool),        \
    CPU_HOST_BINARY_INIT(name, float16, uint32, bool),        \
    CPU_HOST_BINARY_INIT(name, float16, int8, bool),          \
    CPU_HOST_BINARY_INIT(name, float16, int16, bool),         \
    CPU_HOST_BINARY_INIT(name, float16, int32, bool),         \
    CPU_HOST_BINARY_INIT(name, float16, bfloat16, bool),      \
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
    CPU_HOST_BINARY_INIT(name, float32, bfloat16, bool),      \
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
    CPU_HOST_BINARY_INIT(name, float64, bfloat16, bool),      \
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
    CPU_HOST_BINARY_INIT(name, complex32, bfloat16, bool),    \
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
    CPU_HOST_BINARY_INIT(name, complex64, bfloat16, bool),    \
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
    CPU_HOST_BINARY_INIT(name, complex128, bfloat16, bool),   \
    CPU_HOST_BINARY_INIT(name, complex128, float16, bool),    \
    CPU_HOST_BINARY_INIT(name, complex128, float32, bool),    \
    CPU_HOST_BINARY_INIT(name, complex128, float64, bool),    \
    CPU_HOST_BINARY_INIT(name, complex128, complex32, bool),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex64, bool),  \
    CPU_HOST_BINARY_INIT(name, complex128, complex128, bool)

#define CPU_HOST_ALL_EQUALN_INIT(name) \
    CPU_HOST_EQUALN_INIT(name, uint8, uint8, bool),           \
    CPU_HOST_EQUALN_INIT(name, uint8, uint16, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint8, uint32, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint8, uint64, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint8, int8, bool),            \
    CPU_HOST_EQUALN_INIT(name, uint8, int16, bool),           \
    CPU_HOST_EQUALN_INIT(name, uint8, int32, bool),           \
    CPU_HOST_EQUALN_INIT(name, uint8, int64, bool),           \
    CPU_HOST_EQUALN_INIT(name, uint8, bfloat16, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint8, float16, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint8, float32, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint8, float64, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint8, complex32, bool),       \
    CPU_HOST_EQUALN_INIT(name, uint8, complex64, bool),       \
    CPU_HOST_EQUALN_INIT(name, uint8, complex128, bool),      \
                                                              \
    CPU_HOST_EQUALN_INIT(name, uint16, uint8, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint16, uint16, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint16, uint32, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint16, uint64, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint16, int8, bool),           \
    CPU_HOST_EQUALN_INIT(name, uint16, int16, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint16, int32, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint16, int64, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint16, bfloat16, bool),       \
    CPU_HOST_EQUALN_INIT(name, uint16, float16, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint16, float32, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint16, float64, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint16, complex32, bool),      \
    CPU_HOST_EQUALN_INIT(name, uint16, complex64, bool),      \
    CPU_HOST_EQUALN_INIT(name, uint16, complex128, bool),     \
                                                              \
    CPU_HOST_EQUALN_INIT(name, uint32, uint8, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint32, uint16, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint32, uint32, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint32, uint64, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint32, int8, bool),           \
    CPU_HOST_EQUALN_INIT(name, uint32, int16, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint32, int32, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint32, int64, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint32, bfloat16, bool),       \
    CPU_HOST_EQUALN_INIT(name, uint32, float16, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint32, float32, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint32, float64, bool),        \
    CPU_HOST_EQUALN_INIT(name, uint32, complex32, bool),      \
    CPU_HOST_EQUALN_INIT(name, uint32, complex64, bool),      \
    CPU_HOST_EQUALN_INIT(name, uint32, complex128, bool),     \
                                                              \
    CPU_HOST_EQUALN_INIT(name, uint64, uint8, bool),          \
    CPU_HOST_EQUALN_INIT(name, uint64, uint16, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint64, uint32, bool),         \
    CPU_HOST_EQUALN_INIT(name, uint64, uint64, bool),         \
                                                              \
    CPU_HOST_EQUALN_INIT(name, int8, uint8, bool),            \
    CPU_HOST_EQUALN_INIT(name, int8, uint16, bool),           \
    CPU_HOST_EQUALN_INIT(name, int8, uint32, bool),           \
    CPU_HOST_EQUALN_INIT(name, int8, int8, bool),             \
    CPU_HOST_EQUALN_INIT(name, int8, int16, bool),            \
    CPU_HOST_EQUALN_INIT(name, int8, int32, bool),            \
    CPU_HOST_EQUALN_INIT(name, int8, int64, bool),            \
    CPU_HOST_EQUALN_INIT(name, int8, bfloat16, bool),         \
    CPU_HOST_EQUALN_INIT(name, int8, float16, bool),          \
    CPU_HOST_EQUALN_INIT(name, int8, float32, bool),          \
    CPU_HOST_EQUALN_INIT(name, int8, float64, bool),          \
    CPU_HOST_EQUALN_INIT(name, int8, complex32, bool),        \
    CPU_HOST_EQUALN_INIT(name, int8, complex64, bool),        \
    CPU_HOST_EQUALN_INIT(name, int8, complex128, bool),       \
                                                              \
    CPU_HOST_EQUALN_INIT(name, int16, uint8, bool),           \
    CPU_HOST_EQUALN_INIT(name, int16, uint16, bool),          \
    CPU_HOST_EQUALN_INIT(name, int16, uint32, bool),          \
    CPU_HOST_EQUALN_INIT(name, int16, int8, bool),            \
    CPU_HOST_EQUALN_INIT(name, int16, int16, bool),           \
    CPU_HOST_EQUALN_INIT(name, int16, int32, bool),           \
    CPU_HOST_EQUALN_INIT(name, int16, int64, bool),           \
    CPU_HOST_EQUALN_INIT(name, int16, bfloat16, bool),        \
    CPU_HOST_EQUALN_INIT(name, int16, float16, bool),         \
    CPU_HOST_EQUALN_INIT(name, int16, float32, bool),         \
    CPU_HOST_EQUALN_INIT(name, int16, float64, bool),         \
    CPU_HOST_EQUALN_INIT(name, int16, complex32, bool),       \
    CPU_HOST_EQUALN_INIT(name, int16, complex64, bool),       \
    CPU_HOST_EQUALN_INIT(name, int16, complex128, bool),      \
                                                              \
    CPU_HOST_EQUALN_INIT(name, int32, uint8, bool),           \
    CPU_HOST_EQUALN_INIT(name, int32, uint16, bool),          \
    CPU_HOST_EQUALN_INIT(name, int32, uint32, bool),          \
    CPU_HOST_EQUALN_INIT(name, int32, int8, bool),            \
    CPU_HOST_EQUALN_INIT(name, int32, int16, bool),           \
    CPU_HOST_EQUALN_INIT(name, int32, int32, bool),           \
    CPU_HOST_EQUALN_INIT(name, int32, int64, bool),           \
    CPU_HOST_EQUALN_INIT(name, int32, bfloat16, bool),        \
    CPU_HOST_EQUALN_INIT(name, int32, float16, bool),         \
    CPU_HOST_EQUALN_INIT(name, int32, float32, bool),         \
    CPU_HOST_EQUALN_INIT(name, int32, float64, bool),         \
    CPU_HOST_EQUALN_INIT(name, int32, complex32, bool),       \
    CPU_HOST_EQUALN_INIT(name, int32, complex64, bool),       \
    CPU_HOST_EQUALN_INIT(name, int32, complex128, bool),      \
                                                              \
    CPU_HOST_EQUALN_INIT(name, int64, uint8, bool),           \
    CPU_HOST_EQUALN_INIT(name, int64, uint16, bool),          \
    CPU_HOST_EQUALN_INIT(name, int64, uint32, bool),          \
    CPU_HOST_EQUALN_INIT(name, int64, int8, bool),            \
    CPU_HOST_EQUALN_INIT(name, int64, int16, bool),           \
    CPU_HOST_EQUALN_INIT(name, int64, int32, bool),           \
    CPU_HOST_EQUALN_INIT(name, int64, int64, bool),           \
                                                              \
    CPU_HOST_EQUALN_INIT(name, bfloat16, uint8, bool),        \
    CPU_HOST_EQUALN_INIT(name, bfloat16, uint16, bool),       \
    CPU_HOST_EQUALN_INIT(name, bfloat16, uint32, bool),       \
    CPU_HOST_EQUALN_INIT(name, bfloat16, int8, bool),         \
    CPU_HOST_EQUALN_INIT(name, bfloat16, int16, bool),        \
    CPU_HOST_EQUALN_INIT(name, bfloat16, int32, bool),        \
    CPU_HOST_EQUALN_INIT(name, bfloat16, bfloat16, bool),     \
    CPU_HOST_EQUALN_INIT(name, bfloat16, float16, bool),      \
    CPU_HOST_EQUALN_INIT(name, bfloat16, float32, bool),      \
    CPU_HOST_EQUALN_INIT(name, bfloat16, float64, bool),      \
    CPU_HOST_EQUALN_INIT(name, bfloat16, complex32, bool),    \
    CPU_HOST_EQUALN_INIT(name, bfloat16, complex64, bool),    \
    CPU_HOST_EQUALN_INIT(name, bfloat16, complex128, bool),   \
                                                              \
    CPU_HOST_EQUALN_INIT(name, float16, uint8, bool),         \
    CPU_HOST_EQUALN_INIT(name, float16, uint16, bool),        \
    CPU_HOST_EQUALN_INIT(name, float16, uint32, bool),        \
    CPU_HOST_EQUALN_INIT(name, float16, int8, bool),          \
    CPU_HOST_EQUALN_INIT(name, float16, int16, bool),         \
    CPU_HOST_EQUALN_INIT(name, float16, int32, bool),         \
    CPU_HOST_EQUALN_INIT(name, float16, bfloat16, bool),      \
    CPU_HOST_EQUALN_INIT(name, float16, float16, bool),       \
    CPU_HOST_EQUALN_INIT(name, float16, float32, bool),       \
    CPU_HOST_EQUALN_INIT(name, float16, float64, bool),       \
    CPU_HOST_EQUALN_INIT(name, float16, complex32, bool),     \
    CPU_HOST_EQUALN_INIT(name, float16, complex64, bool),     \
    CPU_HOST_EQUALN_INIT(name, float16, complex128, bool),    \
                                                              \
    CPU_HOST_EQUALN_INIT(name, float32, uint8, bool),         \
    CPU_HOST_EQUALN_INIT(name, float32, uint16, bool),        \
    CPU_HOST_EQUALN_INIT(name, float32, uint32, bool),        \
    CPU_HOST_EQUALN_INIT(name, float32, int8, bool),          \
    CPU_HOST_EQUALN_INIT(name, float32, int16, bool),         \
    CPU_HOST_EQUALN_INIT(name, float32, int32, bool),         \
    CPU_HOST_EQUALN_INIT(name, float32, bfloat16, bool),      \
    CPU_HOST_EQUALN_INIT(name, float32, float16, bool),       \
    CPU_HOST_EQUALN_INIT(name, float32, float32, bool),       \
    CPU_HOST_EQUALN_INIT(name, float32, float64, bool),       \
    CPU_HOST_EQUALN_INIT(name, float32, complex32, bool),     \
    CPU_HOST_EQUALN_INIT(name, float32, complex64, bool),     \
    CPU_HOST_EQUALN_INIT(name, float32, complex128, bool),    \
                                                              \
    CPU_HOST_EQUALN_INIT(name, float64, uint8, bool),         \
    CPU_HOST_EQUALN_INIT(name, float64, uint16, bool),        \
    CPU_HOST_EQUALN_INIT(name, float64, uint32, bool),        \
    CPU_HOST_EQUALN_INIT(name, float64, int8, bool),          \
    CPU_HOST_EQUALN_INIT(name, float64, int16, bool),         \
    CPU_HOST_EQUALN_INIT(name, float64, int32, bool),         \
    CPU_HOST_EQUALN_INIT(name, float64, bfloat16, bool),      \
    CPU_HOST_EQUALN_INIT(name, float64, float16, bool),       \
    CPU_HOST_EQUALN_INIT(name, float64, float32, bool),       \
    CPU_HOST_EQUALN_INIT(name, float64, float64, bool),       \
    CPU_HOST_EQUALN_INIT(name, float64, complex32, bool),     \
    CPU_HOST_EQUALN_INIT(name, float64, complex64, bool),     \
    CPU_HOST_EQUALN_INIT(name, float64, complex128, bool),    \
                                                              \
    CPU_HOST_EQUALN_INIT(name, complex32, uint8, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex32, uint16, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex32, uint32, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex32, int8, bool),        \
    CPU_HOST_EQUALN_INIT(name, complex32, int16, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex32, int32, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex32, bfloat16, bool),    \
    CPU_HOST_EQUALN_INIT(name, complex32, float16, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex32, float32, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex32, float64, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex32, complex32, bool),   \
    CPU_HOST_EQUALN_INIT(name, complex32, complex64, bool),   \
    CPU_HOST_EQUALN_INIT(name, complex32, complex128, bool),  \
                                                              \
    CPU_HOST_EQUALN_INIT(name, complex64, uint8, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex64, uint16, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex64, uint32, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex64, int8, bool),        \
    CPU_HOST_EQUALN_INIT(name, complex64, int16, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex64, int32, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex64, bfloat16, bool),    \
    CPU_HOST_EQUALN_INIT(name, complex64, float16, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex64, float32, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex64, float64, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex64, complex32, bool),   \
    CPU_HOST_EQUALN_INIT(name, complex64, complex64, bool),   \
    CPU_HOST_EQUALN_INIT(name, complex64, complex128, bool),  \
                                                              \
    CPU_HOST_EQUALN_INIT(name, complex128, uint8, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex128, uint16, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex128, uint32, bool),     \
    CPU_HOST_EQUALN_INIT(name, complex128, int8, bool),       \
    CPU_HOST_EQUALN_INIT(name, complex128, int16, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex128, int32, bool),      \
    CPU_HOST_EQUALN_INIT(name, complex128, bfloat16, bool),   \
    CPU_HOST_EQUALN_INIT(name, complex128, float16, bool),    \
    CPU_HOST_EQUALN_INIT(name, complex128, float32, bool),    \
    CPU_HOST_EQUALN_INIT(name, complex128, float64, bool),    \
    CPU_HOST_EQUALN_INIT(name, complex128, complex32, bool),  \
    CPU_HOST_EQUALN_INIT(name, complex128, complex64, bool),  \
    CPU_HOST_EQUALN_INIT(name, complex128, complex128, bool)


CPU_HOST_ALL_COMPARISON(less)
CPU_HOST_ALL_COMPARISON(less_equal)
CPU_HOST_ALL_COMPARISON(greater_equal)
CPU_HOST_ALL_COMPARISON(greater)
CPU_HOST_ALL_COMPARISON(equal)
CPU_HOST_ALL_COMPARISON(not_equal)
CPU_HOST_ALL_COMPARISON(equaln)


static const gm_kernel_init_t binary_kernels[] = {
  CPU_HOST_ALL_ARITHMETIC_INIT(add),
  CPU_HOST_ALL_ARITHMETIC_INIT(subtract),
  CPU_HOST_ALL_ARITHMETIC_INIT(multiply),
  CPU_HOST_ALL_ARITHMETIC_INIT(floor_divide),
  CPU_HOST_ALL_ARITHMETIC_INIT(remainder),
  CPU_HOST_ALL_ARITHMETIC_FLOAT_RETURN_INIT(divide),
  CPU_HOST_ALL_ARITHMETIC_INIT(power),
  CPU_HOST_ALL_COMPARISON_INIT(less),
  CPU_HOST_ALL_COMPARISON_INIT(less_equal),
  CPU_HOST_ALL_COMPARISON_INIT(greater_equal),
  CPU_HOST_ALL_COMPARISON_INIT(greater),
  CPU_HOST_ALL_COMPARISON_INIT(equal),
  CPU_HOST_ALL_COMPARISON_INIT(not_equal),
  CPU_HOST_ALL_EQUALN_INIT(equaln),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Bitwise                                 */
/*****************************************************************************/

#define CPU_HOST_ALL_BITWISE(name) \
    CPU_HOST_BINARY(name, bool, bool, bool)       \
    CPU_HOST_BINARY(name, bool, uint8, uint8)     \
    CPU_HOST_BINARY(name, bool, uint16, uint16)   \
    CPU_HOST_BINARY(name, bool, uint32, uint32)   \
    CPU_HOST_BINARY(name, bool, uint64, uint64)   \
    CPU_HOST_BINARY(name, bool, int8, int8)       \
    CPU_HOST_BINARY(name, bool, int16, int16)     \
    CPU_HOST_BINARY(name, bool, int32, int32)     \
    CPU_HOST_BINARY(name, bool, int64, int64)     \
                                                  \
    CPU_HOST_BINARY(name, uint8, bool, uint8)     \
    CPU_HOST_BINARY(name, uint8, uint8, uint8)    \
    CPU_HOST_BINARY(name, uint8, uint16, uint16)  \
    CPU_HOST_BINARY(name, uint8, uint32, uint32)  \
    CPU_HOST_BINARY(name, uint8, uint64, uint64)  \
    CPU_HOST_BINARY(name, uint8, int8, int16)     \
    CPU_HOST_BINARY(name, uint8, int16, int16)    \
    CPU_HOST_BINARY(name, uint8, int32, int32)    \
    CPU_HOST_BINARY(name, uint8, int64, int64)    \
                                                  \
    CPU_HOST_BINARY(name, uint16, bool, uint16)   \
    CPU_HOST_BINARY(name, uint16, uint8, uint16)  \
    CPU_HOST_BINARY(name, uint16, uint16, uint16) \
    CPU_HOST_BINARY(name, uint16, uint32, uint32) \
    CPU_HOST_BINARY(name, uint16, uint64, uint64) \
    CPU_HOST_BINARY(name, uint16, int8, int32)    \
    CPU_HOST_BINARY(name, uint16, int16, int32)   \
    CPU_HOST_BINARY(name, uint16, int32, int32)   \
    CPU_HOST_BINARY(name, uint16, int64, int64)   \
                                                  \
    CPU_HOST_BINARY(name, uint32, bool, uint32)   \
    CPU_HOST_BINARY(name, uint32, uint8, uint32)  \
    CPU_HOST_BINARY(name, uint32, uint16, uint32) \
    CPU_HOST_BINARY(name, uint32, uint32, uint32) \
    CPU_HOST_BINARY(name, uint32, uint64, uint64) \
    CPU_HOST_BINARY(name, uint32, int8, int64)    \
    CPU_HOST_BINARY(name, uint32, int16, int64)   \
    CPU_HOST_BINARY(name, uint32, int32, int64)   \
    CPU_HOST_BINARY(name, uint32, int64, int64)   \
                                                  \
    CPU_HOST_BINARY(name, uint64, bool, uint64)   \
    CPU_HOST_BINARY(name, uint64, uint8, uint64)  \
    CPU_HOST_BINARY(name, uint64, uint16, uint64) \
    CPU_HOST_BINARY(name, uint64, uint32, uint64) \
    CPU_HOST_BINARY(name, uint64, uint64, uint64) \
                                                  \
    CPU_HOST_BINARY(name, int8, bool, int8)       \
    CPU_HOST_BINARY(name, int8, uint8, int16)     \
    CPU_HOST_BINARY(name, int8, uint16, int32)    \
    CPU_HOST_BINARY(name, int8, uint32, int64)    \
    CPU_HOST_BINARY(name, int8, int8, int8)       \
    CPU_HOST_BINARY(name, int8, int16, int16)     \
    CPU_HOST_BINARY(name, int8, int32, int32)     \
    CPU_HOST_BINARY(name, int8, int64, int64)     \
                                                  \
    CPU_HOST_BINARY(name, int16, bool, int16)     \
    CPU_HOST_BINARY(name, int16, uint8, int16)    \
    CPU_HOST_BINARY(name, int16, uint16, int32)   \
    CPU_HOST_BINARY(name, int16, uint32, int64)   \
    CPU_HOST_BINARY(name, int16, int8, int16)     \
    CPU_HOST_BINARY(name, int16, int16, int16)    \
    CPU_HOST_BINARY(name, int16, int32, int32)    \
    CPU_HOST_BINARY(name, int16, int64, int64)    \
                                                  \
    CPU_HOST_BINARY(name, int32, bool, int32)     \
    CPU_HOST_BINARY(name, int32, uint8, int32)    \
    CPU_HOST_BINARY(name, int32, uint16, int32)   \
    CPU_HOST_BINARY(name, int32, uint32, int64)   \
    CPU_HOST_BINARY(name, int32, int8, int32)     \
    CPU_HOST_BINARY(name, int32, int16, int32)    \
    CPU_HOST_BINARY(name, int32, int32, int32)    \
    CPU_HOST_BINARY(name, int32, int64, int64)    \
                                                  \
    CPU_HOST_BINARY(name, int64, bool, int64)     \
    CPU_HOST_BINARY(name, int64, uint8, int64)    \
    CPU_HOST_BINARY(name, int64, uint16, int64)   \
    CPU_HOST_BINARY(name, int64, uint32, int64)   \
    CPU_HOST_BINARY(name, int64, int8, int64)     \
    CPU_HOST_BINARY(name, int64, int16, int64)    \
    CPU_HOST_BINARY(name, int64, int32, int64)    \
    CPU_HOST_BINARY(name, int64, int64, int64)

#define CPU_HOST_ALL_BITWISE_INIT(name) \
    CPU_HOST_BINARY_INIT(name, bool, bool, bool),       \
    CPU_HOST_BINARY_INIT(name, bool, uint8, uint8),     \
    CPU_HOST_BINARY_INIT(name, bool, uint16, uint16),   \
    CPU_HOST_BINARY_INIT(name, bool, uint32, uint32),   \
    CPU_HOST_BINARY_INIT(name, bool, uint64, uint64),   \
    CPU_HOST_BINARY_INIT(name, bool, int8, int8),       \
    CPU_HOST_BINARY_INIT(name, bool, int16, int16),     \
    CPU_HOST_BINARY_INIT(name, bool, int32, int32),     \
    CPU_HOST_BINARY_INIT(name, bool, int64, int64),     \
                                                        \
    CPU_HOST_BINARY_INIT(name, uint8, bool, uint8),     \
    CPU_HOST_BINARY_INIT(name, uint8, uint8, uint8),    \
    CPU_HOST_BINARY_INIT(name, uint8, uint16, uint16),  \
    CPU_HOST_BINARY_INIT(name, uint8, uint32, uint32),  \
    CPU_HOST_BINARY_INIT(name, uint8, uint64, uint64),  \
    CPU_HOST_BINARY_INIT(name, uint8, int8, int16),     \
    CPU_HOST_BINARY_INIT(name, uint8, int16, int16),    \
    CPU_HOST_BINARY_INIT(name, uint8, int32, int32),    \
    CPU_HOST_BINARY_INIT(name, uint8, int64, int64),    \
                                                        \
    CPU_HOST_BINARY_INIT(name, uint16, bool, uint16),   \
    CPU_HOST_BINARY_INIT(name, uint16, uint8, uint16),  \
    CPU_HOST_BINARY_INIT(name, uint16, uint16, uint16), \
    CPU_HOST_BINARY_INIT(name, uint16, uint32, uint32), \
    CPU_HOST_BINARY_INIT(name, uint16, uint64, uint64), \
    CPU_HOST_BINARY_INIT(name, uint16, int8, int32),    \
    CPU_HOST_BINARY_INIT(name, uint16, int16, int32),   \
    CPU_HOST_BINARY_INIT(name, uint16, int32, int32),   \
    CPU_HOST_BINARY_INIT(name, uint16, int64, int64),   \
                                                        \
    CPU_HOST_BINARY_INIT(name, uint32, bool, uint32),   \
    CPU_HOST_BINARY_INIT(name, uint32, uint8, uint32),  \
    CPU_HOST_BINARY_INIT(name, uint32, uint16, uint32), \
    CPU_HOST_BINARY_INIT(name, uint32, uint32, uint32), \
    CPU_HOST_BINARY_INIT(name, uint32, uint64, uint64), \
    CPU_HOST_BINARY_INIT(name, uint32, int8, int64),    \
    CPU_HOST_BINARY_INIT(name, uint32, int16, int64),   \
    CPU_HOST_BINARY_INIT(name, uint32, int32, int64),   \
    CPU_HOST_BINARY_INIT(name, uint32, int64, int64),   \
                                                        \
    CPU_HOST_BINARY_INIT(name, uint64, bool, uint64),   \
    CPU_HOST_BINARY_INIT(name, uint64, uint8, uint64),  \
    CPU_HOST_BINARY_INIT(name, uint64, uint16, uint64), \
    CPU_HOST_BINARY_INIT(name, uint64, uint32, uint64), \
    CPU_HOST_BINARY_INIT(name, uint64, uint64, uint64), \
                                                        \
    CPU_HOST_BINARY_INIT(name, int8, bool, int8),       \
    CPU_HOST_BINARY_INIT(name, int8, uint8, int16),     \
    CPU_HOST_BINARY_INIT(name, int8, uint16, int32),    \
    CPU_HOST_BINARY_INIT(name, int8, uint32, int64),    \
    CPU_HOST_BINARY_INIT(name, int8, int8, int8),       \
    CPU_HOST_BINARY_INIT(name, int8, int16, int16),     \
    CPU_HOST_BINARY_INIT(name, int8, int32, int32),     \
    CPU_HOST_BINARY_INIT(name, int8, int64, int64),     \
                                                        \
    CPU_HOST_BINARY_INIT(name, int16, bool, int16),     \
    CPU_HOST_BINARY_INIT(name, int16, uint8, int16),    \
    CPU_HOST_BINARY_INIT(name, int16, uint16, int32),   \
    CPU_HOST_BINARY_INIT(name, int16, uint32, int64),   \
    CPU_HOST_BINARY_INIT(name, int16, int8, int16),     \
    CPU_HOST_BINARY_INIT(name, int16, int16, int16),    \
    CPU_HOST_BINARY_INIT(name, int16, int32, int32),    \
    CPU_HOST_BINARY_INIT(name, int16, int64, int64),    \
                                                        \
    CPU_HOST_BINARY_INIT(name, int32, bool, int32),     \
    CPU_HOST_BINARY_INIT(name, int32, uint8, int32),    \
    CPU_HOST_BINARY_INIT(name, int32, uint16, int32),   \
    CPU_HOST_BINARY_INIT(name, int32, uint32, int64),   \
    CPU_HOST_BINARY_INIT(name, int32, int8, int32),     \
    CPU_HOST_BINARY_INIT(name, int32, int16, int32),    \
    CPU_HOST_BINARY_INIT(name, int32, int32, int32),    \
    CPU_HOST_BINARY_INIT(name, int32, int64, int64),    \
                                                        \
    CPU_HOST_BINARY_INIT(name, int64, bool, int64),     \
    CPU_HOST_BINARY_INIT(name, int64, uint8, int64),    \
    CPU_HOST_BINARY_INIT(name, int64, uint16, int64),   \
    CPU_HOST_BINARY_INIT(name, int64, uint32, int64),   \
    CPU_HOST_BINARY_INIT(name, int64, int8, int64),     \
    CPU_HOST_BINARY_INIT(name, int64, int16, int64),    \
    CPU_HOST_BINARY_INIT(name, int64, int32, int64),    \
    CPU_HOST_BINARY_INIT(name, int64, int64, int64)


CPU_HOST_ALL_BITWISE(bitwise_and)
CPU_HOST_ALL_BITWISE(bitwise_or)
CPU_HOST_ALL_BITWISE(bitwise_xor)


static const gm_kernel_init_t bitwise_kernels[] = {
  CPU_HOST_ALL_BITWISE_INIT(bitwise_and),
  CPU_HOST_ALL_BITWISE_INIT(bitwise_or),
  CPU_HOST_ALL_BITWISE_INIT(bitwise_xor),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                               Two return values                          */
/****************************************************************************/

#define CPU_HOST_BINARY_MV(name, t0, t1, t2, t3) \
static int                                                                                   \
gm_cpu_host_fixed_1D_C_##name##_##t0##_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                            \
    const char *a0 = apply_index(&stack[0]);                                                 \
    const char *a1 = apply_index(&stack[1]);                                                 \
    char *a2 = apply_index(&stack[2]);                                                       \
    char *a3 = apply_index(&stack[3]);                                                       \
    int64_t N = xnd_fixed_shape(&stack[0]);                                                  \
    (void)ctx;                                                                               \
                                                                                             \
    gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2##_##t3(                             \
        a0, a1, a2, a3, N);                                                                  \
                                                                                             \
    return 0;                                                                                \
}                                                                                            \
                                                                                             \
static int                                                                                   \
gm_cpu_host_0D_##name##_##t0##_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx)         \
{                                                                                            \
    const char *a0 = stack[0].ptr;                                                           \
    const char *a1 = stack[1].ptr;                                                           \
    char *a2 = stack[2].ptr;                                                                 \
    char *a3 = stack[3].ptr;                                                                 \
    (void)ctx;                                                                               \
                                                                                             \
    gm_cpu_device_0D_##name##_##t0##_##t1##_##t2##_##t3(a0, a1, a2, a3);                     \
                                                                                             \
    return 0;                                                                                \
}

#define CPU_HOST_BINARY_MV_INIT(func, t0, t1, t2, t3) \
  { .name = STRINGIZE(func),                                           \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> "      \
           "... * " STRINGIZE(t2) ", ... * " STRINGIZE(t3),            \
    .OptC = gm_cpu_host_fixed_1D_C_##func##_##t0##_##t1##_##t2##_##t3, \
    .Xnd = gm_cpu_host_0D_##func##_##t0##_##t1##_##t2##_##t3 }

#define CPU_HOST_ALL_BINARY_MV(name) \
    CPU_HOST_BINARY_MV(name, uint8, uint8, uint8, uint8)             \
    CPU_HOST_BINARY_MV(name, uint16, uint16, uint16, uint16)         \
    CPU_HOST_BINARY_MV(name, uint32, uint32, uint32, uint32)         \
    CPU_HOST_BINARY_MV(name, uint64, uint64, uint64, uint64)         \
    CPU_HOST_BINARY_MV(name, int8, int8, int8, int8)                 \
    CPU_HOST_BINARY_MV(name, int16, int16, int16, int16)             \
    CPU_HOST_BINARY_MV(name, int32, int32, int32, int32)             \
    CPU_HOST_BINARY_MV(name, int64, int64, int64, int64)             \
    CPU_HOST_BINARY_MV(name, bfloat16, bfloat16, bfloat16, bfloat16) \
    CPU_HOST_BINARY_MV(name, float32, float32, float32, float32)     \
    CPU_HOST_BINARY_MV(name, float64, float64, float64, float64)

#define CPU_HOST_ALL_BINARY_MV_INIT(name) \
    CPU_HOST_BINARY_MV_INIT(name, uint8, uint8, uint8, uint8),             \
    CPU_HOST_BINARY_MV_INIT(name, uint16, uint16, uint16, uint16),         \
    CPU_HOST_BINARY_MV_INIT(name, uint32, uint32, uint32, uint32),         \
    CPU_HOST_BINARY_MV_INIT(name, uint64, uint64, uint64, uint64),         \
    CPU_HOST_BINARY_MV_INIT(name, int8, int8, int8, int8),                 \
    CPU_HOST_BINARY_MV_INIT(name, int16, int16, int16, int16),             \
    CPU_HOST_BINARY_MV_INIT(name, int32, int32, int32, int32),             \
    CPU_HOST_BINARY_MV_INIT(name, int64, int64, int64, int64),             \
    CPU_HOST_BINARY_MV_INIT(name, bfloat16, bfloat16, bfloat16, bfloat16), \
    CPU_HOST_BINARY_MV_INIT(name, float32, float32, float32, float32),     \
    CPU_HOST_BINARY_MV_INIT(name, float64, float64, float64, float64)

CPU_HOST_ALL_BINARY_MV(divmod)


static const gm_kernel_init_t binary_mv_kernels[] = {
  CPU_HOST_ALL_BINARY_MV_INIT(divmod),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

typedef _Bool bool;

static const gm_kernel_set_t *
binary_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                 const int64_t li[], int nin, int nout, bool check_broadcast,
                 ndt_context_t *ctx)
{
    return cpu_binary_typecheck(binary_kernel_location, spec, f, types, li,
                                nin, nout, check_broadcast, ctx);
}

static const gm_kernel_set_t *
bitwise_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                  const int64_t li[], int nin, int nout, bool check_broadcast,
                  ndt_context_t *ctx)
{
    return cpu_binary_typecheck(bitwise_kernel_location, spec, f, types, li,
                                nin, nout, check_broadcast, ctx);
}


int
gm_init_cpu_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = binary_kernels; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &binary_typecheck) < 0) {
             return -1;
        }
    }

    for (k = bitwise_kernels; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &bitwise_typecheck) < 0) {
             return -1;
        }
    }

    for (k = binary_mv_kernels; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
             return -1;
        }
    }

    return 0;
}

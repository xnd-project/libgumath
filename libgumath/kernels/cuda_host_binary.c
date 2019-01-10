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
#include "cuda_device_binary.h"


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
        case Uint16: return 4;
        case Uint32: return 8;
        case Uint64: return 12;

        case Int8: return 16;
        case Int16: return 20;
        case Int32: return 24;
        case Int64: return 28;

        case BFloat16: return 32;
        case Float16: return 36;
        case Float32: return 40;
        case Float64: return 44;

        case Complex32: return 48;
        case Complex64: return 52;
        case Complex128: return 56;

        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Uint8: return 60;
        case Uint16: return 64;
        case Uint32: return 68;
        case Uint64: return 72;

        case Int8: return 76;
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
    case Uint32: {
        switch (t1->tag) {
        case Uint8: return 120;
        case Uint16: return 124;
        case Uint32: return 128;
        case Uint64: return 132;

        case Int8: return 136;
        case Int16: return 140;
        case Int32: return 144;
        case Int64: return 148;

        case BFloat16: return 152;
        case Float16: return 156;
        case Float32: return 160;
        case Float64: return 164;

        case Complex32: return 168;
        case Complex64: return 172;
        case Complex128: return 176;

        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Uint8: return 180;
        case Uint16: return 184;
        case Uint32: return 188;
        case Uint64: return 192;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Uint8: return 196;
        case Uint16: return 200;
        case Uint32: return 204;

        case Int8: return 208;
        case Int16: return 212;
        case Int32: return 216;
        case Int64: return 220;

        case BFloat16: return 224;
        case Float16: return 228;
        case Float32: return 232;
        case Float64: return 236;

        case Complex32: return 240;
        case Complex64: return 244;
        case Complex128: return 248;

        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Uint8: return 252;
        case Uint16: return 256;
        case Uint32: return 260;

        case Int8: return 264;
        case Int16: return 268;
        case Int32: return 272;
        case Int64: return 276;

        case BFloat16: return 280;
        case Float16: return 284;
        case Float32: return 288;
        case Float64: return 292;

        case Complex32: return 296;
        case Complex64: return 300;
        case Complex128: return 304;

        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Uint8: return 308;
        case Uint16: return 312;
        case Uint32: return 316;

        case Int8: return 320;
        case Int16: return 324;
        case Int32: return 328;
        case Int64: return 332;

        case BFloat16: return 336;
        case Float16: return 340;
        case Float32: return 344;
        case Float64: return 348;

        case Complex32: return 352;
        case Complex64: return 356;
        case Complex128: return 360;

        default: goto invalid_combination;
        }
    }
    case Int64: {
        switch (t1->tag) {
        case Uint8: return 364;
        case Uint16: return 368;
        case Uint32: return 372;

        case Int8: return 376;
        case Int16: return 380;
        case Int32: return 384;
        case Int64: return 388;

        default: goto invalid_combination;
        }
    }

    case BFloat16: {
        switch (t1->tag) {
        case Uint8: return 392;
        case Uint16: return 396;
        case Uint32: return 400;

        case Int8: return 404;
        case Int16: return 408;
        case Int32: return 412;

        case BFloat16: return 416;
        case Float16: return 420;
        case Float32: return 424;
        case Float64: return 428;

        case Complex32: return 432;
        case Complex64: return 436;
        case Complex128: return 440;

        default: goto invalid_combination;
        }
    }

    case Float16: {
        switch (t1->tag) {
        case Uint8: return 444;
        case Uint16: return 448;
        case Uint32: return 452;

        case Int8: return 456;
        case Int16: return 460;
        case Int32: return 464;

        case BFloat16: return 468;
        case Float16: return 472;
        case Float32: return 476;
        case Float64: return 480;

        case Complex32: return 484;
        case Complex64: return 488;
        case Complex128: return 492;

        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (t1->tag) {
        case Uint8: return 496;
        case Uint16: return 500;
        case Uint32: return 504;

        case Int8: return 508;
        case Int16: return 512;
        case Int32: return 516;

        case BFloat16: return 520;
        case Float16: return 524;
        case Float32: return 528;
        case Float64: return 532;

        case Complex32: return 536;
        case Complex64: return 540;
        case Complex128: return 544;

        default: goto invalid_combination;
        }
    }

    case Float64: {
        switch (t1->tag) {
        case Uint8: return 548;
        case Uint16: return 552;
        case Uint32: return 556;

        case Int8: return 560;
        case Int16: return 564;
        case Int32: return 568;

        case BFloat16: return 572;
        case Float16: return 576;
        case Float32: return 580;
        case Float64: return 584;

        case Complex32: return 588;
        case Complex64: return 592;
        case Complex128: return 596;

        default: goto invalid_combination;
        }
    }

    case Complex32: {
        switch (t1->tag) {
        case Uint8: return 600;
        case Uint16: return 604;
        case Uint32: return 608;

        case Int8: return 612;
        case Int16: return 616;
        case Int32: return 620;

        case BFloat16: return 624;
        case Float16: return 628;
        case Float32: return 632;
        case Float64: return 636;

        case Complex32: return 640;
        case Complex64: return 644;
        case Complex128: return 648;

        default: goto invalid_combination;
        }
    }

    case Complex64: {
        switch (t1->tag) {
        case Uint8: return 652;
        case Uint16: return 656;
        case Uint32: return 660;

        case Int8: return 664;
        case Int16: return 668;
        case Int32: return 672;

        case BFloat16: return 676;
        case Float16: return 680;
        case Float32: return 684;
        case Float64: return 688;

        case Complex32: return 692;
        case Complex64: return 696;
        case Complex128: return 700;

        default: goto invalid_combination;
        }
    }

    case Complex128: {
        switch (t1->tag) {
        case Uint8: return 704;
        case Uint16: return 708;
        case Uint32: return 712;

        case Int8: return 716;
        case Int16: return 720;
        case Int32: return 724;

        case BFloat16: return 728;
        case Float16: return 732;
        case Float32: return 736;
        case Float64: return 740;

        case Complex32: return 744;
        case Complex64: return 748;
        case Complex128: return 752;

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

        case Uint8: return 4;
        case Uint16: return 8;
        case Uint32: return 12;
        case Uint64: return 16;

        case Int8: return 20;
        case Int16: return 24;
        case Int32: return 28;
        case Int64: return 32;

        default: goto invalid_combination;
        }
    }

    case Uint8: {
        switch (t1->tag) {
        case Bool: return 36;

        case Uint8: return 40;
        case Uint16: return 44;
        case Uint32: return 48;
        case Uint64: return 52;

        case Int8: return 56;
        case Int16: return 60;
        case Int32: return 64;
        case Int64: return 68;

        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Bool: return 72;

        case Int8: return 76;
        case Int16: return 80;
        case Int32: return 84;
        case Int64: return 88;

        case Uint8: return 92;
        case Uint16: return 96;
        case Uint32: return 100;
        case Uint64: return 104;

        default: goto invalid_combination;
        }
    }
    case Uint32: {
        switch (t1->tag) {
        case Bool: return 108;

        case Uint8: return 112;
        case Uint16: return 116;
        case Uint32: return 120;
        case Uint64: return 124;

        case Int8: return 128;
        case Int16: return 132;
        case Int32: return 136;
        case Int64: return 140;

        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Bool: return 144;

        case Uint8: return 148;
        case Uint16: return 152;
        case Uint32: return 156;
        case Uint64: return 160;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Bool: return 164;

        case Uint8: return 168;
        case Uint16: return 172;
        case Uint32: return 176;

        case Int8: return 180;
        case Int16: return 184;
        case Int32: return 188;
        case Int64: return 192;

        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Bool: return 196;

        case Uint8: return 200;
        case Uint16: return 204;
        case Uint32: return 208;

        case Int8: return 212;
        case Int16: return 216;
        case Int32: return 220;
        case Int64: return 224;

        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Bool: return 228;

        case Uint8: return 232;
        case Uint16: return 236;
        case Uint32: return 240;

        case Int8: return 244;
        case Int16: return 248;
        case Int32: return 252;
        case Int64: return 256;

        default: goto invalid_combination;
        }
    }

    case Int64: {
        switch (t1->tag) {
        case Bool: return 260;

        case Uint8: return 264;
        case Uint16: return 268;
        case Uint32: return 272;

        case Int8: return 276;
        case Int16: return 280;
        case Int32: return 284;
        case Int64: return 288;

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

static inline enum cuda_binary
get_step_tag(const ndt_t *t0, const ndt_t *t1)
{
    int64_t s0 = t0->Concrete.FixedDim.step;
    int64_t s1 = t1->Concrete.FixedDim.step;

    if (s0 == 0) {
        return s1 == 0 ? ZeroStepIn0In1 : ZeroStepIn0;
    }

    return s1 == 0 ? ZeroStepIn1 : ZeroStepNone;
}

#define CUDA_HOST_BINARY(name, t0, t1, t2) \
static int                                                                        \
gm_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx)      \
{                                                                                 \
    const char *in0 = apply_index(&stack[0]);                                     \
    const char *in1 = apply_index(&stack[1]);                                     \
    char *out = apply_index(&stack[2]);                                           \
    int64_t N = xnd_fixed_shape(&stack[0]);                                       \
    enum cuda_binary tag = get_step_tag(stack[0].type, stack[1].type);            \
    (void)ctx;                                                                    \
                                                                                  \
    gm_cuda_device_fixed_1D_C_##name##_##t0##_##t1##_##t2(in0, in1, out, N, tag); \
                                                                                  \
    if (ndt_is_optional(ndt_dtype(stack[2].type))) {                              \
        binary_update_bitmap1D(stack);                                            \
    }                                                                             \
                                                                                  \
    return 0;                                                                     \
}

#define CUDA_HOST_NOIMPL(name, t0, t1, t2) \
static int                                                                   \
gm_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                            \
    (void)stack;                                                             \
                                                                             \
    ndt_err_format(ctx, NDT_NotImplementedError,                             \
        "implementation for " STRINGIZE(name) " : "                          \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2)                \
        " currently requires double rounding");                              \
                                                                             \
    return -1;                                                               \
}

#define CUDA_HOST_NOKERN(name, t0, t1, t2) \
static int                                                                   \
gm_fixed_1D_C_##name##_##t0##_##t1##_##t2(xnd_t stack[], ndt_context_t *ctx) \
{                                                                            \
    (void)stack;                                                             \
                                                                             \
    ndt_err_format(ctx, NDT_TypeError,                                       \
        "no kernel for " STRINGIZE(name) " : "                               \
        STRINGIZE(t0) ", " STRINGIZE(t1) " -> " STRINGIZE(t2));              \
                                                                             \
    return -1;                                                               \
}

#define CUDA_HOST_BINARY_INIT(func, t0, t1, t2) \
  { .name = STRINGIZE(func),                                                              \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * " STRINGIZE(t2),    \
    .Opt = gm_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                     \
    .C = NULL },                                                                          \
                                                                                          \
   { .name = STRINGIZE(func),                                                             \
    .sig = "... * ?" STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),  \
    .Opt = gm_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                     \
    .C = NULL },                                                                          \
                                                                                          \
   { .name = STRINGIZE(func),                                                             \
    .sig = "... * " STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2),  \
    .Opt = gm_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                     \
    .C = NULL },                                                                          \
                                                                                          \
   { .name = STRINGIZE(func),                                                             \
    .sig = "... * ?" STRINGIZE(t0) ", ... * ?" STRINGIZE(t1) " -> ... * ?" STRINGIZE(t2), \
    .Opt = gm_fixed_1D_C_##func##_##t0##_##t1##_##t2,                                     \
    .C = NULL }


/*****************************************************************************/
/*                                 Arithmetic                                */
/*****************************************************************************/

#define CUDA_HOST_ALL_ARITHMETIC(name) \
    CUDA_HOST_BINARY(name, uint8, uint8, uint8)                \
    CUDA_HOST_BINARY(name, uint8, uint16, uint16)              \
    CUDA_HOST_BINARY(name, uint8, uint32, uint32)              \
    CUDA_HOST_BINARY(name, uint8, uint64, uint64)              \
    CUDA_HOST_BINARY(name, uint8, int8, int16)                 \
    CUDA_HOST_BINARY(name, uint8, int16, int16)                \
    CUDA_HOST_BINARY(name, uint8, int32, int32)                \
    CUDA_HOST_BINARY(name, uint8, int64, int64)                \
    CUDA_HOST_BINARY(name, uint8, bfloat16, bfloat16)          \
    CUDA_HOST_BINARY(name, uint8, float16, float16)            \
    CUDA_HOST_BINARY(name, uint8, float32, float32)            \
    CUDA_HOST_BINARY(name, uint8, float64, float64)            \
    CUDA_HOST_NOIMPL(name, uint8, complex32, complex32)        \
    CUDA_HOST_BINARY(name, uint8, complex64, complex64)        \
    CUDA_HOST_BINARY(name, uint8, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, uint16, uint8, uint16)              \
    CUDA_HOST_BINARY(name, uint16, uint16, uint16)             \
    CUDA_HOST_BINARY(name, uint16, uint32, uint32)             \
    CUDA_HOST_BINARY(name, uint16, uint64, uint64)             \
    CUDA_HOST_BINARY(name, uint16, int8, int32)                \
    CUDA_HOST_BINARY(name, uint16, int16, int32)               \
    CUDA_HOST_BINARY(name, uint16, int32, int32)               \
    CUDA_HOST_BINARY(name, uint16, int64, int64)               \
    CUDA_HOST_BINARY(name, uint16, bfloat16, float32)          \
    CUDA_HOST_BINARY(name, uint16, float16, float32)           \
    CUDA_HOST_BINARY(name, uint16, float32, float32)           \
    CUDA_HOST_BINARY(name, uint16, float64, float64)           \
    CUDA_HOST_NOIMPL(name, uint16, complex32, complex64)       \
    CUDA_HOST_BINARY(name, uint16, complex64, complex64)       \
    CUDA_HOST_BINARY(name, uint16, complex128, complex128)     \
                                                               \
    CUDA_HOST_BINARY(name, uint32, uint8, uint32)              \
    CUDA_HOST_BINARY(name, uint32, uint16, uint32)             \
    CUDA_HOST_BINARY(name, uint32, uint32, uint32)             \
    CUDA_HOST_BINARY(name, uint32, uint64, uint64)             \
    CUDA_HOST_BINARY(name, uint32, int8, int64)                \
    CUDA_HOST_BINARY(name, uint32, int16, int64)               \
    CUDA_HOST_BINARY(name, uint32, int32, int64)               \
    CUDA_HOST_BINARY(name, uint32, int64, int64)               \
    CUDA_HOST_BINARY(name, uint32, bfloat16, float64)          \
    CUDA_HOST_BINARY(name, uint32, float16, float64)           \
    CUDA_HOST_BINARY(name, uint32, float32, float64)           \
    CUDA_HOST_BINARY(name, uint32, float64, float64)           \
    CUDA_HOST_NOIMPL(name, uint32, complex32, complex128)      \
    CUDA_HOST_BINARY(name, uint32, complex64, complex128)      \
    CUDA_HOST_BINARY(name, uint32, complex128, complex128)     \
                                                               \
    CUDA_HOST_BINARY(name, uint64, uint8, uint64)              \
    CUDA_HOST_BINARY(name, uint64, uint16, uint64)             \
    CUDA_HOST_BINARY(name, uint64, uint32, uint64)             \
    CUDA_HOST_BINARY(name, uint64, uint64, uint64)             \
                                                               \
    CUDA_HOST_BINARY(name, int8, uint8, int16)                 \
    CUDA_HOST_BINARY(name, int8, uint16, int32)                \
    CUDA_HOST_BINARY(name, int8, uint32, int64)                \
    CUDA_HOST_BINARY(name, int8, int8, int8)                   \
    CUDA_HOST_BINARY(name, int8, int16, int16)                 \
    CUDA_HOST_BINARY(name, int8, int32, int32)                 \
    CUDA_HOST_BINARY(name, int8, int64, int64)                 \
    CUDA_HOST_BINARY(name, int8, bfloat16, bfloat16)           \
    CUDA_HOST_BINARY(name, int8, float16, float16)             \
    CUDA_HOST_BINARY(name, int8, float32, float32)             \
    CUDA_HOST_BINARY(name, int8, float64, float64)             \
    CUDA_HOST_NOIMPL(name, int8, complex32, complex32)         \
    CUDA_HOST_BINARY(name, int8, complex64, complex64)         \
    CUDA_HOST_BINARY(name, int8, complex128, complex128)       \
                                                               \
    CUDA_HOST_BINARY(name, int16, uint8, int16)                \
    CUDA_HOST_BINARY(name, int16, uint16, int32)               \
    CUDA_HOST_BINARY(name, int16, uint32, int64)               \
    CUDA_HOST_BINARY(name, int16, int8, int16)                 \
    CUDA_HOST_BINARY(name, int16, int16, int16)                \
    CUDA_HOST_BINARY(name, int16, int32, int32)                \
    CUDA_HOST_BINARY(name, int16, int64, int64)                \
    CUDA_HOST_BINARY(name, int16, bfloat16, float32)           \
    CUDA_HOST_BINARY(name, int16, float16, float32)            \
    CUDA_HOST_BINARY(name, int16, float32, float32)            \
    CUDA_HOST_BINARY(name, int16, float64, float64)            \
    CUDA_HOST_NOIMPL(name, int16, complex32, complex64)        \
    CUDA_HOST_BINARY(name, int16, complex64, complex64)        \
    CUDA_HOST_BINARY(name, int16, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, int32, uint8, int32)                \
    CUDA_HOST_BINARY(name, int32, uint16, int32)               \
    CUDA_HOST_BINARY(name, int32, uint32, int64)               \
    CUDA_HOST_BINARY(name, int32, int8, int32)                 \
    CUDA_HOST_BINARY(name, int32, int16, int32)                \
    CUDA_HOST_BINARY(name, int32, int32, int32)                \
    CUDA_HOST_BINARY(name, int32, int64, int64)                \
    CUDA_HOST_BINARY(name, int32, bfloat16, float64)           \
    CUDA_HOST_BINARY(name, int32, float16, float64)            \
    CUDA_HOST_BINARY(name, int32, float32, float64)            \
    CUDA_HOST_BINARY(name, int32, float64, float64)            \
    CUDA_HOST_NOIMPL(name, int32, complex32, complex128)       \
    CUDA_HOST_BINARY(name, int32, complex64, complex128)       \
    CUDA_HOST_BINARY(name, int32, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, int64, uint8, int64)                \
    CUDA_HOST_BINARY(name, int64, uint16, int64)               \
    CUDA_HOST_BINARY(name, int64, uint32, int64)               \
    CUDA_HOST_BINARY(name, int64, int8, int64)                 \
    CUDA_HOST_BINARY(name, int64, int16, int64)                \
    CUDA_HOST_BINARY(name, int64, int32, int64)                \
    CUDA_HOST_BINARY(name, int64, int64, int64)                \
                                                               \
    CUDA_HOST_BINARY(name, bfloat16, uint8, bfloat16)          \
    CUDA_HOST_BINARY(name, bfloat16, uint16, float32)          \
    CUDA_HOST_BINARY(name, bfloat16, uint32, float64)          \
    CUDA_HOST_BINARY(name, bfloat16, int8, bfloat16)           \
    CUDA_HOST_BINARY(name, bfloat16, int16, float32)           \
    CUDA_HOST_BINARY(name, bfloat16, int32, float64)           \
    CUDA_HOST_BINARY(name, bfloat16, bfloat16, bfloat16)       \
    CUDA_HOST_BINARY(name, bfloat16, float16, float32)         \
    CUDA_HOST_BINARY(name, bfloat16, float32, float32)         \
    CUDA_HOST_BINARY(name, bfloat16, float64, float64)         \
    CUDA_HOST_NOIMPL(name, bfloat16, complex32, complex64)     \
    CUDA_HOST_BINARY(name, bfloat16, complex64, complex64)     \
    CUDA_HOST_BINARY(name, bfloat16, complex128, complex128)   \
                                                               \
    CUDA_HOST_BINARY(name, float16, uint8, float16)            \
    CUDA_HOST_BINARY(name, float16, uint16, float32)           \
    CUDA_HOST_BINARY(name, float16, uint32, float64)           \
    CUDA_HOST_BINARY(name, float16, int8, float16)             \
    CUDA_HOST_BINARY(name, float16, int16, float32)            \
    CUDA_HOST_BINARY(name, float16, int32, float64)            \
    CUDA_HOST_BINARY(name, float16, bfloat16, float32)         \
    CUDA_HOST_BINARY(name, float16, float16, float16)          \
    CUDA_HOST_BINARY(name, float16, float32, float32)          \
    CUDA_HOST_BINARY(name, float16, float64, float64)          \
    CUDA_HOST_NOIMPL(name, float16, complex32, complex32)      \
    CUDA_HOST_BINARY(name, float16, complex64, complex64)      \
    CUDA_HOST_BINARY(name, float16, complex128, complex128)    \
                                                               \
    CUDA_HOST_BINARY(name, float32, uint8, float32)            \
    CUDA_HOST_BINARY(name, float32, uint16, float32)           \
    CUDA_HOST_BINARY(name, float32, uint32, float64)           \
    CUDA_HOST_BINARY(name, float32, int8, float32)             \
    CUDA_HOST_BINARY(name, float32, int16, float32)            \
    CUDA_HOST_BINARY(name, float32, int32, float64)            \
    CUDA_HOST_BINARY(name, float32, bfloat16, float32)         \
    CUDA_HOST_BINARY(name, float32, float16, float32)          \
    CUDA_HOST_BINARY(name, float32, float32, float32)          \
    CUDA_HOST_BINARY(name, float32, float64, float64)          \
    CUDA_HOST_NOIMPL(name, float32, complex32, complex64)      \
    CUDA_HOST_BINARY(name, float32, complex64, complex64)      \
    CUDA_HOST_BINARY(name, float32, complex128, complex128)    \
                                                               \
    CUDA_HOST_BINARY(name, float64, uint8, float64)            \
    CUDA_HOST_BINARY(name, float64, uint16, float64)           \
    CUDA_HOST_BINARY(name, float64, uint32, float64)           \
    CUDA_HOST_BINARY(name, float64, int8, float64)             \
    CUDA_HOST_BINARY(name, float64, int16, float64)            \
    CUDA_HOST_BINARY(name, float64, int32, float64)            \
    CUDA_HOST_BINARY(name, float64, bfloat16, float64)         \
    CUDA_HOST_BINARY(name, float64, float16, float64)          \
    CUDA_HOST_BINARY(name, float64, float32, float64)          \
    CUDA_HOST_BINARY(name, float64, float64, float64)          \
    CUDA_HOST_NOIMPL(name, float64, complex32, complex128)     \
    CUDA_HOST_BINARY(name, float64, complex64, complex128)     \
    CUDA_HOST_BINARY(name, float64, complex128, complex128)    \
                                                               \
    CUDA_HOST_NOIMPL(name, complex32, uint8, complex32)        \
    CUDA_HOST_NOIMPL(name, complex32, uint16, complex64)       \
    CUDA_HOST_NOIMPL(name, complex32, uint32, complex128)      \
    CUDA_HOST_NOIMPL(name, complex32, int8, complex32)         \
    CUDA_HOST_NOIMPL(name, complex32, int16, complex64)        \
    CUDA_HOST_NOIMPL(name, complex32, int32, complex128)       \
    CUDA_HOST_NOIMPL(name, complex32, bfloat16, complex64)     \
    CUDA_HOST_NOIMPL(name, complex32, float16, complex32)      \
    CUDA_HOST_NOIMPL(name, complex32, float32, complex64)      \
    CUDA_HOST_NOIMPL(name, complex32, float64, complex128)     \
    CUDA_HOST_NOIMPL(name, complex32, complex32, complex32)    \
    CUDA_HOST_NOIMPL(name, complex32, complex64, complex64)    \
    CUDA_HOST_NOIMPL(name, complex32, complex128, complex128)  \
                                                               \
    CUDA_HOST_BINARY(name, complex64, uint8, complex64)        \
    CUDA_HOST_BINARY(name, complex64, uint16, complex64)       \
    CUDA_HOST_BINARY(name, complex64, uint32, complex128)      \
    CUDA_HOST_BINARY(name, complex64, int8, complex64)         \
    CUDA_HOST_BINARY(name, complex64, int16, complex64)        \
    CUDA_HOST_BINARY(name, complex64, int32, complex128)       \
    CUDA_HOST_BINARY(name, complex64, bfloat16, complex64)     \
    CUDA_HOST_BINARY(name, complex64, float16, complex64)      \
    CUDA_HOST_BINARY(name, complex64, float32, complex64)      \
    CUDA_HOST_BINARY(name, complex64, float64, complex128)     \
    CUDA_HOST_NOIMPL(name, complex64, complex32, complex64)    \
    CUDA_HOST_BINARY(name, complex64, complex64, complex64)    \
    CUDA_HOST_BINARY(name, complex64, complex128, complex128)  \
                                                               \
    CUDA_HOST_BINARY(name, complex128, uint8, complex128)      \
    CUDA_HOST_BINARY(name, complex128, uint16, complex128)     \
    CUDA_HOST_BINARY(name, complex128, uint32, complex128)     \
    CUDA_HOST_BINARY(name, complex128, int8, complex128)       \
    CUDA_HOST_BINARY(name, complex128, int16, complex128)      \
    CUDA_HOST_BINARY(name, complex128, int32, complex128)      \
    CUDA_HOST_BINARY(name, complex128, bfloat16, complex128)   \
    CUDA_HOST_BINARY(name, complex128, float16, complex128)    \
    CUDA_HOST_BINARY(name, complex128, float32, complex128)    \
    CUDA_HOST_BINARY(name, complex128, float64, complex128)    \
    CUDA_HOST_NOIMPL(name, complex128, complex32, complex128)  \
    CUDA_HOST_BINARY(name, complex128, complex64, complex128)  \
    CUDA_HOST_BINARY(name, complex128, complex128, complex128)

#define CUDA_HOST_ALL_ARITHMETIC_NO_COMPLEX(name) \
    CUDA_HOST_BINARY(name, uint8, uint8, uint8)                \
    CUDA_HOST_BINARY(name, uint8, uint16, uint16)              \
    CUDA_HOST_BINARY(name, uint8, uint32, uint32)              \
    CUDA_HOST_BINARY(name, uint8, uint64, uint64)              \
    CUDA_HOST_BINARY(name, uint8, int8, int16)                 \
    CUDA_HOST_BINARY(name, uint8, int16, int16)                \
    CUDA_HOST_BINARY(name, uint8, int32, int32)                \
    CUDA_HOST_BINARY(name, uint8, int64, int64)                \
    CUDA_HOST_BINARY(name, uint8, bfloat16, bfloat16)          \
    CUDA_HOST_NOIMPL(name, uint8, float16, float16)            \
    CUDA_HOST_BINARY(name, uint8, float32, float32)            \
    CUDA_HOST_BINARY(name, uint8, float64, float64)            \
    CUDA_HOST_NOKERN(name, uint8, complex32, complex32)        \
    CUDA_HOST_NOKERN(name, uint8, complex64, complex64)        \
    CUDA_HOST_NOKERN(name, uint8, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, uint16, uint8, uint16)              \
    CUDA_HOST_BINARY(name, uint16, uint16, uint16)             \
    CUDA_HOST_BINARY(name, uint16, uint32, uint32)             \
    CUDA_HOST_BINARY(name, uint16, uint64, uint64)             \
    CUDA_HOST_BINARY(name, uint16, int8, int32)                \
    CUDA_HOST_BINARY(name, uint16, int16, int32)               \
    CUDA_HOST_BINARY(name, uint16, int32, int32)               \
    CUDA_HOST_BINARY(name, uint16, int64, int64)               \
    CUDA_HOST_BINARY(name, uint16, bfloat16, float32)          \
    CUDA_HOST_BINARY(name, uint16, float16, float32)           \
    CUDA_HOST_BINARY(name, uint16, float32, float32)           \
    CUDA_HOST_BINARY(name, uint16, float64, float64)           \
    CUDA_HOST_NOKERN(name, uint16, complex32, complex64)       \
    CUDA_HOST_NOKERN(name, uint16, complex64, complex64)       \
    CUDA_HOST_NOKERN(name, uint16, complex128, complex128)     \
                                                               \
    CUDA_HOST_BINARY(name, uint32, uint8, uint32)              \
    CUDA_HOST_BINARY(name, uint32, uint16, uint32)             \
    CUDA_HOST_BINARY(name, uint32, uint32, uint32)             \
    CUDA_HOST_BINARY(name, uint32, uint64, uint64)             \
    CUDA_HOST_BINARY(name, uint32, int8, int64)                \
    CUDA_HOST_BINARY(name, uint32, int16, int64)               \
    CUDA_HOST_BINARY(name, uint32, int32, int64)               \
    CUDA_HOST_BINARY(name, uint32, int64, int64)               \
    CUDA_HOST_BINARY(name, uint32, bfloat16, float64)          \
    CUDA_HOST_BINARY(name, uint32, float16, float64)           \
    CUDA_HOST_BINARY(name, uint32, float32, float64)           \
    CUDA_HOST_BINARY(name, uint32, float64, float64)           \
    CUDA_HOST_NOKERN(name, uint32, complex32, complex128)      \
    CUDA_HOST_NOKERN(name, uint32, complex64, complex128)      \
    CUDA_HOST_NOKERN(name, uint32, complex128, complex128)     \
                                                               \
    CUDA_HOST_BINARY(name, uint64, uint8, uint64)              \
    CUDA_HOST_BINARY(name, uint64, uint16, uint64)             \
    CUDA_HOST_BINARY(name, uint64, uint32, uint64)             \
    CUDA_HOST_BINARY(name, uint64, uint64, uint64)             \
                                                               \
    CUDA_HOST_BINARY(name, int8, uint8, int16)                 \
    CUDA_HOST_BINARY(name, int8, uint16, int32)                \
    CUDA_HOST_BINARY(name, int8, uint32, int64)                \
    CUDA_HOST_BINARY(name, int8, int8, int8)                   \
    CUDA_HOST_BINARY(name, int8, int16, int16)                 \
    CUDA_HOST_BINARY(name, int8, int32, int32)                 \
    CUDA_HOST_BINARY(name, int8, int64, int64)                 \
    CUDA_HOST_BINARY(name, int8, bfloat16, bfloat16)           \
    CUDA_HOST_NOIMPL(name, int8, float16, float16)             \
    CUDA_HOST_BINARY(name, int8, float32, float32)             \
    CUDA_HOST_BINARY(name, int8, float64, float64)             \
    CUDA_HOST_NOKERN(name, int8, complex32, complex32)         \
    CUDA_HOST_NOKERN(name, int8, complex64, complex64)         \
    CUDA_HOST_NOKERN(name, int8, complex128, complex128)       \
                                                               \
    CUDA_HOST_BINARY(name, int16, uint8, int16)                \
    CUDA_HOST_BINARY(name, int16, uint16, int32)               \
    CUDA_HOST_BINARY(name, int16, uint32, int64)               \
    CUDA_HOST_BINARY(name, int16, int8, int16)                 \
    CUDA_HOST_BINARY(name, int16, int16, int16)                \
    CUDA_HOST_BINARY(name, int16, int32, int32)                \
    CUDA_HOST_BINARY(name, int16, int64, int64)                \
    CUDA_HOST_BINARY(name, int16, bfloat16, float32)           \
    CUDA_HOST_BINARY(name, int16, float16, float32)            \
    CUDA_HOST_BINARY(name, int16, float32, float32)            \
    CUDA_HOST_BINARY(name, int16, float64, float64)            \
    CUDA_HOST_NOKERN(name, int16, complex32, complex64)        \
    CUDA_HOST_NOKERN(name, int16, complex64, complex64)        \
    CUDA_HOST_NOKERN(name, int16, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, int32, uint8, int32)                \
    CUDA_HOST_BINARY(name, int32, uint16, int32)               \
    CUDA_HOST_BINARY(name, int32, uint32, int64)               \
    CUDA_HOST_BINARY(name, int32, int8, int32)                 \
    CUDA_HOST_BINARY(name, int32, int16, int32)                \
    CUDA_HOST_BINARY(name, int32, int32, int32)                \
    CUDA_HOST_BINARY(name, int32, int64, int64)                \
    CUDA_HOST_BINARY(name, int32, bfloat16, float64)           \
    CUDA_HOST_BINARY(name, int32, float16, float64)            \
    CUDA_HOST_BINARY(name, int32, float32, float64)            \
    CUDA_HOST_BINARY(name, int32, float64, float64)            \
    CUDA_HOST_NOKERN(name, int32, complex32, complex128)       \
    CUDA_HOST_NOKERN(name, int32, complex64, complex128)       \
    CUDA_HOST_NOKERN(name, int32, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, int64, uint8, int64)                \
    CUDA_HOST_BINARY(name, int64, uint16, int64)               \
    CUDA_HOST_BINARY(name, int64, uint32, int64)               \
    CUDA_HOST_BINARY(name, int64, int8, int64)                 \
    CUDA_HOST_BINARY(name, int64, int16, int64)                \
    CUDA_HOST_BINARY(name, int64, int32, int64)                \
    CUDA_HOST_BINARY(name, int64, int64, int64)                \
                                                               \
    CUDA_HOST_BINARY(name, bfloat16, uint8, bfloat16)          \
    CUDA_HOST_BINARY(name, bfloat16, uint16, float32)          \
    CUDA_HOST_BINARY(name, bfloat16, uint32, float64)          \
    CUDA_HOST_BINARY(name, bfloat16, int8, bfloat16)           \
    CUDA_HOST_BINARY(name, bfloat16, int16, float32)           \
    CUDA_HOST_BINARY(name, bfloat16, int32, float64)           \
    CUDA_HOST_BINARY(name, bfloat16, bfloat16, bfloat16)       \
    CUDA_HOST_BINARY(name, bfloat16, float16, float32)         \
    CUDA_HOST_BINARY(name, bfloat16, float32, float32)         \
    CUDA_HOST_BINARY(name, bfloat16, float64, float64)         \
    CUDA_HOST_NOKERN(name, bfloat16, complex32, complex64)     \
    CUDA_HOST_NOKERN(name, bfloat16, complex64, complex64)     \
    CUDA_HOST_NOKERN(name, bfloat16, complex128, complex128)   \
                                                               \
    CUDA_HOST_NOIMPL(name, float16, uint8, float16)            \
    CUDA_HOST_BINARY(name, float16, uint16, float32)           \
    CUDA_HOST_BINARY(name, float16, uint32, float64)           \
    CUDA_HOST_NOIMPL(name, float16, int8, float16)             \
    CUDA_HOST_BINARY(name, float16, int16, float32)            \
    CUDA_HOST_BINARY(name, float16, int32, float64)            \
    CUDA_HOST_NOIMPL(name, float16, bfloat16, float32)         \
    CUDA_HOST_NOIMPL(name, float16, float16, float16)          \
    CUDA_HOST_BINARY(name, float16, float32, float32)          \
    CUDA_HOST_BINARY(name, float16, float64, float64)          \
    CUDA_HOST_NOKERN(name, float16, complex32, complex32)      \
    CUDA_HOST_NOKERN(name, float16, complex64, complex64)      \
    CUDA_HOST_NOKERN(name, float16, complex128, complex128)    \
                                                               \
    CUDA_HOST_BINARY(name, float32, uint8, float32)            \
    CUDA_HOST_BINARY(name, float32, uint16, float32)           \
    CUDA_HOST_BINARY(name, float32, uint32, float64)           \
    CUDA_HOST_BINARY(name, float32, int8, float32)             \
    CUDA_HOST_BINARY(name, float32, int16, float32)            \
    CUDA_HOST_BINARY(name, float32, int32, float64)            \
    CUDA_HOST_BINARY(name, float32, bfloat16, float32)         \
    CUDA_HOST_BINARY(name, float32, float16, float32)          \
    CUDA_HOST_BINARY(name, float32, float32, float32)          \
    CUDA_HOST_BINARY(name, float32, float64, float64)          \
    CUDA_HOST_NOKERN(name, float32, complex32, complex64)      \
    CUDA_HOST_NOKERN(name, float32, complex64, complex64)      \
    CUDA_HOST_NOKERN(name, float32, complex128, complex128)    \
                                                               \
    CUDA_HOST_BINARY(name, float64, uint8, float64)            \
    CUDA_HOST_BINARY(name, float64, uint16, float64)           \
    CUDA_HOST_BINARY(name, float64, uint32, float64)           \
    CUDA_HOST_BINARY(name, float64, int8, float64)             \
    CUDA_HOST_BINARY(name, float64, int16, float64)            \
    CUDA_HOST_BINARY(name, float64, int32, float64)            \
    CUDA_HOST_BINARY(name, float64, bfloat16, float64)         \
    CUDA_HOST_BINARY(name, float64, float16, float64)          \
    CUDA_HOST_BINARY(name, float64, float32, float64)          \
    CUDA_HOST_BINARY(name, float64, float64, float64)          \
    CUDA_HOST_NOKERN(name, float64, complex32, complex128)     \
    CUDA_HOST_NOKERN(name, float64, complex64, complex128)     \
    CUDA_HOST_NOKERN(name, float64, complex128, complex128)    \
                                                               \
    CUDA_HOST_NOKERN(name, complex32, uint8, complex32)        \
    CUDA_HOST_NOKERN(name, complex32, uint16, complex64)       \
    CUDA_HOST_NOKERN(name, complex32, uint32, complex128)      \
    CUDA_HOST_NOKERN(name, complex32, int8, complex32)         \
    CUDA_HOST_NOKERN(name, complex32, int16, complex64)        \
    CUDA_HOST_NOKERN(name, complex32, int32, complex128)       \
    CUDA_HOST_NOKERN(name, complex32, bfloat16, complex64)     \
    CUDA_HOST_NOKERN(name, complex32, float16, complex32)      \
    CUDA_HOST_NOKERN(name, complex32, float32, complex64)      \
    CUDA_HOST_NOKERN(name, complex32, float64, complex128)     \
    CUDA_HOST_NOKERN(name, complex32, complex32, complex32)    \
    CUDA_HOST_NOKERN(name, complex32, complex64, complex64)    \
    CUDA_HOST_NOKERN(name, complex32, complex128, complex128)  \
                                                               \
    CUDA_HOST_NOKERN(name, complex64, uint8, complex64)        \
    CUDA_HOST_NOKERN(name, complex64, uint16, complex64)       \
    CUDA_HOST_NOKERN(name, complex64, uint32, complex128)      \
    CUDA_HOST_NOKERN(name, complex64, int8, complex64)         \
    CUDA_HOST_NOKERN(name, complex64, int16, complex64)        \
    CUDA_HOST_NOKERN(name, complex64, int32, complex128)       \
    CUDA_HOST_NOKERN(name, complex64, bfloat16, complex64)     \
    CUDA_HOST_NOKERN(name, complex64, float16, complex64)      \
    CUDA_HOST_NOKERN(name, complex64, float32, complex64)      \
    CUDA_HOST_NOKERN(name, complex64, float64, complex128)     \
    CUDA_HOST_NOKERN(name, complex64, complex32, complex64)    \
    CUDA_HOST_NOKERN(name, complex64, complex64, complex64)    \
    CUDA_HOST_NOKERN(name, complex64, complex128, complex128)  \
                                                               \
    CUDA_HOST_NOKERN(name, complex128, uint8, complex128)      \
    CUDA_HOST_NOKERN(name, complex128, uint16, complex128)     \
    CUDA_HOST_NOKERN(name, complex128, uint32, complex128)     \
    CUDA_HOST_NOKERN(name, complex128, int8, complex128)       \
    CUDA_HOST_NOKERN(name, complex128, int16, complex128)      \
    CUDA_HOST_NOKERN(name, complex128, int32, complex128)      \
    CUDA_HOST_NOKERN(name, complex128, bfloat16, complex128)   \
    CUDA_HOST_NOKERN(name, complex128, float16, complex128)    \
    CUDA_HOST_NOKERN(name, complex128, float32, complex128)    \
    CUDA_HOST_NOKERN(name, complex128, float64, complex128)    \
    CUDA_HOST_NOKERN(name, complex128, complex32, complex128)  \
    CUDA_HOST_NOKERN(name, complex128, complex64, complex128)  \
    CUDA_HOST_NOKERN(name, complex128, complex128, complex128)

#define CUDA_HOST_ALL_ARITHMETIC_FLOAT_RETURN(name) \
    CUDA_HOST_BINARY(name, uint8, uint8, float16)              \
    CUDA_HOST_BINARY(name, uint8, uint16, float32)             \
    CUDA_HOST_BINARY(name, uint8, uint32, float64)             \
    CUDA_HOST_NOKERN(name, uint8, uint64, uint64)              \
    CUDA_HOST_BINARY(name, uint8, int8, float16)               \
    CUDA_HOST_BINARY(name, uint8, int16, float32)              \
    CUDA_HOST_BINARY(name, uint8, int32, float64)              \
    CUDA_HOST_NOKERN(name, uint8, int64, int64)                \
    CUDA_HOST_BINARY(name, uint8, bfloat16, bfloat16)          \
    CUDA_HOST_BINARY(name, uint8, float16, float16)            \
    CUDA_HOST_BINARY(name, uint8, float32, float32)            \
    CUDA_HOST_BINARY(name, uint8, float64, float64)            \
    CUDA_HOST_NOIMPL(name, uint8, complex32, complex32)        \
    CUDA_HOST_BINARY(name, uint8, complex64, complex64)        \
    CUDA_HOST_BINARY(name, uint8, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, uint16, uint8, float32)             \
    CUDA_HOST_BINARY(name, uint16, uint16, float32)            \
    CUDA_HOST_BINARY(name, uint16, uint32, float64)            \
    CUDA_HOST_NOKERN(name, uint16, uint64, uint64)             \
    CUDA_HOST_BINARY(name, uint16, int8, float32)              \
    CUDA_HOST_BINARY(name, uint16, int16, float32)             \
    CUDA_HOST_BINARY(name, uint16, int32, float64)             \
    CUDA_HOST_NOKERN(name, uint16, int64, int64)               \
    CUDA_HOST_BINARY(name, uint16, bfloat16, float32)          \
    CUDA_HOST_BINARY(name, uint16, float16, float32)           \
    CUDA_HOST_BINARY(name, uint16, float32, float32)           \
    CUDA_HOST_BINARY(name, uint16, float64, float64)           \
    CUDA_HOST_NOIMPL(name, uint16, complex32, complex64)       \
    CUDA_HOST_BINARY(name, uint16, complex64, complex64)       \
    CUDA_HOST_BINARY(name, uint16, complex128, complex128)     \
                                                               \
    CUDA_HOST_BINARY(name, uint32, uint8, float64)             \
    CUDA_HOST_BINARY(name, uint32, uint16, float64)            \
    CUDA_HOST_BINARY(name, uint32, uint32, float64)            \
    CUDA_HOST_NOKERN(name, uint32, uint64, uint64)             \
    CUDA_HOST_BINARY(name, uint32, int8, float64)              \
    CUDA_HOST_BINARY(name, uint32, int16, float64)             \
    CUDA_HOST_BINARY(name, uint32, int32, float64)             \
    CUDA_HOST_NOKERN(name, uint32, int64, int64)               \
    CUDA_HOST_BINARY(name, uint32, bfloat16, float64)          \
    CUDA_HOST_BINARY(name, uint32, float16, float64)           \
    CUDA_HOST_BINARY(name, uint32, float32, float64)           \
    CUDA_HOST_BINARY(name, uint32, float64, float64)           \
    CUDA_HOST_NOIMPL(name, uint32, complex32, complex128)      \
    CUDA_HOST_BINARY(name, uint32, complex64, complex128)      \
    CUDA_HOST_BINARY(name, uint32, complex128, complex128)     \
                                                               \
    CUDA_HOST_NOKERN(name, uint64, uint8, uint64)              \
    CUDA_HOST_NOKERN(name, uint64, uint16, uint64)             \
    CUDA_HOST_NOKERN(name, uint64, uint32, uint64)             \
    CUDA_HOST_NOKERN(name, uint64, uint64, uint64)             \
                                                               \
    CUDA_HOST_BINARY(name, int8, uint8, float16)               \
    CUDA_HOST_BINARY(name, int8, uint16, float32)              \
    CUDA_HOST_BINARY(name, int8, uint32, float64)              \
    CUDA_HOST_BINARY(name, int8, int8, float16)                \
    CUDA_HOST_BINARY(name, int8, int16, float32)               \
    CUDA_HOST_BINARY(name, int8, int32, float64)               \
    CUDA_HOST_NOKERN(name, int8, int64, int64)                 \
    CUDA_HOST_BINARY(name, int8, bfloat16, bfloat16)           \
    CUDA_HOST_BINARY(name, int8, float16, float16)             \
    CUDA_HOST_BINARY(name, int8, float32, float32)             \
    CUDA_HOST_BINARY(name, int8, float64, float64)             \
    CUDA_HOST_NOIMPL(name, int8, complex32, complex32)         \
    CUDA_HOST_BINARY(name, int8, complex64, complex64)         \
    CUDA_HOST_BINARY(name, int8, complex128, complex128)       \
                                                               \
    CUDA_HOST_BINARY(name, int16, uint8, float32)              \
    CUDA_HOST_BINARY(name, int16, uint16, float32)             \
    CUDA_HOST_BINARY(name, int16, uint32, float64)             \
    CUDA_HOST_BINARY(name, int16, int8, float32)               \
    CUDA_HOST_BINARY(name, int16, int16, float32)              \
    CUDA_HOST_BINARY(name, int16, int32, float64)              \
    CUDA_HOST_NOKERN(name, int16, int64, int64)                \
    CUDA_HOST_BINARY(name, int16, bfloat16, float32)           \
    CUDA_HOST_BINARY(name, int16, float16, float32)            \
    CUDA_HOST_BINARY(name, int16, float32, float32)            \
    CUDA_HOST_BINARY(name, int16, float64, float64)            \
    CUDA_HOST_NOIMPL(name, int16, complex32, complex64)        \
    CUDA_HOST_BINARY(name, int16, complex64, complex64)        \
    CUDA_HOST_BINARY(name, int16, complex128, complex128)      \
                                                               \
    CUDA_HOST_BINARY(name, int32, uint8, float64)              \
    CUDA_HOST_BINARY(name, int32, uint16, float64)             \
    CUDA_HOST_BINARY(name, int32, uint32, float64)             \
    CUDA_HOST_BINARY(name, int32, int8, float64)               \
    CUDA_HOST_BINARY(name, int32, int16, float64)              \
    CUDA_HOST_BINARY(name, int32, int32, float64)              \
    CUDA_HOST_NOKERN(name, int32, int64, int64)                \
    CUDA_HOST_BINARY(name, int32, bfloat16, float64)           \
    CUDA_HOST_BINARY(name, int32, float16, float64)            \
    CUDA_HOST_BINARY(name, int32, float32, float64)            \
    CUDA_HOST_BINARY(name, int32, float64, float64)            \
    CUDA_HOST_NOIMPL(name, int32, complex32, complex128)       \
    CUDA_HOST_BINARY(name, int32, complex64, complex128)       \
    CUDA_HOST_BINARY(name, int32, complex128, complex128)      \
                                                               \
    CUDA_HOST_NOKERN(name, int64, uint8, int64)                \
    CUDA_HOST_NOKERN(name, int64, uint16, int64)               \
    CUDA_HOST_NOKERN(name, int64, uint32, int64)               \
    CUDA_HOST_NOKERN(name, int64, int8, int64)                 \
    CUDA_HOST_NOKERN(name, int64, int16, int64)                \
    CUDA_HOST_NOKERN(name, int64, int32, int64)                \
    CUDA_HOST_NOKERN(name, int64, int64, int64)                \
                                                               \
    CUDA_HOST_BINARY(name, bfloat16, uint8, bfloat16)          \
    CUDA_HOST_BINARY(name, bfloat16, uint16, float32)          \
    CUDA_HOST_BINARY(name, bfloat16, uint32, float64)          \
    CUDA_HOST_BINARY(name, bfloat16, int8, bfloat16)           \
    CUDA_HOST_BINARY(name, bfloat16, int16, float32)           \
    CUDA_HOST_BINARY(name, bfloat16, int32, float64)           \
    CUDA_HOST_BINARY(name, bfloat16, bfloat16, bfloat16)       \
    CUDA_HOST_BINARY(name, bfloat16, float16, float32)         \
    CUDA_HOST_BINARY(name, bfloat16, float32, float32)         \
    CUDA_HOST_BINARY(name, bfloat16, float64, float64)         \
    CUDA_HOST_NOIMPL(name, bfloat16, complex32, complex64)     \
    CUDA_HOST_BINARY(name, bfloat16, complex64, complex64)     \
    CUDA_HOST_BINARY(name, bfloat16, complex128, complex128)   \
                                                               \
    CUDA_HOST_BINARY(name, float16, uint8, float16)            \
    CUDA_HOST_BINARY(name, float16, uint16, float32)           \
    CUDA_HOST_BINARY(name, float16, uint32, float64)           \
    CUDA_HOST_BINARY(name, float16, int8, float16)             \
    CUDA_HOST_BINARY(name, float16, int16, float32)            \
    CUDA_HOST_BINARY(name, float16, int32, float64)            \
    CUDA_HOST_BINARY(name, float16, bfloat16, float32)         \
    CUDA_HOST_BINARY(name, float16, float16, float16)          \
    CUDA_HOST_BINARY(name, float16, float32, float32)          \
    CUDA_HOST_BINARY(name, float16, float64, float64)          \
    CUDA_HOST_NOIMPL(name, float16, complex32, complex32)      \
    CUDA_HOST_BINARY(name, float16, complex64, complex64)      \
    CUDA_HOST_BINARY(name, float16, complex128, complex128)    \
                                                               \
    CUDA_HOST_BINARY(name, float32, uint8, float32)            \
    CUDA_HOST_BINARY(name, float32, uint16, float32)           \
    CUDA_HOST_BINARY(name, float32, uint32, float64)           \
    CUDA_HOST_BINARY(name, float32, int8, float32)             \
    CUDA_HOST_BINARY(name, float32, int16, float32)            \
    CUDA_HOST_BINARY(name, float32, int32, float64)            \
    CUDA_HOST_BINARY(name, float32, bfloat16, float32)         \
    CUDA_HOST_BINARY(name, float32, float16, float32)          \
    CUDA_HOST_BINARY(name, float32, float32, float32)          \
    CUDA_HOST_BINARY(name, float32, float64, float64)          \
    CUDA_HOST_NOIMPL(name, float32, complex32, complex64)      \
    CUDA_HOST_BINARY(name, float32, complex64, complex64)      \
    CUDA_HOST_BINARY(name, float32, complex128, complex128)    \
                                                               \
    CUDA_HOST_BINARY(name, float64, uint8, float64)            \
    CUDA_HOST_BINARY(name, float64, uint16, float64)           \
    CUDA_HOST_BINARY(name, float64, uint32, float64)           \
    CUDA_HOST_BINARY(name, float64, int8, float64)             \
    CUDA_HOST_BINARY(name, float64, int16, float64)            \
    CUDA_HOST_BINARY(name, float64, int32, float64)            \
    CUDA_HOST_BINARY(name, float64, bfloat16, float64)         \
    CUDA_HOST_BINARY(name, float64, float16, float64)          \
    CUDA_HOST_BINARY(name, float64, float32, float64)          \
    CUDA_HOST_BINARY(name, float64, float64, float64)          \
    CUDA_HOST_NOIMPL(name, float64, complex32, complex128)     \
    CUDA_HOST_BINARY(name, float64, complex64, complex128)     \
    CUDA_HOST_BINARY(name, float64, complex128, complex128)    \
                                                               \
    CUDA_HOST_NOIMPL(name, complex32, uint8, complex32)        \
    CUDA_HOST_NOIMPL(name, complex32, uint16, complex64)       \
    CUDA_HOST_NOIMPL(name, complex32, uint32, complex128)      \
    CUDA_HOST_NOIMPL(name, complex32, int8, complex32)         \
    CUDA_HOST_NOIMPL(name, complex32, int16, complex64)        \
    CUDA_HOST_NOIMPL(name, complex32, int32, complex128)       \
    CUDA_HOST_NOIMPL(name, complex32, bfloat16, complex64)     \
    CUDA_HOST_NOIMPL(name, complex32, float16, complex32)      \
    CUDA_HOST_NOIMPL(name, complex32, float32, complex64)      \
    CUDA_HOST_NOIMPL(name, complex32, float64, complex128)     \
    CUDA_HOST_NOIMPL(name, complex32, complex32, complex32)    \
    CUDA_HOST_NOIMPL(name, complex32, complex64, complex64)    \
    CUDA_HOST_NOIMPL(name, complex32, complex128, complex128)  \
                                                               \
    CUDA_HOST_BINARY(name, complex64, uint8, complex64)        \
    CUDA_HOST_BINARY(name, complex64, uint16, complex64)       \
    CUDA_HOST_BINARY(name, complex64, uint32, complex128)      \
    CUDA_HOST_BINARY(name, complex64, int8, complex64)         \
    CUDA_HOST_BINARY(name, complex64, int16, complex64)        \
    CUDA_HOST_BINARY(name, complex64, int32, complex128)       \
    CUDA_HOST_BINARY(name, complex64, bfloat16, complex64)     \
    CUDA_HOST_BINARY(name, complex64, float16, complex64)      \
    CUDA_HOST_BINARY(name, complex64, float32, complex64)      \
    CUDA_HOST_BINARY(name, complex64, float64, complex128)     \
    CUDA_HOST_NOIMPL(name, complex64, complex32, complex64)    \
    CUDA_HOST_BINARY(name, complex64, complex64, complex64)    \
    CUDA_HOST_BINARY(name, complex64, complex128, complex128)  \
                                                               \
    CUDA_HOST_BINARY(name, complex128, uint8, complex128)      \
    CUDA_HOST_BINARY(name, complex128, uint16, complex128)     \
    CUDA_HOST_BINARY(name, complex128, uint32, complex128)     \
    CUDA_HOST_BINARY(name, complex128, int8, complex128)       \
    CUDA_HOST_BINARY(name, complex128, int16, complex128)      \
    CUDA_HOST_BINARY(name, complex128, int32, complex128)      \
    CUDA_HOST_BINARY(name, complex128, bfloat16, complex128)   \
    CUDA_HOST_BINARY(name, complex128, float16, complex128)    \
    CUDA_HOST_BINARY(name, complex128, float32, complex128)    \
    CUDA_HOST_BINARY(name, complex128, float64, complex128)    \
    CUDA_HOST_NOIMPL(name, complex128, complex32, complex128)  \
    CUDA_HOST_BINARY(name, complex128, complex64, complex128)  \
    CUDA_HOST_BINARY(name, complex128, complex128, complex128)

#define CUDA_HOST_ALL_ARITHMETIC_INIT(name) \
    CUDA_HOST_BINARY_INIT(name, uint8, uint8, uint8),                \
    CUDA_HOST_BINARY_INIT(name, uint8, uint16, uint16),              \
    CUDA_HOST_BINARY_INIT(name, uint8, uint32, uint32),              \
    CUDA_HOST_BINARY_INIT(name, uint8, uint64, uint64),              \
    CUDA_HOST_BINARY_INIT(name, uint8, int8, int16),                 \
    CUDA_HOST_BINARY_INIT(name, uint8, int16, int16),                \
    CUDA_HOST_BINARY_INIT(name, uint8, int32, int32),                \
    CUDA_HOST_BINARY_INIT(name, uint8, int64, int64),                \
    CUDA_HOST_BINARY_INIT(name, uint8, bfloat16, bfloat16),          \
    CUDA_HOST_BINARY_INIT(name, uint8, float16, float16),            \
    CUDA_HOST_BINARY_INIT(name, uint8, float32, float32),            \
    CUDA_HOST_BINARY_INIT(name, uint8, float64, float64),            \
    CUDA_HOST_BINARY_INIT(name, uint8, complex32, complex32),        \
    CUDA_HOST_BINARY_INIT(name, uint8, complex64, complex64),        \
    CUDA_HOST_BINARY_INIT(name, uint8, complex128, complex128),      \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, uint16, uint8, uint16),              \
    CUDA_HOST_BINARY_INIT(name, uint16, uint16, uint16),             \
    CUDA_HOST_BINARY_INIT(name, uint16, uint32, uint32),             \
    CUDA_HOST_BINARY_INIT(name, uint16, uint64, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint16, int8, int32),                \
    CUDA_HOST_BINARY_INIT(name, uint16, int16, int32),               \
    CUDA_HOST_BINARY_INIT(name, uint16, int32, int32),               \
    CUDA_HOST_BINARY_INIT(name, uint16, int64, int64),               \
    CUDA_HOST_BINARY_INIT(name, uint16, bfloat16, float32),          \
    CUDA_HOST_BINARY_INIT(name, uint16, float16, float32),           \
    CUDA_HOST_BINARY_INIT(name, uint16, float32, float32),           \
    CUDA_HOST_BINARY_INIT(name, uint16, float64, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint16, complex32, complex64),       \
    CUDA_HOST_BINARY_INIT(name, uint16, complex64, complex64),       \
    CUDA_HOST_BINARY_INIT(name, uint16, complex128, complex128),     \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, uint32, uint8, uint32),              \
    CUDA_HOST_BINARY_INIT(name, uint32, uint16, uint32),             \
    CUDA_HOST_BINARY_INIT(name, uint32, uint32, uint32),             \
    CUDA_HOST_BINARY_INIT(name, uint32, uint64, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint32, int8, int64),                \
    CUDA_HOST_BINARY_INIT(name, uint32, int16, int64),               \
    CUDA_HOST_BINARY_INIT(name, uint32, int32, int64),               \
    CUDA_HOST_BINARY_INIT(name, uint32, int64, int64),               \
    CUDA_HOST_BINARY_INIT(name, uint32, bfloat16, float64),          \
    CUDA_HOST_BINARY_INIT(name, uint32, float16, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint32, float32, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint32, float64, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint32, complex32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, uint32, complex64, complex128),      \
    CUDA_HOST_BINARY_INIT(name, uint32, complex128, complex128),     \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, uint64, uint8, uint64),              \
    CUDA_HOST_BINARY_INIT(name, uint64, uint16, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint64, uint32, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint64, uint64, uint64),             \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int8, uint8, int16),                 \
    CUDA_HOST_BINARY_INIT(name, int8, uint16, int32),                \
    CUDA_HOST_BINARY_INIT(name, int8, uint32, int64),                \
    CUDA_HOST_BINARY_INIT(name, int8, int8, int8),                   \
    CUDA_HOST_BINARY_INIT(name, int8, int16, int16),                 \
    CUDA_HOST_BINARY_INIT(name, int8, int32, int32),                 \
    CUDA_HOST_BINARY_INIT(name, int8, int64, int64),                 \
    CUDA_HOST_BINARY_INIT(name, int8, bfloat16, bfloat16),           \
    CUDA_HOST_BINARY_INIT(name, int8, float16, float16),             \
    CUDA_HOST_BINARY_INIT(name, int8, float32, float32),             \
    CUDA_HOST_BINARY_INIT(name, int8, float64, float64),             \
    CUDA_HOST_BINARY_INIT(name, int8, complex32, complex32),         \
    CUDA_HOST_BINARY_INIT(name, int8, complex64, complex64),         \
    CUDA_HOST_BINARY_INIT(name, int8, complex128, complex128),       \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int16, uint8, int16),                \
    CUDA_HOST_BINARY_INIT(name, int16, uint16, int32),               \
    CUDA_HOST_BINARY_INIT(name, int16, uint32, int64),               \
    CUDA_HOST_BINARY_INIT(name, int16, int8, int16),                 \
    CUDA_HOST_BINARY_INIT(name, int16, int16, int16),                \
    CUDA_HOST_BINARY_INIT(name, int16, int32, int32),                \
    CUDA_HOST_BINARY_INIT(name, int16, int64, int64),                \
    CUDA_HOST_BINARY_INIT(name, int16, bfloat16, float32),           \
    CUDA_HOST_BINARY_INIT(name, int16, float16, float32),            \
    CUDA_HOST_BINARY_INIT(name, int16, float32, float32),            \
    CUDA_HOST_BINARY_INIT(name, int16, float64, float64),            \
    CUDA_HOST_BINARY_INIT(name, int16, complex32, complex64),        \
    CUDA_HOST_BINARY_INIT(name, int16, complex64, complex64),        \
    CUDA_HOST_BINARY_INIT(name, int16, complex128, complex128),      \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int32, uint8, int32),                \
    CUDA_HOST_BINARY_INIT(name, int32, uint16, int32),               \
    CUDA_HOST_BINARY_INIT(name, int32, uint32, int64),               \
    CUDA_HOST_BINARY_INIT(name, int32, int8, int32),                 \
    CUDA_HOST_BINARY_INIT(name, int32, int16, int32),                \
    CUDA_HOST_BINARY_INIT(name, int32, int32, int32),                \
    CUDA_HOST_BINARY_INIT(name, int32, int64, int64),                \
    CUDA_HOST_BINARY_INIT(name, int32, bfloat16, float64),           \
    CUDA_HOST_BINARY_INIT(name, int32, float16, float64),            \
    CUDA_HOST_BINARY_INIT(name, int32, float32, float64),            \
    CUDA_HOST_BINARY_INIT(name, int32, float64, float64),            \
    CUDA_HOST_BINARY_INIT(name, int32, complex32, complex128),       \
    CUDA_HOST_BINARY_INIT(name, int32, complex64, complex128),       \
    CUDA_HOST_BINARY_INIT(name, int32, complex128, complex128),      \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int64, uint8, int64),                \
    CUDA_HOST_BINARY_INIT(name, int64, uint16, int64),               \
    CUDA_HOST_BINARY_INIT(name, int64, uint32, int64),               \
    CUDA_HOST_BINARY_INIT(name, int64, int8, int64),                 \
    CUDA_HOST_BINARY_INIT(name, int64, int16, int64),                \
    CUDA_HOST_BINARY_INIT(name, int64, int32, int64),                \
    CUDA_HOST_BINARY_INIT(name, int64, int64, int64),                \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint8, bfloat16),          \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint16, float32),          \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint32, float64),          \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int8, bfloat16),           \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int16, float32),           \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int32, float64),           \
    CUDA_HOST_BINARY_INIT(name, bfloat16, bfloat16, bfloat16),       \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float16, float32),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float32, float32),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float64, float64),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex32, complex64),     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex64, complex64),     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex128, complex128),   \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, float16, uint8, float16),            \
    CUDA_HOST_BINARY_INIT(name, float16, uint16, float32),           \
    CUDA_HOST_BINARY_INIT(name, float16, uint32, float64),           \
    CUDA_HOST_BINARY_INIT(name, float16, int8, float16),             \
    CUDA_HOST_BINARY_INIT(name, float16, int16, float32),            \
    CUDA_HOST_BINARY_INIT(name, float16, int32, float64),            \
    CUDA_HOST_BINARY_INIT(name, float16, bfloat16, float32),         \
    CUDA_HOST_BINARY_INIT(name, float16, float16, float16),          \
    CUDA_HOST_BINARY_INIT(name, float16, float32, float32),          \
    CUDA_HOST_BINARY_INIT(name, float16, float64, float64),          \
    CUDA_HOST_BINARY_INIT(name, float16, complex32, complex32),      \
    CUDA_HOST_BINARY_INIT(name, float16, complex64, complex64),      \
    CUDA_HOST_BINARY_INIT(name, float16, complex128, complex128),    \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, float32, uint8, float32),            \
    CUDA_HOST_BINARY_INIT(name, float32, uint16, float32),           \
    CUDA_HOST_BINARY_INIT(name, float32, uint32, float64),           \
    CUDA_HOST_BINARY_INIT(name, float32, int8, float32),             \
    CUDA_HOST_BINARY_INIT(name, float32, int16, float32),            \
    CUDA_HOST_BINARY_INIT(name, float32, int32, float64),            \
    CUDA_HOST_BINARY_INIT(name, float32, bfloat16, float32),         \
    CUDA_HOST_BINARY_INIT(name, float32, float16, float32),          \
    CUDA_HOST_BINARY_INIT(name, float32, float32, float32),          \
    CUDA_HOST_BINARY_INIT(name, float32, float64, float64),          \
    CUDA_HOST_BINARY_INIT(name, float32, complex32, complex64),      \
    CUDA_HOST_BINARY_INIT(name, float32, complex64, complex64),      \
    CUDA_HOST_BINARY_INIT(name, float32, complex128, complex128),    \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, float64, uint8, float64),            \
    CUDA_HOST_BINARY_INIT(name, float64, uint16, float64),           \
    CUDA_HOST_BINARY_INIT(name, float64, uint32, float64),           \
    CUDA_HOST_BINARY_INIT(name, float64, int8, float64),             \
    CUDA_HOST_BINARY_INIT(name, float64, int16, float64),            \
    CUDA_HOST_BINARY_INIT(name, float64, int32, float64),            \
    CUDA_HOST_BINARY_INIT(name, float64, bfloat16, float64),         \
    CUDA_HOST_BINARY_INIT(name, float64, float16, float64),          \
    CUDA_HOST_BINARY_INIT(name, float64, float32, float64),          \
    CUDA_HOST_BINARY_INIT(name, float64, float64, float64),          \
    CUDA_HOST_BINARY_INIT(name, float64, complex32, complex128),     \
    CUDA_HOST_BINARY_INIT(name, float64, complex64, complex128),     \
    CUDA_HOST_BINARY_INIT(name, float64, complex128, complex128),    \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, complex32, uint8, complex32),        \
    CUDA_HOST_BINARY_INIT(name, complex32, uint16, complex64),       \
    CUDA_HOST_BINARY_INIT(name, complex32, uint32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex32, int8, complex32),         \
    CUDA_HOST_BINARY_INIT(name, complex32, int16, complex64),        \
    CUDA_HOST_BINARY_INIT(name, complex32, int32, complex128),       \
    CUDA_HOST_BINARY_INIT(name, complex32, bfloat16, complex64),     \
    CUDA_HOST_BINARY_INIT(name, complex32, float16, complex32),      \
    CUDA_HOST_BINARY_INIT(name, complex32, float32, complex64),      \
    CUDA_HOST_BINARY_INIT(name, complex32, float64, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex32, complex32, complex32),    \
    CUDA_HOST_BINARY_INIT(name, complex32, complex64, complex64),    \
    CUDA_HOST_BINARY_INIT(name, complex32, complex128, complex128),  \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, complex64, uint8, complex64),        \
    CUDA_HOST_BINARY_INIT(name, complex64, uint16, complex64),       \
    CUDA_HOST_BINARY_INIT(name, complex64, uint32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex64, int8, complex64),         \
    CUDA_HOST_BINARY_INIT(name, complex64, int16, complex64),        \
    CUDA_HOST_BINARY_INIT(name, complex64, int32, complex128),       \
    CUDA_HOST_BINARY_INIT(name, complex64, bfloat16, complex64),     \
    CUDA_HOST_BINARY_INIT(name, complex64, float16, complex64),      \
    CUDA_HOST_BINARY_INIT(name, complex64, float32, complex64),      \
    CUDA_HOST_BINARY_INIT(name, complex64, float64, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex64, complex32, complex64),    \
    CUDA_HOST_BINARY_INIT(name, complex64, complex64, complex64),    \
    CUDA_HOST_BINARY_INIT(name, complex64, complex128, complex128),  \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, complex128, uint8, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex128, uint16, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex128, uint32, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex128, int8, complex128),       \
    CUDA_HOST_BINARY_INIT(name, complex128, int16, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex128, int32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex128, bfloat16, complex128),   \
    CUDA_HOST_BINARY_INIT(name, complex128, float16, complex128),    \
    CUDA_HOST_BINARY_INIT(name, complex128, float32, complex128),    \
    CUDA_HOST_BINARY_INIT(name, complex128, float64, complex128),    \
    CUDA_HOST_BINARY_INIT(name, complex128, complex32, complex128),  \
    CUDA_HOST_BINARY_INIT(name, complex128, complex64, complex128),  \
    CUDA_HOST_BINARY_INIT(name, complex128, complex128, complex128)

#define CUDA_HOST_ALL_ARITHMETIC_FLOAT_RETURN_INIT(name) \
    CUDA_HOST_BINARY_INIT(name, uint8, uint8, float16),              \
    CUDA_HOST_BINARY_INIT(name, uint8, uint16, float32),             \
    CUDA_HOST_BINARY_INIT(name, uint8, uint32, float64),             \
    CUDA_HOST_BINARY_INIT(name, uint8, uint64, uint64),              \
    CUDA_HOST_BINARY_INIT(name, uint8, int8, float16),               \
    CUDA_HOST_BINARY_INIT(name, uint8, int16, float32),              \
    CUDA_HOST_BINARY_INIT(name, uint8, int32, float64),              \
    CUDA_HOST_BINARY_INIT(name, uint8, int64, int64),                \
    CUDA_HOST_BINARY_INIT(name, uint8, bfloat16, bfloat16),          \
    CUDA_HOST_BINARY_INIT(name, uint8, float16, float16),            \
    CUDA_HOST_BINARY_INIT(name, uint8, float32, float32),            \
    CUDA_HOST_BINARY_INIT(name, uint8, float64, float64),            \
    CUDA_HOST_BINARY_INIT(name, uint8, complex32, complex32),        \
    CUDA_HOST_BINARY_INIT(name, uint8, complex64, complex64),        \
    CUDA_HOST_BINARY_INIT(name, uint8, complex128, complex128),      \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, uint16, uint8, float32),             \
    CUDA_HOST_BINARY_INIT(name, uint16, uint16, float32),            \
    CUDA_HOST_BINARY_INIT(name, uint16, uint32, float64),            \
    CUDA_HOST_BINARY_INIT(name, uint16, uint64, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint16, int8, float32),              \
    CUDA_HOST_BINARY_INIT(name, uint16, int16, float32),             \
    CUDA_HOST_BINARY_INIT(name, uint16, int32, float64),             \
    CUDA_HOST_BINARY_INIT(name, uint16, int64, int64),               \
    CUDA_HOST_BINARY_INIT(name, uint16, bfloat16, float32),          \
    CUDA_HOST_BINARY_INIT(name, uint16, float16, float32),           \
    CUDA_HOST_BINARY_INIT(name, uint16, float32, float32),           \
    CUDA_HOST_BINARY_INIT(name, uint16, float64, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint16, complex32, complex64),       \
    CUDA_HOST_BINARY_INIT(name, uint16, complex64, complex64),       \
    CUDA_HOST_BINARY_INIT(name, uint16, complex128, complex128),     \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, uint32, uint8, float64),             \
    CUDA_HOST_BINARY_INIT(name, uint32, uint16, float64),            \
    CUDA_HOST_BINARY_INIT(name, uint32, uint32, float64),            \
    CUDA_HOST_BINARY_INIT(name, uint32, uint64, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint32, int8, float64),              \
    CUDA_HOST_BINARY_INIT(name, uint32, int16, float64),             \
    CUDA_HOST_BINARY_INIT(name, uint32, int32, float64),             \
    CUDA_HOST_BINARY_INIT(name, uint32, int64, int64),               \
    CUDA_HOST_BINARY_INIT(name, uint32, bfloat16, float64),          \
    CUDA_HOST_BINARY_INIT(name, uint32, float16, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint32, float32, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint32, float64, float64),           \
    CUDA_HOST_BINARY_INIT(name, uint32, complex32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, uint32, complex64, complex128),      \
    CUDA_HOST_BINARY_INIT(name, uint32, complex128, complex128),     \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, uint64, uint8, uint64),              \
    CUDA_HOST_BINARY_INIT(name, uint64, uint16, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint64, uint32, uint64),             \
    CUDA_HOST_BINARY_INIT(name, uint64, uint64, uint64),             \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int8, uint8, float16),               \
    CUDA_HOST_BINARY_INIT(name, int8, uint16, float32),              \
    CUDA_HOST_BINARY_INIT(name, int8, uint32, float64),              \
    CUDA_HOST_BINARY_INIT(name, int8, int8, float16),                \
    CUDA_HOST_BINARY_INIT(name, int8, int16, float32),               \
    CUDA_HOST_BINARY_INIT(name, int8, int32, float64),               \
    CUDA_HOST_BINARY_INIT(name, int8, int64, int64),                 \
    CUDA_HOST_BINARY_INIT(name, int8, bfloat16, bfloat16),           \
    CUDA_HOST_BINARY_INIT(name, int8, float16, float16),             \
    CUDA_HOST_BINARY_INIT(name, int8, float32, float32),             \
    CUDA_HOST_BINARY_INIT(name, int8, float64, float64),             \
    CUDA_HOST_BINARY_INIT(name, int8, complex32, complex32),         \
    CUDA_HOST_BINARY_INIT(name, int8, complex64, complex64),         \
    CUDA_HOST_BINARY_INIT(name, int8, complex128, complex128),       \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int16, uint8, float32),              \
    CUDA_HOST_BINARY_INIT(name, int16, uint16, float32),             \
    CUDA_HOST_BINARY_INIT(name, int16, uint32, float64),             \
    CUDA_HOST_BINARY_INIT(name, int16, int8, float32),               \
    CUDA_HOST_BINARY_INIT(name, int16, int16, float32),              \
    CUDA_HOST_BINARY_INIT(name, int16, int32, float64),              \
    CUDA_HOST_BINARY_INIT(name, int16, int64, int64),                \
    CUDA_HOST_BINARY_INIT(name, int16, bfloat16, float32),           \
    CUDA_HOST_BINARY_INIT(name, int16, float16, float32),            \
    CUDA_HOST_BINARY_INIT(name, int16, float32, float32),            \
    CUDA_HOST_BINARY_INIT(name, int16, float64, float64),            \
    CUDA_HOST_BINARY_INIT(name, int16, complex32, complex64),        \
    CUDA_HOST_BINARY_INIT(name, int16, complex64, complex64),        \
    CUDA_HOST_BINARY_INIT(name, int16, complex128, complex128),      \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int32, uint8, float64),              \
    CUDA_HOST_BINARY_INIT(name, int32, uint16, float64),             \
    CUDA_HOST_BINARY_INIT(name, int32, uint32, float64),             \
    CUDA_HOST_BINARY_INIT(name, int32, int8, float64),               \
    CUDA_HOST_BINARY_INIT(name, int32, int16, float64),              \
    CUDA_HOST_BINARY_INIT(name, int32, int32, float64),              \
    CUDA_HOST_BINARY_INIT(name, int32, int64, int64),                \
    CUDA_HOST_BINARY_INIT(name, int32, bfloat16, float64),           \
    CUDA_HOST_BINARY_INIT(name, int32, float16, float64),            \
    CUDA_HOST_BINARY_INIT(name, int32, float32, float64),            \
    CUDA_HOST_BINARY_INIT(name, int32, float64, float64),            \
    CUDA_HOST_BINARY_INIT(name, int32, complex32, complex128),       \
    CUDA_HOST_BINARY_INIT(name, int32, complex64, complex128),       \
    CUDA_HOST_BINARY_INIT(name, int32, complex128, complex128),      \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, int64, uint8, int64),                \
    CUDA_HOST_BINARY_INIT(name, int64, uint16, int64),               \
    CUDA_HOST_BINARY_INIT(name, int64, uint32, int64),               \
    CUDA_HOST_BINARY_INIT(name, int64, int8, int64),                 \
    CUDA_HOST_BINARY_INIT(name, int64, int16, int64),                \
    CUDA_HOST_BINARY_INIT(name, int64, int32, int64),                \
    CUDA_HOST_BINARY_INIT(name, int64, int64, int64),                \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint8, bfloat16),          \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint16, float32),          \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint32, float64),          \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int8, bfloat16),           \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int16, float32),           \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int32, float64),           \
    CUDA_HOST_BINARY_INIT(name, bfloat16, bfloat16, bfloat16),       \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float16, float32),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float32, float32),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float64, float64),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex32, complex64),     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex64, complex64),     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex128, complex128),   \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, float16, uint8, float16),            \
    CUDA_HOST_BINARY_INIT(name, float16, uint16, float32),           \
    CUDA_HOST_BINARY_INIT(name, float16, uint32, float64),           \
    CUDA_HOST_BINARY_INIT(name, float16, int8, float16),             \
    CUDA_HOST_BINARY_INIT(name, float16, int16, float32),            \
    CUDA_HOST_BINARY_INIT(name, float16, int32, float64),            \
    CUDA_HOST_BINARY_INIT(name, float16, bfloat16, float32),         \
    CUDA_HOST_BINARY_INIT(name, float16, float16, float16),          \
    CUDA_HOST_BINARY_INIT(name, float16, float32, float32),          \
    CUDA_HOST_BINARY_INIT(name, float16, float64, float64),          \
    CUDA_HOST_BINARY_INIT(name, float16, complex32, complex32),      \
    CUDA_HOST_BINARY_INIT(name, float16, complex64, complex64),      \
    CUDA_HOST_BINARY_INIT(name, float16, complex128, complex128),    \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, float32, uint8, float32),            \
    CUDA_HOST_BINARY_INIT(name, float32, uint16, float32),           \
    CUDA_HOST_BINARY_INIT(name, float32, uint32, float64),           \
    CUDA_HOST_BINARY_INIT(name, float32, int8, float32),             \
    CUDA_HOST_BINARY_INIT(name, float32, int16, float32),            \
    CUDA_HOST_BINARY_INIT(name, float32, int32, float64),            \
    CUDA_HOST_BINARY_INIT(name, float32, bfloat16, float32),         \
    CUDA_HOST_BINARY_INIT(name, float32, float16, float32),          \
    CUDA_HOST_BINARY_INIT(name, float32, float32, float32),          \
    CUDA_HOST_BINARY_INIT(name, float32, float64, float64),          \
    CUDA_HOST_BINARY_INIT(name, float32, complex32, complex64),      \
    CUDA_HOST_BINARY_INIT(name, float32, complex64, complex64),      \
    CUDA_HOST_BINARY_INIT(name, float32, complex128, complex128),    \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, float64, uint8, float64),            \
    CUDA_HOST_BINARY_INIT(name, float64, uint16, float64),           \
    CUDA_HOST_BINARY_INIT(name, float64, uint32, float64),           \
    CUDA_HOST_BINARY_INIT(name, float64, int8, float64),             \
    CUDA_HOST_BINARY_INIT(name, float64, int16, float64),            \
    CUDA_HOST_BINARY_INIT(name, float64, int32, float64),            \
    CUDA_HOST_BINARY_INIT(name, float64, bfloat16, float64),         \
    CUDA_HOST_BINARY_INIT(name, float64, float16, float64),          \
    CUDA_HOST_BINARY_INIT(name, float64, float32, float64),          \
    CUDA_HOST_BINARY_INIT(name, float64, float64, float64),          \
    CUDA_HOST_BINARY_INIT(name, float64, complex32, complex128),     \
    CUDA_HOST_BINARY_INIT(name, float64, complex64, complex128),     \
    CUDA_HOST_BINARY_INIT(name, float64, complex128, complex128),    \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, complex32, uint8, complex32),        \
    CUDA_HOST_BINARY_INIT(name, complex32, uint16, complex64),       \
    CUDA_HOST_BINARY_INIT(name, complex32, uint32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex32, int8, complex32),         \
    CUDA_HOST_BINARY_INIT(name, complex32, int16, complex64),        \
    CUDA_HOST_BINARY_INIT(name, complex32, int32, complex128),       \
    CUDA_HOST_BINARY_INIT(name, complex32, bfloat16, complex64),     \
    CUDA_HOST_BINARY_INIT(name, complex32, float16, complex32),      \
    CUDA_HOST_BINARY_INIT(name, complex32, float32, complex64),      \
    CUDA_HOST_BINARY_INIT(name, complex32, float64, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex32, complex32, complex32),    \
    CUDA_HOST_BINARY_INIT(name, complex32, complex64, complex64),    \
    CUDA_HOST_BINARY_INIT(name, complex32, complex128, complex128),  \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, complex64, uint8, complex64),        \
    CUDA_HOST_BINARY_INIT(name, complex64, uint16, complex64),       \
    CUDA_HOST_BINARY_INIT(name, complex64, uint32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex64, int8, complex64),         \
    CUDA_HOST_BINARY_INIT(name, complex64, int16, complex64),        \
    CUDA_HOST_BINARY_INIT(name, complex64, int32, complex128),       \
    CUDA_HOST_BINARY_INIT(name, complex64, bfloat16, complex64),     \
    CUDA_HOST_BINARY_INIT(name, complex64, float16, complex64),      \
    CUDA_HOST_BINARY_INIT(name, complex64, float32, complex64),      \
    CUDA_HOST_BINARY_INIT(name, complex64, float64, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex64, complex32, complex64),    \
    CUDA_HOST_BINARY_INIT(name, complex64, complex64, complex64),    \
    CUDA_HOST_BINARY_INIT(name, complex64, complex128, complex128),  \
                                                                     \
    CUDA_HOST_BINARY_INIT(name, complex128, uint8, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex128, uint16, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex128, uint32, complex128),     \
    CUDA_HOST_BINARY_INIT(name, complex128, int8, complex128),       \
    CUDA_HOST_BINARY_INIT(name, complex128, int16, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex128, int32, complex128),      \
    CUDA_HOST_BINARY_INIT(name, complex128, bfloat16, complex128),   \
    CUDA_HOST_BINARY_INIT(name, complex128, float16, complex128),    \
    CUDA_HOST_BINARY_INIT(name, complex128, float32, complex128),    \
    CUDA_HOST_BINARY_INIT(name, complex128, float64, complex128),    \
    CUDA_HOST_BINARY_INIT(name, complex128, complex32, complex128),  \
    CUDA_HOST_BINARY_INIT(name, complex128, complex64, complex128),  \
    CUDA_HOST_BINARY_INIT(name, complex128, complex128, complex128)


#define add(x, y) x + y
CUDA_HOST_ALL_ARITHMETIC(add)

#define subtract(x, y) x - y
CUDA_HOST_ALL_ARITHMETIC(subtract)

#define multiply(x, y) x * y
CUDA_HOST_ALL_ARITHMETIC(multiply)

#define floor_divide(x, y) x / y
CUDA_HOST_ALL_ARITHMETIC_NO_COMPLEX(floor_divide)

#define remainder(x, y) x % y
CUDA_HOST_ALL_ARITHMETIC_NO_COMPLEX(remainder)

#define divide(x, y) x / y
CUDA_HOST_ALL_ARITHMETIC_FLOAT_RETURN(divide)


/*****************************************************************************/
/*                                 Comparison                                */
/*****************************************************************************/

#define CUDA_HOST_ALL_COMPARISON(name) \
    CUDA_HOST_BINARY(name, uint8, uint8, bool)           \
    CUDA_HOST_BINARY(name, uint8, uint16, bool)          \
    CUDA_HOST_BINARY(name, uint8, uint32, bool)          \
    CUDA_HOST_BINARY(name, uint8, uint64, bool)          \
    CUDA_HOST_BINARY(name, uint8, int8, bool)            \
    CUDA_HOST_BINARY(name, uint8, int16, bool)           \
    CUDA_HOST_BINARY(name, uint8, int32, bool)           \
    CUDA_HOST_BINARY(name, uint8, int64, bool)           \
    CUDA_HOST_BINARY(name, uint8, bfloat16, bool)        \
    CUDA_HOST_BINARY(name, uint8, float16, bool)         \
    CUDA_HOST_BINARY(name, uint8, float32, bool)         \
    CUDA_HOST_BINARY(name, uint8, float64, bool)         \
    CUDA_HOST_NOIMPL(name, uint8, complex32, bool)       \
    CUDA_HOST_BINARY(name, uint8, complex64, bool)       \
    CUDA_HOST_BINARY(name, uint8, complex128, bool)      \
                                                         \
    CUDA_HOST_BINARY(name, uint16, uint8, bool)          \
    CUDA_HOST_BINARY(name, uint16, uint16, bool)         \
    CUDA_HOST_BINARY(name, uint16, uint32, bool)         \
    CUDA_HOST_BINARY(name, uint16, uint64, bool)         \
    CUDA_HOST_BINARY(name, uint16, int8, bool)           \
    CUDA_HOST_BINARY(name, uint16, int16, bool)          \
    CUDA_HOST_BINARY(name, uint16, int32, bool)          \
    CUDA_HOST_BINARY(name, uint16, int64, bool)          \
    CUDA_HOST_BINARY(name, uint16, bfloat16, bool)       \
    CUDA_HOST_BINARY(name, uint16, float16, bool)        \
    CUDA_HOST_BINARY(name, uint16, float32, bool)        \
    CUDA_HOST_BINARY(name, uint16, float64, bool)        \
    CUDA_HOST_NOIMPL(name, uint16, complex32, bool)      \
    CUDA_HOST_BINARY(name, uint16, complex64, bool)      \
    CUDA_HOST_BINARY(name, uint16, complex128, bool)     \
                                                         \
    CUDA_HOST_BINARY(name, uint32, uint8, bool)          \
    CUDA_HOST_BINARY(name, uint32, uint16, bool)         \
    CUDA_HOST_BINARY(name, uint32, uint32, bool)         \
    CUDA_HOST_BINARY(name, uint32, uint64, bool)         \
    CUDA_HOST_BINARY(name, uint32, int8, bool)           \
    CUDA_HOST_BINARY(name, uint32, int16, bool)          \
    CUDA_HOST_BINARY(name, uint32, int32, bool)          \
    CUDA_HOST_BINARY(name, uint32, int64, bool)          \
    CUDA_HOST_BINARY(name, uint32, bfloat16, bool)       \
    CUDA_HOST_BINARY(name, uint32, float16, bool)        \
    CUDA_HOST_BINARY(name, uint32, float32, bool)        \
    CUDA_HOST_BINARY(name, uint32, float64, bool)        \
    CUDA_HOST_NOIMPL(name, uint32, complex32, bool)      \
    CUDA_HOST_BINARY(name, uint32, complex64, bool)      \
    CUDA_HOST_BINARY(name, uint32, complex128, bool)     \
                                                         \
    CUDA_HOST_BINARY(name, uint64, uint8, bool)          \
    CUDA_HOST_BINARY(name, uint64, uint16, bool)         \
    CUDA_HOST_BINARY(name, uint64, uint32, bool)         \
    CUDA_HOST_BINARY(name, uint64, uint64, bool)         \
                                                         \
    CUDA_HOST_BINARY(name, int8, uint8, bool)            \
    CUDA_HOST_BINARY(name, int8, uint16, bool)           \
    CUDA_HOST_BINARY(name, int8, uint32, bool)           \
    CUDA_HOST_BINARY(name, int8, int8, bool)             \
    CUDA_HOST_BINARY(name, int8, int16, bool)            \
    CUDA_HOST_BINARY(name, int8, int32, bool)            \
    CUDA_HOST_BINARY(name, int8, int64, bool)            \
    CUDA_HOST_BINARY(name, int8, bfloat16, bool)         \
    CUDA_HOST_BINARY(name, int8, float16, bool)          \
    CUDA_HOST_BINARY(name, int8, float32, bool)          \
    CUDA_HOST_BINARY(name, int8, float64, bool)          \
    CUDA_HOST_NOIMPL(name, int8, complex32, bool)        \
    CUDA_HOST_BINARY(name, int8, complex64, bool)        \
    CUDA_HOST_BINARY(name, int8, complex128, bool)       \
                                                         \
    CUDA_HOST_BINARY(name, int16, uint8, bool)           \
    CUDA_HOST_BINARY(name, int16, uint16, bool)          \
    CUDA_HOST_BINARY(name, int16, uint32, bool)          \
    CUDA_HOST_BINARY(name, int16, int8, bool)            \
    CUDA_HOST_BINARY(name, int16, int16, bool)           \
    CUDA_HOST_BINARY(name, int16, int32, bool)           \
    CUDA_HOST_BINARY(name, int16, int64, bool)           \
    CUDA_HOST_BINARY(name, int16, bfloat16, bool)        \
    CUDA_HOST_BINARY(name, int16, float16, bool)         \
    CUDA_HOST_BINARY(name, int16, float32, bool)         \
    CUDA_HOST_BINARY(name, int16, float64, bool)         \
    CUDA_HOST_NOIMPL(name, int16, complex32, bool)       \
    CUDA_HOST_BINARY(name, int16, complex64, bool)       \
    CUDA_HOST_BINARY(name, int16, complex128, bool)      \
                                                         \
    CUDA_HOST_BINARY(name, int32, uint8, bool)           \
    CUDA_HOST_BINARY(name, int32, uint16, bool)          \
    CUDA_HOST_BINARY(name, int32, uint32, bool)          \
    CUDA_HOST_BINARY(name, int32, int8, bool)            \
    CUDA_HOST_BINARY(name, int32, int16, bool)           \
    CUDA_HOST_BINARY(name, int32, int32, bool)           \
    CUDA_HOST_BINARY(name, int32, int64, bool)           \
    CUDA_HOST_BINARY(name, int32, bfloat16, bool)        \
    CUDA_HOST_BINARY(name, int32, float16, bool)         \
    CUDA_HOST_BINARY(name, int32, float32, bool)         \
    CUDA_HOST_BINARY(name, int32, float64, bool)         \
    CUDA_HOST_NOIMPL(name, int32, complex32, bool)       \
    CUDA_HOST_BINARY(name, int32, complex64, bool)       \
    CUDA_HOST_BINARY(name, int32, complex128, bool)      \
                                                         \
    CUDA_HOST_BINARY(name, int64, uint8, bool)           \
    CUDA_HOST_BINARY(name, int64, uint16, bool)          \
    CUDA_HOST_BINARY(name, int64, uint32, bool)          \
    CUDA_HOST_BINARY(name, int64, int8, bool)            \
    CUDA_HOST_BINARY(name, int64, int16, bool)           \
    CUDA_HOST_BINARY(name, int64, int32, bool)           \
    CUDA_HOST_BINARY(name, int64, int64, bool)           \
                                                         \
    CUDA_HOST_BINARY(name, bfloat16, uint8, bool)        \
    CUDA_HOST_BINARY(name, bfloat16, uint16, bool)       \
    CUDA_HOST_BINARY(name, bfloat16, uint32, bool)       \
    CUDA_HOST_BINARY(name, bfloat16, int8, bool)         \
    CUDA_HOST_BINARY(name, bfloat16, int16, bool)        \
    CUDA_HOST_BINARY(name, bfloat16, int32, bool)        \
    CUDA_HOST_BINARY(name, bfloat16, bfloat16, bool)     \
    CUDA_HOST_BINARY(name, bfloat16, float16, bool)      \
    CUDA_HOST_BINARY(name, bfloat16, float32, bool)      \
    CUDA_HOST_BINARY(name, bfloat16, float64, bool)      \
    CUDA_HOST_NOIMPL(name, bfloat16, complex32, bool)    \
    CUDA_HOST_BINARY(name, bfloat16, complex64, bool)    \
    CUDA_HOST_BINARY(name, bfloat16, complex128, bool)   \
                                                         \
    CUDA_HOST_BINARY(name, float16, uint8, bool)         \
    CUDA_HOST_BINARY(name, float16, uint16, bool)        \
    CUDA_HOST_BINARY(name, float16, uint32, bool)        \
    CUDA_HOST_BINARY(name, float16, int8, bool)          \
    CUDA_HOST_BINARY(name, float16, int16, bool)         \
    CUDA_HOST_BINARY(name, float16, int32, bool)         \
    CUDA_HOST_BINARY(name, float16, bfloat16, bool)      \
    CUDA_HOST_BINARY(name, float16, float16, bool)       \
    CUDA_HOST_BINARY(name, float16, float32, bool)       \
    CUDA_HOST_BINARY(name, float16, float64, bool)       \
    CUDA_HOST_NOIMPL(name, float16, complex32, bool)     \
    CUDA_HOST_BINARY(name, float16, complex64, bool)     \
    CUDA_HOST_BINARY(name, float16, complex128, bool)    \
                                                         \
    CUDA_HOST_BINARY(name, float32, uint8, bool)         \
    CUDA_HOST_BINARY(name, float32, uint16, bool)        \
    CUDA_HOST_BINARY(name, float32, uint32, bool)        \
    CUDA_HOST_BINARY(name, float32, int8, bool)          \
    CUDA_HOST_BINARY(name, float32, int16, bool)         \
    CUDA_HOST_BINARY(name, float32, int32, bool)         \
    CUDA_HOST_BINARY(name, float32, bfloat16, bool)      \
    CUDA_HOST_BINARY(name, float32, float16, bool)       \
    CUDA_HOST_BINARY(name, float32, float32, bool)       \
    CUDA_HOST_BINARY(name, float32, float64, bool)       \
    CUDA_HOST_NOIMPL(name, float32, complex32, bool)     \
    CUDA_HOST_BINARY(name, float32, complex64, bool)     \
    CUDA_HOST_BINARY(name, float32, complex128, bool)    \
                                                         \
    CUDA_HOST_BINARY(name, float64, uint8, bool)         \
    CUDA_HOST_BINARY(name, float64, uint16, bool)        \
    CUDA_HOST_BINARY(name, float64, uint32, bool)        \
    CUDA_HOST_BINARY(name, float64, int8, bool)          \
    CUDA_HOST_BINARY(name, float64, int16, bool)         \
    CUDA_HOST_BINARY(name, float64, int32, bool)         \
    CUDA_HOST_BINARY(name, float64, bfloat16, bool)      \
    CUDA_HOST_BINARY(name, float64, float16, bool)       \
    CUDA_HOST_BINARY(name, float64, float32, bool)       \
    CUDA_HOST_BINARY(name, float64, float64, bool)       \
    CUDA_HOST_NOIMPL(name, float64, complex32, bool)     \
    CUDA_HOST_BINARY(name, float64, complex64, bool)     \
    CUDA_HOST_BINARY(name, float64, complex128, bool)    \
                                                         \
    CUDA_HOST_NOIMPL(name, complex32, uint8, bool)       \
    CUDA_HOST_NOIMPL(name, complex32, uint16, bool)      \
    CUDA_HOST_NOIMPL(name, complex32, uint32, bool)      \
    CUDA_HOST_NOIMPL(name, complex32, int8, bool)        \
    CUDA_HOST_NOIMPL(name, complex32, int16, bool)       \
    CUDA_HOST_NOIMPL(name, complex32, int32, bool)       \
    CUDA_HOST_NOIMPL(name, complex32, bfloat16, bool)    \
    CUDA_HOST_NOIMPL(name, complex32, float16, bool)     \
    CUDA_HOST_NOIMPL(name, complex32, float32, bool)     \
    CUDA_HOST_NOIMPL(name, complex32, float64, bool)     \
    CUDA_HOST_NOIMPL(name, complex32, complex32, bool)   \
    CUDA_HOST_NOIMPL(name, complex32, complex64, bool)   \
    CUDA_HOST_NOIMPL(name, complex32, complex128, bool)  \
                                                         \
    CUDA_HOST_BINARY(name, complex64, uint8, bool)       \
    CUDA_HOST_BINARY(name, complex64, uint16, bool)      \
    CUDA_HOST_BINARY(name, complex64, uint32, bool)      \
    CUDA_HOST_BINARY(name, complex64, int8, bool)        \
    CUDA_HOST_BINARY(name, complex64, int16, bool)       \
    CUDA_HOST_BINARY(name, complex64, int32, bool)       \
    CUDA_HOST_BINARY(name, complex64, bfloat16, bool)    \
    CUDA_HOST_BINARY(name, complex64, float16, bool)     \
    CUDA_HOST_BINARY(name, complex64, float32, bool)     \
    CUDA_HOST_BINARY(name, complex64, float64, bool)     \
    CUDA_HOST_NOIMPL(name, complex64, complex32, bool)   \
    CUDA_HOST_BINARY(name, complex64, complex64, bool)   \
    CUDA_HOST_BINARY(name, complex64, complex128, bool)  \
                                                         \
    CUDA_HOST_BINARY(name, complex128, uint8, bool)      \
    CUDA_HOST_BINARY(name, complex128, uint16, bool)     \
    CUDA_HOST_BINARY(name, complex128, uint32, bool)     \
    CUDA_HOST_BINARY(name, complex128, int8, bool)       \
    CUDA_HOST_BINARY(name, complex128, int16, bool)      \
    CUDA_HOST_BINARY(name, complex128, int32, bool)      \
    CUDA_HOST_BINARY(name, complex128, bfloat16, bool)   \
    CUDA_HOST_BINARY(name, complex128, float16, bool)    \
    CUDA_HOST_BINARY(name, complex128, float32, bool)    \
    CUDA_HOST_BINARY(name, complex128, float64, bool)    \
    CUDA_HOST_NOIMPL(name, complex128, complex32, bool)  \
    CUDA_HOST_BINARY(name, complex128, complex64, bool)  \
    CUDA_HOST_BINARY(name, complex128, complex128, bool)

#define CUDA_HOST_ALL_COMPARISON_INIT(name) \
    CUDA_HOST_BINARY_INIT(name, uint8, uint8, bool),           \
    CUDA_HOST_BINARY_INIT(name, uint8, uint16, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint8, uint32, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint8, uint64, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint8, int8, bool),            \
    CUDA_HOST_BINARY_INIT(name, uint8, int16, bool),           \
    CUDA_HOST_BINARY_INIT(name, uint8, int32, bool),           \
    CUDA_HOST_BINARY_INIT(name, uint8, int64, bool),           \
    CUDA_HOST_BINARY_INIT(name, uint8, bfloat16, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint8, float16, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint8, float32, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint8, float64, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint8, complex32, bool),       \
    CUDA_HOST_BINARY_INIT(name, uint8, complex64, bool),       \
    CUDA_HOST_BINARY_INIT(name, uint8, complex128, bool),      \
                                                               \
    CUDA_HOST_BINARY_INIT(name, uint16, uint8, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint16, uint16, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint16, uint32, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint16, uint64, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint16, int8, bool),           \
    CUDA_HOST_BINARY_INIT(name, uint16, int16, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint16, int32, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint16, int64, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint16, bfloat16, bool),       \
    CUDA_HOST_BINARY_INIT(name, uint16, float16, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint16, float32, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint16, float64, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint16, complex32, bool),      \
    CUDA_HOST_BINARY_INIT(name, uint16, complex64, bool),      \
    CUDA_HOST_BINARY_INIT(name, uint16, complex128, bool),     \
                                                               \
    CUDA_HOST_BINARY_INIT(name, uint32, uint8, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint32, uint16, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint32, uint32, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint32, uint64, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint32, int8, bool),           \
    CUDA_HOST_BINARY_INIT(name, uint32, int16, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint32, int32, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint32, int64, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint32, bfloat16, bool),       \
    CUDA_HOST_BINARY_INIT(name, uint32, float16, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint32, float32, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint32, float64, bool),        \
    CUDA_HOST_BINARY_INIT(name, uint32, complex32, bool),      \
    CUDA_HOST_BINARY_INIT(name, uint32, complex64, bool),      \
    CUDA_HOST_BINARY_INIT(name, uint32, complex128, bool),     \
                                                               \
    CUDA_HOST_BINARY_INIT(name, uint64, uint8, bool),          \
    CUDA_HOST_BINARY_INIT(name, uint64, uint16, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint64, uint32, bool),         \
    CUDA_HOST_BINARY_INIT(name, uint64, uint64, bool),         \
                                                               \
    CUDA_HOST_BINARY_INIT(name, int8, uint8, bool),            \
    CUDA_HOST_BINARY_INIT(name, int8, uint16, bool),           \
    CUDA_HOST_BINARY_INIT(name, int8, uint32, bool),           \
    CUDA_HOST_BINARY_INIT(name, int8, int8, bool),             \
    CUDA_HOST_BINARY_INIT(name, int8, int16, bool),            \
    CUDA_HOST_BINARY_INIT(name, int8, int32, bool),            \
    CUDA_HOST_BINARY_INIT(name, int8, int64, bool),            \
    CUDA_HOST_BINARY_INIT(name, int8, bfloat16, bool),         \
    CUDA_HOST_BINARY_INIT(name, int8, float16, bool),          \
    CUDA_HOST_BINARY_INIT(name, int8, float32, bool),          \
    CUDA_HOST_BINARY_INIT(name, int8, float64, bool),          \
    CUDA_HOST_BINARY_INIT(name, int8, complex32, bool),        \
    CUDA_HOST_BINARY_INIT(name, int8, complex64, bool),        \
    CUDA_HOST_BINARY_INIT(name, int8, complex128, bool),       \
                                                               \
    CUDA_HOST_BINARY_INIT(name, int16, uint8, bool),           \
    CUDA_HOST_BINARY_INIT(name, int16, uint16, bool),          \
    CUDA_HOST_BINARY_INIT(name, int16, uint32, bool),          \
    CUDA_HOST_BINARY_INIT(name, int16, int8, bool),            \
    CUDA_HOST_BINARY_INIT(name, int16, int16, bool),           \
    CUDA_HOST_BINARY_INIT(name, int16, int32, bool),           \
    CUDA_HOST_BINARY_INIT(name, int16, int64, bool),           \
    CUDA_HOST_BINARY_INIT(name, int16, bfloat16, bool),        \
    CUDA_HOST_BINARY_INIT(name, int16, float16, bool),         \
    CUDA_HOST_BINARY_INIT(name, int16, float32, bool),         \
    CUDA_HOST_BINARY_INIT(name, int16, float64, bool),         \
    CUDA_HOST_BINARY_INIT(name, int16, complex32, bool),       \
    CUDA_HOST_BINARY_INIT(name, int16, complex64, bool),       \
    CUDA_HOST_BINARY_INIT(name, int16, complex128, bool),      \
                                                               \
    CUDA_HOST_BINARY_INIT(name, int32, uint8, bool),           \
    CUDA_HOST_BINARY_INIT(name, int32, uint16, bool),          \
    CUDA_HOST_BINARY_INIT(name, int32, uint32, bool),          \
    CUDA_HOST_BINARY_INIT(name, int32, int8, bool),            \
    CUDA_HOST_BINARY_INIT(name, int32, int16, bool),           \
    CUDA_HOST_BINARY_INIT(name, int32, int32, bool),           \
    CUDA_HOST_BINARY_INIT(name, int32, int64, bool),           \
    CUDA_HOST_BINARY_INIT(name, int32, bfloat16, bool),        \
    CUDA_HOST_BINARY_INIT(name, int32, float16, bool),         \
    CUDA_HOST_BINARY_INIT(name, int32, float32, bool),         \
    CUDA_HOST_BINARY_INIT(name, int32, float64, bool),         \
    CUDA_HOST_BINARY_INIT(name, int32, complex32, bool),       \
    CUDA_HOST_BINARY_INIT(name, int32, complex64, bool),       \
    CUDA_HOST_BINARY_INIT(name, int32, complex128, bool),      \
                                                               \
    CUDA_HOST_BINARY_INIT(name, int64, uint8, bool),           \
    CUDA_HOST_BINARY_INIT(name, int64, uint16, bool),          \
    CUDA_HOST_BINARY_INIT(name, int64, uint32, bool),          \
    CUDA_HOST_BINARY_INIT(name, int64, int8, bool),            \
    CUDA_HOST_BINARY_INIT(name, int64, int16, bool),           \
    CUDA_HOST_BINARY_INIT(name, int64, int32, bool),           \
    CUDA_HOST_BINARY_INIT(name, int64, int64, bool),           \
                                                               \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint8, bool),        \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint16, bool),       \
    CUDA_HOST_BINARY_INIT(name, bfloat16, uint32, bool),       \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int8, bool),         \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int16, bool),        \
    CUDA_HOST_BINARY_INIT(name, bfloat16, int32, bool),        \
    CUDA_HOST_BINARY_INIT(name, bfloat16, bfloat16, bool),     \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float16, bool),      \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float32, bool),      \
    CUDA_HOST_BINARY_INIT(name, bfloat16, float64, bool),      \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex32, bool),    \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex64, bool),    \
    CUDA_HOST_BINARY_INIT(name, bfloat16, complex128, bool),   \
                                                               \
    CUDA_HOST_BINARY_INIT(name, float16, uint8, bool),         \
    CUDA_HOST_BINARY_INIT(name, float16, uint16, bool),        \
    CUDA_HOST_BINARY_INIT(name, float16, uint32, bool),        \
    CUDA_HOST_BINARY_INIT(name, float16, int8, bool),          \
    CUDA_HOST_BINARY_INIT(name, float16, int16, bool),         \
    CUDA_HOST_BINARY_INIT(name, float16, int32, bool),         \
    CUDA_HOST_BINARY_INIT(name, float16, bfloat16, bool),      \
    CUDA_HOST_BINARY_INIT(name, float16, float16, bool),       \
    CUDA_HOST_BINARY_INIT(name, float16, float32, bool),       \
    CUDA_HOST_BINARY_INIT(name, float16, float64, bool),       \
    CUDA_HOST_BINARY_INIT(name, float16, complex32, bool),     \
    CUDA_HOST_BINARY_INIT(name, float16, complex64, bool),     \
    CUDA_HOST_BINARY_INIT(name, float16, complex128, bool),    \
                                                               \
    CUDA_HOST_BINARY_INIT(name, float32, uint8, bool),         \
    CUDA_HOST_BINARY_INIT(name, float32, uint16, bool),        \
    CUDA_HOST_BINARY_INIT(name, float32, uint32, bool),        \
    CUDA_HOST_BINARY_INIT(name, float32, int8, bool),          \
    CUDA_HOST_BINARY_INIT(name, float32, int16, bool),         \
    CUDA_HOST_BINARY_INIT(name, float32, int32, bool),         \
    CUDA_HOST_BINARY_INIT(name, float32, bfloat16, bool),      \
    CUDA_HOST_BINARY_INIT(name, float32, float16, bool),       \
    CUDA_HOST_BINARY_INIT(name, float32, float32, bool),       \
    CUDA_HOST_BINARY_INIT(name, float32, float64, bool),       \
    CUDA_HOST_BINARY_INIT(name, float32, complex32, bool),     \
    CUDA_HOST_BINARY_INIT(name, float32, complex64, bool),     \
    CUDA_HOST_BINARY_INIT(name, float32, complex128, bool),    \
                                                               \
    CUDA_HOST_BINARY_INIT(name, float64, uint8, bool),         \
    CUDA_HOST_BINARY_INIT(name, float64, uint16, bool),        \
    CUDA_HOST_BINARY_INIT(name, float64, uint32, bool),        \
    CUDA_HOST_BINARY_INIT(name, float64, int8, bool),          \
    CUDA_HOST_BINARY_INIT(name, float64, int16, bool),         \
    CUDA_HOST_BINARY_INIT(name, float64, int32, bool),         \
    CUDA_HOST_BINARY_INIT(name, float64, bfloat16, bool),      \
    CUDA_HOST_BINARY_INIT(name, float64, float16, bool),       \
    CUDA_HOST_BINARY_INIT(name, float64, float32, bool),       \
    CUDA_HOST_BINARY_INIT(name, float64, float64, bool),       \
    CUDA_HOST_BINARY_INIT(name, float64, complex32, bool),     \
    CUDA_HOST_BINARY_INIT(name, float64, complex64, bool),     \
    CUDA_HOST_BINARY_INIT(name, float64, complex128, bool),    \
                                                               \
    CUDA_HOST_BINARY_INIT(name, complex32, uint8, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex32, uint16, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex32, uint32, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex32, int8, bool),        \
    CUDA_HOST_BINARY_INIT(name, complex32, int16, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex32, int32, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex32, bfloat16, bool),    \
    CUDA_HOST_BINARY_INIT(name, complex32, float16, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex32, float32, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex32, float64, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex32, complex32, bool),   \
    CUDA_HOST_BINARY_INIT(name, complex32, complex64, bool),   \
    CUDA_HOST_BINARY_INIT(name, complex32, complex128, bool),  \
                                                               \
    CUDA_HOST_BINARY_INIT(name, complex64, uint8, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex64, uint16, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex64, uint32, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex64, int8, bool),        \
    CUDA_HOST_BINARY_INIT(name, complex64, int16, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex64, int32, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex64, bfloat16, bool),    \
    CUDA_HOST_BINARY_INIT(name, complex64, float16, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex64, float32, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex64, float64, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex64, complex32, bool),   \
    CUDA_HOST_BINARY_INIT(name, complex64, complex64, bool),   \
    CUDA_HOST_BINARY_INIT(name, complex64, complex128, bool),  \
                                                               \
    CUDA_HOST_BINARY_INIT(name, complex128, uint8, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex128, uint16, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex128, uint32, bool),     \
    CUDA_HOST_BINARY_INIT(name, complex128, int8, bool),       \
    CUDA_HOST_BINARY_INIT(name, complex128, int16, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex128, int32, bool),      \
    CUDA_HOST_BINARY_INIT(name, complex128, bfloat16, bool),   \
    CUDA_HOST_BINARY_INIT(name, complex128, float16, bool),    \
    CUDA_HOST_BINARY_INIT(name, complex128, float32, bool),    \
    CUDA_HOST_BINARY_INIT(name, complex128, float64, bool),    \
    CUDA_HOST_BINARY_INIT(name, complex128, complex32, bool),  \
    CUDA_HOST_BINARY_INIT(name, complex128, complex64, bool),  \
    CUDA_HOST_BINARY_INIT(name, complex128, complex128, bool)


#undef bool
#define bool_t _Bool

#define less(x, y) x < y
CUDA_HOST_ALL_COMPARISON(less)

#define less_equal(x, y) x <= y
CUDA_HOST_ALL_COMPARISON(less_equal)

#define greater_equal(x, y) x >= y
CUDA_HOST_ALL_COMPARISON(greater_equal)

#define greater(x, y) x > y
CUDA_HOST_ALL_COMPARISON(greater)


static const gm_kernel_init_t binary_kernels[] = {
  CUDA_HOST_ALL_ARITHMETIC_INIT(add),
  CUDA_HOST_ALL_ARITHMETIC_INIT(subtract),
  CUDA_HOST_ALL_ARITHMETIC_INIT(multiply),
  CUDA_HOST_ALL_ARITHMETIC_INIT(floor_divide),
  CUDA_HOST_ALL_ARITHMETIC_INIT(remainder),
  CUDA_HOST_ALL_ARITHMETIC_FLOAT_RETURN_INIT(divide),
  CUDA_HOST_ALL_COMPARISON_INIT(less),
  CUDA_HOST_ALL_COMPARISON_INIT(less_equal),
  CUDA_HOST_ALL_COMPARISON_INIT(greater_equal),
  CUDA_HOST_ALL_COMPARISON_INIT(greater),

  { .name = NULL, .sig = NULL }
};


/*****************************************************************************/
/*                                   Bitwise                                 */
/*****************************************************************************/

#define CUDA_HOST_ALL_BITWISE(name) \
    CUDA_HOST_BINARY(name, bool, bool, bool)       \
    CUDA_HOST_BINARY(name, bool, uint8, uint8)     \
    CUDA_HOST_BINARY(name, bool, uint16, uint16)   \
    CUDA_HOST_BINARY(name, bool, uint32, uint32)   \
    CUDA_HOST_BINARY(name, bool, uint64, uint64)   \
    CUDA_HOST_BINARY(name, bool, int8, int8)       \
    CUDA_HOST_BINARY(name, bool, int16, int16)     \
    CUDA_HOST_BINARY(name, bool, int32, int32)     \
    CUDA_HOST_BINARY(name, bool, int64, int64)     \
                                                  \
    CUDA_HOST_BINARY(name, uint8, bool, uint8)     \
    CUDA_HOST_BINARY(name, uint8, uint8, uint8)    \
    CUDA_HOST_BINARY(name, uint8, uint16, uint16)  \
    CUDA_HOST_BINARY(name, uint8, uint32, uint32)  \
    CUDA_HOST_BINARY(name, uint8, uint64, uint64)  \
    CUDA_HOST_BINARY(name, uint8, int8, int16)     \
    CUDA_HOST_BINARY(name, uint8, int16, int16)    \
    CUDA_HOST_BINARY(name, uint8, int32, int32)    \
    CUDA_HOST_BINARY(name, uint8, int64, int64)    \
                                                  \
    CUDA_HOST_BINARY(name, uint16, bool, uint16)   \
    CUDA_HOST_BINARY(name, uint16, uint8, uint16)  \
    CUDA_HOST_BINARY(name, uint16, uint16, uint16) \
    CUDA_HOST_BINARY(name, uint16, uint32, uint32) \
    CUDA_HOST_BINARY(name, uint16, uint64, uint64) \
    CUDA_HOST_BINARY(name, uint16, int8, int32)    \
    CUDA_HOST_BINARY(name, uint16, int16, int32)   \
    CUDA_HOST_BINARY(name, uint16, int32, int32)   \
    CUDA_HOST_BINARY(name, uint16, int64, int64)   \
                                                  \
    CUDA_HOST_BINARY(name, uint32, bool, uint32)   \
    CUDA_HOST_BINARY(name, uint32, uint8, uint32)  \
    CUDA_HOST_BINARY(name, uint32, uint16, uint32) \
    CUDA_HOST_BINARY(name, uint32, uint32, uint32) \
    CUDA_HOST_BINARY(name, uint32, uint64, uint64) \
    CUDA_HOST_BINARY(name, uint32, int8, int64)    \
    CUDA_HOST_BINARY(name, uint32, int16, int64)   \
    CUDA_HOST_BINARY(name, uint32, int32, int64)   \
    CUDA_HOST_BINARY(name, uint32, int64, int64)   \
                                                  \
    CUDA_HOST_BINARY(name, uint64, bool, uint64)   \
    CUDA_HOST_BINARY(name, uint64, uint8, uint64)  \
    CUDA_HOST_BINARY(name, uint64, uint16, uint64) \
    CUDA_HOST_BINARY(name, uint64, uint32, uint64) \
    CUDA_HOST_BINARY(name, uint64, uint64, uint64) \
                                                  \
    CUDA_HOST_BINARY(name, int8, bool, int8)       \
    CUDA_HOST_BINARY(name, int8, uint8, int16)     \
    CUDA_HOST_BINARY(name, int8, uint16, int32)    \
    CUDA_HOST_BINARY(name, int8, uint32, int64)    \
    CUDA_HOST_BINARY(name, int8, int8, int8)       \
    CUDA_HOST_BINARY(name, int8, int16, int16)     \
    CUDA_HOST_BINARY(name, int8, int32, int32)     \
    CUDA_HOST_BINARY(name, int8, int64, int64)     \
                                                  \
    CUDA_HOST_BINARY(name, int16, bool, int16)     \
    CUDA_HOST_BINARY(name, int16, uint8, int16)    \
    CUDA_HOST_BINARY(name, int16, uint16, int32)   \
    CUDA_HOST_BINARY(name, int16, uint32, int64)   \
    CUDA_HOST_BINARY(name, int16, int8, int16)     \
    CUDA_HOST_BINARY(name, int16, int16, int16)    \
    CUDA_HOST_BINARY(name, int16, int32, int32)    \
    CUDA_HOST_BINARY(name, int16, int64, int64)    \
                                                  \
    CUDA_HOST_BINARY(name, int32, bool, int32)     \
    CUDA_HOST_BINARY(name, int32, uint8, int32)    \
    CUDA_HOST_BINARY(name, int32, uint16, int32)   \
    CUDA_HOST_BINARY(name, int32, uint32, int64)   \
    CUDA_HOST_BINARY(name, int32, int8, int32)     \
    CUDA_HOST_BINARY(name, int32, int16, int32)    \
    CUDA_HOST_BINARY(name, int32, int32, int32)    \
    CUDA_HOST_BINARY(name, int32, int64, int64)    \
                                                  \
    CUDA_HOST_BINARY(name, int64, bool, int64)     \
    CUDA_HOST_BINARY(name, int64, uint8, int64)    \
    CUDA_HOST_BINARY(name, int64, uint16, int64)   \
    CUDA_HOST_BINARY(name, int64, uint32, int64)   \
    CUDA_HOST_BINARY(name, int64, int8, int64)     \
    CUDA_HOST_BINARY(name, int64, int16, int64)    \
    CUDA_HOST_BINARY(name, int64, int32, int64)    \
    CUDA_HOST_BINARY(name, int64, int64, int64)

#define CUDA_HOST_ALL_BITWISE_INIT(name) \
    CUDA_HOST_BINARY_INIT(name, bool, bool, bool),       \
    CUDA_HOST_BINARY_INIT(name, bool, uint8, uint8),     \
    CUDA_HOST_BINARY_INIT(name, bool, uint16, uint16),   \
    CUDA_HOST_BINARY_INIT(name, bool, uint32, uint32),   \
    CUDA_HOST_BINARY_INIT(name, bool, uint64, uint64),   \
    CUDA_HOST_BINARY_INIT(name, bool, int8, int8),       \
    CUDA_HOST_BINARY_INIT(name, bool, int16, int16),     \
    CUDA_HOST_BINARY_INIT(name, bool, int32, int32),     \
    CUDA_HOST_BINARY_INIT(name, bool, int64, int64),     \
                                                        \
    CUDA_HOST_BINARY_INIT(name, uint8, bool, uint8),     \
    CUDA_HOST_BINARY_INIT(name, uint8, uint8, uint8),    \
    CUDA_HOST_BINARY_INIT(name, uint8, uint16, uint16),  \
    CUDA_HOST_BINARY_INIT(name, uint8, uint32, uint32),  \
    CUDA_HOST_BINARY_INIT(name, uint8, uint64, uint64),  \
    CUDA_HOST_BINARY_INIT(name, uint8, int8, int16),     \
    CUDA_HOST_BINARY_INIT(name, uint8, int16, int16),    \
    CUDA_HOST_BINARY_INIT(name, uint8, int32, int32),    \
    CUDA_HOST_BINARY_INIT(name, uint8, int64, int64),    \
                                                        \
    CUDA_HOST_BINARY_INIT(name, uint16, bool, uint16),   \
    CUDA_HOST_BINARY_INIT(name, uint16, uint8, uint16),  \
    CUDA_HOST_BINARY_INIT(name, uint16, uint16, uint16), \
    CUDA_HOST_BINARY_INIT(name, uint16, uint32, uint32), \
    CUDA_HOST_BINARY_INIT(name, uint16, uint64, uint64), \
    CUDA_HOST_BINARY_INIT(name, uint16, int8, int32),    \
    CUDA_HOST_BINARY_INIT(name, uint16, int16, int32),   \
    CUDA_HOST_BINARY_INIT(name, uint16, int32, int32),   \
    CUDA_HOST_BINARY_INIT(name, uint16, int64, int64),   \
                                                        \
    CUDA_HOST_BINARY_INIT(name, uint32, bool, uint32),   \
    CUDA_HOST_BINARY_INIT(name, uint32, uint8, uint32),  \
    CUDA_HOST_BINARY_INIT(name, uint32, uint16, uint32), \
    CUDA_HOST_BINARY_INIT(name, uint32, uint32, uint32), \
    CUDA_HOST_BINARY_INIT(name, uint32, uint64, uint64), \
    CUDA_HOST_BINARY_INIT(name, uint32, int8, int64),    \
    CUDA_HOST_BINARY_INIT(name, uint32, int16, int64),   \
    CUDA_HOST_BINARY_INIT(name, uint32, int32, int64),   \
    CUDA_HOST_BINARY_INIT(name, uint32, int64, int64),   \
                                                        \
    CUDA_HOST_BINARY_INIT(name, uint64, bool, uint64),   \
    CUDA_HOST_BINARY_INIT(name, uint64, uint8, uint64),  \
    CUDA_HOST_BINARY_INIT(name, uint64, uint16, uint64), \
    CUDA_HOST_BINARY_INIT(name, uint64, uint32, uint64), \
    CUDA_HOST_BINARY_INIT(name, uint64, uint64, uint64), \
                                                        \
    CUDA_HOST_BINARY_INIT(name, int8, bool, int8),       \
    CUDA_HOST_BINARY_INIT(name, int8, uint8, int16),     \
    CUDA_HOST_BINARY_INIT(name, int8, uint16, int32),    \
    CUDA_HOST_BINARY_INIT(name, int8, uint32, int64),    \
    CUDA_HOST_BINARY_INIT(name, int8, int8, int8),       \
    CUDA_HOST_BINARY_INIT(name, int8, int16, int16),     \
    CUDA_HOST_BINARY_INIT(name, int8, int32, int32),     \
    CUDA_HOST_BINARY_INIT(name, int8, int64, int64),     \
                                                        \
    CUDA_HOST_BINARY_INIT(name, int16, bool, int16),     \
    CUDA_HOST_BINARY_INIT(name, int16, uint8, int16),    \
    CUDA_HOST_BINARY_INIT(name, int16, uint16, int32),   \
    CUDA_HOST_BINARY_INIT(name, int16, uint32, int64),   \
    CUDA_HOST_BINARY_INIT(name, int16, int8, int16),     \
    CUDA_HOST_BINARY_INIT(name, int16, int16, int16),    \
    CUDA_HOST_BINARY_INIT(name, int16, int32, int32),    \
    CUDA_HOST_BINARY_INIT(name, int16, int64, int64),    \
                                                        \
    CUDA_HOST_BINARY_INIT(name, int32, bool, int32),     \
    CUDA_HOST_BINARY_INIT(name, int32, uint8, int32),    \
    CUDA_HOST_BINARY_INIT(name, int32, uint16, int32),   \
    CUDA_HOST_BINARY_INIT(name, int32, uint32, int64),   \
    CUDA_HOST_BINARY_INIT(name, int32, int8, int32),     \
    CUDA_HOST_BINARY_INIT(name, int32, int16, int32),    \
    CUDA_HOST_BINARY_INIT(name, int32, int32, int32),    \
    CUDA_HOST_BINARY_INIT(name, int32, int64, int64),    \
                                                        \
    CUDA_HOST_BINARY_INIT(name, int64, bool, int64),     \
    CUDA_HOST_BINARY_INIT(name, int64, uint8, int64),    \
    CUDA_HOST_BINARY_INIT(name, int64, uint16, int64),   \
    CUDA_HOST_BINARY_INIT(name, int64, uint32, int64),   \
    CUDA_HOST_BINARY_INIT(name, int64, int8, int64),     \
    CUDA_HOST_BINARY_INIT(name, int64, int16, int64),    \
    CUDA_HOST_BINARY_INIT(name, int64, int32, int64),    \
    CUDA_HOST_BINARY_INIT(name, int64, int64, int64)


CUDA_HOST_ALL_BITWISE(bitwise_and)
CUDA_HOST_ALL_BITWISE(bitwise_or)
CUDA_HOST_ALL_BITWISE(bitwise_xor)


static const gm_kernel_init_t bitwise_kernels[] = {
  CUDA_HOST_ALL_BITWISE_INIT(bitwise_and),
  CUDA_HOST_ALL_BITWISE_INIT(bitwise_or),
  CUDA_HOST_ALL_BITWISE_INIT(bitwise_xor),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                               Two return values                          */
/****************************************************************************/

#define CUDA_HOST_BINARY_MV(name, t0, t1, t2, t3) \
static int                                                                          \
gm_fixed_1D_C_##name##_##t0##_##t1##_##t2##_##t3(xnd_t stack[], ndt_context_t *ctx) \
{                                                                                   \
    const char *in0 = apply_index(&stack[0]);                                       \
    const char *in1 = apply_index(&stack[1]);                                       \
    char *out0 = apply_index(&stack[2]);                                            \
    char *out1 = apply_index(&stack[3]);                                            \
    int64_t N = xnd_fixed_shape(&stack[0]);                                         \
    enum cuda_binary tag = get_step_tag(stack[0].type, stack[1].type);              \
    (void)ctx;                                                                      \
                                                                                    \
    gm_cuda_device_fixed_1D_C_##name##_##t0##_##t1##_##t2##_##t3(                   \
        in0, in1, out0, out1, N, tag);                                              \
                                                                                    \
    return 0;                                                                       \
}

#define CUDA_HOST_BINARY_MV_INIT(func, t0, t1, t2, t3) \
  { .name = STRINGIZE(func),                                       \
    .sig = "... * " STRINGIZE(t0) ", ... * " STRINGIZE(t1) " -> "  \
           "... * " STRINGIZE(t2) ", ... * " STRINGIZE(t3),        \
    .Opt = gm_fixed_1D_C_##func##_##t0##_##t1##_##t2##_##t3,       \
    .C = NULL }

#define CUDA_HOST_ALL_BINARY_MV(name) \
    CUDA_HOST_BINARY_MV(name, uint8, uint8, uint8, uint8)             \
    CUDA_HOST_BINARY_MV(name, uint16, uint16, uint16, uint16)         \
    CUDA_HOST_BINARY_MV(name, uint32, uint32, uint32, uint32)         \
    CUDA_HOST_BINARY_MV(name, uint64, uint64, uint64, uint64)         \
    CUDA_HOST_BINARY_MV(name, int8, int8, int8, int8)                 \
    CUDA_HOST_BINARY_MV(name, int16, int16, int16, int16)             \
    CUDA_HOST_BINARY_MV(name, int32, int32, int32, int32)             \
    CUDA_HOST_BINARY_MV(name, int64, int64, int64, int64)             \
    CUDA_HOST_BINARY_MV(name, bfloat16, bfloat16, bfloat16, bfloat16) \
    CUDA_HOST_BINARY_MV(name, float32, float32, float32, float32)     \
    CUDA_HOST_BINARY_MV(name, float64, float64, float64, float64)

#define CUDA_HOST_ALL_BINARY_MV_INIT(name) \
    CUDA_HOST_BINARY_MV_INIT(name, uint8, uint8, uint8, uint8),             \
    CUDA_HOST_BINARY_MV_INIT(name, uint16, uint16, uint16, uint16),         \
    CUDA_HOST_BINARY_MV_INIT(name, uint32, uint32, uint32, uint32),         \
    CUDA_HOST_BINARY_MV_INIT(name, uint64, uint64, uint64, uint64),         \
    CUDA_HOST_BINARY_MV_INIT(name, int8, int8, int8, int8),                 \
    CUDA_HOST_BINARY_MV_INIT(name, int16, int16, int16, int16),             \
    CUDA_HOST_BINARY_MV_INIT(name, int32, int32, int32, int32),             \
    CUDA_HOST_BINARY_MV_INIT(name, int64, int64, int64, int64),             \
    CUDA_HOST_BINARY_MV_INIT(name, bfloat16, bfloat16, bfloat16, bfloat16), \
    CUDA_HOST_BINARY_MV_INIT(name, float32, float32, float32, float32),     \
    CUDA_HOST_BINARY_MV_INIT(name, float64, float64, float64, float64)

CUDA_HOST_ALL_BINARY_MV(divmod)


static const gm_kernel_init_t binary_mv_kernels[] = {
  CUDA_HOST_ALL_BINARY_MV_INIT(divmod),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

static const gm_kernel_set_t *
binary_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                 const int64_t li[], int nin, int nout, ndt_context_t *ctx)
{
    return cuda_binary_typecheck(binary_kernel_location, spec, f, types, li,
                                 nin, nout, ctx);
}

static const gm_kernel_set_t *
bitwise_typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *in[],
          const int64_t li[], int nin, int nout, ndt_context_t *ctx)
{
    return cuda_binary_typecheck(bitwise_kernel_location, spec, f, in, li,
                                 nin, nout, ctx);
}


int
gm_init_cuda_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
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

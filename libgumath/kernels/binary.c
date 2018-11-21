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

        case Float32: return 64;
        case Float64: return 72;

        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Uint8: return 80;
        case Uint16: return 88;
        case Uint32: return 96;
        case Uint64: return 104;

        case Int8: return 112;
        case Int16: return 120;
        case Int32: return 128;
        case Int64: return 136;

        case Float32: return 144;
        case Float64: return 152;

        default: goto invalid_combination;
        }
    }
    case Uint32: {
        switch (t1->tag) {
        case Uint8: return 160;
        case Uint16: return 168;
        case Uint32: return 176;
        case Uint64: return 184;

        case Int8: return 192;
        case Int16: return 200;
        case Int32: return 208;
        case Int64: return 216;

        case Float32: return 224;
        case Float64: return 232;

        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Uint8: return 240;
        case Uint16: return 248;
        case Uint32: return 256;
        case Uint64: return 264;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Uint8: return 272;
        case Uint16: return 280;
        case Uint32: return 288;

        case Int8: return 296;
        case Int16: return 304;
        case Int32: return 312;
        case Int64: return 320;

        case Float32: return 328;
        case Float64: return 336;

        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Uint8: return 344;
        case Uint16: return 352;
        case Uint32: return 360;

        case Int8: return 368;
        case Int16: return 376;
        case Int32: return 384;
        case Int64: return 392;

        case Float32: return 400;
        case Float64: return 408;

        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Uint8: return 416;
        case Uint16: return 424;
        case Uint32: return 432;

        case Int8: return 440;
        case Int16: return 448;
        case Int32: return 456;
        case Int64: return 464;

        case Float32: return 472;
        case Float64: return 480;

        default: goto invalid_combination;
        }
    }
    case Int64: {
        switch (t1->tag) {
        case Uint8: return 488;
        case Uint16: return 496;
        case Uint32: return 504;

        case Int8: return 512;
        case Int16: return 520;
        case Int32: return 528;
        case Int64: return 536;

        default: goto invalid_combination;
        }
    }

    case Float32: {
        switch (t1->tag) {
        case Uint8: return 544;
        case Uint16: return 552;
        case Uint32: return 560;

        case Int8: return 568;
        case Int16: return 576;
        case Int32: return 584;

        case Float32: return 592;
        case Float64: return 600;

        default: goto invalid_combination;
        }
    }
    case Float64: {
        switch (t1->tag) {
        case Uint8: return 608;
        case Uint16: return 616;
        case Uint32: return 624;

        case Int8: return 632;
        case Int16: return 640;
        case Int32: return 648;

        case Float32: return 656;
        case Float64: return 664;

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


/****************************************************************************/
/*                                 Arithmetic                               */
/****************************************************************************/

#define XND_ALL_BINARY(name) \
    XND_BINARY(name, uint8, uint8, uint8, uint8)         \
    XND_BINARY(name, uint8, uint16, uint16, uint16)      \
    XND_BINARY(name, uint8, uint32, uint32, uint32)      \
    XND_BINARY(name, uint8, uint64, uint64, uint64)      \
    XND_BINARY(name, uint8, int8, int16, int16)          \
    XND_BINARY(name, uint8, int16, int16, int16)         \
    XND_BINARY(name, uint8, int32, int32, int32)         \
    XND_BINARY(name, uint8, int64, int64, int64)         \
    XND_BINARY(name, uint8, float32, float32, float32)   \
    XND_BINARY(name, uint8, float64, float64, float64)   \
                                                         \
    XND_BINARY(name, uint16, uint8, uint16, uint16)      \
    XND_BINARY(name, uint16, uint16, uint16, uint16)     \
    XND_BINARY(name, uint16, uint32, uint32, uint32)     \
    XND_BINARY(name, uint16, uint64, uint64, uint64)     \
    XND_BINARY(name, uint16, int8, int32, int32)         \
    XND_BINARY(name, uint16, int16, int32, int32)        \
    XND_BINARY(name, uint16, int32, int32, int32)        \
    XND_BINARY(name, uint16, int64, int64, int64)        \
    XND_BINARY(name, uint16, float32, float32, float32)  \
    XND_BINARY(name, uint16, float64, float64, float64)  \
                                                         \
    XND_BINARY(name, uint32, uint8, uint32, uint32)      \
    XND_BINARY(name, uint32, uint16, uint32, uint32)     \
    XND_BINARY(name, uint32, uint32, uint32, uint32)     \
    XND_BINARY(name, uint32, uint64, uint64, uint64)     \
    XND_BINARY(name, uint32, int8, int64, int64)         \
    XND_BINARY(name, uint32, int16, int64, int64)        \
    XND_BINARY(name, uint32, int32, int64, int64)        \
    XND_BINARY(name, uint32, int64, int64, int64)        \
    XND_BINARY(name, uint32, float32, float64, float64)  \
    XND_BINARY(name, uint32, float64, float64, float64)  \
                                                         \
    XND_BINARY(name, uint64, uint8, uint64, uint64)      \
    XND_BINARY(name, uint64, uint16, uint64, uint64)     \
    XND_BINARY(name, uint64, uint32, uint64, uint64)     \
    XND_BINARY(name, uint64, uint64, uint64, uint64)     \
                                                         \
    XND_BINARY(name, int8, uint8, int16, int16)          \
    XND_BINARY(name, int8, uint16, int32, int32)         \
    XND_BINARY(name, int8, uint32, int64, int64)         \
    XND_BINARY(name, int8, int8, int8, int8)             \
    XND_BINARY(name, int8, int16, int16, int16)          \
    XND_BINARY(name, int8, int32, int32, int32)          \
    XND_BINARY(name, int8, int64, int64, int64)          \
    XND_BINARY(name, int8, float32, float32, float32)    \
    XND_BINARY(name, int8, float64, float64, float64)    \
                                                         \
    XND_BINARY(name, int16, uint8, int16, int16)         \
    XND_BINARY(name, int16, uint16, int32, int32)        \
    XND_BINARY(name, int16, uint32, int64, int64)        \
    XND_BINARY(name, int16, int8, int16, int16)          \
    XND_BINARY(name, int16, int16, int16, int16)         \
    XND_BINARY(name, int16, int32, int32, int32)         \
    XND_BINARY(name, int16, int64, int64, int64)         \
    XND_BINARY(name, int16, float32, float32, float32)   \
    XND_BINARY(name, int16, float64, float64, float64)   \
                                                         \
    XND_BINARY(name, int32, uint8, int32, int32)         \
    XND_BINARY(name, int32, uint16, int32, int32)        \
    XND_BINARY(name, int32, uint32, int64, int64)        \
    XND_BINARY(name, int32, int8, int32, int32)          \
    XND_BINARY(name, int32, int16, int32, int32)         \
    XND_BINARY(name, int32, int32, int32, int32)         \
    XND_BINARY(name, int32, int64, int64, int64)         \
    XND_BINARY(name, int32, float32, float64, float64)   \
    XND_BINARY(name, int32, float64, float64, float64)   \
                                                         \
    XND_BINARY(name, int64, uint8, int64, int64)         \
    XND_BINARY(name, int64, uint16, int64, int64)        \
    XND_BINARY(name, int64, uint32, int64, int64)        \
    XND_BINARY(name, int64, int8, int64, int64)          \
    XND_BINARY(name, int64, int16, int64, int64)         \
    XND_BINARY(name, int64, int32, int64, int64)         \
    XND_BINARY(name, int64, int64, int64, int64)         \
                                                         \
    XND_BINARY(name, float32, uint8, float32, float32)   \
    XND_BINARY(name, float32, uint16, float32, float32)  \
    XND_BINARY(name, float32, uint32, float64, float64)  \
    XND_BINARY(name, float32, int8, float32, float32)    \
    XND_BINARY(name, float32, int16, float32, float32)   \
    XND_BINARY(name, float32, int32, float64, float64)   \
    XND_BINARY(name, float32, float32, float32, float32) \
    XND_BINARY(name, float32, float64, float64, float64) \
                                                         \
    XND_BINARY(name, float64, uint8, float64, float64)   \
    XND_BINARY(name, float64, uint16, float64, float64)  \
    XND_BINARY(name, float64, uint32, float64, float64)  \
    XND_BINARY(name, float64, int8, float64, float64)    \
    XND_BINARY(name, float64, int16, float64, float64)   \
    XND_BINARY(name, float64, int32, float64, float64)   \
    XND_BINARY(name, float64, float32, float64, float64) \
    XND_BINARY(name, float64, float64, float64, float64)

#define XND_ALL_BINARY_INIT(name) \
    XND_BINARY_INIT(name, uint8, uint8, uint8),       \
    XND_BINARY_INIT(name, uint8, uint16, uint16),     \
    XND_BINARY_INIT(name, uint8, uint32, uint32),     \
    XND_BINARY_INIT(name, uint8, uint64, uint64),     \
    XND_BINARY_INIT(name, uint8, int8, int16),        \
    XND_BINARY_INIT(name, uint8, int16, int16),       \
    XND_BINARY_INIT(name, uint8, int32, int32),       \
    XND_BINARY_INIT(name, uint8, int64, int64),       \
    XND_BINARY_INIT(name, uint8, float32, float32),   \
    XND_BINARY_INIT(name, uint8, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, uint16, uint8, uint16),     \
    XND_BINARY_INIT(name, uint16, uint16, uint16),    \
    XND_BINARY_INIT(name, uint16, uint32, uint32),    \
    XND_BINARY_INIT(name, uint16, uint64, uint64),    \
    XND_BINARY_INIT(name, uint16, int8, int32),       \
    XND_BINARY_INIT(name, uint16, int16, int32),      \
    XND_BINARY_INIT(name, uint16, int32, int32),      \
    XND_BINARY_INIT(name, uint16, int64, int64),      \
    XND_BINARY_INIT(name, uint16, float32, float32),  \
    XND_BINARY_INIT(name, uint16, float64, float64),  \
                                                      \
    XND_BINARY_INIT(name, uint32, uint8, uint32),     \
    XND_BINARY_INIT(name, uint32, uint16, uint32),    \
    XND_BINARY_INIT(name, uint32, uint32, uint32),    \
    XND_BINARY_INIT(name, uint32, uint64, uint64),    \
    XND_BINARY_INIT(name, uint32, int8, int64),       \
    XND_BINARY_INIT(name, uint32, int16, int64),      \
    XND_BINARY_INIT(name, uint32, int32, int64),      \
    XND_BINARY_INIT(name, uint32, int64, int64),      \
    XND_BINARY_INIT(name, uint32, float32, float64),  \
    XND_BINARY_INIT(name, uint32, float64, float64),  \
                                                      \
    XND_BINARY_INIT(name, uint64, uint8, uint64),     \
    XND_BINARY_INIT(name, uint64, uint16, uint64),    \
    XND_BINARY_INIT(name, uint64, uint32, uint64),    \
    XND_BINARY_INIT(name, uint64, uint64, uint64),    \
                                                      \
    XND_BINARY_INIT(name, int8, uint8, int16),        \
    XND_BINARY_INIT(name, int8, uint16, int32),       \
    XND_BINARY_INIT(name, int8, uint32, int64),       \
    XND_BINARY_INIT(name, int8, int8, int8),          \
    XND_BINARY_INIT(name, int8, int16, int16),        \
    XND_BINARY_INIT(name, int8, int32, int32),        \
    XND_BINARY_INIT(name, int8, int64, int64),        \
    XND_BINARY_INIT(name, int8, float32, float32),    \
    XND_BINARY_INIT(name, int8, float64, float64),    \
                                                      \
    XND_BINARY_INIT(name, int16, uint8, int16),       \
    XND_BINARY_INIT(name, int16, uint16, int32),      \
    XND_BINARY_INIT(name, int16, uint32, int64),      \
    XND_BINARY_INIT(name, int16, int8, int16),        \
    XND_BINARY_INIT(name, int16, int16, int16),       \
    XND_BINARY_INIT(name, int16, int32, int32),       \
    XND_BINARY_INIT(name, int16, int64, int64),       \
    XND_BINARY_INIT(name, int16, float32, float32),   \
    XND_BINARY_INIT(name, int16, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, int32, uint8, int32),       \
    XND_BINARY_INIT(name, int32, uint16, int32),      \
    XND_BINARY_INIT(name, int32, uint32, int64),      \
    XND_BINARY_INIT(name, int32, int8, int32),        \
    XND_BINARY_INIT(name, int32, int16, int32),       \
    XND_BINARY_INIT(name, int32, int32, int32),       \
    XND_BINARY_INIT(name, int32, int64, int64),       \
    XND_BINARY_INIT(name, int32, float32, float64),   \
    XND_BINARY_INIT(name, int32, float64, float64),   \
                                                      \
    XND_BINARY_INIT(name, int64, uint8, int64),       \
    XND_BINARY_INIT(name, int64, uint16, int64),      \
    XND_BINARY_INIT(name, int64, uint32, int64),      \
    XND_BINARY_INIT(name, int64, int8, int64),        \
    XND_BINARY_INIT(name, int64, int16, int64),       \
    XND_BINARY_INIT(name, int64, int32, int64),       \
    XND_BINARY_INIT(name, int64, int64, int64),       \
                                                      \
    XND_BINARY_INIT(name, float32, uint8, float32),   \
    XND_BINARY_INIT(name, float32, uint16, float32),  \
    XND_BINARY_INIT(name, float32, uint32, float64),  \
    XND_BINARY_INIT(name, float32, int8, float32),    \
    XND_BINARY_INIT(name, float32, int16, float32),   \
    XND_BINARY_INIT(name, float32, int32, float64),   \
    XND_BINARY_INIT(name, float32, float32, float32), \
    XND_BINARY_INIT(name, float32, float64, float64), \
                                                      \
    XND_BINARY_INIT(name, float64, uint8, float64),   \
    XND_BINARY_INIT(name, float64, uint16, float64),  \
    XND_BINARY_INIT(name, float64, uint32, float64),  \
    XND_BINARY_INIT(name, float64, int8, float64),    \
    XND_BINARY_INIT(name, float64, int16, float64),   \
    XND_BINARY_INIT(name, float64, int32, float64),   \
    XND_BINARY_INIT(name, float64, float32, float64), \
    XND_BINARY_INIT(name, float64, float64, float64)


#define add(x, y) x + y
XND_ALL_BINARY(add)

#define subtract(x, y) x - y
XND_ALL_BINARY(subtract)

#define multiply(x, y) x * y
XND_ALL_BINARY(multiply)

#define divide(x, y) x / y
XND_ALL_BINARY(divide)


/****************************************************************************/
/*                                Comparison                                */
/****************************************************************************/

#define XND_ALL_COMPARISON(name) \
    XND_BINARY(name, uint8, uint8, bool, uint8)       \
    XND_BINARY(name, uint8, uint16, bool, uint16)     \
    XND_BINARY(name, uint8, uint32, bool, uint32)     \
    XND_BINARY(name, uint8, uint64, bool, uint64)     \
    XND_BINARY(name, uint8, int8, bool, int16)        \
    XND_BINARY(name, uint8, int16, bool, int16)       \
    XND_BINARY(name, uint8, int32, bool, int32)       \
    XND_BINARY(name, uint8, int64, bool, int64)       \
    XND_BINARY(name, uint8, float32, bool, float32)   \
    XND_BINARY(name, uint8, float64, bool, float64)   \
                                                      \
    XND_BINARY(name, uint16, uint8, bool, uint16)     \
    XND_BINARY(name, uint16, uint16, bool, uint16)    \
    XND_BINARY(name, uint16, uint32, bool, uint32)    \
    XND_BINARY(name, uint16, uint64, bool, uint64)    \
    XND_BINARY(name, uint16, int8, bool, int32)       \
    XND_BINARY(name, uint16, int16, bool, int32)      \
    XND_BINARY(name, uint16, int32, bool, int32)      \
    XND_BINARY(name, uint16, int64, bool, int64)      \
    XND_BINARY(name, uint16, float32, bool, float32)  \
    XND_BINARY(name, uint16, float64, bool, float64)  \
                                                      \
    XND_BINARY(name, uint32, uint8, bool, uint32)     \
    XND_BINARY(name, uint32, uint16, bool, uint32)    \
    XND_BINARY(name, uint32, uint32, bool, uint32)    \
    XND_BINARY(name, uint32, uint64, bool, uint64)    \
    XND_BINARY(name, uint32, int8, bool, int64)       \
    XND_BINARY(name, uint32, int16, bool, int64)      \
    XND_BINARY(name, uint32, int32, bool, int64)      \
    XND_BINARY(name, uint32, int64, bool, int64)      \
    XND_BINARY(name, uint32, float32, bool, float64)  \
    XND_BINARY(name, uint32, float64, bool, float64)  \
                                                      \
    XND_BINARY(name, uint64, uint8, bool, uint64)     \
    XND_BINARY(name, uint64, uint16, bool, uint64)    \
    XND_BINARY(name, uint64, uint32, bool, uint64)    \
    XND_BINARY(name, uint64, uint64, bool, uint64)    \
                                                      \
    XND_BINARY(name, int8, uint8, bool, int16)        \
    XND_BINARY(name, int8, uint16, bool, int32)       \
    XND_BINARY(name, int8, uint32, bool, int64)       \
    XND_BINARY(name, int8, int8, bool, int8)          \
    XND_BINARY(name, int8, int16, bool, int16)        \
    XND_BINARY(name, int8, int32, bool, int32)        \
    XND_BINARY(name, int8, int64, bool, int64)        \
    XND_BINARY(name, int8, float32, bool, float32)    \
    XND_BINARY(name, int8, float64, bool, float64)    \
                                                      \
    XND_BINARY(name, int16, uint8, bool, int16)       \
    XND_BINARY(name, int16, uint16, bool, int32)      \
    XND_BINARY(name, int16, uint32, bool, int64)      \
    XND_BINARY(name, int16, int8, bool, int16)        \
    XND_BINARY(name, int16, int16, bool, int16)       \
    XND_BINARY(name, int16, int32, bool, int32)       \
    XND_BINARY(name, int16, int64, bool, int64)       \
    XND_BINARY(name, int16, float32, bool, float32)   \
    XND_BINARY(name, int16, float64, bool, float64)   \
                                                      \
    XND_BINARY(name, int32, uint8, bool, int32)       \
    XND_BINARY(name, int32, uint16, bool, int32)      \
    XND_BINARY(name, int32, uint32, bool, int64)      \
    XND_BINARY(name, int32, int8, bool, int32)        \
    XND_BINARY(name, int32, int16, bool, int32)       \
    XND_BINARY(name, int32, int32, bool, int32)       \
    XND_BINARY(name, int32, int64, bool, int64)       \
    XND_BINARY(name, int32, float32, bool, float64)   \
    XND_BINARY(name, int32, float64, bool, float64)   \
                                                      \
    XND_BINARY(name, int64, uint8, bool, int64)       \
    XND_BINARY(name, int64, uint16, bool, int64)      \
    XND_BINARY(name, int64, uint32, bool, int64)      \
    XND_BINARY(name, int64, int8, bool, int64)        \
    XND_BINARY(name, int64, int16, bool, int64)       \
    XND_BINARY(name, int64, int32, bool, int64)       \
    XND_BINARY(name, int64, int64, bool, int64)       \
                                                      \
    XND_BINARY(name, float32, uint8, bool, float32)   \
    XND_BINARY(name, float32, uint16, bool, float32)  \
    XND_BINARY(name, float32, uint32, bool, float64)  \
    XND_BINARY(name, float32, int8, bool, float32)    \
    XND_BINARY(name, float32, int16, bool, float32)   \
    XND_BINARY(name, float32, int32, bool, float64)   \
    XND_BINARY(name, float32, float32, bool, float32) \
    XND_BINARY(name, float32, float64, bool, float64) \
                                                      \
    XND_BINARY(name, float64, uint8, bool, float64)   \
    XND_BINARY(name, float64, uint16, bool, float64)  \
    XND_BINARY(name, float64, uint32, bool, float64)  \
    XND_BINARY(name, float64, int8, bool, float64)    \
    XND_BINARY(name, float64, int16, bool, float64)   \
    XND_BINARY(name, float64, int32, bool, float64)   \
    XND_BINARY(name, float64, float32, bool, float64) \
    XND_BINARY(name, float64, float64, bool, float64)

#define XND_ALL_COMPARISON_INIT(name) \
    XND_BINARY_INIT(name, uint8, uint8, bool),     \
    XND_BINARY_INIT(name, uint8, uint16, bool),    \
    XND_BINARY_INIT(name, uint8, uint32, bool),    \
    XND_BINARY_INIT(name, uint8, uint64, bool),    \
    XND_BINARY_INIT(name, uint8, int8, bool),      \
    XND_BINARY_INIT(name, uint8, int16, bool),     \
    XND_BINARY_INIT(name, uint8, int32, bool),     \
    XND_BINARY_INIT(name, uint8, int64, bool),     \
    XND_BINARY_INIT(name, uint8, float32, bool),   \
    XND_BINARY_INIT(name, uint8, float64, bool),   \
                                                   \
    XND_BINARY_INIT(name, uint16, uint8, bool),    \
    XND_BINARY_INIT(name, uint16, uint16, bool),   \
    XND_BINARY_INIT(name, uint16, uint32, bool),   \
    XND_BINARY_INIT(name, uint16, uint64, bool),   \
    XND_BINARY_INIT(name, uint16, int8, bool),     \
    XND_BINARY_INIT(name, uint16, int16, bool),    \
    XND_BINARY_INIT(name, uint16, int32, bool),    \
    XND_BINARY_INIT(name, uint16, int64, bool),    \
    XND_BINARY_INIT(name, uint16, float32, bool),  \
    XND_BINARY_INIT(name, uint16, float64, bool),  \
                                                   \
    XND_BINARY_INIT(name, uint32, uint8, bool),    \
    XND_BINARY_INIT(name, uint32, uint16, bool),   \
    XND_BINARY_INIT(name, uint32, uint32, bool),   \
    XND_BINARY_INIT(name, uint32, uint64, bool),   \
    XND_BINARY_INIT(name, uint32, int8, bool),     \
    XND_BINARY_INIT(name, uint32, int16, bool),    \
    XND_BINARY_INIT(name, uint32, int32, bool),    \
    XND_BINARY_INIT(name, uint32, int64, bool),    \
    XND_BINARY_INIT(name, uint32, float32, bool),  \
    XND_BINARY_INIT(name, uint32, float64, bool),  \
                                                   \
    XND_BINARY_INIT(name, uint64, uint8, bool),    \
    XND_BINARY_INIT(name, uint64, uint16, bool),   \
    XND_BINARY_INIT(name, uint64, uint32, bool),   \
    XND_BINARY_INIT(name, uint64, uint64, bool),   \
                                                   \
    XND_BINARY_INIT(name, int8, uint8, bool),      \
    XND_BINARY_INIT(name, int8, uint16, bool),     \
    XND_BINARY_INIT(name, int8, uint32, bool),     \
    XND_BINARY_INIT(name, int8, int8, bool),       \
    XND_BINARY_INIT(name, int8, int16, bool),      \
    XND_BINARY_INIT(name, int8, int32, bool),      \
    XND_BINARY_INIT(name, int8, int64, bool),      \
    XND_BINARY_INIT(name, int8, float32, bool),    \
    XND_BINARY_INIT(name, int8, float64, bool),    \
                                                   \
    XND_BINARY_INIT(name, int16, uint8, bool),     \
    XND_BINARY_INIT(name, int16, uint16, bool),    \
    XND_BINARY_INIT(name, int16, uint32, bool),    \
    XND_BINARY_INIT(name, int16, int8, bool),      \
    XND_BINARY_INIT(name, int16, int16, bool),     \
    XND_BINARY_INIT(name, int16, int32, bool),     \
    XND_BINARY_INIT(name, int16, int64, bool),     \
    XND_BINARY_INIT(name, int16, float32, bool),   \
    XND_BINARY_INIT(name, int16, float64, bool),   \
                                                   \
    XND_BINARY_INIT(name, int32, uint8, bool),     \
    XND_BINARY_INIT(name, int32, uint16, bool),    \
    XND_BINARY_INIT(name, int32, uint32, bool),    \
    XND_BINARY_INIT(name, int32, int8, bool),      \
    XND_BINARY_INIT(name, int32, int16, bool),     \
    XND_BINARY_INIT(name, int32, int32, bool),     \
    XND_BINARY_INIT(name, int32, int64, bool),     \
    XND_BINARY_INIT(name, int32, float32, bool),   \
    XND_BINARY_INIT(name, int32, float64, bool),   \
                                                   \
    XND_BINARY_INIT(name, int64, uint8, bool),     \
    XND_BINARY_INIT(name, int64, uint16, bool),    \
    XND_BINARY_INIT(name, int64, uint32, bool),    \
    XND_BINARY_INIT(name, int64, int8, bool),      \
    XND_BINARY_INIT(name, int64, int16, bool),     \
    XND_BINARY_INIT(name, int64, int32, bool),     \
    XND_BINARY_INIT(name, int64, int64, bool),     \
                                                   \
    XND_BINARY_INIT(name, float32, uint8, bool),   \
    XND_BINARY_INIT(name, float32, uint16, bool),  \
    XND_BINARY_INIT(name, float32, uint32, bool),  \
    XND_BINARY_INIT(name, float32, int8, bool),    \
    XND_BINARY_INIT(name, float32, int16, bool),   \
    XND_BINARY_INIT(name, float32, int32, bool),   \
    XND_BINARY_INIT(name, float32, float32, bool), \
    XND_BINARY_INIT(name, float32, float64, bool), \
                                                   \
    XND_BINARY_INIT(name, float64, uint8, bool),   \
    XND_BINARY_INIT(name, float64, uint16, bool),  \
    XND_BINARY_INIT(name, float64, uint32, bool),  \
    XND_BINARY_INIT(name, float64, int8, bool),    \
    XND_BINARY_INIT(name, float64, int16, bool),   \
    XND_BINARY_INIT(name, float64, int32, bool),   \
    XND_BINARY_INIT(name, float64, float32, bool), \
    XND_BINARY_INIT(name, float64, float64, bool)


#undef bool
#define bool_t _Bool

#define greater(x, y) x > y
XND_ALL_COMPARISON(greater)

#define greater_equal(x, y) x >= y
XND_ALL_COMPARISON(greater_equal)

#define less(x, y) x < y
XND_ALL_COMPARISON(less)

#define less_equal(x, y) x <= y
XND_ALL_COMPARISON(less_equal)


static const gm_kernel_init_t kernels[] = {
  XND_ALL_BINARY_INIT(add),
  XND_ALL_BINARY_INIT(subtract),
  XND_ALL_BINARY_INIT(multiply),
  XND_ALL_BINARY_INIT(divide),
  XND_ALL_COMPARISON_INIT(greater),
  XND_ALL_COMPARISON_INIT(greater_equal),
  XND_ALL_COMPARISON_INIT(less),
  XND_ALL_COMPARISON_INIT(less_equal),

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

static const gm_kernel_set_t *
typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *in[], int nin, ndt_context_t *ctx)
{
    return binary_typecheck(kernel_location, spec, f, in, nin, ctx);
}

int
gm_init_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &typecheck) < 0) {
             return -1;
        }
    }

    return 0;
}

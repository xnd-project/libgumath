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
    case Bool: {
        switch (t1->tag) {
        case Bool: return 0;

        case Uint8: return 8;
        case Uint16: return 16;
        case Uint32: return 24;
        case Uint64: return 32;

        case Int8: return 40;
        case Int16: return 48;
        case Int32: return 56;
        case Int64: return 64;

        default: goto invalid_combination;
        }
    }

    case Uint8: {
        switch (t1->tag) {
        case Bool: return 72;

        case Uint8: return 80;
        case Uint16: return 88;
        case Uint32: return 96;
        case Uint64: return 104;

        case Int8: return 112;
        case Int16: return 120;
        case Int32: return 128;
        case Int64: return 136;

        default: goto invalid_combination;
        }
    }
    case Uint16: {
        switch (t1->tag) {
        case Bool: return 144;

        case Int8: return 152;
        case Int16: return 160;
        case Int32: return 168;
        case Int64: return 176;

        case Uint8: return 184;
        case Uint16: return 192;
        case Uint32: return 200;
        case Uint64: return 208;

        default: goto invalid_combination;
        }
    }
    case Uint32: {
        switch (t1->tag) {
        case Bool: return 216;

        case Uint8: return 224;
        case Uint16: return 232;
        case Uint32: return 240;
        case Uint64: return 248;

        case Int8: return 256;
        case Int16: return 264;
        case Int32: return 272;
        case Int64: return 280;

        default: goto invalid_combination;
        }
    }
    case Uint64: {
        switch (t1->tag) {
        case Bool: return 288;

        case Uint8: return 296;
        case Uint16: return 304;
        case Uint32: return 312;
        case Uint64: return 320;

        default: goto invalid_combination;
        }
    }

    case Int8: {
        switch (t1->tag) {
        case Bool: return 328;

        case Uint8: return 336;
        case Uint16: return 344;
        case Uint32: return 352;

        case Int8: return 360;
        case Int16: return 368;
        case Int32: return 376;
        case Int64: return 384;

        default: goto invalid_combination;
        }
    }
    case Int16: {
        switch (t1->tag) {
        case Bool: return 392;

        case Uint8: return 400;
        case Uint16: return 408;
        case Uint32: return 416;

        case Int8: return 424;
        case Int16: return 432;
        case Int32: return 440;
        case Int64: return 448;

        default: goto invalid_combination;
        }
    }
    case Int32: {
        switch (t1->tag) {
        case Bool: return 456;

        case Uint8: return 464;
        case Uint16: return 472;
        case Uint32: return 480;

        case Int8: return 488;
        case Int16: return 496;
        case Int32: return 504;
        case Int64: return 512;

        default: goto invalid_combination;
        }
    }

    case Int64: {
        switch (t1->tag) {
        case Bool: return 520;

        case Uint8: return 528;
        case Uint16: return 536;
        case Uint32: return 544;

        case Int8: return 552;
        case Int16: return 560;
        case Int32: return 568;
        case Int64: return 576;

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
/*                                  Bitwise                                 */
/****************************************************************************/

#undef bool
#define bool_t _Bool

#define XND_ALL_BITWISE(name) \
    XND_BINARY(name, bool, bool, bool, bool)         \
    XND_BINARY(name, bool, uint8, uint8, uint8)      \
    XND_BINARY(name, bool, uint16, uint16, uint16)   \
    XND_BINARY(name, bool, uint32, uint32, uint32)   \
    XND_BINARY(name, bool, uint64, uint64, uint64)   \
    XND_BINARY(name, bool, int8, int8, int8)         \
    XND_BINARY(name, bool, int16, int16, int16)      \
    XND_BINARY(name, bool, int32, int32, int32)      \
    XND_BINARY(name, bool, int64, int64, int64)      \
                                                     \
    XND_BINARY(name, uint8, bool, uint8, uint8)      \
    XND_BINARY(name, uint8, uint8, uint8, uint8)     \
    XND_BINARY(name, uint8, uint16, uint16, uint16)  \
    XND_BINARY(name, uint8, uint32, uint32, uint32)  \
    XND_BINARY(name, uint8, uint64, uint64, uint64)  \
    XND_BINARY(name, uint8, int8, int16, int16)      \
    XND_BINARY(name, uint8, int16, int16, int16)     \
    XND_BINARY(name, uint8, int32, int32, int32)     \
    XND_BINARY(name, uint8, int64, int64, int64)     \
                                                     \
    XND_BINARY(name, uint16, bool, uint16, uint16)   \
    XND_BINARY(name, uint16, uint8, uint16, uint16)  \
    XND_BINARY(name, uint16, uint16, uint16, uint16) \
    XND_BINARY(name, uint16, uint32, uint32, uint32) \
    XND_BINARY(name, uint16, uint64, uint64, uint64) \
    XND_BINARY(name, uint16, int8, int32, int32)     \
    XND_BINARY(name, uint16, int16, int32, int32)    \
    XND_BINARY(name, uint16, int32, int32, int32)    \
    XND_BINARY(name, uint16, int64, int64, int64)    \
                                                     \
    XND_BINARY(name, uint32, bool, uint32, uint32)   \
    XND_BINARY(name, uint32, uint8, uint32, uint32)  \
    XND_BINARY(name, uint32, uint16, uint32, uint32) \
    XND_BINARY(name, uint32, uint32, uint32, uint32) \
    XND_BINARY(name, uint32, uint64, uint64, uint64) \
    XND_BINARY(name, uint32, int8, int64, int64)     \
    XND_BINARY(name, uint32, int16, int64, int64)    \
    XND_BINARY(name, uint32, int32, int64, int64)    \
    XND_BINARY(name, uint32, int64, int64, int64)    \
                                                     \
    XND_BINARY(name, uint64, bool, uint64, uint64)   \
    XND_BINARY(name, uint64, uint8, uint64, uint64)  \
    XND_BINARY(name, uint64, uint16, uint64, uint64) \
    XND_BINARY(name, uint64, uint32, uint64, uint64) \
    XND_BINARY(name, uint64, uint64, uint64, uint64) \
                                                     \
    XND_BINARY(name, int8, bool, int8, int8)         \
    XND_BINARY(name, int8, uint8, int16, int16)      \
    XND_BINARY(name, int8, uint16, int32, int32)     \
    XND_BINARY(name, int8, uint32, int64, int64)     \
    XND_BINARY(name, int8, int8, int8, int8)         \
    XND_BINARY(name, int8, int16, int16, int16)      \
    XND_BINARY(name, int8, int32, int32, int32)      \
    XND_BINARY(name, int8, int64, int64, int64)      \
                                                     \
    XND_BINARY(name, int16, bool, int16, int16)      \
    XND_BINARY(name, int16, uint8, int16, int16)     \
    XND_BINARY(name, int16, uint16, int32, int32)    \
    XND_BINARY(name, int16, uint32, int64, int64)    \
    XND_BINARY(name, int16, int8, int16, int16)      \
    XND_BINARY(name, int16, int16, int16, int16)     \
    XND_BINARY(name, int16, int32, int32, int32)     \
    XND_BINARY(name, int16, int64, int64, int64)     \
                                                     \
    XND_BINARY(name, int32, bool, int32, int32)      \
    XND_BINARY(name, int32, uint8, int32, int32)     \
    XND_BINARY(name, int32, uint16, int32, int32)    \
    XND_BINARY(name, int32, uint32, int64, int64)    \
    XND_BINARY(name, int32, int8, int32, int32)      \
    XND_BINARY(name, int32, int16, int32, int32)     \
    XND_BINARY(name, int32, int32, int32, int32)     \
    XND_BINARY(name, int32, int64, int64, int64)     \
                                                     \
    XND_BINARY(name, int64, bool, int64, int64)      \
    XND_BINARY(name, int64, uint8, int64, int64)     \
    XND_BINARY(name, int64, uint16, int64, int64)    \
    XND_BINARY(name, int64, uint32, int64, int64)    \
    XND_BINARY(name, int64, int8, int64, int64)      \
    XND_BINARY(name, int64, int16, int64, int64)     \
    XND_BINARY(name, int64, int32, int64, int64)     \
    XND_BINARY(name, int64, int64, int64, int64)     \

#define XND_ALL_BITWISE_INIT(name) \
    XND_BINARY_INIT(name, bool, bool, bool),       \
    XND_BINARY_INIT(name, bool, uint8, uint8),     \
    XND_BINARY_INIT(name, bool, uint16, uint16),   \
    XND_BINARY_INIT(name, bool, uint32, uint32),   \
    XND_BINARY_INIT(name, bool, uint64, uint64),   \
    XND_BINARY_INIT(name, bool, int8, int8),       \
    XND_BINARY_INIT(name, bool, int16, int16),     \
    XND_BINARY_INIT(name, bool, int32, int32),     \
    XND_BINARY_INIT(name, bool, int64, int64),     \
                                                   \
    XND_BINARY_INIT(name, uint8, bool, uint8),     \
    XND_BINARY_INIT(name, uint8, uint8, uint8),    \
    XND_BINARY_INIT(name, uint8, uint16, uint16),  \
    XND_BINARY_INIT(name, uint8, uint32, uint32),  \
    XND_BINARY_INIT(name, uint8, uint64, uint64),  \
    XND_BINARY_INIT(name, uint8, int8, int16),     \
    XND_BINARY_INIT(name, uint8, int16, int16),    \
    XND_BINARY_INIT(name, uint8, int32, int32),    \
    XND_BINARY_INIT(name, uint8, int64, int64),    \
                                                   \
    XND_BINARY_INIT(name, uint16, bool, uint16),   \
    XND_BINARY_INIT(name, uint16, uint8, uint16),  \
    XND_BINARY_INIT(name, uint16, uint16, uint16), \
    XND_BINARY_INIT(name, uint16, uint32, uint32), \
    XND_BINARY_INIT(name, uint16, uint64, uint64), \
    XND_BINARY_INIT(name, uint16, int8, int32),    \
    XND_BINARY_INIT(name, uint16, int16, int32),   \
    XND_BINARY_INIT(name, uint16, int32, int32),   \
    XND_BINARY_INIT(name, uint16, int64, int64),   \
                                                   \
    XND_BINARY_INIT(name, uint32, bool, uint32),   \
    XND_BINARY_INIT(name, uint32, uint8, uint32),  \
    XND_BINARY_INIT(name, uint32, uint16, uint32), \
    XND_BINARY_INIT(name, uint32, uint32, uint32), \
    XND_BINARY_INIT(name, uint32, uint64, uint64), \
    XND_BINARY_INIT(name, uint32, int8, int64),    \
    XND_BINARY_INIT(name, uint32, int16, int64),   \
    XND_BINARY_INIT(name, uint32, int32, int64),   \
    XND_BINARY_INIT(name, uint32, int64, int64),   \
                                                   \
    XND_BINARY_INIT(name, uint64, bool, uint64),   \
    XND_BINARY_INIT(name, uint64, uint8, uint64),  \
    XND_BINARY_INIT(name, uint64, uint16, uint64), \
    XND_BINARY_INIT(name, uint64, uint32, uint64), \
    XND_BINARY_INIT(name, uint64, uint64, uint64), \
                                                   \
    XND_BINARY_INIT(name, int8, bool, int8),       \
    XND_BINARY_INIT(name, int8, uint8, int16),     \
    XND_BINARY_INIT(name, int8, uint16, int32),    \
    XND_BINARY_INIT(name, int8, uint32, int64),    \
    XND_BINARY_INIT(name, int8, int8, int8),       \
    XND_BINARY_INIT(name, int8, int16, int16),     \
    XND_BINARY_INIT(name, int8, int32, int32),     \
    XND_BINARY_INIT(name, int8, int64, int64),     \
                                                   \
    XND_BINARY_INIT(name, int16, bool, int16),     \
    XND_BINARY_INIT(name, int16, uint8, int16),    \
    XND_BINARY_INIT(name, int16, uint16, int32),   \
    XND_BINARY_INIT(name, int16, uint32, int64),   \
    XND_BINARY_INIT(name, int16, int8, int16),     \
    XND_BINARY_INIT(name, int16, int16, int16),    \
    XND_BINARY_INIT(name, int16, int32, int32),    \
    XND_BINARY_INIT(name, int16, int64, int64),    \
                                                   \
    XND_BINARY_INIT(name, int32, bool, int32),     \
    XND_BINARY_INIT(name, int32, uint8, int32),    \
    XND_BINARY_INIT(name, int32, uint16, int32),   \
    XND_BINARY_INIT(name, int32, uint32, int64),   \
    XND_BINARY_INIT(name, int32, int8, int32),     \
    XND_BINARY_INIT(name, int32, int16, int32),    \
    XND_BINARY_INIT(name, int32, int32, int32),    \
    XND_BINARY_INIT(name, int32, int64, int64),    \
                                                   \
    XND_BINARY_INIT(name, int64, bool, int64),     \
    XND_BINARY_INIT(name, int64, uint8, int64),    \
    XND_BINARY_INIT(name, int64, uint16, int64),   \
    XND_BINARY_INIT(name, int64, uint32, int64),   \
    XND_BINARY_INIT(name, int64, int8, int64),     \
    XND_BINARY_INIT(name, int64, int16, int64),    \
    XND_BINARY_INIT(name, int64, int32, int64),    \
    XND_BINARY_INIT(name, int64, int64, int64)

#define bitwise_and(x, y) x & y
XND_ALL_BITWISE(bitwise_and)

#define bitwise_or(x, y) x | y
XND_ALL_BITWISE(bitwise_or)

#define bitwise_xor(x, y) x ^ y
XND_ALL_BITWISE(bitwise_xor)

static const gm_kernel_init_t kernels[] = {
  XND_ALL_BITWISE_INIT(bitwise_and),
  XND_ALL_BITWISE_INIT(bitwise_or),
  XND_ALL_BITWISE_INIT(bitwise_xor),

  { .name = NULL, .sig = NULL }
};


static const gm_kernel_set_t *
typecheck(ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *in[],
          const int64_t li[], int nin, ndt_context_t *ctx)
{
    return cpu_binary_typecheck(kernel_location, spec, f, in, li, nin, ctx);
}


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_bitwise_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel_typecheck(tbl, k, ctx, &typecheck) < 0) {
             return -1;
        }
    }

    return 0;
}

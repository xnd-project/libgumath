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


gm_kernel_set_t empty_kernel_set =
 { .sig = NULL,
   .C = NULL,
   .Fortran = NULL,
   .Strided = NULL,
   .Xnd = NULL };


/****************************************************************************/
/*                              Example kernels                             */
/****************************************************************************/

/* NumPy signatures */
#define NP_STRIDED(func, srctype, desttype) \
static int                                                                     \
gm_##func##_strided_##srctype##_##desttype(                                    \
    char *args[], int64_t dimensions[], int64_t steps[], void *data GM_UNUSED) \
{                                                                              \
    const char *src = args[0];                                                 \
    char *dest = args[1];                                                      \
    int64_t n = dimensions[0];                                                 \
    int64_t i;                                                                 \
                                                                               \
    for (i = 0; i < n; i++) {                                                  \
        *(desttype##_t *)dest = func(*(const srctype##_t *)src);               \
        src += steps[0];                                                       \
        dest += steps[1];                                                      \
    }                                                                          \
                                                                               \
    return 0;                                                                  \
}

#define NP_COPY_STRIDED(srctype, desttype) \
static int                                                                     \
gm_copy_strided_##srctype##_##desttype(                                        \
    char *args[], int64_t dimensions[], int64_t steps[], void *data GM_UNUSED) \
{                                                                              \
    const char *src = args[0];                                                 \
    char *dest = args[1];                                                      \
    int64_t n = dimensions[0];                                                 \
    int64_t i;                                                                 \
                                                                               \
    for (i = 0; i < n; i++) {                                                  \
        *(desttype##_t *)dest = *(const srctype##_t *)src;                     \
        src += steps[0];                                                       \
        dest += steps[1];                                                      \
    }                                                                          \
    return 0;                                                                  \
}

static int
gm_multiply_strided_q64_q64(char *args[], int64_t dimensions[], int64_t steps[], void *data GM_UNUSED)
{
    const char *src1 = args[0];
    const char *src2 = args[1];
    char *dest = args[2];
    const ndt_complex64_t (*s1)[2];
    const ndt_complex64_t (*s2)[2];
    ndt_complex64_t (*d1)[2];
    int64_t n = dimensions[0];
    int64_t i, j, k, l;

    for (i = 0; i < n; i++) {
      s1 = (const ndt_complex64_t (*)[2])src1;
      s2 = (const ndt_complex64_t (*)[2])src2;
      d1 = (ndt_complex64_t (*)[2])dest;
      for (j = 0; j < 2; j++){
        for (k = 0; k < 2; k++) {
          ndt_complex64_t sum = 0;
          for (l = 0; l < 2; l++) {
            sum += s1[j][l] * s2[l][k];
          }
          d1[j][k] = sum;
        }
      }
      src1 += steps[0];
      src2 += steps[1];
      dest += steps[2];
    }

    return 0;
}

static int
gm_multiply_strided_q128_q128(char *args[], int64_t dimensions[], int64_t steps[], void *data GM_UNUSED)
{
    const char *src1 = args[0];
    const char *src2 = args[1];
    char *dest = args[2];
    const ndt_complex128_t (*s1)[2];
    const ndt_complex128_t (*s2)[2];
    ndt_complex128_t (*d1)[2];
    int64_t n = dimensions[0];
    int64_t i, j, k, l;

    for (i = 0; i < n; i++) {
      s1 = (const ndt_complex128_t (*)[2])src1;
      s2 = (const ndt_complex128_t (*)[2])src2;
      d1 = (ndt_complex128_t (*)[2])dest;
      for (j = 0; j < 2; j++){
        for (k = 0; k < 2; k++) {
          ndt_complex128_t sum = 0;
          for (l = 0; l < 2; l++) {
            sum += s1[j][l] * s2[l][k];
          }
          d1[j][k] = sum;
        }
      }
      src1 += steps[0];
      src2 += steps[1];
      dest += steps[2];
    }

    return 0;
}

NP_STRIDED(sin, float32, float64)
NP_STRIDED(sin, float64, float64)
NP_STRIDED(sin, uint8, float64)
NP_STRIDED(sin, uint16, float64)
NP_STRIDED(sin, uint32, float64)
NP_STRIDED(sin, uint64, float64)
NP_STRIDED(sin, int8, float64)
NP_STRIDED(sin, int16, float64)
NP_STRIDED(sin, int32, float64)
NP_STRIDED(sin, int64, float64)

NP_STRIDED(sinf, float32, float32)

NP_COPY_STRIDED(uint8, uint8)
NP_COPY_STRIDED(uint16, uint16)
NP_COPY_STRIDED(uint32, uint32)
NP_COPY_STRIDED(uint64, uint64)
NP_COPY_STRIDED(int8, int8)
NP_COPY_STRIDED(int16, int16)
NP_COPY_STRIDED(int32, int32)
NP_COPY_STRIDED(int64, int64)
NP_COPY_STRIDED(float32, float32)
NP_COPY_STRIDED(float64, float64)


static const gm_typedef_init_t typedefs[] = {
  { .name = "quaternion64", .type = "2 * 2 * complex64" },
  { .name = "quaternion128", .type = "2 * 2 * complex128" },
  { .name = NULL, .type = NULL }
};

static const gm_kernel_init_t kernels[] = {
  /* COPY */
  { .name = "copy", .sig = "... * uint8 -> ... * uint8", .Strided = gm_copy_strided_uint8_uint8 },
  { .name = "copy", .sig = "... * uint16 -> ... * uint16", .Strided = gm_copy_strided_uint16_uint16 },
  { .name = "copy", .sig = "... * uint32 -> ... * uint32", .Strided = gm_copy_strided_uint32_uint32 },
  { .name = "copy", .sig = "... * uint64 -> ... * uint64", .Strided = gm_copy_strided_uint64_uint64 },

  { .name = "copy", .sig = "... * int8 -> ... * int8", .Strided = gm_copy_strided_int8_int8 },
  { .name = "copy", .sig = "... * int16 -> ... * int16", .Strided = gm_copy_strided_int16_int16 },
  { .name = "copy", .sig = "... * int32 -> ... * int32", .Strided = gm_copy_strided_int32_int32 },
  { .name = "copy", .sig = "... * int64 -> ... * int64", .Strided = gm_copy_strided_int64_int64 },

  { .name = "copy", .sig = "... * float32 -> ... * float32", .Strided = gm_copy_strided_float32_float32 },
  { .name = "copy", .sig = "... * float64 -> ... * float64", .Strided = gm_copy_strided_float64_float64 },

  /* SIN */
  /* return float32 */
  { .name = "sin", .sig = "... * float32 -> ... * float32", .Strided = gm_sinf_strided_float32_float32 },

  /* return float64 */
  { .name = "sin", .sig = "... * uint8 -> ... * float64", .Strided = gm_sin_strided_uint8_float64 },
  { .name = "sin", .sig = "... * uint16 -> ... * float64", .Strided = gm_sin_strided_uint16_float64 },
  { .name = "sin", .sig = "... * uint32 -> ... * float64", .Strided = gm_sin_strided_uint32_float64 },
  { .name = "sin", .sig = "... * uint64 -> ... * float64", .Strided = gm_sin_strided_uint64_float64 },

  { .name = "sin", .sig = "... * int8 -> ... * float64", .Strided = gm_sin_strided_int8_float64 },
  { .name = "sin", .sig = "... * int16 -> ... * float64", .Strided = gm_sin_strided_int16_float64 },
  { .name = "sin", .sig = "... * int32 -> ... * float64", .Strided = gm_sin_strided_int32_float64 },
  { .name = "sin", .sig = "... * int64 -> ... * float64", .Strided = gm_sin_strided_int64_float64 },

  { .name = "sin", .sig = "... * float32 -> ... * float64", .Strided = gm_sin_strided_float32_float64 },
  { .name = "sin", .sig = "... * float64 -> ... * float64", .Strided = gm_sin_strided_float64_float64 },

  /* MULTIPLY */
  /* quaternions */
  { .name = "multiply",
    .sig = "... * Q64(2 * 2 * complex64), ... * Q64(2 * 2 * complex64) -> ... * Q64(2 * 2 * complex64)",
    .Strided = gm_multiply_strided_q64_q64 },

  { .name = "multiply",
    .sig = "... * quaternion128, ... * quaternion128 -> ... * quaternion128",
    .Strided = gm_multiply_strided_q128_q128 },

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_kernels(ndt_context_t *ctx)
{
    const gm_typedef_init_t *t;
    const gm_kernel_init_t *k;

    for (t = typedefs; t->name != NULL; t++) {
        if (ndt_typedef_from_string(t->name, t->type, ctx) < 0) {
            return -1;
        }
    }

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel(k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

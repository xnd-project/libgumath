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


static int
gm_multiply_strided_1D_q64_q64(char *args[], intptr_t dimensions[], intptr_t steps[],
                               void *data GM_UNUSED)
{
    const char *src1 = args[0];
    const char *src2 = args[1];
    char *dest = args[2];
    const ndt_complex64_t (*s1)[2];
    const ndt_complex64_t (*s2)[2];
    ndt_complex64_t (*d1)[2];
    intptr_t n = dimensions[0];
    intptr_t i, j, k, l;

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
gm_multiply_strided_1D_q128_q128(char *args[], intptr_t dimensions[], intptr_t steps[],
                                 void *data GM_UNUSED)
{
    const char *src1 = args[0];
    const char *src2 = args[1];
    char *dest = args[2];
    const ndt_complex128_t (*s1)[2];
    const ndt_complex128_t (*s2)[2];
    ndt_complex128_t (*d1)[2];
    intptr_t n = dimensions[0];
    intptr_t i, j, k, l;

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


static const gm_typedef_init_t typedefs[] = {
  { .name = "quaternion64", .type = "2 * 2 * complex64", .meth=NULL },
  { .name = "quaternion128", .type = "2 * 2 * complex128", .meth=NULL },
  { .name = NULL, .type = NULL }
};

static const gm_kernel_init_t kernels[] = {
  { .name = "multiply",
    .sig = "... * N * quaternion64, ... * N * quaternion64 -> ... * N * quaternion64",
    .Strided = gm_multiply_strided_1D_q64_q64 },

  { .name = "multiply",
    .sig = "... * N * quaternion128, ... * N * quaternion128 -> ... * N * quaternion128",
    .Strided = gm_multiply_strided_1D_q128_q128 },

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_quaternion_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_typedef_init_t *t;
    const gm_kernel_init_t *k;

    for (t = typedefs; t->name != NULL; t++) {
        if (ndt_typedef_from_string(t->name, t->type, t->meth, ctx) < 0) {
            return -1;
        }
    }

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

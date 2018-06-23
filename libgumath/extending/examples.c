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
/*                               Xnd kernels                                */
/****************************************************************************/

/*
 * Count valid/missing values in a 1D array of records and return the result
 * as a record.
 *
 * Signature:
 *    "... * N * {index: int64, name: string, value: ?int64} -> ... * {valid: int64, missing: int64}"
 */
static int
count_valid_missing(xnd_t stack[], ndt_context_t *ctx)
{
    const xnd_t *array = &stack[0];
    int64_t N = array->type->FixedDim.shape; /* N in the above signature */
    xnd_t *out = &stack[1];
    int64_t ok = 0;
    int64_t na = 0;

    for (int64_t i = 0; i < N; i++) {
        const xnd_t record = xnd_fixed_dim_next(array, i);
        const xnd_t value = xnd_record_next(&record, 2, ctx);
        if (value.ptr == NULL) {
            return -1;
        }

        if (xnd_is_na(&value)) {
            na++;
        }
        else {
            ok++;
        }
    }

    xnd_t valid = xnd_record_next(out, 0, ctx);
    *(int64_t *)(valid.ptr) = ok;

    xnd_t missing = xnd_record_next(out, 1, ctx);
    *(int64_t *)(missing.ptr) = na;

    return 0;
}

int
gm_0D_add_scalar(xnd_t stack[], ndt_context_t *ctx)
{
    const xnd_t *x = &stack[0];
    const xnd_t *y = &stack[1];
    xnd_t *z = &stack[2];
    int64_t N = xnd_fixed_shape(x);
    int64_t yy = *(int64_t *)y->ptr;
    (void)ctx;

    for (int64_t i = 0; i < N; i++) {
        const xnd_t xx = xnd_fixed_dim_next(x, i);
        const xnd_t zz = xnd_fixed_dim_next(z, i);
        *(int64_t *)zz.ptr = *(int64_t *)xx.ptr + yy;
    }

    return 0;
}

int
gm_randint(xnd_t stack[], ndt_context_t *ctx)
{
    const xnd_t *x = &stack[0];
    (void)ctx;

    *(int32_t *)x->ptr = rand();
    return 0;
}

int
gm_randtuple(xnd_t stack[], ndt_context_t *ctx)
{
    const xnd_t *x = &stack[0];
    const xnd_t *y = &stack[1];
    (void)ctx;

    *(int32_t *)x->ptr = rand();
    *(int32_t *)y->ptr = rand();
    return 0;
}

int
gm_divmod10(xnd_t stack[], ndt_context_t *ctx)
{
    const xnd_t *x = &stack[0];
    int64_t xx = *(int64_t *)x->ptr;
    const xnd_t *y = &stack[1];
    const xnd_t *z = &stack[2];
    (void)ctx;

    *(int64_t *)y->ptr = xx / 10;
    *(int64_t *)z->ptr = xx % 10;
    return 0;
}


static const gm_kernel_init_t kernels[] = {
  { .name = "add_scalar", .sig = "... * N * int64, ... * int64 -> ... * N * int64", .Xnd = gm_0D_add_scalar },

  { .name = "count_valid_missing",
    .sig = "... * N * {index: int64, name: string, value: ?int64} -> ... * {valid: int64, missing: int64}",
    .Xnd = count_valid_missing },

  { .name = "randint", .sig = "void -> int32", .Xnd = gm_randint },
  { .name = "randtuple", .sig = "void -> int32, int32", .Xnd = gm_randtuple },
  { .name = "divmod10", .sig = "int64 -> int64, int64", .Xnd = gm_divmod10 },

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_example_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

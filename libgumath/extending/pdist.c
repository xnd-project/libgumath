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


/*****************************************************************************/
/*                         Euclidian pairwise distance                       */
/*****************************************************************************/

/* Numerically unstable: This is just an example for the constraint feature. */

static int
pdist(xnd_t stack[], ndt_context_t *ctx)
{
    const ndt_t *t = stack[0].type;
    const ndt_t *u = t->FixedDim.type;
    const int64_t N = t->FixedDim.shape;
    const int64_t M = u->FixedDim.shape;
    int64_t l = 0;
    (void)ctx;

    for (int64_t i = 0; i < N; i++) {
        const xnd_t vector1 = xnd_fixed_dim_next(&stack[0], i);
        for (int64_t j = i+1; j < N; j++) {
            const xnd_t vector2 = xnd_fixed_dim_next(&stack[0], j);
            double sum = 0.0;
            for (int64_t k = 0; k < M; k++) {
                const xnd_t value1 = xnd_fixed_dim_next(&vector1, k);
                const xnd_t value2 = xnd_fixed_dim_next(&vector2, k);
                const double v1 = *(double *)value1.ptr;
                const double v2 = *(double *)value2.ptr;
                sum += pow(v1-v2, 2);
            }
            const xnd_t res = xnd_fixed_dim_next(&stack[1], l++);
            *(double *)res.ptr = sqrt(sum);
        }
    }

    return 0;
}

/*
 * Validate N, M and compute unknown output dimension P.
 *
 * shape[0] = N
 * shape[1] = N
 * shape[2] = P
 *
 * 'args' is unused here.  Other functions may inspect the incoming xnd
 * arguments in order to resolve further constraints based on values.
 */
static int
pdist_constraint(int64_t *shapes, const void *args, ndt_context_t *ctx)
{
    (void)args;

    if (shapes[0] == 0 || shapes[1] == 0) {
        ndt_err_format(ctx, NDT_ValueError,
            "euclidian_pdist() requires a non-empty matrix");
        return -1;
    }

    shapes[2] = (shapes[0] * (shapes[0]-1)) / 2;
    return 0;
}

static const ndt_constraint_t constraint = {
  .f = pdist_constraint,
  .nin = 2,
  .nout = 1,
  .symbols = {"N", "M", "P"}
};


static const gm_kernel_init_t kernels[] = {
  { .name = "euclidian_pdist",
    .sig = "N * M * float64 -> P * float64",
    .constraint = &constraint,
    .Xnd = pdist },

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_pdist_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_kernel_init_t *k;

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}

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
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"


gm_kernel_set_t empty_kernel_set =
 { .sig = NULL,
   .Elementwise = NULL,
   .C = NULL,
   .Fortran = NULL,
   .Strided = NULL,
   .Xnd = NULL };


/****************************************************************************/
/*                              Example kernels                             */
/****************************************************************************/

static int
gm_sin_strided_d_d(char *args[], int64_t dimensions[], int64_t steps[],
                   void *data GM_UNUSED)
{
    const char *src = args[0];
    char *dest = args[1];
    int64_t n = dimensions[0];
    int64_t i;

    assert(n == dimensions[1]);

    for (i = 0; i < n; i++) {
        *(double *)dest = sin(*(const double *)src);
        src += steps[0];
        dest += steps[1];
    }

    return 1;
}

static int
gm_sin_strided_f_f(char **args, int64_t *dimensions, int64_t *steps,
                   void *data GM_UNUSED)
{
    const char *src = args[0];
    char *dest = args[1];
    int64_t n = dimensions[0];
    int64_t i;

    assert(n == dimensions[1]);

    for (i = 0; i < n; i++) {
        *(float *)dest = sinf(*(const float *)src);
        src += steps[0];
        dest += steps[1];
    }

    return 1;
}


/****************************************************************************/
/*                               Example init                               */
/****************************************************************************/

int
gm_sin_init(ndt_context_t *ctx)
{
    gm_kernel_set_t set;

    if (gm_add_func("sin", ctx) < 0) {
        return -1;
    }

    set = empty_kernel_set;
    set.sig = ndt_from_string("... * float64 => ... * float64", ctx);
    if (set.sig == NULL) {
        return -1;
    }

    set.Elementwise = gm_sin_strided_d_d;
    set.Strided = gm_sin_strided_d_d;
    if (gm_add_kernel("sin", set, ctx) < 0) {
        return -1;
    }

    set = empty_kernel_set;
    set.sig = ndt_from_string("... * float32 => ... * float32", ctx);
    if (set.sig == NULL) {
        return -1;
    }

    set.Elementwise = gm_sin_strided_f_f;
    set.Strided = gm_sin_strided_f_f;
    if (gm_add_kernel("sin", set, ctx) < 0) {
        return -1;
    }

    return 0;
}

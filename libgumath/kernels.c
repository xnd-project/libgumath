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


/****************************************************************************/
/*                              Example kernels                             */
/****************************************************************************/

static inline void
gm_sin_c_d_d(xnd_ndarray_t stack[], int64_t n)
{
    const xnd_ndarray_t *src = &stack[0];
    xnd_ndarray_t *dst = &stack[1];
    const double *s = (const double *)src->ptr;
    double *d = (double *)dst->ptr;
    int64_t i;

    for (i = 0; i < n-3; i += 4) {
        d[i] = sin(s[i]);
        d[i+1] = sin(s[i+1]);
        d[i+2] = sin(s[i+2]);
        d[i+3] = sin(s[i+3]);
    }

    for (; i < n; i++) {
        d[i] = sin(s[i]);
    }
}

static inline void
gm_sin_strided_d_d(xnd_ndarray_t stack[], int64_t n)
{
    const xnd_ndarray_t *src = &stack[0];
    xnd_ndarray_t *dst = &stack[1];
    const double *s = (const double *)src->ptr;
    double *d = (double *)dst->ptr;
    int64_t i;

    for (i = 0; i < n; i++) {
        d[i] = sin(s[i]);
        s += src->strides[0];
        d += dst->strides[0];
    }
}

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


#ifndef COMMON_H
#define COMMON_H


#define XSTRINGIZE(v) #v
#define STRINGIZE(v) XSTRINGIZE(v)


/*****************************************************************************/
/*              Apply linear index to the data pointer (1D kernels)          */
/*****************************************************************************/

static inline char *
apply_index(const xnd_t *x)
{
    return xnd_fixed_apply_index(x);
}


/*****************************************************************************/
/*                          Optimized bitmap handling                        */
/*****************************************************************************/

static inline uint8_t *
get_bitmap(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->ndim == 0);
    return ndt_is_optional(t) ? x->bitmap.data : NULL;
}

static inline uint8_t *
get_bitmap1D(const xnd_t *x)
{
    const ndt_t *t = x->type;
    assert(t->ndim == 1 && t->tag == FixedDim);
    return ndt_is_optional(ndt_dtype(t)) ? x->bitmap.data : NULL;
}

static inline int
is_valid(const uint8_t *data, int64_t n)
{
    return data[n / 8] & ((uint8_t)1 << (n % 8));
}

static inline void
set_valid(uint8_t *data, int64_t n)
{
    data[n / 8] |= ((uint8_t)1 << (n % 8));
}

static inline int64_t
linear_index1D(const xnd_t *x, const int64_t i)
{
    const ndt_t *t = x->type;
    const int64_t step = i * t->Concrete.FixedDim.step;
    return x->index + step;
}


/*****************************************************************************/
/*                              Binary typecheck                             */
/*****************************************************************************/

/* LOCAL SCOPE */
NDT_PRAGMA(NDT_HIDE_SYMBOLS_START)

void unary_update_bitmap1D(xnd_t stack[]);
void unary_update_bitmap(xnd_t stack[]);

void binary_update_bitmap1D(xnd_t stack[]);
void binary_update_bitmap(xnd_t stack[]);

const gm_kernel_set_t *cpu_unary_typecheck(int (*kernel_location)(const ndt_t *, ndt_context_t *),
                                           ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                                           const int64_t li[], int nin, int nout, bool check_broadcast,
                                           ndt_context_t *ctx);

const gm_kernel_set_t *cuda_unary_typecheck(int (*kernel_location)(const ndt_t *, ndt_context_t *),
                                            ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                                            const int64_t li[], int nin, int nout, bool check_broadcast,
                                            ndt_context_t *ctx);

const gm_kernel_set_t *cpu_binary_typecheck(int (*kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                                            ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                                            const int64_t li[], int nin, int nout, bool check_broadcast,
                                            ndt_context_t *ctx);

const gm_kernel_set_t *cuda_binary_typecheck(int (* kernel_location)(const ndt_t *in0, const ndt_t *in1, ndt_context_t *ctx),
                                             ndt_apply_spec_t *spec, const gm_func_t *f, const ndt_t *types[],
                                             const int64_t li[], int nin, int nout, bool check_broadcast,
                                             ndt_context_t *ctx);

/* END LOCAL SCOPE */
NDT_PRAGMA(NDT_HIDE_SYMBOLS_END)


#endif /* COMMON_H */

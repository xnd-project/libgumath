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


#ifndef GUMATH_H
#define GUMATH_H


#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
  #include <cstdint>
#else
  #include <stdint.h>
#endif

#include "ndtypes.h"
#include "xnd.h"


#ifdef _MSC_VER
  #if defined (GM_EXPORT)
    #define GM_API __declspec(dllexport)
  #elif defined(GM_IMPORT)
    #define GM_API __declspec(dllimport)
  #else
    #define GM_API
  #endif

  #ifndef GM_UNUSED
    #define GM_UNUSED
  #endif

  #include "malloc.h"
  #define ALLOCA(type, name, nmemb) type *name = _alloca(nmemb * sizeof(type))
#else
  #define GM_API
  #if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define GM_UNUSED __attribute__((unused))
  #else
    #define GM_UNUSED
  #endif

  #define ALLOCA(type, name, nmemb) type name[nmemb]
#endif


#define GM_MAX_KERNELS 8192
#define GM_THREAD_CUTOFF 1000000

typedef float float32_t;
typedef double float64_t;


typedef int (* gm_xnd_kernel_t)(xnd_t stack[], ndt_context_t *ctx);
typedef int (* gm_strided_kernel_t)(char **args, intptr_t *dimensions, intptr_t *steps, void *data);

/*
 * Collection of specialized kernels for a single function signature.
 *
 * NOTE: The specialized kernel lookup scheme is transitional and may
 * be replaced by something else.
 *
 * This should be considered as a first version of a kernel request
 * protocol.
 */
typedef struct {
    const ndt_t *sig;
    const ndt_constraint_t *constraint;

    /* Xnd signatures */
    gm_xnd_kernel_t OptC;    /* C in inner+1 dimensions */
    gm_xnd_kernel_t OptZ;    /* C in inner dimensions, C or zero stride in (inner+1)th. */
    gm_xnd_kernel_t OptS;    /* strided in (inner+1)th. */
    gm_xnd_kernel_t C;       /* C in inner dimensions */
    gm_xnd_kernel_t Fortran; /* Fortran in inner dimensions */
    gm_xnd_kernel_t Xnd;     /* selected if non-contiguous or the other fields are NULL */

    /* NumPy signature */
    gm_strided_kernel_t Strided;
} gm_kernel_set_t;

typedef struct {
    const char *name;
    const char *type;
    const ndt_methods_t *meth;
} gm_typedef_init_t;

typedef struct {
    const char *name;
    const char *sig;
    const ndt_constraint_t *constraint;
    uint32_t cap;

    /* Xnd signatures */
    gm_xnd_kernel_t OptC;
    gm_xnd_kernel_t OptZ;
    gm_xnd_kernel_t OptS;
    gm_xnd_kernel_t C;
    gm_xnd_kernel_t Fortran;
    gm_xnd_kernel_t Xnd;

    /* NumPy signature */
    gm_strided_kernel_t Strided;
} gm_kernel_init_t;

/* Actual kernel selected for application */
typedef struct {
    uint32_t flag;
    const gm_kernel_set_t *set;
} gm_kernel_t;

/* Multimethod with associated kernels */
typedef struct gm_func gm_func_t;
typedef const gm_kernel_set_t *(*gm_typecheck_t)(ndt_apply_spec_t *spec, const gm_func_t *f,
                                                 const ndt_t *in[], const int64_t li[],
                                                 int nin, int nout, bool check_broadcast,
                                                 ndt_context_t *ctx);
struct gm_func {
    char *name;
    gm_typecheck_t typecheck; /* Experimental optimized type-checking, may be NULL. */
    int nkernels;
    gm_kernel_set_t kernels[GM_MAX_KERNELS];
};


typedef struct _gm_tbl gm_tbl_t;


/******************************************************************************/
/*                                  Functions                                 */
/******************************************************************************/

GM_API gm_func_t *gm_func_new(const char *name, ndt_context_t *ctx);
GM_API void gm_func_del(gm_func_t *f);

GM_API gm_func_t *gm_add_func(gm_tbl_t *tbl, const char *name, ndt_context_t *ctx);
GM_API int gm_add_kernel(gm_tbl_t *tbl, const gm_kernel_init_t *kernel, ndt_context_t *ctx);
GM_API int gm_add_kernel_typecheck(gm_tbl_t *tbl, const gm_kernel_init_t *kernel, ndt_context_t *ctx, gm_typecheck_t f);

GM_API gm_kernel_t gm_select(ndt_apply_spec_t *spec, const gm_tbl_t *tbl, const char *name,
                             const ndt_t *types[], const int64_t li[], int nin, int nout,
                             bool check_broadcast, const xnd_t args[], ndt_context_t *ctx);
GM_API int gm_apply(const gm_kernel_t *kernel, xnd_t stack[], int outer_dims, ndt_context_t *ctx);
GM_API int gm_apply_thread(const gm_kernel_t *kernel, xnd_t stack[], int outer_dims, const int64_t nthreads, ndt_context_t *ctx);


/******************************************************************************/
/*                                NumPy loops                                 */
/******************************************************************************/

GM_API int gm_np_flatten(char **args, const int nargs,
                         int64_t *dimensions, int64_t *strides, const xnd_t stack[],
                         ndt_context_t *ctx);

GM_API int gm_np_convert_xnd(char **args, const int nargs,
                             intptr_t *dimensions, const int dims_size,
                             intptr_t *steps, const int steps_size,
                             xnd_t stack[], const int outer_dims,
                             ndt_context_t *ctx);

GM_API int gm_np_map(const gm_strided_kernel_t f,
                     char **args, int nargs,
                     intptr_t *dimensions,
                     intptr_t *steps,
                     void *data,
                     int outer_dims);


/******************************************************************************/
/*                                  Xnd loops                                 */
/******************************************************************************/

GM_API int array_shape_check(xnd_t *x, const int64_t shape, ndt_context_t *ctx);
GM_API int gm_xnd_map(const gm_xnd_kernel_t f, xnd_t stack[], const int nargs,
                      const int outer_dims, ndt_context_t *ctx);


/******************************************************************************/
/*                                Gufunc table                                */
/******************************************************************************/
GM_API gm_tbl_t *gm_tbl_new(ndt_context_t *ctx);
GM_API void gm_tbl_del(gm_tbl_t *t);

GM_API int gm_tbl_add(gm_tbl_t *tbl, const char *key, gm_func_t *value, ndt_context_t *ctx);
GM_API gm_func_t *gm_tbl_find(const gm_tbl_t *tbl, const char *key, ndt_context_t *ctx);
GM_API int gm_tbl_map(const gm_tbl_t *tbl, int (*f)(const gm_func_t *, void *state), void *state);


/******************************************************************************/
/*                       Library initialization and tables                    */
/******************************************************************************/

GM_API void gm_init(void);
GM_API int gm_init_cpu_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);
GM_API int gm_init_cpu_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);
GM_API int gm_init_bitwise_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);

GM_API int gm_init_cuda_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);
GM_API int gm_init_cuda_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);

GM_API int gm_init_example_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);
GM_API int gm_init_graph_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);
GM_API int gm_init_quaternion_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);
GM_API int gm_init_pdist_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);

GM_API void gm_finalize(void);


#ifdef __cplusplus
} /* END extern "C" */
#endif


#endif /* GUMATH_H */

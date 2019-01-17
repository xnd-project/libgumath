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


#ifndef PYGUMATH_H
#define PYGUMATH_H
#ifdef __cplusplus
extern "C" {
#endif


#include <Python.h>
#include "gumath.h"


/****************************************************************************/
/*                               Gufunc Object                              */
/****************************************************************************/

/* Exposed here for the benefit of Numba. The API should not be regarded
   stable across versions. */

#define GM_CPU_FUNC  0x0001U
#define GM_CUDA_MANAGED_FUNC 0x0002U

typedef struct {
    PyObject_HEAD
    const gm_tbl_t *tbl; /* kernel table */
    uint32_t flags;      /* memory target */
    char *name;          /* function name */
} GufuncObject;


/****************************************************************************/
/*                                Capsule API                               */
/****************************************************************************/

#define Gufunc_CheckExact_INDEX 0
#define Gufunc_CheckExact_RETURN int
#define Gufunc_CheckExact_ARGS (const PyObject *)

#define Gufunc_Check_INDEX 1
#define Gufunc_Check_RETURN int
#define Gufunc_Check_ARGS (const PyObject *)

#define Gumath_AddFunctions_INDEX 2
#define Gumath_AddFunctions_RETURN int
#define Gumath_AddFunctions_ARGS (PyObject *, const gm_tbl_t *)

#define Gumath_AddCudaFunctions_INDEX 3
#define Gumath_AddCudaFunctions_RETURN int
#define Gumath_AddCudaFunctions_ARGS (PyObject *, const gm_tbl_t *)

#define GUMATH_MAX_API 4


#ifdef GUMATH_MODULE
static Gufunc_CheckExact_RETURN Gufunc_CheckExact Gufunc_CheckExact_ARGS;
static Gufunc_Check_RETURN Gufunc_Check Gufunc_Check_ARGS;
static Gumath_AddFunctions_RETURN Gumath_AddFunctions Gumath_AddFunctions_ARGS;
static Gumath_AddCudaFunctions_RETURN Gumath_AddCudaFunctions Gumath_AddCudaFunctions_ARGS;
#else
static void **_gumath_api;

#define Gufunc_CheckExact \
    (*(Gufunc_CheckExact_RETURN (*)Gufunc_CheckExact_ARGS) _gumath_api[Gufunc_CheckExact_INDEX])

#define Gufunc_Check \
    (*(Gufunc_Check_RETURN (*)Gufunc_Check_ARGS) _gumath_api[Gufunc_Check_INDEX])

#define Gumath_AddFunctions \
    (*(Gumath_AddFunctions_RETURN (*)Gumath_AddFunctions_ARGS) _gumath_api[Gumath_AddFunctions_INDEX])

#define Gumath_AddCudaFunctions \
    (*(Gumath_AddCudaFunctions_RETURN (*)Gumath_AddCudaFunctions_ARGS) _gumath_api[Gumath_AddCudaFunctions_INDEX])


static int
import_gumath(void)
{
    _gumath_api = (void **)PyCapsule_Import("gumath._gumath._API", 0);
    if (_gumath_api == NULL) {
        return -1;
    }

    return 0;
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* PYGUMATH_H */

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


#include "gumath.h"


#define Gumath_AddFunctions_INDEX 0
#define Gumath_AddFunctions_RETURN int
#define Gumath_AddFunctions_ARGS (PyObject *, const gm_tbl_t *)

#define GUMATH_MAX_API 1


#ifdef GUMATH_MODULE
static Gumath_AddFunctions_RETURN Gumath_AddFunctions Gumath_AddFunctions_ARGS;
#else
static void **_gumath_api;

#define Gumath_AddFunctions \
    (*(Gumath_AddFunctions_RETURN (*)Gumath_AddFunctions_ARGS) _gumath_api[Gumath_AddFunctions_INDEX])


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

/* BSD 3-Clause License
 *
 * Copyright (c) 2018, Quansight and Sameer Deshmukh
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
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
#ifndef GUFUNC_OBJECT_H
#define GUFUNC_OBJECT_H

#include "ruby_gumath_internal.h"

typedef struct {
  const gm_tbl_t *table;          /* kernel table */
  char *name;                     /* function name */
} GufuncObject;

extern const rb_data_type_t GufuncObject_type;
extern VALUE cGumath_GufuncObject;

#define GET_GUOBJ(obj, guobj_p) do {                              \
    TypedData_Get_Struct((obj), GufuncObject,                     \
                         &GufuncObject_type, guobj_p);            \
  } while (0)
#define MAKE_GUOBJ(klass, guobj_p) TypedData_Make_Struct(klass, GufuncObject, \
                                                          &GufuncObject_type, guobj_p)
#define WRAP_GUOBJ(klass, guobj_p) TypedData_Wrap_Struct(klass,         \
                                                          &GufuncObject_type, guobj_p)

VALUE GufuncObject_alloc(const gm_tbl_t *table, const char *name);

#endif

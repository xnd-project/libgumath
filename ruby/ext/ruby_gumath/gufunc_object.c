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

#include "gufunc_object.h"

/****************************************************************************/
/*                               Gufunc Object                              */
/****************************************************************************/

VALUE cGumath_GufuncObject;

static void
GufuncObject_dfree(void *self)
{
  GufuncObject *guobj = (GufuncObject*)self;

  ndt_free(guobj->name);
}

static size_t
GufuncObject_dsize(const void *self)
{
  return sizeof(GufuncObject);
}

const rb_data_type_t GufuncObject_type = {
  .wrap_struct_name = "GufuncObject",
  .function = {
    .dmark = NULL,
    .dfree = GufuncObject_dfree,
    .dsize = GufuncObject_dsize,
    .reserved = {0,0},
  },
  .parent = 0,
  .flags = RUBY_TYPED_FREE_IMMEDIATELY,
};

VALUE
GufuncObject_alloc(const gm_tbl_t *table, const char *name)
{
  NDT_STATIC_CONTEXT(ctx);
  GufuncObject *guobj_p;
  VALUE guobj;

  guobj = MAKE_GUOBJ(cGumath_GufuncObject, guobj_p);
  guobj_p->table = table;
  guobj_p->name = ndt_strdup(name, &ctx);
  if (guobj_p->name == NULL) {
    seterr(&ctx);
  }

  return guobj;
}

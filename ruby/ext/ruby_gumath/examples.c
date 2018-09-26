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
#include <ruby.h>
#include "ndtypes.h"
#include "ruby_ndtypes.h"
#include "gumath.h"
#include "gufunc_object.h"
#include "ruby_gumath.h"
#include "util.h"

/****************************************************************************/
/*                              Static globals                              */
/****************************************************************************/

/* Function table */
static gm_tbl_t *table = NULL;
static VALUE mGumath_Examples;
static int initialized = 0;

/****************************************************************************/
/*                              Singleton methods                           */
/****************************************************************************/
static VALUE
mGumath_Examples_s_method_missing(int argc, VALUE *argv, VALUE module)
{
  VALUE method_name = argv[0];
  VALUE method_hash = rb_ivar_get(module, GUMATH_FUNCTION_HASH);
  VALUE gumath_method = rb_hash_aref(method_hash, method_name);
  int is_present = RTEST(gumath_method);
  
  if (is_present) {
    rb_funcall2(gumath_method, rb_intern("call"), argc-1, &argv[1]);
  }
  else {
    VALUE str = rb_funcall(method_name, rb_intern("to_s"), 0, NULL);
    rb_raise(rb_eNoMethodError, "Method %s not present in this gumath module.",
             StringValueCStr(str));
  }
}

void
Init_gumath_examples(void)
{
  NDT_STATIC_CONTEXT(ctx);

  if (!initialized) {
    if (!ndt_exists()) {
      rb_raise(rb_eLoadError, "NDT is needed for making gumath work.");
    }

    if (!xnd_exists()) {
      rb_raise(rb_eLoadError, "XND is needed for making gumath work.");
    }

    table = gm_tbl_new(&ctx);
    if (table == NULL) {
      seterr(&ctx);
      raise_error();
    }
   
    /* load custom examples */
    if (gm_init_example_kernels(table, &ctx) < 0) {
      seterr(&ctx);
      raise_error();
    }

    /* extending examples */
    if (gm_init_graph_kernels(table, &ctx) < 0) {
      seterr(&ctx);
      raise_error();
    }

#ifndef _MSC_VER
    if (gm_init_quaternion_kernels(table, &ctx) < 0) {
      seterr(&ctx);
      raise_error();
    }
#endif
    if (gm_init_pdist_kernels(table, &ctx) < 0) {
      seterr(&ctx);
      raise_error(); 
    }

    initialized = 1;
  }

  mGumath_Examples = rb_define_module_under(cGumath, "Examples");
  rb_ivar_set(mGumath_Examples, GUMATH_FUNCTION_HASH, rb_hash_new());

  if (rb_gumath_add_functions(mGumath_Examples, table) < 0) {
    mGumath_Examples = Qundef;
    rb_raise(rb_eLoadError, "failed to load functions into Gumath::Examples module.");
  }

  /* Singleton methods */
  rb_define_singleton_method(mGumath_Examples, "method_missing",
                             mGumath_Examples_s_method_missing, -1);
}

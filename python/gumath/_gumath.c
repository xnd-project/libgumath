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


#include <Python.h>
#include "ndtypes.h"
#include "pyndtypes.h"
#include "xnd.h"
#include "pyxnd.h"
#include "gumath.h"


#ifdef _MSC_VER
  #ifndef UNUSED
    #define UNUSED
  #endif
#else
  #if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define UNUSED __attribute__((unused))
  #else
    #define UNUSED
  #endif
#endif


/* libxnd.so is not linked without at least one xnd symbol. The -no-as-needed
 * linker option is difficult to integrate into setup.py. */
const void *dummy = &xnd_error;


/****************************************************************************/
/*                               Error handling                             */
/****************************************************************************/

static PyObject *
seterr(ndt_context_t *ctx)
{
    return Ndt_SetError(ctx);
}


/****************************************************************************/
/*                              Function object                             */
/****************************************************************************/

typedef struct {
    PyObject_HEAD
    char *name;
} GufuncObject;

static PyTypeObject Gufunc_Type;

static PyObject *
gufunc_new(const char *name)
{
    NDT_STATIC_CONTEXT(ctx);
    GufuncObject *self;

    self = PyObject_New(GufuncObject, &Gufunc_Type);
    if (self == NULL) {
        return NULL;
    }

    self->name = ndt_strdup(name, &ctx);
    if (self->name == NULL) {
        return seterr(&ctx);
    }

    return (PyObject *)self;
}

static void
gufunc_dealloc(GufuncObject *self)
{
    ndt_free(self->name);
    PyObject_Del(self);
}


/****************************************************************************/
/*                              Function calls                              */
/****************************************************************************/

static void
clear_objects(PyObject **a, Py_ssize_t len)
{
    Py_ssize_t i;

    for (i = 0; i < len; i++) {
        Py_CLEAR(a[i]);
    }
}

static PyObject *
gufunc_call(GufuncObject *self, PyObject *args, PyObject *kwds)
{
    NDT_STATIC_CONTEXT(ctx);
    const Py_ssize_t nin = PyTuple_GET_SIZE(args);
    PyObject **a = &PyTuple_GET_ITEM(args, 0);
    PyObject *result[NDT_MAX_ARGS];
    ndt_apply_spec_t spec = ndt_apply_spec_empty;
    const ndt_t *in_types[NDT_MAX_ARGS];
    xnd_t stack[NDT_MAX_ARGS];
    gm_kernel_t kernel;
    int i;

    if (kwds) {
        PyErr_SetString(PyExc_TypeError,
            "gufunc calls do not support keywords");
        return NULL;
    }

    if (nin > NDT_MAX_ARGS) {
        PyErr_SetString(PyExc_TypeError,
            "invalid number of arguments");
        return NULL;
    }

    for (i = 0; i < nin; i++) {
        if (!Xnd_Check(a[i])) {
            PyErr_SetString(PyExc_TypeError, "arguments must be xnd");
            return NULL;
        }
        stack[i] = *CONST_XND(a[i]);
        in_types[i] = stack[i].type;
    }

    kernel = gm_select(&spec, self->name, in_types, nin, &ctx);
    if (kernel.set == NULL) {
        return seterr(&ctx);
    }

    if (spec.nbroadcast > 0) {
        for (i = 0; i < nin; i++) {
            stack[i].type = spec.broadcast[i];
        }
    }

    for (i = 0; i < spec.nout; i++) {
        PyObject *x = Xnd_EmptyFromType(Py_TYPE(a[i]), spec.out[i]);
        if (x == NULL) {
            clear_objects(result, i);
            ndt_apply_spec_clear(&spec);
            return NULL;
        }
        result[i] = x;
        stack[nin+i] = *CONST_XND(x);
    }

    if (gm_apply(&kernel, stack, spec.outer_dims, &ctx) < 0) {
        for (i = 0; i < spec.nout; i++) {
            Py_DECREF(result[i]);
        }
        return seterr(&ctx);
    }

    switch (spec.nout) {
    case 0: Py_RETURN_NONE;
    case 1: return result[0];
    default: {
        PyObject *tuple = PyTuple_New(spec.nout);
        if (tuple == NULL) {
            clear_objects(result, spec.nout);
            return NULL;
        }
        for (i = 0; i < spec.nout; i++) {
            PyTuple_SET_ITEM(tuple, i, result[i]);
        }
        return tuple;
      }
    }
}

static PyObject *
gufunc_kernels(GufuncObject *self, PyObject *args UNUSED)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *list, *tmp;
    const gm_func_t *f;
    char *s;
    int i;

    f = gm_tbl_find(self->name, &ctx);
    if (f == NULL) {
        return seterr(&ctx);
    }

    list = PyList_New(f->nkernels);
    if (list == NULL) {
        return NULL;
    }

    for (i = 0; i < f->nkernels; i++) {
        s = ndt_as_string(f->kernels[i].sig, &ctx);
        if (s == NULL) {
            Py_DECREF(list);
            return seterr(&ctx);
        }

        tmp = PyUnicode_FromString(s);
        ndt_free(s);
        if (tmp == NULL) {
            Py_DECREF(list);
            return NULL;
        }

        PyList_SET_ITEM(list, i, tmp);
    }

    return list;
}

static int
add_function(const gm_func_t *f, void *state)
{
    PyObject *m = (PyObject *)state;
    PyObject *func;

    func = gufunc_new(f->name);
    if (func == NULL) {
        return -1;
    }

    return PyModule_AddObject(m, f->name, func);
}


static PyGetSetDef gufunc_getsets [] =
{
  { "kernels", (getter)gufunc_kernels, NULL, NULL, NULL},
  {NULL}
};


static PyTypeObject Gufunc_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_gumath.gufunc",
    .tp_basicsize = sizeof(GufuncObject),
    .tp_dealloc = (destructor)gufunc_dealloc,
    .tp_hash = PyObject_HashNotImplemented,
    .tp_call = (ternaryfunc)gufunc_call,
    .tp_getattro = PyObject_GenericGetAttr,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_getset = gufunc_getsets
};


/****************************************************************************/
/*                                  Module                                  */
/****************************************************************************/

static struct PyModuleDef gumath_module = {
    PyModuleDef_HEAD_INIT,        /* m_base */
    "_gumath",                    /* m_name */
    NULL,                         /* m_doc */
    -1,                           /* m_size */
    NULL,                         /* m_methods */
    NULL,                         /* m_slots */
    NULL,                         /* m_traverse */
    NULL,                         /* m_clear */
    NULL                          /* m_free */
};


PyMODINIT_FUNC
PyInit__gumath(void)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *m = NULL;
    static int initialized = 0;

    if (!initialized) {
       if (import_ndtypes() < 0) {
            return NULL;
       }
       if (import_xnd() < 0) {
            return NULL;
       }
       if (gm_init(&ctx) < 0) {
           return seterr(&ctx);
       }
       if (gm_sin_init(&ctx) < 0) {
           return seterr(&ctx);
       }
       initialized = 1;
    }

    if (PyType_Ready(&Gufunc_Type) < 0) {
        return NULL;
    }

    m = PyModule_Create(&gumath_module);
    if (m == NULL) {
        goto error;
    }

    if (gm_tbl_map(add_function, m) < 0) {
        goto error;
    }


    return m;

error:
    Py_CLEAR(m);
    return NULL;
}

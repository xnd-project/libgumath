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

#define GUMATH_MODULE
#include "pygumath.h"


/* libxnd.so is not linked without at least one xnd symbol. The -no-as-needed
 * linker option is difficult to integrate into setup.py. */
const void *dummy = NULL;


/****************************************************************************/
/*                              Module globals                              */
/****************************************************************************/

/* Function table */
static gm_tbl_t *table = NULL;

/* Xnd type */
static PyTypeObject *xnd = NULL;


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

static PyTypeObject Gufunc_Type;

static PyObject *
gufunc_new(const gm_tbl_t *tbl, const char *name)
{
    NDT_STATIC_CONTEXT(ctx);
    GufuncObject *self;

    self = PyObject_New(GufuncObject, &Gufunc_Type);
    if (self == NULL) {
        return NULL;
    }

    self->tbl = tbl;

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
    int i, k;

    if (kwds && PyDict_Size(kwds) > 0) {
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

    kernel = gm_select(&spec, self->tbl, self->name, in_types, (int)nin, stack, &ctx);
    if (kernel.set == NULL) {
        return seterr(&ctx);
    }

    if (spec.nbroadcast > 0) {
        for (i = 0; i < nin; i++) {
            stack[i].type = spec.broadcast[i];
        }
    }

    for (i = 0; i < spec.nout; i++) {
        if (ndt_is_concrete(spec.out[i])) {
            PyObject *x = Xnd_EmptyFromType(xnd, spec.out[i]);
            if (x == NULL) {
                clear_objects(result, i);
                ndt_apply_spec_clear(&spec);
                return NULL;
            }
            result[i] = x;
            stack[nin+i] = *CONST_XND(x);
         }
         else {
            result[i] = NULL;
            stack[nin+i] = xnd_error;
         }
    }

    if (gm_apply(&kernel, stack, spec.outer_dims, &ctx) < 0) {
        clear_objects(result, spec.nout);
        return seterr(&ctx);
    }

    for (i = 0; i < spec.nout; i++) {
        if (ndt_is_abstract(spec.out[i])) {
            ndt_del(spec.out[i]);
            PyObject *x = Xnd_FromXnd(xnd, &stack[nin+i]);
            stack[nin+i] = xnd_error;
            if (x == NULL) {
                clear_objects(result, i);
                for (k = i+1; k < spec.nout; k++) {
                    if (ndt_is_abstract(spec.out[k])) {
                        xnd_del_buffer(&stack[nin+k], XND_OWN_ALL);
                    }
                }
            }
            result[i] = x;
        }
    }

    if (spec.nbroadcast > 0) {
        for (i = 0; i < nin; i++) {
            ndt_del(spec.broadcast[i]);
        }
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
gufunc_kernels(GufuncObject *self, PyObject *args GM_UNUSED)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *list, *tmp;
    const gm_func_t *f;
    char *s;
    int i;

    f = gm_tbl_find(self->tbl, self->name, &ctx);
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
/*                                   C-API                                  */
/****************************************************************************/

static void **gumath_api[GUMATH_MAX_API];

struct map_args {
    PyObject *module;
    const gm_tbl_t *tbl;
};

static int
add_function(const gm_func_t *f, void *args)
{
    struct map_args *a = (struct map_args *)args;
    PyObject *func;

    func = gufunc_new(a->tbl, f->name);
    if (func == NULL) {
        return -1;
    }

    return PyModule_AddObject(a->module, f->name, func);
}

static int
Gumath_AddFunctions(PyObject *m, const gm_tbl_t *tbl)
{
    struct map_args args = {m, tbl};

    if (gm_tbl_map(tbl, add_function, &args) < 0) {
        return -1;
    }

    return 0;
}

static PyObject *
init_api(void)
{
    gumath_api[Gumath_AddFunctions_INDEX] = (void *)Gumath_AddFunctions;

    return PyCapsule_New(gumath_api, "gumath._gumath._API", NULL);
}


/****************************************************************************/
/*                                  Module                                  */
/****************************************************************************/

static PyObject *
unsafe_add_kernel(PyObject *m GM_UNUSED, PyObject *args, PyObject *kwds)
{
    NDT_STATIC_CONTEXT(ctx);
    static char *kwlist[] = {"name", "sig", "tag", "ptr", NULL};
    gm_kernel_init_t k = {NULL};
    gm_func_t *f;
    char *name;
    char *sig;
    char *tag;
    PyObject *ptr;
    void *p;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sssO", kwlist, &name, &sig,
        &tag, &ptr)) {
        return NULL;
    }

    p = PyLong_AsVoidPtr(ptr);
    if (p == NULL) {
        return NULL;
    }

    k.name = name;
    k.sig = sig;

    if (strcmp(tag, "C") == 0) {
        k.C = p;
    }
    else if (strcmp(tag, "Fortran") == 0) {
        k.Fortran = p;
    }
    else if (strcmp(tag, "Xnd") == 0) {
        k.Xnd = p;
    }
    else if (strcmp(tag, "Strided") == 0) {
        k.Strided = p;
    }
    else {
        PyErr_SetString(PyExc_ValueError,
            "tag must be 'C', 'Fortran', 'Xnd' or 'Strided'");
        return NULL;
    }

    if (gm_add_kernel(table, &k, &ctx) < 0) {
        return seterr(&ctx);
    }

    f = gm_tbl_find(table, name, &ctx);
    if (f == NULL) {
        return seterr(&ctx);
    }

    return gufunc_new(table, f->name);
}

static PyMethodDef gumath_methods [] =
{
  /* Methods */
  { "unsafe_add_kernel", (PyCFunction)unsafe_add_kernel, METH_VARARGS|METH_KEYWORDS, NULL },
  { NULL, NULL, 1 }
};


static struct PyModuleDef gumath_module = {
    PyModuleDef_HEAD_INIT,        /* m_base */
    "_gumath",                    /* m_name */
    NULL,                         /* m_doc */
    -1,                           /* m_size */
    gumath_methods,               /* m_methods */
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
    static PyObject *capsule = NULL;
    static int initialized = 0;
    PyObject *obj = NULL;

    if (!initialized) {
       dummy = &xnd_error;

       gm_init();

       if (import_ndtypes() < 0) {
            return NULL;
       }
       if (import_xnd() < 0) {
            return NULL;
       }

       capsule = init_api();
       if (capsule == NULL) {
            return NULL;
       }

       table = gm_tbl_new(&ctx);
       if (table == NULL) {
           return seterr(&ctx);
       }

       initialized = 1;
    }

    if (PyType_Ready(&Gufunc_Type) < 0) {
        return NULL;
    }

    obj = PyImport_ImportModule("xnd");
    if (obj == NULL) {
        return NULL;
    }
    xnd = (PyTypeObject *)PyObject_GetAttrString(obj, "xnd");
    Py_CLEAR(obj);
    if (xnd == NULL) {
        goto error;
    }

    m = PyModule_Create(&gumath_module);
    if (m == NULL) {
        goto error;
    }

    Py_INCREF(capsule);
    if (PyModule_AddObject(m, "_API", capsule) < 0) {
        goto error;
    }

    if (Gumath_AddFunctions(m, table) < 0) {
        goto error;
    }

    return m;

error:
    Py_CLEAR(xnd);
    Py_CLEAR(m);
    return NULL;
}

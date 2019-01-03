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

#ifndef _MSC_VER
  #include "config.h"
#endif

#define GUMATH_MODULE
#include "pygumath.h"


#ifdef _MSC_VER
  #ifndef UNUSED
    #define UNUSED
  #endif
  #include <float.h>
  #include <fenv.h>
  #pragma fenv_access(on)
#else
  #if defined(__GNUC__) && !defined(__INTEL_COMPILER)
    #define UNUSED __attribute__((unused))
  #else
    #define UNUSED
  #endif
  #include <fenv.h>
  #if 0 /* Not supported by gcc and clang. */
    #pragma STDC FENV_ACCESS ON
  #endif
#endif


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

/* Maximum number of threads */
static int64_t max_threads = 1;


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
gufunc_new(const gm_tbl_t *tbl, const char *name, const uint32_t flags)
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

    self->flags = flags;

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
clear_pystack(PyObject *pystack[], Py_ssize_t len)
{
    for (Py_ssize_t i = 0; i < len; i++) {
        Py_CLEAR(pystack[i]);
    }
}

static PyObject *
get_item_with_error(PyObject *d, const char *key)
{
    PyObject *s, *v;

    s = PyUnicode_FromString(key);
    if (s == NULL) {
        return NULL;
    }

    v = PyDict_GetItemWithError(d, s);
    Py_DECREF(s);
    return v;
}

static int
parse_args(PyObject *pystack[NDT_MAX_ARGS], int *py_nin, int *py_nout, int *py_nargs,
           PyObject *args, PyObject *kwargs)
{
    Py_ssize_t nin;
    Py_ssize_t nout;

    if (!args || !PyTuple_Check(args)) {
        const char *tuple = args ? Py_TYPE(args)->tp_name : "NULL";
        PyErr_Format(PyExc_SystemError,
            "internal error: expected tuple, got '%.200s'",
            tuple);
        return -1;
    }

    if (kwargs && !PyDict_Check(kwargs)) {
        PyErr_Format(PyExc_SystemError,
            "internal error: expected dict, got '%.200s'",
            Py_TYPE(kwargs)->tp_name);
        return -1;
    }

    nin = PyTuple_GET_SIZE(args);
    if (nin > NDT_MAX_ARGS) {
        PyErr_Format(PyExc_TypeError,
            "maximum number of arguments is %d, got %n", NDT_MAX_ARGS, nin);
        return -1;
    }

    for (Py_ssize_t i = 0; i < nin; i++) {
        PyObject *v = PyTuple_GET_ITEM(args, i);
        if (!Xnd_Check(v)) {
            PyErr_Format(PyExc_TypeError,
                "expected xnd argument, got '%.200s'", Py_TYPE(v)->tp_name);
            return -1;
        }

        pystack[i] = v;
    }

    if (kwargs == NULL || PyDict_Size(kwargs) == 0) {
        nout = 0;
    }
    else if (PyDict_Size(kwargs) == 1) {
        PyObject *out = get_item_with_error(kwargs, "out");
        if (out == NULL) {
            if (PyErr_Occurred()) {
                return -1;
            }
            PyErr_SetString(PyExc_TypeError,
                "the only supported keyword argument is 'out'");
            return -1;
        }

        if (out == Py_None) {
            nout = 0;
        }
        else if (Xnd_Check(out)) {
            nout = 1;
            if (nin+nout > NDT_MAX_ARGS) {
                PyErr_Format(PyExc_TypeError,
                    "maximum number of arguments is %d, got %n", NDT_MAX_ARGS, nin+nout);
                return -1;
            }

            pystack[nin] = out;
        }
        else if (PyTuple_Check(out)) {
            nout = PyTuple_GET_SIZE(out);
            if (nout > NDT_MAX_ARGS || nin+nout > NDT_MAX_ARGS) {
                PyErr_Format(PyExc_TypeError,
                    "maximum number of arguments is %d, got %n", NDT_MAX_ARGS, nin+nout);
                return -1;
            }

            for (Py_ssize_t i = 0; i < nout; i++) {
                PyObject *v = PyTuple_GET_ITEM(out, i);
                if (!Xnd_Check(v)) {
                    PyErr_Format(PyExc_TypeError,
                        "expected xnd argument, got '%.200s'", Py_TYPE(v)->tp_name);
                    return -1;
                }

                pystack[nin+i] = v;
            }
        }
        else {
            PyErr_Format(PyExc_TypeError,
                "'out' argument must be xnd or a tuple of xnd, got '%.200s'",
                Py_TYPE(out)->tp_name);
            return -1;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
            "the only supported keyword argument is 'out'");
        return -1;
    }

    for (int i = 0; i < nin+nout; i++) {
        Py_INCREF(pystack[i]);
    }

    *py_nin = (int)nin;
    *py_nout = (int)nout;
    *py_nargs = (int)nin+(int)nout;

    return 0;
}

static PyObject *
gufunc_call(GufuncObject *self, PyObject *args, PyObject *kwargs)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *pystack[NDT_MAX_ARGS];
    xnd_t stack[NDT_MAX_ARGS];
    const ndt_t *types[NDT_MAX_ARGS];
    int64_t li[NDT_MAX_ARGS];
    ndt_apply_spec_t spec = ndt_apply_spec_empty;
    gm_kernel_t kernel;
    bool have_cpu_device = false;
    int nin, nout, nargs;

    if (parse_args(pystack, &nin, &nout, &nargs, args, kwargs) < 0) {
        return NULL;
    }

    for (int i = 0; i < nargs; i++) {
        const XndObject *x = (XndObject *)pystack[i];
        if (!(x->mblock->xnd->flags&XND_CUDA_MANAGED)) {
            have_cpu_device = true;
        }

        stack[i] = *CONST_XND((PyObject *)x);
        types[i] = stack[i].type;
        li[i] = stack[i].index;
    }

    if (have_cpu_device) {
        if (self->flags & GM_CUDA_MANAGED_FUNC) {
            PyErr_SetString(PyExc_ValueError,
                "cannot run a cuda function on xnd objects with cpu memory");
            clear_pystack(pystack, nargs);
            return NULL;
        }
    }

    kernel = gm_select(&spec, self->tbl, self->name, types, li, nin, nout, stack, &ctx);
    if (kernel.set == NULL) {
        return seterr(&ctx);
    }

    /*
     * Replace args/kwargs types with types after substitution and broadcasting.
     * This includes 'out' types, if explicitly passed as kwargs.
     */
    for (int i = 0; i < spec.nargs; i++) {
        stack[i].type = spec.types[i];
    }

    if (nout == 0) {
        /* 'out' types have been inferred, create new XndObjects. */
        for (int i = 0; i < spec.nout; i++) {
            if (ndt_is_concrete(spec.types[nin+i])) {
                uint32_t flags = self->flags == GM_CUDA_MANAGED_FUNC ? XND_CUDA_MANAGED : 0;
                PyObject *x = Xnd_EmptyFromType(xnd, spec.types[nin+i], flags);
                if (x == NULL) {
                    clear_pystack(pystack, nin+i);
                    ndt_apply_spec_clear(&spec);
                return NULL;
            }
            pystack[nin+i] = x;
            stack[nin+i] = *CONST_XND(x);
            }
            else {
                clear_pystack(pystack, nin+i);
                ndt_apply_spec_clear(&spec);
                PyErr_SetString(PyExc_ValueError,
                    "arguments with abstract types are temporarily disabled");
                return NULL;
            }
        }
    }

    if (self->flags == GM_CUDA_MANAGED_FUNC) {
    #if HAVE_CUDA
        const int ret = gm_apply(&kernel, stack, spec.outer_dims, &ctx);

        if (xnd_cuda_device_synchronize(&ctx) < 0 || ret < 0) {
            clear_pystack(pystack, spec.nargs);
            ndt_apply_spec_clear(&spec);
            return seterr(&ctx);
        }
    #else
        ndt_err_format(&ctx, NDT_RuntimeError,
           "internal error: GM_CUDA_MANAGED_FUNC set in a build without cuda support");
        clear_pystack(pystack, spec.nargs);
        ndt_apply_spec_clear(&spec);
        return seterr(&ctx);
    #endif
    }
    else {
    #ifdef HAVE_PTHREAD_H
        const int rounding = fegetround();
        fesetround(FE_TONEAREST);

        const int ret = gm_apply_thread(&kernel, stack, spec.outer_dims,
                                        spec.flags, max_threads, &ctx);
        fesetround(rounding);

        if (ret < 0) {
            clear_pystack(pystack, spec.nargs);
            ndt_apply_spec_clear(&spec);
            return seterr(&ctx);
        }
    #else
        const int rounding = fegetround();
        fesetround(FE_TONEAREST);

        const int ret = gm_apply(&kernel, stack, spec.outer_dims, &ctx);

        fesetround(rounding);

        if (ret < 0) {
            clear_pystack(pystack, spec.nargs);
            ndt_apply_spec_clear(&spec);
            return seterr(&ctx);
        }
    #endif
    }

    nin = spec.nin;
    nout = spec.nout;
    nargs = spec.nargs;
    ndt_apply_spec_clear(&spec);

    switch (nout) {
    case 0: {
        clear_pystack(pystack, nargs);
        Py_RETURN_NONE;
    }
    case 1: {
        clear_pystack(pystack, nin);
        return pystack[nin];
    }
    default: {
        PyObject *tuple = PyTuple_New(nout);
        if (tuple == NULL) {
            clear_pystack(pystack, nargs);
            return NULL;
        }
        for (int i = 0; i < nout; i++) {
            PyTuple_SET_ITEM(tuple, i, pystack[nin+i]);
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

    func = gufunc_new(a->tbl, f->name, GM_CPU_FUNC);
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

static int
add_cuda_function(const gm_func_t *f, void *args)
{
    struct map_args *a = (struct map_args *)args;
    PyObject *func;

    func = gufunc_new(a->tbl, f->name, GM_CUDA_MANAGED_FUNC);
    if (func == NULL) {
        return -1;
    }

    return PyModule_AddObject(a->module, f->name, func);
}

static int
Gumath_AddCudaFunctions(PyObject *m, const gm_tbl_t *tbl)
{
    struct map_args args = {m, tbl};

    if (gm_tbl_map(tbl, add_cuda_function, &args) < 0) {
        return -1;
    }

    return 0;
}

static PyObject *
init_api(void)
{
    gumath_api[Gumath_AddFunctions_INDEX] = (void *)Gumath_AddFunctions;
    gumath_api[Gumath_AddCudaFunctions_INDEX] = (void *)Gumath_AddCudaFunctions;

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

    if (strcmp(tag, "Opt") == 0) {
        k.Opt = p;
    }
    else if (strcmp(tag, "C") == 0) {
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
            "tag must be 'Opt', 'C', 'Fortran', 'Xnd' or 'Strided'");
        return NULL;
    }

    if (gm_add_kernel(table, &k, &ctx) < 0) {
        return seterr(&ctx);
    }

    f = gm_tbl_find(table, name, &ctx);
    if (f == NULL) {
        return seterr(&ctx);
    }

    return gufunc_new(table, f->name, GM_CPU_FUNC);
}

static void
init_max_threads(void)
{
    PyObject *os = NULL;
    PyObject *n = NULL;
    int64_t i64;

    os = PyImport_ImportModule("os");
    if (os == NULL) {
        goto error;
    }

    n = PyObject_CallMethod(os, "cpu_count", "()");
    if (n == NULL) {
        goto error;
    }

    i64 = PyLong_AsLongLong(n);
    if (i64 < 1) {
        goto error;
    }

    max_threads = i64;

out:
    Py_XDECREF(os);
    Py_XDECREF(n);
    return;

error:
    if (PyErr_Occurred()) {
        PyErr_Clear();
    }
    PyErr_WarnEx(PyExc_RuntimeWarning,
        "could not get cpu count: using max_threads==1", 1);
    goto out;
}

static PyObject *
get_max_threads(PyObject *m UNUSED, PyObject *args UNUSED)
{
    return PyLong_FromLongLong(max_threads);
}

static PyObject *
set_max_threads(PyObject *m UNUSED, PyObject *obj)
{
    int64_t n;

    n = PyLong_AsLongLong(obj);
    if (n == -1 && PyErr_Occurred()) {
        return NULL;
    }

    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError,
            "max_threads must be greater than 0");
        return NULL;
    }

    max_threads = n;

    Py_RETURN_NONE;
}


static PyMethodDef gumath_methods [] =
{
  /* Methods */
  { "unsafe_add_kernel", (PyCFunction)unsafe_add_kernel, METH_VARARGS|METH_KEYWORDS, NULL },
  { "get_max_threads", (PyCFunction)get_max_threads, METH_NOARGS, NULL },
  { "set_max_threads", (PyCFunction)set_max_threads, METH_O, NULL },
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

       init_max_threads();

       initialized = 1;
    }

    if (PyType_Ready(&Gufunc_Type) < 0) {
        return NULL;
    }

    xnd = Xnd_GetType();
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

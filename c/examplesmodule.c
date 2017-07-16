
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define THIS_MODULE_NAME examples

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>

#include "gufuncs.h"

/* Some misc macros */
#define _CONCAT(a,b) a ## b
#define CONCAT(a,b) _CONCAT(a,b)
#define _STR(a) # a
#define STR(a) _STR(a)

#if defined(__GNUC__)
#  define UNUSED_VAR(x) CONCAT(UNUSED_, x) __attribute__((unused))
#elif defined(__LCLINT__)
#  define UNUSED_VAR(x) /*@unused@*/ CONCAT(UNUSED_, x)
#elif defined(__cplusplus)
#  define UNUSED_VAR(x)
#else
#  define UNUSED_VAR(x) CONCAT(UNUSED_, x)
#endif 

/* Python 3 support */
#if PY_MAJOR_VERSION >= 3
#   define PYTHON3
#   define MOD_INIT(name) PyMODINIT_FUNC CONCAT(PyInit_, name)(void)
#   define MOD_RETURN(val) do { return val; } while(0)
#else
#   define MOD_INIT(name) PyMODINIT_FUNC CONCAT(init, name)(void)
#   define MOD_RETURN(val) do {} while(0)
#endif



/* The method table */
static struct PyMethodDef methods[] = {
    { NULL, NULL, 0, NULL }   /* sentinel */
};

#if defined(PYTHON3)
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    STR(THIS_MODULE_NAME),
    NULL,
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif 

MOD_INIT(THIS_MODULE_NAME)
{
    PyObject *m = NULL;

#if defined(PYTHON3)
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule(STR(THIS_MODULE_NAME), methods);
#endif /* PYTHON3 */
    if (m == NULL)
        MOD_RETURN(m);

    /* add whatever modules we may want */
    if (add_example_gufuncs(m) != 0) {
        Py_DECREF(m);
        m = NULL;
        MOD_RETURN(m);
    }

    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Can not load " STR(THIS_MODULE_NAME) " module.");
    }

    MOD_RETURN(m);
}




#include <Python.h>
#include "ndtypes.h"
#include "pyndtypes.h"
#include "gumath.h"
#include "pygumath.h"


/****************************************************************************/
/*                              Module globals                              */
/****************************************************************************/

/* Function table */
static gm_tbl_t *table = NULL;


/****************************************************************************/
/*                                  Module                                  */
/****************************************************************************/

static struct PyModuleDef examples_module = {
    PyModuleDef_HEAD_INIT,        /* m_base */
    "examples",                   /* m_name */
    NULL,                         /* m_doc */
    -1,                           /* m_size */
    NULL,                         /* m_methods */
    NULL,                         /* m_slots */
    NULL,                         /* m_traverse */
    NULL,                         /* m_clear */
    NULL                          /* m_free */
};


PyMODINIT_FUNC
PyInit_examples(void)
{
    NDT_STATIC_CONTEXT(ctx);
    PyObject *m = NULL;
    static int initialized = 0;

    if (!initialized) {
       if (import_ndtypes() < 0) {
            return NULL;
       }
       if (import_gumath() < 0) {
            return NULL;
       }

       table = gm_tbl_new(&ctx);
       if (table == NULL) {
           return Ndt_SetError(&ctx);
       }

       /* custom examples */
       if (gm_init_example_kernels(table, &ctx) < 0) {
           return Ndt_SetError(&ctx);
       }

       /* extending examples */
       if (gm_init_graph_kernels(table, &ctx) < 0) {
           return Ndt_SetError(&ctx);
       }
#ifndef _MSC_VER
       if (gm_init_quaternion_kernels(table, &ctx) < 0) {
           return Ndt_SetError(&ctx);
       }
#endif
       if (gm_init_pdist_kernels(table, &ctx) < 0) {
           return Ndt_SetError(&ctx);
       }

       initialized = 1;
    }

    m = PyModule_Create(&examples_module);
    if (m == NULL) {
        goto error;
    }

    if (Gumath_AddFunctions(m, table) < 0) {
        goto error;
    }

    return m;

error:
    Py_CLEAR(m);
    return NULL;
}

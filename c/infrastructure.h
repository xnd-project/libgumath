#ifndef infrastructure_h_
#define infrastructure_h_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "Python.h"
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
/* #include "npy_config.h" */
/* #include "numpy/arrayobject.h" */
/* #include "numpy/ufuncobject.h" */
/* #include "numpy/arrayscalars.h" */
/* #include "lowlevel_strided_loops.h" */
/* #include "ufunc_type_resolution.h" */
/* #include "reduction.h" */
/* #include "mem_overlap.h" */

/* #include "ufunc_object.h" */
/* #include "override.h" */
/* #include "npy_import.h" */

PyObject *
PyGUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
				     char *types, int ntypes,
				     int nin, int nout, int identity,
				     const char *name, const char *doc,
				     int unused, const char *signature);

int
PyGUFunc_DefaultLegacyInnerLoopSelector(PyUFuncObject *ufunc,
                                       PyArray_Descr **dtypes,
                                       PyUFuncGenericFunction *out_innerloop,
                                       void **out_innerloopdata,
                                       int *out_needs_api);

int
PyGUFunc_DefaultMaskedInnerLoopSelector(PyUFuncObject *ufunc,
                                      PyArray_Descr **dtypes,
                                      PyArray_Descr *mask_dtypes,
                                      npy_intp *NPY_UNUSED(fixed_strides),
                                      npy_intp NPY_UNUSED(fixed_mask_stride),
                                      PyUFunc_MaskedStridedInnerLoopFunc 
                                      **out_innerloop,
                                      NpyAuxData **out_innerloopdata,
                                      int *out_needs_api);


#endif

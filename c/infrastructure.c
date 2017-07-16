#include "infrastructure.h"
//#include "lowlevel_strided_loops.h"
#include "npy_3kcompat.h"

/* Return the position of next non-white-space char in the string */
static int
_next_non_white_space(const char* str, int offset)
{
    int ret = offset;
    while (str[ret] == ' ' || str[ret] == '\t') {
        ret++;
    }
    return ret;
}

static int
_is_alpha_underscore(char ch)
{
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || ch == '_';
}

static int
_is_alnum_underscore(char ch)
{
    return _is_alpha_underscore(ch) || (ch >= '0' && ch <= '9');
}


/*
 * Return the ending position of a variable name
 */
static int
_get_end_of_name(const char* str, int offset)
{
    int ret = offset;
    while (_is_alnum_underscore(str[ret])) {
        ret++;
    }
    return ret;
}
/*
 * Returns 1 if the dimension names pointed by s1 and s2 are the same,
 * otherwise returns 0.
 */
static int
_is_same_name(const char* s1, const char* s2)
{
    while (_is_alnum_underscore(*s1) && _is_alnum_underscore(*s2)) {
        if (*s1 != *s2) {
            return 0;
        }
        s1++;
        s2++;
    }
    return !_is_alnum_underscore(*s1) && !_is_alnum_underscore(*s2);
}


/*
 * Sets core_num_dim_ix, core_num_dims, core_dim_ixs, core_offsets,
 * and core_signature in PyUFuncObject "ufunc".  Returns 0 unless an
 * error occurred.
 */
static int
_parse_signature(PyUFuncObject *ufunc, const char *signature)
{
    size_t len;
    char const **var_names;
    int nd = 0;             /* number of dimension of the current argument */
    int cur_arg = 0;        /* index into core_num_dims&core_offsets */
    int cur_core_dim = 0;   /* index into core_dim_ixs */
    int i = 0;
    char *parse_error = NULL;

    if (signature == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_parse_signature with NULL signature");
        return -1;
    }

    len = strlen(signature);
    ufunc->core_signature = PyArray_malloc(sizeof(char) * (len+1));
    if (ufunc->core_signature) {
        strcpy(ufunc->core_signature, signature);
    }
    /* Allocate sufficient memory to store pointers to all dimension names */
    var_names = PyArray_malloc(sizeof(char const*) * len);
    if (var_names == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    ufunc->core_enabled = 1;
    ufunc->core_num_dim_ix = 0;
    ufunc->core_num_dims = PyArray_malloc(sizeof(int) * ufunc->nargs);
    ufunc->core_dim_ixs = PyArray_malloc(sizeof(int) * len); /* shrink this later */
    ufunc->core_offsets = PyArray_malloc(sizeof(int) * ufunc->nargs);
    if (ufunc->core_num_dims == NULL || ufunc->core_dim_ixs == NULL
        || ufunc->core_offsets == NULL) {
        PyErr_NoMemory();
        goto fail;
    }

    i = _next_non_white_space(signature, 0);
    while (signature[i] != '\0') {
        /* loop over input/output arguments */
        if (cur_arg == ufunc->nin) {
            /* expect "->" */
            if (signature[i] != '-' || signature[i+1] != '>') {
                parse_error = "expect '->'";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 2);
        }

        /*
         * parse core dimensions of one argument,
         * e.g. "()", "(i)", or "(i,j)"
         */
        if (signature[i] != '(') {
            parse_error = "expect '('";
            goto fail;
        }
        i = _next_non_white_space(signature, i + 1);
        while (signature[i] != ')') {
            /* loop over core dimensions */
            int j = 0;
            if (!_is_alpha_underscore(signature[i])) {
                parse_error = "expect dimension name";
                goto fail;
            }
            while (j < ufunc->core_num_dim_ix) {
                if (_is_same_name(signature+i, var_names[j])) {
                    break;
                }
                j++;
            }
            if (j >= ufunc->core_num_dim_ix) {
                var_names[j] = signature+i;
                ufunc->core_num_dim_ix++;
            }
            ufunc->core_dim_ixs[cur_core_dim] = j;
            cur_core_dim++;
            nd++;
            i = _get_end_of_name(signature, i);
            i = _next_non_white_space(signature, i);
            if (signature[i] != ',' && signature[i] != ')') {
                parse_error = "expect ',' or ')'";
                goto fail;
            }
            if (signature[i] == ',')
            {
                i = _next_non_white_space(signature, i + 1);
                if (signature[i] == ')') {
                    parse_error = "',' must not be followed by ')'";
                    goto fail;
                }
            }
        }
        ufunc->core_num_dims[cur_arg] = nd;
        ufunc->core_offsets[cur_arg] = cur_core_dim-nd;
        cur_arg++;
        nd = 0;

        i = _next_non_white_space(signature, i + 1);
        if (cur_arg != ufunc->nin && cur_arg != ufunc->nargs) {
            /*
             * The list of input arguments (or output arguments) was
             * only read partially
             */
            if (signature[i] != ',') {
                parse_error = "expect ','";
                goto fail;
            }
            i = _next_non_white_space(signature, i + 1);
        }
    }
    if (cur_arg != ufunc->nargs) {
        parse_error = "incomplete signature: not all arguments found";
        goto fail;
    }
    ufunc->core_dim_ixs = PyArray_realloc(ufunc->core_dim_ixs,
            sizeof(int)*cur_core_dim);
    /* check for trivial core-signature, e.g. "(),()->()" */
    if (cur_core_dim == 0) {
        ufunc->core_enabled = 0;
    }
    PyArray_free((void*)var_names);
    return 0;

fail:
    PyArray_free((void*)var_names);
    if (parse_error) {
        PyErr_Format(PyExc_ValueError,
                     "%s at position %d in \"%s\"",
                     parse_error, i, signature);
    }
    return -1;
}


/*
 * memchr with stride and invert argument
 * intended for small searches where a call out to libc memchr is costly.
 * stride must be a multiple of size.
 * compared to memchr it returns one stride past end instead of NULL if needle
 * is not found.
 */
static NPY_INLINE char *
npy_memchr(char * haystack, char needle,
           npy_intp stride, npy_intp size, npy_intp * psubloopsize, int invert)
{
    char * p = haystack;
    npy_intp subloopsize = 0;

    if (!invert) {
        /*
         * this is usually the path to determine elements to process,
         * performance less important here.
         * memchr has large setup cost if 0 byte is close to start.
         */
        while (subloopsize < size && *p != needle) {
            subloopsize++;
            p += stride;
        }
    }
    else {
        /* usually find elements to skip path */
        if (NPY_CPU_HAVE_UNALIGNED_ACCESS && needle == 0 && stride == 1) {
            /* iterate until last multiple of 4 */
            char * block_end = haystack + size - (size % sizeof(unsigned int));
            while (p < block_end) {
                unsigned int  v = *(unsigned int*)p;
                if (v != 0) {
                    break;
                }
                p += sizeof(unsigned int);
            }
            /* handle rest */
            subloopsize = (p - haystack);
        }
        while (subloopsize < size && *p == needle) {
            subloopsize++;
            p += stride;
        }
    }

    *psubloopsize = subloopsize;

    return p;
}



/*UFUNC_API*/
NPY_NO_EXPORT PyObject *
PyGUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     int unused, const char *signature)
{
    PyUFuncObject *ufunc;

    if (nin + nout > NPY_MAXARGS) {
        PyErr_Format(PyExc_ValueError,
                     "Cannot construct a ufunc with more than %d operands "
                     "(requested number were: inputs = %d and outputs = %d)",
                     NPY_MAXARGS, nin, nout);
        return NULL;
    }

    ufunc = PyArray_malloc(sizeof(PyUFuncObject));
    if (ufunc == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)ufunc, &PyUFunc_Type);

    ufunc->reserved1 = 0;
    ufunc->reserved2 = NULL;

    ufunc->nin = nin;
    ufunc->nout = nout;
    ufunc->nargs = nin+nout;
    ufunc->identity = identity;

    ufunc->functions = func;
    ufunc->data = data;
    ufunc->types = types;
    ufunc->ntypes = ntypes;
    ufunc->ptr = NULL;
    ufunc->obj = NULL;
    ufunc->userloops=NULL;

    /* Type resolution and inner loop selection functions */
    ufunc->type_resolver = &PyUFunc_DefaultTypeResolver;
    ufunc->legacy_inner_loop_selector = &PyGUFunc_DefaultLegacyInnerLoopSelector;
    ufunc->masked_inner_loop_selector = &PyGUFunc_DefaultMaskedInnerLoopSelector;

    if (name == NULL) {
        ufunc->name = "?";
    }
    else {
        ufunc->name = name;
    }
    ufunc->doc = doc;

    ufunc->op_flags = PyArray_malloc(sizeof(npy_uint32)*ufunc->nargs);
    if (ufunc->op_flags == NULL) {
        return PyErr_NoMemory();
    }
    memset(ufunc->op_flags, 0, sizeof(npy_uint32)*ufunc->nargs);

    ufunc->iter_flags = 0;

    /* generalized ufunc */
    ufunc->core_enabled = 0;
    ufunc->core_num_dim_ix = 0;
    ufunc->core_num_dims = NULL;
    ufunc->core_dim_ixs = NULL;
    ufunc->core_offsets = NULL;
    ufunc->core_signature = NULL;
    if (signature != NULL) {
        if (_parse_signature(ufunc, signature) != 0) {
            Py_DECREF(ufunc);
            return NULL;
        }
    }
    return (PyObject *)ufunc;
}

static int
find_userloop(PyUFuncObject *ufunc,
                PyArray_Descr **dtypes,
                PyUFuncGenericFunction *out_innerloop,
                void **out_innerloopdata)
{
    npy_intp i, nin = ufunc->nin, j, nargs = nin + ufunc->nout;
    PyUFunc_Loop1d *funcdata;

    /* Use this to try to avoid repeating the same userdef loop search */
    int last_userdef = -1;

    for (i = 0; i < nargs; ++i) {
        int type_num;

        /* no more ufunc arguments to check */
        if (dtypes[i] == NULL) {
            break;
        }

        type_num = dtypes[i]->type_num;
        if (type_num != last_userdef &&
                (PyTypeNum_ISUSERDEF(type_num) || type_num == NPY_VOID)) {
            PyObject *key, *obj;

            last_userdef = type_num;

            key = PyInt_FromLong(type_num);
            if (key == NULL) {
                return -1;
            }
            obj = PyDict_GetItem(ufunc->userloops, key);
            Py_DECREF(key);
            if (obj == NULL) {
                continue;
            }
            funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(obj);
            while (funcdata != NULL) {
                int *types = funcdata->arg_types;

                for (j = 0; j < nargs; ++j) {
                    if (types[j] != dtypes[j]->type_num) {
                        break;
                    }
                }
                /* It matched */
                if (j == nargs) {
                    *out_innerloop = funcdata->func;
                    *out_innerloopdata = funcdata->data;
                    return 1;
                }

                funcdata = funcdata->next;
            }
        }
    }

    /* Didn't find a match */
    return 0;
}



NPY_NO_EXPORT int
PyGUFunc_DefaultLegacyInnerLoopSelector(PyUFuncObject *ufunc,
                                PyArray_Descr **dtypes,
                                PyUFuncGenericFunction *out_innerloop,
                                void **out_innerloopdata,
                                int *out_needs_api)
{
    int nargs = ufunc->nargs;
    char *types;
    const char *ufunc_name;
    PyObject *errmsg;
    int i, j;

    ufunc_name = ufunc->name ? ufunc->name : "(unknown)";

    /*
     * If there are user-loops search them first.
     * TODO: There needs to be a loop selection acceleration structure,
     *       like a hash table.
     */
    if (ufunc->userloops) {
        switch (find_userloop(ufunc, dtypes,
                    out_innerloop, out_innerloopdata)) {
            /* Error */
            case -1:
                return -1;
            /* Found a loop */
            case 1:
                return 0;
        }
    }

    types = ufunc->types;
    for (i = 0; i < ufunc->ntypes; ++i) {
        /* Copy the types into an int array for matching */
        for (j = 0; j < nargs; ++j) {
            if (types[j] != dtypes[j]->type_num) {
                break;
            }
        }
        if (j == nargs) {
            *out_innerloop = ufunc->functions[i];
            *out_innerloopdata = ufunc->data[i];
            return 0;
        }

        types += nargs;
    }

    errmsg = PyUString_FromFormat("ufunc '%s' did not contain a loop "
                    "with signature matching types ", ufunc_name);
    for (i = 0; i < nargs; ++i) {
        PyUString_ConcatAndDel(&errmsg,
                PyObject_Repr((PyObject *)dtypes[i]));
        if (i < nargs - 1) {
            PyUString_ConcatAndDel(&errmsg, PyUString_FromString(" "));
        }
    }
    PyErr_SetObject(PyExc_TypeError, errmsg);
    Py_DECREF(errmsg);

    return -1;
}

typedef struct {
    NpyAuxData base;
    PyUFuncGenericFunction unmasked_innerloop;
    void *unmasked_innerloopdata;
    int nargs;
} _ufunc_masker_data;

static NpyAuxData *
ufunc_masker_data_clone(NpyAuxData *data)
{
    _ufunc_masker_data *n;

    /* Allocate a new one */
    n = (_ufunc_masker_data *)PyArray_malloc(sizeof(_ufunc_masker_data));
    if (n == NULL) {
        return NULL;
    }

    /* Copy the data (unmasked data doesn't have object semantics) */
    memcpy(n, data, sizeof(_ufunc_masker_data));

    return (NpyAuxData *)n;
}

/*
 * This function wraps a regular unmasked ufunc inner loop as a
 * masked ufunc inner loop, only calling the function for
 * elements where the mask is True.
 */
static void
unmasked_ufunc_loop_as_masked(
             char **dataptrs, npy_intp *strides,
             char *mask, npy_intp mask_stride,
             npy_intp loopsize,
             NpyAuxData *innerloopdata)
{
    _ufunc_masker_data *data;
    int iargs, nargs;
    PyUFuncGenericFunction unmasked_innerloop;
    void *unmasked_innerloopdata;
    npy_intp subloopsize;

    /* Put the aux data into local variables */
    data = (_ufunc_masker_data *)innerloopdata;
    unmasked_innerloop = data->unmasked_innerloop;
    unmasked_innerloopdata = data->unmasked_innerloopdata;
    nargs = data->nargs;

    /* Process the data as runs of unmasked values */
    do {
        /* Skip masked values */
        mask = npy_memchr(mask, 0, mask_stride, loopsize, &subloopsize, 1);
        for (iargs = 0; iargs < nargs; ++iargs) {
            dataptrs[iargs] += subloopsize * strides[iargs];
        }
        loopsize -= subloopsize;
        /*
         * Process unmasked values (assumes unmasked loop doesn't
         * mess with the 'args' pointer values)
         */
        mask = npy_memchr(mask, 0, mask_stride, loopsize, &subloopsize, 0);
        unmasked_innerloop(dataptrs, &subloopsize, strides,
                                        unmasked_innerloopdata);
        for (iargs = 0; iargs < nargs; ++iargs) {
            dataptrs[iargs] += subloopsize * strides[iargs];
        }
        loopsize -= subloopsize;
    } while (loopsize > 0);
}


/*
 * This function wraps a legacy inner loop so it becomes masked.
 *
 * Returns 0 on success, -1 on error.
 */
NPY_NO_EXPORT int
PyGUFunc_DefaultMaskedInnerLoopSelector(PyUFuncObject *ufunc,
                            PyArray_Descr **dtypes,
                            PyArray_Descr *mask_dtype,
                            npy_intp *NPY_UNUSED(fixed_strides),
                            npy_intp NPY_UNUSED(fixed_mask_stride),
                            PyUFunc_MaskedStridedInnerLoopFunc **out_innerloop,
                            NpyAuxData **out_innerloopdata,
                            int *out_needs_api)
{
    int retcode;
    _ufunc_masker_data *data;

    if (ufunc->legacy_inner_loop_selector == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "the ufunc default masked inner loop selector doesn't "
                "yet support wrapping the new inner loop selector, it "
                "still only wraps the legacy inner loop selector");
        return -1;
    }

    if (mask_dtype->type_num != NPY_BOOL) {
        PyErr_SetString(PyExc_ValueError,
                "only boolean masks are supported in ufunc inner loops "
                "presently");
        return -1;
    }

    /* Create a new NpyAuxData object for the masker data */
    data = (_ufunc_masker_data *)PyArray_malloc(sizeof(_ufunc_masker_data));
    if (data == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memset(data, 0, sizeof(_ufunc_masker_data));
    data->base.free = (NpyAuxData_FreeFunc *)&PyArray_free;
    data->base.clone = &ufunc_masker_data_clone;
    data->nargs = ufunc->nin + ufunc->nout;

    /* Get the unmasked ufunc inner loop */
    retcode = ufunc->legacy_inner_loop_selector(ufunc, dtypes,
                    &data->unmasked_innerloop, &data->unmasked_innerloopdata,
                    out_needs_api);
    if (retcode < 0) {
        PyArray_free(data);
        return retcode;
    }

    /* Return the loop function + aux data */
    *out_innerloop = &unmasked_ufunc_loop_as_masked;
    *out_innerloopdata = (NpyAuxData *)data;
    return 0;
}


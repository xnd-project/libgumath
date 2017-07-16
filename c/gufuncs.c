#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <stdio.h>

#include "gufuncs.h"
#include "infrastructure.h"

 
static void example_core_loop_f(char **args,
                                npy_intp *dimensions,
                                npy_intp *steps,
                                void *NPY_UNUSED(user_data));

int
add_example_gufuncs(PyObject *m)
{
    PyObject *module_dict = NULL;
    PyObject *gufunc = NULL;

    import_array();
    import_ufunc();

    module_dict = PyModule_GetDict(m);

    /* To create a gufunc, PyUFunc_FromFuncAndDataAndSignature. The arguments are
     *  (in order):
     *
     *  * func   - a function table of "core loops". This table contains the
     *             function pointers to the different functions implementing the
     *             core loop for the different type signatures (one entry per
     *             type signature).
     *
     *  * data   - a void* table of user data. One entry per type signature.
     *
     *  * types  - a char table with core loop type specificiation in numpy's
     *             letter format. Note that the table contains values in the form
     *             NPY_FLOAT, NPY_DOUBLE, etc and not letters (from the NPY_TYPES
     *             enum). There is one char per operand per signature). The way
     *             to access them is: types[n_op + n_sig*nops], with n_op being
     *             the operand number, n_sig the signature number and nops the
     *             number of operands of the gufunc (nin + nout).
     *
     *  * ntypes - the number of different kernel functions (size of the above 
     *             tables).
     *
     *  * nin    - number of input operands for the kernel.
     *
     *  * nout   - number of output operands for the kernel.
     *
     *  * identity - value to use as "identity" (for reductions). This can either
     *               be PyUFunc_Zero, PyUFunc_One, PyUFunc_MinusOne,
     *               PyUFunc_None, PyUFunc_ReorderableNone.
     *
     *  * name   - gufunc's name.
     *
     *  * doc    - gufunc's docstring
     *
     *  * unused - <unused>
     *
     *  * signature - the dimension signature. This is the shape signature string
     *                that takes a form like "(M,N)->(N,M)", for example.
     *
     * Note that all tables must be static, as the underlying function doesn't
     * copy them.
     */
    static PyUFuncGenericFunction core_loops[] = {
        example_core_loop_f,
    };

    static void *core_loop_data[] = {
        NULL,
    };

    static char core_loop_type_table[] = {
        NPY_FLOAT, NPY_FLOAT
    };

    const size_t core_loop_count = sizeof(core_loops)/sizeof(core_loops[0]);
    const char *gufunc_name="example_gufunc";

    gufunc = PyGUFunc_FromFuncAndDataAndSignature(core_loops,
                                                 core_loop_data,
                                                 core_loop_type_table,
                                                 core_loop_count,
                                                 1, /* input count */
                                                 1, /* output count */
                                                 PyUFunc_None, /* identity */
                                                 gufunc_name,
                                                 "no doc",
                                                 0,
                                                 "(M,N)->(N)");
    if (gufunc) {
        PyDict_SetItemString(module_dict, gufunc_name, gufunc);
        Py_DECREF(gufunc);
    } else {
        PyErr_Format(PyExc_RuntimeError,
                     "Failed to register gufunc %s.", gufunc_name);
        return -1;
    }                                            
    
    return 0;
}


/* This is an example of a gufunc kernel function. Note that all parameters are
   packed into args, plus a set of dimensions and steps. Note the "user_data"
   (not used here) comes from the user_data array provided when registering the
   gufunc.

   The inner loop receives the following:
   
   - args: an array of pointers. There will be as many as operand in the gufunc,
           that is, nin + nout. In our sample, as there is one input and one
           output there will be two entries.

   - dimensions: an array of integers. This will be the bound dimensions of the
           dimension variables found in the signature. There will be one entry
           per dimension variable in the same order as they appear in the
           signature when reading from left to right. In our example, the
           signature is (M,N)->(N). That means there will be two dimensions (M
           and N) and that they will appear in that order. Dimensions also
           include the loop dimension that is placed first. So they array will
           contain LOOP_DIM, M, N; where LOOP_DIM is the number of elements
           executed in this loop.

   - steps: the steps required when for iterating dimensions of all operands,
            including steps for the core dimensions of all operands plus the
            steps for the first outer dimension of each operand. The layout is
            as follows:

            a) The first NOPS steps, where NOPS is the total number of operands
               in the gufunc, are the steps of the data corresponding to the
               latest dimension in the iteration shape. These should be used
               to implemente the outer iteration of the gufunc.

            b) Following the steps in (a), each of the operands will have the
               dimensions of their core dimensions in order, from inner dimension
               to outer

            Note that all steps are given in bytes.
*/

static void example_core_loop_f(char **args,
                                npy_intp *dimensions,
                                npy_intp *steps,
                                void *NPY_UNUSED(user_data))
{
    /* signature is (M,N)->(N) */
    npy_intp outer_step_in = *steps++;
    npy_intp outer_step_out = *steps++;
    npy_intp step_in_outer = *steps++;
    npy_intp step_in_inner = *steps++;
    npy_intp step_out_inner = *steps;
    npy_intp LOOP_SIZE = *dimensions++;
    npy_intp M = dimensions[0];
    npy_intp N = dimensions[1];
    char *arg_in = args[0];
    char *arg_out = args[1];

    printf("LOOP_SIZE: %zd M: %zd N: %zd\n", LOOP_SIZE, M, N);
    printf("step_in_outer: %zd step_in_inner: %zd step_out_inner: %zd\n",
           step_in_outer, step_in_inner, step_out_inner);
    /* outer loop for the operation: for each element */
    for (npy_intp i = 0; i < LOOP_SIZE; i++) {
        /* compute base address of the input element and the output element */
        char *el_in = arg_in + i*outer_step_in;
        char *el_out = arg_out + i*outer_step_out;

        /* implement the operation for a single element. This dummy gufunc will
           just accumulate the values in the same column.
        */
        for (npy_intp c = 0; c < N; c++) {
            float acc = 0.0f;
            for (npy_intp r = 0; r < M; r++) {
                acc += *(float*)(el_in + r*step_in_outer + c*step_in_inner);
            }
            *(float*)(el_out + c*step_out_inner) = acc;
        }
        /* end of the operation for a single element. */
    }

}

Gufunc
======

A gufunc is defined as a name and a collection of associated kernels.  Gufunc
structs are in a lookup table with their names as keys:

.. code-block:: c

   typedef struct {
       char *name;
       int nkernels;
       gm_kernel_t kernels[GM_MAX_KERNELS];
   } gm_func_t;

Since each kernel has its own type signature, gufuncs are essentially multimethods.
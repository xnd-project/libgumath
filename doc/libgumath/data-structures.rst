.. meta::
   :robots: index,follow
   :description: libndtypes documentation

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Data structures
===============

libgumath is a lightweight library for managing and dispatching computational
kernels that target XND containers.

Functions are multimethods in a lookup table. Typically, applications that
use libgumath should create a new lookup table for each namespace. For example,
Python modules generally should have a module-specific lookup table.


Kernel signatures
-----------------

.. topic:: kernels

.. code-block:: c

   typedef int (* gm_xnd_kernel_t)(xnd_t stack[], ndt_context_t *ctx);

The signature of an *xnd* kernel. *stack* contains incoming and outgoing
arguments. In case of an error, kernels are expected to set a context error
message and return *-1*.

In case of success, the return value is *0*.


.. code-block:: c

   typedef int (* gm_strided_kernel_t)(char **args, intptr_t *dimensions, intptr_t *steps, void *data);

The signature of a NumPy compatible kernel.  These signatures are for
applications that want to use existing NumPy compatible kernels on XND
containers.

XND containers are automatically converted to a temporary ndarray before
kernel application.


Kernel set
----------

.. code-block:: c

   /* Collection of specialized kernels for a single function signature. */
   typedef struct {
      ndt_t *sig;
      const ndt_constraint_t *constraint;

      /* Xnd signatures */
      gm_xnd_kernel_t C;       /* dispatch ensures c-contiguous */
      gm_xnd_kernel_t Fortran; /* dispatch ensures f-contiguous */
      gm_xnd_kernel_t Xnd;     /* selected if non-contiguous or both C and Fortran are NULL */

      /* NumPy signature */
      gm_strided_kernel_t Strided;
   } gm_kernel_set_t;

A kernel set contains the function signature, an optional constraint function,
and up to four specialized kernels, each of which may be *NULL*.

The dispatch calls the kernels in the following order of preference:

If the inner dimensions of the incoming arguments are C-contiguous, the *C*
kernel is called first. In case of *Fortran* inner dimensions, *Fortran*
is called first.

If an *Xnd* kernel is present, it is called next, then the *Strided* kernel.


Kernel set initialization
-------------------------

.. topic:: kernel set initialization

.. code-block:: c

   typedef struct {
      const char *name;
      const char *sig;
      const ndt_constraint_t *constraint;

      gm_xnd_kernel_t C;
      gm_xnd_kernel_t Fortran;
      gm_xnd_kernel_t Xnd;
      gm_strided_kernel_t Strided;
   } gm_kernel_init_t;

   int gm_add_kernel(gm_tbl_t *tbl, const gm_kernel_init_t *kernel, ndt_context_t *ctx);

The *gm_kernel_init_t* is used for initializing a kernel set.  Usually, a C
translation unit contains an array of hundreds of *gm_kernel_init_t* structs
together with a function that initializes a specific lookup table.


Multimethod struct
------------------

.. topic:: kernel set initialization

.. code-block:: c

   /* Multimethod with associated kernels */
   typedef struct gm_func gm_func_t;
   typedef const gm_kernel_set_t *(*gm_typecheck_t)(ndt_apply_spec_t *spec,
                     const gm_func_t *f, const ndt_t *in[], int nin, ndt_context_t *ctx);
   struct gm_func {
      char *name;
      gm_typecheck_t typecheck; /* Experimental optimized type-checking, may be NULL. */
      int nkernels;
      gm_kernel_set_t kernels[GM_MAX_KERNELS];
   };

This is the multimethod struct for a given function name.  Each multimethod has
a *nkernels* associated kernel sets with unique type signatures.

If *typecheck* is *NULL*, the generic libndtypes multimethod dispatch is used
to locate the kernel. This is an O(N) operation, whose search time is negligible
for large array operations.

The *typecheck* field can be set to an optimized lookup function that has
internal knowledge of kernel set locations.  The only restriction to the
function is that it must behave exactly as the generic libndtypes typecheck.

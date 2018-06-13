.. meta::
   :robots: index,follow
   :description: libgumath documentation

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Functions
=========

Create a new multimethod
------------------------

.. topic:: gm_add_func

.. code-block:: c

   gm_func_t *gm_add_func(gm_tbl_t *tbl, const char *name, ndt_context_t *ctx);

Add a new multimethod with no associated kernels to a lookup table.  If
*name* is already present in *tbl* or if *name* contains invalid characters,
return *NULL* and set an error.

On success, return the pointer to the new multimethod.  The multimethod
belongs to *tbl*.


Add a kernel to a multimethod
-----------------------------

.. topic:: gm_add_kernel

.. code-block:: c

   int gm_add_kernel(gm_tbl_t *tbl, const gm_kernel_init_t *kernel, ndt_context_t *ctx);

Add a kernel set to a multimethod.  For convenience the multimethod is
created and inserted into the table if not already present.


.. code-block:: c

   int gm_add_kernel_typecheck(gm_tbl_t *tbl, const gm_kernel_init_t *kernel, ndt_context_t *ctx, gm_typecheck_t f);

Add a kernel set to a multimethod, using a custom typecheck function.
For convenience, the multimethod is created and inserted into the table
if not already present.


Select a kernel based on the input types
----------------------------------------

.. code-block:: c

   gm_kernel_t gm_select(ndt_apply_spec_t *spec, const gm_tbl_t *tbl, const char *name,
                         const ndt_t *in_types[], int nin, const xnd_t args[],
                         ndt_context_t *ctx);

The function looks up a multimethod by *name*, using table *tbl*.  If the
multimethod has an optimized custom typecheck function, it is called on
the input types for kernel selection.

Otherwise, the generic *ndt_typecheck* is called on each kernel associated
with the multimethod in order to find a match for the input arguments.


Apply a kernel to input
-----------------------

.. topic:: gm_apply

.. code-block:: c

   int gm_apply(const gm_kernel_t *kernel, xnd_t stack[], int outer_dims, ndt_context_t *ctx);

Apply a kernel to input arguments. *stack* is expected to contain a list of
input arguments followed by output arguments.  *outer_dims* are the number
of dimensions to traverse before applying the kernel to the inner dimensions.

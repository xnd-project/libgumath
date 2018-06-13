.. meta::
   :robots: index,follow
   :description: libgumath documentation

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Builtin kernels
===============

libgumath has a number of builtin kernels that use optimized type checking
and kernel lookup.


Unary kernels
-------------

.. topic:: unary kernels

.. code-block:: c

   int gm_init_unary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);

Add all builtin unary kernels to *tbl*.  The kernels include *fabs*,
*exp*, *exp2*, *expm1*, *log*, *log2*, *log10*, *log1p*, *logb*, *sqrt*,
*cbrt*, *sin*, *cos*, *tan*, *asin*, *acos*, *atan*, *sinh*, *cosh*, *tanh*,
*asinh*, *acosh*, *atanh*, *erf*, *erfc*, *lgamma*, *tgamma*, *ceil*,
*floor*, *trunc*, *round*, *rearbyint*.


Binary kernels
--------------

.. topic:: binary kernels

.. code-block:: c

   int gm_init_binary_kernels(gm_tbl_t *tbl, ndt_context_t *ctx);

Add all binary kernels to *tbl*.  The kernels currently only include
*add*, *subtract*, *multiply*, *divide*.

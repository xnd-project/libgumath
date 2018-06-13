.. meta::
   :robots: index, follow
   :description: libgumath documentation
   :keywords: libgumath, C, array computing

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


libgumath
---------

libgumath is a library for dispatching computational kernels using
ndtypes function signatures.  Kernels are multimethods and can be
JIT-generated and inserted in lookup tables at runtime.

Kernels target XND containers.

libgumath has a small number of generic math library kernels.


.. toctree::

   data-structures.rst
   functions.rst
   kernels.rst

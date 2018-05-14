.. meta::
   :robots: index, follow
   :description: libgumath documentation
   :keywords: libgumath, C, array computing

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


libgumath
---------

The libgumath library implements support for the function dispatch to the memory blocks defined using the xnd library.

This initial libgumath version displays a simple design. The goal is
to determine whether the kernel signatures and the dispatch model are suitable
for Numba.

Currently there is just one working test with sin().

.. toctree::

   gufunc.rst
   kernel.rst
   numba_integration.rst



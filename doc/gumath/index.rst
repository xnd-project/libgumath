.. meta::
   :robots: index, follow
   :description: gumath documentation
   :keywords: kernels, dispatch, Python

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


gumath
------

The gumath Python module provides the infrastructure for managing and
dispatching libgumath kernels.  Kernels target xnd containers from the
xnd Python module.

gumath supports modular namespaces.  Typically, a namespace is implemented
as one Python module that uses gumath for calling kernels.

The xndtools project automates generating kernels and creating namespace
modules.


.. toctree::
   :maxdepth: 1

   functions.rst

.. meta::
   :robots: index, follow
   :description: libgumath documentation
   :keywords: libgumath, function dispatch, numba, C

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>

Numba integration
=================

The basic idea is that libgumath contains functions that allow inserting
gufuncs and kernels into the lookup table.

Ideally, Numba would jit-compile specialized kernels and call the insertion
function (must be on the C level for safety).

The function is then automatically available to be called on the Python
level via the gumath Python module.


Obstacles
---------

- If the datashape (ndt_t) signatures are given on the Python level (which
  is probably the only sane option), the jit-compiled kernel needs to be
  type-checked against the ndt_t type.
.. meta::
   :robots: index, follow
   :description: libgumath documentation
   :keywords: libgumath, C, array computing

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


========
Releases
========


v0.2.0b1 (January 20th 2018)
============================

The first version of libgumath has a relatively simple design.  The goal is
to determine whether the kernel signatures and the dispatch model are suitable
for Numba.

Currently there is just one working test with sin().

The following list includes some of the issues to be discussed before the next release:

-  If the datashape (ndt_t) signatures are given on the Python level (which
  is probably the only sane option), the jit-compiled kernel needs to be
  type-checked against the ndt_t type.



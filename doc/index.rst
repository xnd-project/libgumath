.. meta::
   :robots: index, follow
   :description: gumath documentation
   :keywords: libgumath, gumath, C, Python, function dispatch

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>

******
gumath
******

This package provides tools to dispatch functions towards the memory containers. These containers can be have a general
structure or a Numpy-like container with a composable, generalized function concept.

Installation
============

To run gumathn `xnd`_ and `ndtypes`_, your computer requires a Python interpreters, either version 2.7 or 3.6
(the lattest stable version).

gumath can be installed using `pip`_::

  python3 -m pip install gumath

Or using anaconda package manager `anaconda`_::

  conda install -c xnd/label/dev gumath

gumath does not depend on third-party Python except for `xnd`_ and `ndtypes`_ (currently, these packages do not have any
external dependensives themselves).

.. _xnd: https://github.com/plures/xnd
.. _ndtypes: https://github.com/plures/ndtypes
.. _pip: https://pip.pypa.io/en/latest/installing


Index
=====

Libgumath
---------

C library.

.. toctree::
   :maxdepth: 1

   libgumath/index.rst


Gumath
------

Python module.

.. toctree::
   :maxdepth: 1

   gumath/index.rst

Releases
--------

.. toctree::
   :maxdepth: 1

   releases/index.rst
.. meta::
   :robots: index, follow
   :description: gumath documentation
   :keywords: gumath, sin, Python

.. sectionauthor:: Vital Fernandez <vital-fernandez at gmail.com>


sin
===

Trigonometric sine, element-wise.

Parameters
----------

Atributes
^^^^^^^^^

x: array_like
Angle, in radians (2*pi rad equals 360 degrees).

Returns
^^^^^^^

y: array_like
The sine of each element of x.

returns: array:like

Example
-------

.. doctest::

   >>> import gumath as gm
   >>> from xnd import xnd
   >>> x = [0.0, 30 * 3.14159/180, 90 * 3.14159/180]
   >>> gm.sin(xnd(x))
   xnd([0.0, 0.49999, 0.99999], type='3 * float64')

Notes
=====

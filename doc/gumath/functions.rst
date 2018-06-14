.. meta::
   :robots: index,follow
   :description: xnd container
   :keywords: xnd, types, examples

.. sectionauthor:: Stefan Krah <skrah at bytereef.org>


Builtin functions
=================

The gumath.functions module wraps the builtin libgumath kernels and serves
as an example of a modular namespace.


All builtin functions
---------------------

.. doctest::

   >>> from gumath import functions as fn
   >>> dir(fn)
   ['__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'acos', 'acosh', 'add', 'asin', 'asinh', 'atan', 'atanh', 'cbrt', 'ceil', 'copy', 'cos', 'cosh', 'divide', 'erf', 'erfc', 'exp', 'exp2', 'expm1', 'fabs', 'floor', 'lgamma', 'log', 'log10', 'log1p', 'log2', 'logb', 'multiply', 'nearbyint', 'round', 'sin', 'sinh', 'sqrt', 'subtract', 'tan', 'tanh', 'tgamma', 'trunc']


Unary functions
---------------

.. doctest::

   >>> from xnd import xnd
   >>> x = xnd([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
   >>> fn.log(x)
   xnd([[0.0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055]],
       type='2 * 3 * float64')

On an array with a *float64* dtype, *log* works as expected.


.. doctest::

   >>> x = xnd([[1, 2, 3], [4, 5, 6]])
   >>> x.type
   ndt("2 * 3 * int64")
   >>> fn.log(x)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   RuntimeError: invalid dtype

This function call would require an implicit inexact conversion from *int64* to
*float64*.  All builtin libgumath kernels only allow exact conversions, so the
example fails.


.. doctest::

   >>> x = xnd([[1, 2, 3], [4, 5, 6]], dtype="int32")
   >>> fn.log(x)
   xnd([[0.0, 0.6931471805599453, 1.0986122886681098], [1.3862943611198906, 1.6094379124341003, 1.791759469228055]],
       type='2 * 3 * float64')

*int32* to *float64* conversions are exact, so the call succeeds.

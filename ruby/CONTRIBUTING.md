# Details

## Functioning of libgumath

Gumath is a wrapper over libgumath. The docs for libgumath can be found [here](https://xnd.readthedocs.io/en/latest/libgumath/index.html).

It allows users to write [multiple dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch) functions for XND containers. 
Multiple dispatch can be thought of as the concept behind function overloading in
languages like C++. However the major difference is that the function call is made
according to the run time types of the arguments in case of multiple dispatch whereas
in overloading it is determined at compile-time and the types of arguments determine
the 'signature' of the function. [This answer](https://cs.stackexchange.com/questions/4660/difference-between-multimethods-and-overloading) sheds some light on the differences
between the two. Here's [another link](http://wiki.c2.com/?MultiMethods) on the same topic.

This functionality is absent in Ruby because in Ruby we don't care about the type of 
the arguments of a function when calling it (aka duck typing).

Gumath functions (or 'kernels') are functions that can accept XND objects of multiple 
data types and compute the result depending on what kind of data type is sent in. The
data type is known only at run time (like determining whether a container is of type
float32 or int etc.) and the same function can handle all sorts of data types.

Libgumath stores functions in a look-up table. Each Ruby module containing functions
should have its own look-up table. A prototype of adding gumath kernels to a Ruby module
`Gumath::Functions` can be found in the functions.c file in this repo.

An XND kernel has the following function signature:
```
typedef int (* gm_xnd_kernel_t)(xnd_t stack[], ndt_context_t *ctx);
```

## Interfacing libgumath with Ruby

Unlike in Python, Ruby functions are not first-class objects and therefore we cannot create
a 'callable' object like we can in Python. Since libgumath kernels are run-time constructs
it is difficult to dynamically add these functions to a Ruby module at runtime since unlike
Ruby, C cannot generate new functions at run-time.

However, Ruby does have support for lambdas and we can store the function implementations
inside lambdas. These lambdas are stored in module variables that are of the same name
as the function name that they implement. When the user calls the lambda with the `call` method
or `.` syntax, the lambda will be passed the argument (after some conversion) and will
return the result given by the gumath kernel.

Another approach is to have a Hash of methods and their corresponding implementations
stored inside each module. Whenever a method is called on the module, it will routed to
the method_missing in the class, which will lookup the hash and call the appropriate
lambda. This seems like the better approach since the user no longer needs to know that
the functions that they're calling inside the module are in fact lambdas.

So each module that contains gumath methods should have a hash called `gumath_functions`
that stores function names as keys (as symbols) and the corresponding values as objects
of type `GufuncObject`. This `GufuncObject` will have a method `call` defined on it that
will be called by the `method_missing` when that object is called. The only extra overhead
will be that of a Hash lookup (O(1) time) and calling an Ruby method `call`.

## Calling gumath kernels

Gumath kernels can be called by using the `gm_apply` function and passing a pointer to
the function along with its arguments. Its multi-threaded counterpart, `gm_apply_thread`
can also be used for this purpose.

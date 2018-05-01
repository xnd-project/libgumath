from ._gumath import unsafe_add_xnd_kernel
from numba import jit
from llvmlite import ir
from llvmlite.ir import PointerType as ptr, LiteralStructType as struct
from toolz.functoolz import thread_first as tf


def jit_to_kernel(gumath_sig, numba_sig):
    """
    JIT compiles a function and returns a 0D XND kernel for it.


    Call with the ndtype function signature and the numba signature of the inner function.

    >>> import math
    >>> import numpy as np
    >>> from xnd import xnd
    >>> @jit_to_kernel('... * float64 -> ... * float64', 'float64(float64)')
    ... def f(x):
    ...     return math.sin(x)
    <_gumath.gufunc at 0x10e3d5f90>
    >>> f(xnd.from_buffer(np.arange(20).astype('float64')))
    xnd([0.0,
         0.8414709848078965,
         0.9092974268256817,
         0.1411200080598672,
         -0.7568024953079282,
         -0.9589242746631385,
         -0.27941549819892586,
         0.6569865987187891,
         0.9893582466233818,
         ...],
        type='20 * float64')
    """
    return lambda fn: _gu_vectorize(fn, gumath_sig, numba_sig)


def build_kernel_wrapper(library, context, fname, signature, envptr):
    """
    Returns a pointer to a llvm function that can be used as an xnd kernel.

    Like build_ufunc_wrapper
    """

    # setup the module and jitted function
    wrapperlib = context.codegen().create_library('gumath_wrapper')
    wrapper_module = wrapperlib.create_ir_module('')

    func_type = context.call_conv.get_function_type(
        signature.return_type, signature.args)
    func = wrapper_module.add_function(func_type, name=fname)
    func.attributes.add("alwaysinline")

    # setup xnd types
    i8, i16, i32, i64 = map(ir.IntType, [8, 16, 32, 64])
    index = lambda i: ir.Constant(i32, i)

    # only add bodies to types if we haven't add them already
    # I need this logic because I am not sure how to declare these types
    # externally to this function only once.
    def get_type(name):
        types = wrapper_module.context.identified_types
        added = name not in types
        return added, wrapper_module.context.get_identified_type(name)

    added, ndt_slice_t = get_type("ndt_slice_t")
    if added:
        ndt_slice_t.set_body(i64, i64, i64)

    added, ndt_t = get_type("_ndt")
    if added:
        ndt_t.set_body(
            i32, i32, i32, i32, i64, i16,
            struct([struct([i64, i64, i64, ptr(ptr(ndt_t))])]),
            struct([struct([struct([i32, i64, i32, ptr(i32), i32, ptr(ndt_slice_t)])])]),
            ir.ArrayType(i8, 16)
        )

    added, xnd_bitmap_t = get_type("xnd_bitmap")
    if added:
        xnd_bitmap_t.set_body(
            ptr(i8),
            i64,
            ptr(xnd_bitmap_t)
        )

    added, xnd_t = get_type("xnd")
    if added:
        xnd_t.set_body(
            xnd_bitmap_t,
            i64,
            ptr(ndt_t),
            ptr(i8)
        )

    added, ndt_context_t = get_type("_ndt_context_t")
    if added:
        ndt_context_t.set_body(
            i32, i32, i32,
            struct([ptr(i8)])
        )

    # create xnd kernel function
    fnty = ir.FunctionType(
        i32,
        (
            ptr(xnd_t),
            ptr(ndt_context_t),
        )
    )

    wrapper = wrapper_module.add_function(fnty, "__gumath__." + func.name)
    stack, ndt_context = wrapper.args
    builder = ir.IRBuilder(wrapper.append_basic_block("entry"))

    inputs = []
    for i, typ in enumerate(signature.args):
        llvm_type = context.get_data_type(typ)
        inputs.append(tf(
            stack,
            (builder.gep, [index(i), index(3)], True),
            (builder.bitcast, ptr(ptr(llvm_type))),
            builder.load,
            builder.load
        ))

    llvm_return_type = context.get_data_type(signature.return_type)
    out = tf(
        stack,
        (builder.gep, [index(len(signature.args)), index(3)], True),
        (builder.bitcast, ptr(ptr(llvm_return_type))),
        builder.load
    )

    status, retval = context.call_conv.call_function(builder, func,
                                                     signature.return_type,
                                                     signature.args, inputs, env=envptr)

    with builder.if_then(status.is_error, likely=False):
        builder.ret(ir.Constant(i32, -1))

    builder.store(retval, out)
    builder.ret(ir.Constant(i32, 0))

    # cleanup and return pointer
    del builder

    # print(wrapper_module)
    wrapperlib.add_ir_module(wrapper_module)
    wrapperlib.add_linking_library(library)
    return wrapperlib.get_pointer_to_function(wrapper.name)

i = 0

def _gu_vectorize(fn, gumath_sig, numba_sig):
    global i
    dispatcher = jit(numba_sig, nopython=True)(fn)
    cres = list(dispatcher.overloads.values())[0]
    llvm_name = cres.fndesc.llvm_func_name

    ctx = cres.target_context
    func_ptr = build_kernel_wrapper(
        library=cres.library,
        context=ctx,
        fname=llvm_name,
        signature=cres.signature,
        envptr=cres.environment.as_pointer(ctx)
    )

    kernel_name = 'numba' + str(i)
    i += 1
    return unsafe_add_xnd_kernel(
        name=kernel_name,
        sig=gumath_sig,
        ptr=func_ptr
    )


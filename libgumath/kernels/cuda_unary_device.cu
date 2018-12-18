/*
* BSD 3-Clause License
*
* Copyright (c) 2017-2018, plures
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <cinttypes>
#include <thrust/complex.h>
#include "cuda_unary_device.h"


/*****************************************************************************/
/*                                  Half float                               */
/*****************************************************************************/

static inline __device__ half
half_abs(half a)
{
    return __hlt(a, 0) ? __hneg(a) : a;
}


/*****************************************************************************/
/*                         Cuda unary device kernels                         */
/*****************************************************************************/

#define CUDA_UNARY(name, func, t0, t1, common) \
static __global__ void                                               \
_##name##_##t0##_##t1(const t0##_t *in0, t1##_t *out, int64_t N)     \
{                                                                    \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;           \
    int64_t stride = blockDim.x * gridDim.x;                         \
    for (int64_t i = index; i < N; i += stride) {                    \
        out[i] = func((common##_t)in0[i]);                           \
    }                                                                \
}                                                                    \
                                                                     \
extern "C" void                                                      \
gm_cuda_unary_device_fixed_##name##_1D_C_##t0##_##t1(                \
    const char *in0, char *out, int64_t N)                           \
{                                                                    \
    const t0##_t *_in0 = (const t0##_t *)in0;                        \
    t1##_t *_out = (t1##_t *)out;                                    \
    int blockSize = 256;                                             \
    int64_t numBlocks = (N + blockSize - 1) / blockSize;             \
                                                                     \
    _##name##_##t0##_##t1<<<numBlocks, blockSize>>>(_in0, _out, N);  \
}

#define CUDA_NOIMPL(name, func, t0, t1, common)


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define copy(x) x

CUDA_UNARY(copy, copy, bool, bool, bool)

CUDA_UNARY(copy, copy, int8, int8, int8)
CUDA_UNARY(copy, copy, int16, int16, int16)
CUDA_UNARY(copy, copy, int32, int32, int32)
CUDA_UNARY(copy, copy, int64, int64, int64)

CUDA_UNARY(copy, copy, uint8, uint8, uint8)
CUDA_UNARY(copy, copy, uint16, uint16, uint16)
CUDA_UNARY(copy, copy, uint32, uint32, uint32)
CUDA_UNARY(copy, copy, uint64, uint64, uint64)

CUDA_UNARY(copy, copy, float16, float16, float16)
CUDA_UNARY(copy, copy, float32, float32, float32)
CUDA_UNARY(copy, copy, float64, float64, float64)

CUDA_NOIMPL(copy, copy, complex32, complex32, complex32)
CUDA_UNARY(copy, copy, complex64, complex64, complex64)
CUDA_UNARY(copy, copy, complex128, complex128, complex128)


/*****************************************************************************/
/*                               Bitwise NOT                                 */
/*****************************************************************************/

#define invert(x) !x
CUDA_UNARY(invert, invert, bool, bool, bool)
#undef invert

#define invert(x) ~x
CUDA_UNARY(invert, invert, int8, int8, int8)
CUDA_UNARY(invert, invert, int16, int16, int16)
CUDA_UNARY(invert, invert, int32, int32, int32)
CUDA_UNARY(invert, invert, int64, int64, int64)

CUDA_UNARY(invert, invert, uint8, uint8, uint8)
CUDA_UNARY(invert, invert, uint16, uint16, uint16)
CUDA_UNARY(invert, invert, uint32, uint32, uint32)
CUDA_UNARY(invert, invert, uint64, uint64, uint64)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

#define negative(x) -x
CUDA_UNARY(negative, negative, int8, int8, int8)
CUDA_UNARY(negative, negative, int16, int16, int16)
CUDA_UNARY(negative, negative, int32, int32, int32)
CUDA_UNARY(negative, negative, int64, int64, int64)

CUDA_UNARY(negative, negative, uint8, int16, int16)
CUDA_UNARY(negative, negative, uint16, int32, int32)
CUDA_UNARY(negative, negative, uint32, int64, int64)

CUDA_UNARY(negative, __hneg, float16, float16, float16)
CUDA_UNARY(negative, negative, float32, float32, float32)
CUDA_UNARY(negative, negative, float64, float64, float64)

CUDA_NOIMPL(negative, negative, complex32, complex32, complex32)
CUDA_UNARY(negative, negative, complex64, complex64, complex64)
CUDA_UNARY(negative, negative, complex128, complex128, complex128)


/*****************************************************************************/
/*                                   Math                                    */
/*****************************************************************************/

#define CUDA_UNARY_DEVICE_ALL_REAL_MATH(name) \
    CUDA_UNARY(name##f, name##f, int16, float32, float32)   \
    CUDA_UNARY(name##f, name##f, uint16, float32, float32)  \
    CUDA_UNARY(name##f, name##f, float32, float32, float32) \
    CUDA_UNARY(name, name, int32, float64, float64)         \
    CUDA_UNARY(name, name, uint32, float64, float64)        \
    CUDA_UNARY(name, name, float64, float64, float64)

#define CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(name) \
    CUDA_UNARY_DEVICE_ALL_REAL_MATH(name)                      \
    CUDA_NOIMPL(name, name, complex32, complex32, complex32)   \
    CUDA_UNARY(name, name, complex64, complex64, complex64)    \
    CUDA_UNARY(name, name, complex128, complex128, complex128)

#define CUDA_UNARY_DEVICE_ALL_HALF_MATH(name, hfunc) \
    CUDA_UNARY(name##f16, hfunc, int8, float16, float16)    \
    CUDA_UNARY(name##f16, hfunc, uint8, float16, float16)   \
    CUDA_UNARY(name##f16, hfunc, float16, float16, float16)

#define CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF(name, hfunc) \
    CUDA_UNARY_DEVICE_ALL_HALF_MATH(name, hfunc)               \
    CUDA_UNARY_DEVICE_ALL_REAL_MATH(name)                      \

#define CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(name, hfunc) \
    CUDA_UNARY_DEVICE_ALL_HALF_MATH(name, hfunc)                  \
    CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(name)                      \


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF(fabs, half_abs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(exp, hexp)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF(exp2, hexp2)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(log, hlog)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(log10, hlog10)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF(log2, hlog2)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(log1p)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(sqrt, hsqrt)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(sin, hsin)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF(cos, hcos)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(tan)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(asin)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(acos)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(sinh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(cosh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(tanh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(asinh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(acosh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_REAL_MATH(erf)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(erfc)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(lgamma)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_REAL_MATH(ceil)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(floor)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(trunc)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(round)
CUDA_UNARY_DEVICE_ALL_REAL_MATH(nearbyint)

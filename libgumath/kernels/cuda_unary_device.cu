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
/*                                  Complex32                                */
/*****************************************************************************/

typedef struct {
    half real;
    half imag;
} complex32_t;

static inline __device__ complex64_t
c32_to_c64(complex32_t c)
{
    return thrust::complex<float>(__half2float(c.real), __half2float(c.imag));
}

static inline __device__ complex32_t
c64_to_c32(complex64_t c)
{
    complex32_t res;

    res.real = __float2half(c.real());
    res.imag = __float2half(c.imag());

    return res;
}


/*****************************************************************************/
/*                         Cuda unary device kernels                         */
/*****************************************************************************/

#define CUDA_EXTERN_UNARY(func, t0, t1) \
extern "C" void                                                                       \
gm_cuda_device_fixed_##func##_1D_C_##t0##_##t1(const char *in0, char *out, int64_t N) \
{                                                                                     \
    const t0##_t *_in0 = (const t0##_t *)in0;                                         \
    t1##_t *_out = (t1##_t *)out;                                                     \
    int blockSize = 256;                                                              \
    int64_t numBlocks = (N + blockSize - 1) / blockSize;                              \
    _##func##_##t0##_##t1<<<numBlocks, blockSize>>>(_in0, _out, N);                   \
}


#define CUDA_DEVICE_UNARY(func, t0, t1, cast) \
static __global__ void                                                                \
_##func##_##t0##_##t1(const t0##_t *in0, t1##_t *out, int64_t N)                      \
{                                                                                     \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;                            \
    int64_t stride = blockDim.x * gridDim.x;                                          \
    for (int64_t i = index; i < N; i += stride) {                                     \
        out[i] = func((cast##_t)in0[i]);                                              \
    }                                                                                 \
}                                                                                     \
                                                                                      \
CUDA_EXTERN_UNARY(func, t0, t1)


#define CUDA_DEVICE_UNARY_HALF(func, t0, t1, conv) \
static __global__ void                                                                \
_##func##_##t0##_##t1(const t0##_t *in0, t1##_t *out, int64_t N)                      \
{                                                                                     \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;                            \
    int64_t stride = blockDim.x * gridDim.x;                                          \
    for (int64_t i = index; i < N; i += stride) {                                     \
        out[i] = __float2half(func(__half2float(conv(in0[i]))));                      \
    }                                                                                 \
}                                                                                     \
                                                                                      \
CUDA_EXTERN_UNARY(func, t0, t1)


#define CUDA_DEVICE_UNARY_HALF_COMPLEX(func) \
static __global__ void                                                                        \
_##func##_complex32_complex32(const complex32_t *in0, complex32_t *out, int64_t N)            \
{                                                                                             \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;                                    \
    int64_t stride = blockDim.x * gridDim.x;                                                  \
    for (int64_t i = index; i < N; i += stride) {                                             \
        out[i] = c64_to_c32(func(c32_to_c64(in0[i])));                                        \
    }                                                                                         \
}                                                                                             \
                                                                                              \
CUDA_EXTERN_UNARY(func, complex32, complex32)


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define copy(x) x

CUDA_DEVICE_UNARY(copy, bool, bool, bool)

CUDA_DEVICE_UNARY(copy, int8, int8, int8)
CUDA_DEVICE_UNARY(copy, int16, int16, int16)
CUDA_DEVICE_UNARY(copy, int32, int32, int32)
CUDA_DEVICE_UNARY(copy, int64, int64, int64)

CUDA_DEVICE_UNARY(copy, uint8, uint8, uint8)
CUDA_DEVICE_UNARY(copy, uint16, uint16, uint16)
CUDA_DEVICE_UNARY(copy, uint32, uint32, uint32)
CUDA_DEVICE_UNARY(copy, uint64, uint64, uint64)

CUDA_DEVICE_UNARY_HALF(copy, float16, float16, copy)
CUDA_DEVICE_UNARY(copy, float32, float32, float32)
CUDA_DEVICE_UNARY(copy, float64, float64, float64)

CUDA_DEVICE_UNARY_HALF_COMPLEX(copy)
CUDA_DEVICE_UNARY(copy, complex64, complex64, complex64)
CUDA_DEVICE_UNARY(copy, complex128, complex128, complex128)


/*****************************************************************************/
/*                               Bitwise NOT                                 */
/*****************************************************************************/

#define invert(x) !x
CUDA_DEVICE_UNARY(invert, bool, bool, bool)
#undef invert

#define invert(x) ~x
CUDA_DEVICE_UNARY(invert, int8, int8, int8)
CUDA_DEVICE_UNARY(invert, int16, int16, int16)
CUDA_DEVICE_UNARY(invert, int32, int32, int32)
CUDA_DEVICE_UNARY(invert, int64, int64, int64)

CUDA_DEVICE_UNARY(invert, uint8, uint8, uint8)
CUDA_DEVICE_UNARY(invert, uint16, uint16, uint16)
CUDA_DEVICE_UNARY(invert, uint32, uint32, uint32)
CUDA_DEVICE_UNARY(invert, uint64, uint64, uint64)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

#define negative(x) -x
CUDA_DEVICE_UNARY(negative, int8, int8, int8)
CUDA_DEVICE_UNARY(negative, int16, int16, int16)
CUDA_DEVICE_UNARY(negative, int32, int32, int32)
CUDA_DEVICE_UNARY(negative, int64, int64, int64)

CUDA_DEVICE_UNARY(negative, uint8, int16, int16)
CUDA_DEVICE_UNARY(negative, uint16, int32, int32)
CUDA_DEVICE_UNARY(negative, uint32, int64, int64)

CUDA_DEVICE_UNARY_HALF(negative, float16, float16, copy)
CUDA_DEVICE_UNARY(negative, float32, float32, float32)
CUDA_DEVICE_UNARY(negative, float64, float64, float64)

CUDA_DEVICE_UNARY_HALF_COMPLEX(negative)
CUDA_DEVICE_UNARY(negative, complex64, complex64, complex64)
CUDA_DEVICE_UNARY(negative, complex128, complex128, complex128)


/*****************************************************************************/
/*                                   Math                                    */
/*****************************************************************************/

#define CUDA_DEVICE_ALL_UNARY_REAL_MATH(name) \
    CUDA_DEVICE_UNARY_HALF(name##f, int8, float16, __short2half_rn)   \
    CUDA_DEVICE_UNARY_HALF(name##f, uint8, float16, __ushort2half_rn) \
    CUDA_DEVICE_UNARY_HALF(name##f, float16, float16, copy)           \
                                                                      \
    CUDA_DEVICE_UNARY(name##f, int16, float32, float32)               \
    CUDA_DEVICE_UNARY(name##f, uint16, float32, float32)              \
    CUDA_DEVICE_UNARY(name##f, float32, float32, float32)             \
                                                                      \
    CUDA_DEVICE_UNARY(name, int32, float64, float64)                  \
    CUDA_DEVICE_UNARY(name, uint32, float64, float64)                 \
    CUDA_DEVICE_UNARY(name, float64, float64, float64)                \

#define CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(name) \
    CUDA_DEVICE_ALL_UNARY_REAL_MATH(name)                       \
                                                                \
    CUDA_DEVICE_UNARY_HALF_COMPLEX(name)                        \
    CUDA_DEVICE_UNARY(name, complex64, complex64, complex64)    \
    CUDA_DEVICE_UNARY(name, complex128, complex128, complex128)


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_REAL_MATH(fabs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(exp)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(exp2)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(log)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(log10)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(log2)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(log1p)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(sqrt)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(sin)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(cos)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(tan)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(asin)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(acos)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(sinh)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(cosh)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(tanh)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(asinh)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(acosh)
CUDA_DEVICE_ALL_UNARY_COMPLEX_MATH(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_REAL_MATH(erf)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(erfc)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(lgamma)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

CUDA_DEVICE_ALL_UNARY_REAL_MATH(ceil)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(floor)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(trunc)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(round)
CUDA_DEVICE_ALL_UNARY_REAL_MATH(nearbyint)

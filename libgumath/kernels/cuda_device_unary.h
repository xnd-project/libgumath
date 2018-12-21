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


#ifndef CUDA_UNARY_DEVICE_H
#define CUDA_UNARY_DEVICE_H


#ifdef __cplusplus
#include <cinttypes>
#include <cuda_fp16.h>
#include <thrust/complex.h>
typedef half float16_t;
typedef thrust::complex<float> complex64_t;
typedef thrust::complex<double> complex128_t;
#else
#include <stdint.h>
#endif


typedef bool bool_t;
typedef float float32_t;
typedef double float64_t;


/*****************************************************************************/
/*                        Cuda device kernel signature                       */
/*****************************************************************************/

#ifdef __cplusplus
  #define CUDA_UNARY_DEVICE_DECL(name, t0, t1) \
  extern "C" void gm_cuda_unary_device_fixed_##name##_1D_C_##t0##_##t1(const char *in0, char *out, int64_t N);
#else
  #define CUDA_UNARY_DEVICE_DECL(name, t0, t1) \
  void gm_cuda_unary_device_fixed_##name##_1D_C_##t0##_##t1(const char *in0, char *out, int64_t N);
#endif

#define CUDA_NOIMPL_DEVICE_DECL(name, t0, t1)


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

CUDA_UNARY_DEVICE_DECL(copy, bool, bool)

CUDA_UNARY_DEVICE_DECL(copy, int8, int8)
CUDA_UNARY_DEVICE_DECL(copy, int16, int16)
CUDA_UNARY_DEVICE_DECL(copy, int32, int32)
CUDA_UNARY_DEVICE_DECL(copy, int64, int64)

CUDA_UNARY_DEVICE_DECL(copy, uint8, uint8)
CUDA_UNARY_DEVICE_DECL(copy, uint16, uint16)
CUDA_UNARY_DEVICE_DECL(copy, uint32, uint32)
CUDA_UNARY_DEVICE_DECL(copy, uint64, uint64)

CUDA_UNARY_DEVICE_DECL(copy, float16, float16)
CUDA_UNARY_DEVICE_DECL(copy, float32, float32)
CUDA_UNARY_DEVICE_DECL(copy, float64, float64)

CUDA_NOIMPL_DEVICE_DECL(copy, complex32, complex32)
CUDA_UNARY_DEVICE_DECL(copy, complex64, complex64)
CUDA_UNARY_DEVICE_DECL(copy, complex128, complex128)


/*****************************************************************************/
/*                               Bitwise NOT                                 */
/*****************************************************************************/

CUDA_UNARY_DEVICE_DECL(invert, bool, bool)

CUDA_UNARY_DEVICE_DECL(invert, int8, int8)
CUDA_UNARY_DEVICE_DECL(invert, int16, int16)
CUDA_UNARY_DEVICE_DECL(invert, int32, int32)
CUDA_UNARY_DEVICE_DECL(invert, int64, int64)

CUDA_UNARY_DEVICE_DECL(invert, uint8, uint8)
CUDA_UNARY_DEVICE_DECL(invert, uint16, uint16)
CUDA_UNARY_DEVICE_DECL(invert, uint32, uint32)
CUDA_UNARY_DEVICE_DECL(invert, uint64, uint64)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

CUDA_UNARY_DEVICE_DECL(negative, int8, int8)
CUDA_UNARY_DEVICE_DECL(negative, int16, int16)
CUDA_UNARY_DEVICE_DECL(negative, int32, int32)
CUDA_UNARY_DEVICE_DECL(negative, int64, int64)

CUDA_UNARY_DEVICE_DECL(negative, uint8, int16)
CUDA_UNARY_DEVICE_DECL(negative, uint16, int32)
CUDA_UNARY_DEVICE_DECL(negative, uint32, int64)

CUDA_UNARY_DEVICE_DECL(negative, float16, float16)
CUDA_UNARY_DEVICE_DECL(negative, float32, float32)
CUDA_UNARY_DEVICE_DECL(negative, float64, float64)

CUDA_NOIMPL_DEVICE_DECL(negative, complex32, complex32)
CUDA_UNARY_DEVICE_DECL(negative, complex64, complex64)
CUDA_UNARY_DEVICE_DECL(negative, complex128, complex128)


/*****************************************************************************/
/*                                    Math                                   */
/*****************************************************************************/

#define CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(name) \
    CUDA_UNARY_DEVICE_DECL(name##f, int16, float32)   \
    CUDA_UNARY_DEVICE_DECL(name##f, uint16, float32)  \
    CUDA_UNARY_DEVICE_DECL(name##f, float32, float32) \
    CUDA_UNARY_DEVICE_DECL(name, int32, float64)      \
    CUDA_UNARY_DEVICE_DECL(name, uint32, float64)     \
    CUDA_UNARY_DEVICE_DECL(name, float64, float64)

#define CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(name) \
    CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(name)           \
    CUDA_UNARY_DEVICE_DECL(name, complex32, complex32)   \
    CUDA_UNARY_DEVICE_DECL(name, complex64, complex64)   \
    CUDA_UNARY_DEVICE_DECL(name, complex128, complex128)

#define CUDA_UNARY_DEVICE_ALL_HALF_MATH_DECL(name) \
    CUDA_UNARY_DEVICE_DECL(name##f16, int8, float16)    \
    CUDA_UNARY_DEVICE_DECL(name##f16, uint8, float16)   \
    CUDA_UNARY_DEVICE_DECL(name##f16, float16, float16)

#define CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF_DECL(name) \
    CUDA_UNARY_DEVICE_ALL_HALF_MATH_DECL(name)               \
    CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(name)               \

#define CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(name) \
    CUDA_UNARY_DEVICE_ALL_HALF_MATH_DECL(name)                  \
    CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(name)               \


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF_DECL(fabs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(exp)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF_DECL(exp2)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(log)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(log10)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_WITH_HALF_DECL(log2)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(log1p)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(sqrt)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(sin)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_WITH_HALF_DECL(cos)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(tan)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(asin)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(acos)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(sinh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(cosh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(tanh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(asinh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(acosh)
CUDA_UNARY_DEVICE_ALL_COMPLEX_MATH_DECL(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(erf)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(erfc)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(lgamma)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(ceil)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(floor)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(trunc)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(round)
CUDA_UNARY_DEVICE_ALL_REAL_MATH_DECL(nearbyint)


#endif /* CUDA_UNARY_DEVICE_H */

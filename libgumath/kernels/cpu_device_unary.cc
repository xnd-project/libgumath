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
#include <cmath>
#include <complex>
#include "cpu_device_unary.h"


/*****************************************************************************/
/*                          CPU device unary kernels                         */
/*****************************************************************************/

#define CPU_DEVICE_UNARY(name, func, t0, t1, common) \
extern "C" void                                                   \
gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1(                    \
    const char *in0, char *out, int64_t N)                        \
{                                                                 \
    const t0##_t *_in0 = (const t0##_t *)in0;                     \
    t1##_t *_out = (t1##_t *)out;                                 \
                                                                  \
    for (int64_t i = 0; i < N; i++) {                             \
        _out[i] = func((common##_t)_in0[i]);                      \
    }                                                             \
}                                                                 \
                                                                  \
extern "C" void                                                   \
gm_cpu_device_0D_##name##_##t0##_##t1(const char *in0, char *out) \
{                                                                 \
    const t0##_t x = *((const t0##_t *)in0);                      \
    *((t1##_t *)out) = func((common##_t)x);                       \
}

#ifdef _MSC_VER
  #define CPU_DEVICE_UNARYC(name, func, t0, t1, common)
#else
  #define CPU_DEVICE_UNARYC(name, func, t0, t1, common) \
    CPU_DEVICE_UNARY(name, func, t0, t1, common)
#endif

#define CPU_DEVICE_NOIMPL(name, func, t0, t1, common)



/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define copy(x) x

CPU_DEVICE_UNARY(copy, copy, bool, bool, bool)

CPU_DEVICE_UNARY(copy, copy, uint8, uint8, uint8)
CPU_DEVICE_UNARY(copy, copy, uint16, uint16, uint16)
CPU_DEVICE_UNARY(copy, copy, uint32, uint32, uint32)
CPU_DEVICE_UNARY(copy, copy, uint64, uint64, uint64)

CPU_DEVICE_UNARY(copy, copy, int8, int8, int8)
CPU_DEVICE_UNARY(copy, copy, int16, int16, int16)
CPU_DEVICE_UNARY(copy, copy, int32, int32, int32)
CPU_DEVICE_UNARY(copy, copy, int64, int64, int64)

CPU_DEVICE_NOIMPL(copy, copy, float16, float16, float16)
CPU_DEVICE_UNARY(copy, copy, float32, float32, float32)
CPU_DEVICE_UNARY(copy, copy, float64, float64, float64)

CPU_DEVICE_NOIMPL(copy, copy, complex32, complex32, complex32)
CPU_DEVICE_UNARYC(copy, copy, complex64, complex64, complex64)
CPU_DEVICE_UNARYC(copy, copy, complex128, complex128, complex128)


/*****************************************************************************/
/*                               Bitwise NOT                                 */
/*****************************************************************************/

#define invert(x) !x
CPU_DEVICE_UNARY(invert, invert, bool, bool, bool)
#undef invert

#define invert(x) ~x
CPU_DEVICE_UNARY(invert, invert, uint8, uint8, uint8)
CPU_DEVICE_UNARY(invert, invert, uint16, uint16, uint16)
CPU_DEVICE_UNARY(invert, invert, uint32, uint32, uint32)
CPU_DEVICE_UNARY(invert, invert, uint64, uint64, uint64)

CPU_DEVICE_UNARY(invert, invert, int8, int8, int8)
CPU_DEVICE_UNARY(invert, invert, int16, int16, int16)
CPU_DEVICE_UNARY(invert, invert, int32, int32, int32)
CPU_DEVICE_UNARY(invert, invert, int64, int64, int64)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

#define negative(x) -x

CPU_DEVICE_UNARY(negative, negative, uint8, int16, int16)
CPU_DEVICE_UNARY(negative, negative, uint16, int32, int32)
CPU_DEVICE_UNARY(negative, negative, uint32, int64, int64)

CPU_DEVICE_UNARY(negative, negative, int8, int8, int8)
CPU_DEVICE_UNARY(negative, negative, int16, int16, int16)
CPU_DEVICE_UNARY(negative, negative, int32, int32, int32)
CPU_DEVICE_UNARY(negative, negative, int64, int64, int64)

CPU_DEVICE_NOIMPL(negative, negative, float16, float16, float16)
CPU_DEVICE_UNARY(negative, negative, float32, float32, float32)
CPU_DEVICE_UNARY(negative, negative, float64, float64, float64)

CPU_DEVICE_NOIMPL(negative, negative, complex32, complex32, complex32)
CPU_DEVICE_UNARYC(negative, negative, complex64, complex64, complex64)
CPU_DEVICE_UNARYC(negative, negative, complex128, complex128, complex128)


/*****************************************************************************/
/*                                   Math                                    */
/*****************************************************************************/

#define CPU_DEVICE_UNARY_ALL_REAL_MATH(name) \
    CPU_DEVICE_UNARY(name##f, name##f, uint16, float32, float32)  \
    CPU_DEVICE_UNARY(name##f, name##f, int16, float32, float32)   \
    CPU_DEVICE_UNARY(name##f, name##f, float32, float32, float32) \
    CPU_DEVICE_UNARY(name, name, uint32, float64, float64)        \
    CPU_DEVICE_UNARY(name, name, int32, float64, float64)         \
    CPU_DEVICE_UNARY(name, name, float64, float64, float64)

#define CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(name) \
    CPU_DEVICE_UNARY_ALL_REAL_MATH(name)                              \
    CPU_DEVICE_NOIMPL(name, name, complex32, complex32, complex32)    \
    CPU_DEVICE_UNARYC(name, name, complex64, complex64, complex64)    \
    CPU_DEVICE_UNARYC(name, name, complex128, complex128, complex128) \

#define CPU_DEVICE_UNARY_ALL_HALF_MATH(name, hfunc) \
    CPU_DEVICE_UNARY(name##f16, hfunc, uint8, float16, float16)   \
    CPU_DEVICE_UNARY(name##f16, hfunc, int8, float16, float16)    \
    CPU_DEVICE_UNARY(name##f16, hfunc, float16, float16, float16)

#define CPU_DEVICE_UNARY_ALL_REAL_MATH_WITH_HALF(name, hfunc) \
    CPU_DEVICE_UNARY_ALL_HALF_MATH(name, hfunc)               \
    CPU_DEVICE_UNARY_ALL_REAL_MATH(name)

#define CPU_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(name, hfunc) \
    CPU_DEVICE_UNARY_ALL_HALF_MATH(name, hfunc)                  \
    CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(name)


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_REAL_MATH(fabs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(exp)
CPU_DEVICE_UNARY_ALL_REAL_MATH(exp2)
CPU_DEVICE_UNARY_ALL_REAL_MATH(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(log)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(log10)
CPU_DEVICE_UNARY_ALL_REAL_MATH(log2)
CPU_DEVICE_UNARY_ALL_REAL_MATH(log1p)
CPU_DEVICE_UNARY_ALL_REAL_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(sqrt)
CPU_DEVICE_UNARY_ALL_REAL_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(sin)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(cos)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(tan)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(asin)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(acos)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(sinh)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(cosh)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(tanh)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(asinh)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(acosh)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_REAL_MATH(erf)
CPU_DEVICE_UNARY_ALL_REAL_MATH(erfc)
CPU_DEVICE_UNARY_ALL_REAL_MATH(lgamma)
CPU_DEVICE_UNARY_ALL_REAL_MATH(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_REAL_MATH(ceil)
CPU_DEVICE_UNARY_ALL_REAL_MATH(floor)
CPU_DEVICE_UNARY_ALL_REAL_MATH(trunc)
CPU_DEVICE_UNARY_ALL_REAL_MATH(round)
CPU_DEVICE_UNARY_ALL_REAL_MATH(nearbyint)

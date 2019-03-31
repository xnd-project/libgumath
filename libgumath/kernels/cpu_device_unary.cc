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
#include "contrib/bfloat16.h"


/*****************************************************************************/
/*                          CPU device unary kernels                         */
/*****************************************************************************/

#define CPU_DEVICE_UNARY(name, func, t0, t1, common) \
extern "C" void                                                                   \
gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                              const int64_t N)                    \
{                                                                                 \
    const t0##_t *x0 = (const t0##_t *)a0;                                        \
    t1##_t *x1 = (t1##_t *)a1;                                                    \
                                                                                  \
    for (int64_t i = 0; i < N; i++) {                                             \
        x1[i] = func((common##_t)x0[i]);                                          \
    }                                                                             \
}                                                                                 \
                                                                                  \
extern "C" void                                                                   \
gm_cpu_device_fixed_1D_S_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                              const int64_t s0, const int64_t s1, \
                                              const int64_t N)                    \
{                                                                                 \
    const t0##_t *x0 = (const t0##_t *)a0;                                        \
    t1##_t *x1 = (t1##_t *)a1;                                                    \
    int64_t i, k0, k1;                                                            \
                                                                                  \
    for (i=0, k0=0, k1=0; i < N; i++, k0+=s0, k1+=s1) {                           \
        x1[k1] = func((common##_t)x0[k0]);                                        \
    }                                                                             \
}                                                                                 \
                                                                                  \
extern "C" void                                                                   \
gm_cpu_device_0D_##name##_##t0##_##t1(const char *a0, char *a1)                   \
{                                                                                 \
    const t0##_t x0 = *((const t0##_t *)a0);                                      \
    t1##_t *x1 = (t1##_t *)a1;                                                    \
    *x1 = func((common##_t)x0);                                                   \
}

#ifdef _MSC_VER
  #define CPU_DEVICE_UNARYC(name, func, t0, t1, common)
#else
  #define CPU_DEVICE_UNARYC(name, func, t0, t1, common) \
    CPU_DEVICE_UNARY(name, func, t0, t1, common)
#endif

#define CPU_DEVICE_NOIMPL(name, func, t0, t1, common)


#define CPU_DEVICE_ALL_UNARY(name, func, ufunc, tfunc, hfunc) \
    CPU_DEVICE_UNARY(name, func, bool, bool, bool)                    \
    CPU_DEVICE_UNARY(name, ufunc, bool, uint8, uint8)                 \
    CPU_DEVICE_UNARY(name, ufunc, bool, uint16, uint16)               \
    CPU_DEVICE_UNARY(name, ufunc, bool, uint32, uint32)               \
    CPU_DEVICE_UNARY(name, ufunc, bool, uint64, uint64)               \
    CPU_DEVICE_UNARY(name, func, bool, int8, int8)                    \
    CPU_DEVICE_UNARY(name, func, bool, int16, int16)                  \
    CPU_DEVICE_UNARY(name, func, bool, int32, int32)                  \
    CPU_DEVICE_UNARY(name, func, bool, int64, int64)                  \
    CPU_DEVICE_UNARY(name, tfunc, bool, bfloat16, bfloat16)           \
    CPU_DEVICE_NOIMPL(name, hfunc, bool, float16, float16)            \
    CPU_DEVICE_UNARY(name, func, bool, float32, float32)              \
    CPU_DEVICE_UNARY(name, func, bool, float64, float64)              \
    CPU_DEVICE_NOIMPL(name, func, bool, complex32, complex32)         \
    CPU_DEVICE_UNARYC(name, func, bool, complex64, complex64)         \
    CPU_DEVICE_UNARYC(name, func, bool, complex128, complex128)       \
                                                                      \
    CPU_DEVICE_UNARY(name, ufunc, uint8, uint8, uint8)                \
    CPU_DEVICE_UNARY(name, ufunc, uint8, uint16, uint16)              \
    CPU_DEVICE_UNARY(name, ufunc, uint8, uint32, uint32)              \
    CPU_DEVICE_UNARY(name, ufunc, uint8, uint64, uint64)              \
    CPU_DEVICE_UNARY(name, func, uint8, int16, int16)                 \
    CPU_DEVICE_UNARY(name, func, uint8, int32, int32)                 \
    CPU_DEVICE_UNARY(name, func, uint8, int64, int64)                 \
    CPU_DEVICE_UNARY(name, tfunc, uint8, bfloat16, bfloat16)          \
    CPU_DEVICE_NOIMPL(name, hfunc, uint8, float16, float16)           \
    CPU_DEVICE_UNARY(name, func, uint8, float32, float32)             \
    CPU_DEVICE_UNARY(name, func, uint8, float64, float64)             \
    CPU_DEVICE_NOIMPL(name, func, uint8, complex32, complex32)        \
    CPU_DEVICE_UNARYC(name, func, uint8, complex64, complex64)        \
    CPU_DEVICE_UNARYC(name, func, uint8, complex128, complex128)      \
                                                                      \
    CPU_DEVICE_UNARY(name, ufunc, uint16, uint16, uint16)             \
    CPU_DEVICE_UNARY(name, ufunc, uint16, uint32, uint32)             \
    CPU_DEVICE_UNARY(name, ufunc, uint16, uint64, uint64)             \
    CPU_DEVICE_UNARY(name, func, uint16, int32, int32)                \
    CPU_DEVICE_UNARY(name, func, uint16, int64, int64)                \
    CPU_DEVICE_UNARY(name, func, uint16, float32, float32)            \
    CPU_DEVICE_UNARY(name, func, uint16, float64, float64)            \
    CPU_DEVICE_UNARYC(name, func, uint16, complex64, complex64)       \
    CPU_DEVICE_UNARYC(name, func, uint16, complex128, complex128)     \
                                                                      \
    CPU_DEVICE_UNARY(name, ufunc, uint32, uint32, uint32)             \
    CPU_DEVICE_UNARY(name, ufunc, uint32, uint64, uint64)             \
    CPU_DEVICE_UNARY(name, func, uint32, int64, int64)                \
    CPU_DEVICE_UNARY(name, func, uint32, float64, float64)            \
    CPU_DEVICE_UNARYC(name, func, uint32, complex128, complex128)     \
                                                                      \
    CPU_DEVICE_UNARY(name, ufunc, uint64, uint64, uint64)             \
                                                                      \
    CPU_DEVICE_UNARY(name, func, int8, int8, int8)                    \
    CPU_DEVICE_UNARY(name, func, int8, int16, int16)                  \
    CPU_DEVICE_UNARY(name, func, int8, int32, int32)                  \
    CPU_DEVICE_UNARY(name, func, int8, int64, int64)                  \
    CPU_DEVICE_UNARY(name, tfunc, int8, bfloat16, bfloat16)           \
    CPU_DEVICE_NOIMPL(name, hfunc, int8, float16, float16)            \
    CPU_DEVICE_UNARY(name, func, int8, float32, float32)              \
    CPU_DEVICE_UNARY(name, func, int8, float64, float64)              \
    CPU_DEVICE_NOIMPL(name, func, int8, complex32, complex32)         \
    CPU_DEVICE_UNARYC(name, func, int8, complex64, complex64)         \
    CPU_DEVICE_UNARYC(name, func, int8, complex128, complex128)       \
                                                                      \
    CPU_DEVICE_UNARY(name, func, int16, int16, int16)                 \
    CPU_DEVICE_UNARY(name, func, int16, int32, int32)                 \
    CPU_DEVICE_UNARY(name, func, int16, int64, int64)                 \
    CPU_DEVICE_UNARY(name, func, int16, float32, float32)             \
    CPU_DEVICE_UNARY(name, func, int16, float64, float64)             \
    CPU_DEVICE_UNARYC(name, func, int16, complex64, complex64)        \
    CPU_DEVICE_UNARYC(name, func, int16, complex128, complex128)      \
                                                                      \
    CPU_DEVICE_UNARY(name, func, int32, int32, int32)                 \
    CPU_DEVICE_UNARY(name, func, int32, int64, int64)                 \
    CPU_DEVICE_UNARY(name, func, int32, float64, float64)             \
    CPU_DEVICE_UNARYC(name, func, int32, complex128, complex128)      \
                                                                      \
    CPU_DEVICE_UNARY(name, func, int64, int64, int64)                 \
                                                                      \
    CPU_DEVICE_UNARY(name, tfunc, bfloat16, bfloat16, bfloat16)       \
    CPU_DEVICE_UNARY(name, func, bfloat16, float32, float32)          \
    CPU_DEVICE_UNARY(name, func, bfloat16, float64, float64)          \
    CPU_DEVICE_UNARYC(name, func, bfloat16, complex64, complex64)     \
    CPU_DEVICE_UNARYC(name, func, bfloat16, complex128, complex128)   \
                                                                      \
    CPU_DEVICE_NOIMPL(name, hfunc, float16, float16, float16)         \
    CPU_DEVICE_NOIMPL(name, func, float16, float32, float32)          \
    CPU_DEVICE_NOIMPL(name, func, float16, float64, float64)          \
    CPU_DEVICE_NOIMPL(name, func, float16, complex32, complex32)      \
    CPU_DEVICE_NOIMPL(name, func, float16, complex64, complex64)      \
    CPU_DEVICE_NOIMPL(name, func, float16, complex128, complex128)    \
                                                                      \
    CPU_DEVICE_UNARY(name, func, float32, float32, float32)           \
    CPU_DEVICE_UNARY(name, func, float32, float64, float64)           \
    CPU_DEVICE_UNARYC(name, func, float32, complex64, complex64)      \
    CPU_DEVICE_UNARYC(name, func, float32, complex128, complex128)    \
                                                                      \
    CPU_DEVICE_UNARY(name, func, float64, float64, float64)           \
    CPU_DEVICE_UNARYC(name, func, float64, complex128, complex128)    \
                                                                      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex32, complex32)    \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex64, complex64)    \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex128, complex128)  \
                                                                      \
    CPU_DEVICE_UNARYC(name, func, complex64, complex64, complex64)    \
    CPU_DEVICE_UNARYC(name, func, complex64, complex128, complex128)  \
                                                                      \
    CPU_DEVICE_UNARYC(name, func, complex128, complex128, complex128)


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define copy(x) x
CPU_DEVICE_ALL_UNARY(copy, copy, copy, copy, copy)


/*****************************************************************************/
/*                                    Abs                                    */
/*****************************************************************************/

CPU_DEVICE_ALL_UNARY(abs, std::abs, copy, tf::fabs, std::abs)


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

CPU_DEVICE_UNARY(negative, negative, bfloat16, bfloat16, bfloat16)
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
    CPU_DEVICE_UNARY(name##f, name##f, uint16, float32, float32)        \
    CPU_DEVICE_UNARY(name##f, name##f, int16, float32, float32)         \
    CPU_DEVICE_UNARY(name##b16, tf::name, bfloat16, bfloat16, bfloat16) \
    CPU_DEVICE_UNARY(name##f, name##f, float32, float32, float32)       \
    CPU_DEVICE_UNARY(name, name, uint32, float64, float64)              \
    CPU_DEVICE_UNARY(name, name, int32, float64, float64)               \
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

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


/*
 * This file contains complex functions that resist compilation with
 * /fp:strict on Visual Studio compilers >= 2015 update 3.
 */


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

#define CPU_DEVICE_UNARYC(name, func, t0, t1, common) \
    CPU_DEVICE_UNARY(name, func, t0, t1, common)

#define CPU_DEVICE_NOIMPL(name, func, t0, t1, common)



/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define copy(x) x

CPU_DEVICE_NOIMPL(copy, copy, complex32, complex32, complex32)
CPU_DEVICE_UNARYC(copy, copy, complex64, complex64, complex64)
CPU_DEVICE_UNARYC(copy, copy, complex128, complex128, complex128)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

#define negative(x) -x

CPU_DEVICE_NOIMPL(negative, negative, complex32, complex32, complex32)
CPU_DEVICE_UNARYC(negative, negative, complex64, complex64, complex64)
CPU_DEVICE_UNARYC(negative, negative, complex128, complex128, complex128)


/*****************************************************************************/
/*                                   Math                                    */
/*****************************************************************************/

#define CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(name) \
    CPU_DEVICE_NOIMPL(name, name, complex32, complex32, complex32)    \
    CPU_DEVICE_UNARYC(name, name, complex64, complex64, complex64)    \
    CPU_DEVICE_UNARYC(name, name, complex128, complex128, complex128) \

#define CPU_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(name, hfunc) \
    CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(name)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(exp)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(log)
CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(log10)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CPU_DEVICE_UNARY_ALL_COMPLEX_MATH(sqrt)


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
/*                Lexicographic comparison for complex numbers               */
/*****************************************************************************/

#undef CPU_DEVICE_NOIMPL_DECL
#undef CPU_DEVICE_NOIMPL
#include "cpu_device_binary.h"

template <class T>
static inline bool
_isnan(T a)
{
    return std::isnan(a.real()) || std::isnan(a.imag());
}

template <class T, class U>
static inline bool
lexorder_le(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() < b.real() || (a.real() == b.real() && a.imag() <= b.imag());
}

template <class T, class U>
static inline bool
lexorder_lt(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
}

template <class T, class U>
static inline bool
lexorder_ge(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() > b.real() || (a.real() == b.real() && a.imag() >= b.imag());
}

template <class T, class U>
static inline bool
lexorder_gt(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() > b.real() || (a.real() == b.real() && a.imag() > b.imag());
}


/*****************************************************************************/
/*                         CPU device binary kernels                         */
/*****************************************************************************/

#define CPU_DEVICE_BINARY(name, func, t0, t1, t2, common) \
extern "C" void                                                          \
gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2(                    \
    const char *in0, const char *in1, char *out, int64_t N)              \
{                                                                        \
    const t0##_t *_in0 = (const t0##_t *)in0;                            \
    const t1##_t *_in1 = (const t1##_t *)in1;                            \
    t2##_t *_out = (t2##_t *)out;                                        \
    int64_t i;                                                           \
                                                                         \
    for (i = 0; i < N-7; i += 8) {                                       \
        _out[i] = func((common##_t)_in0[i], (common##_t)_in1[i]);        \
        _out[i+1] = func((common##_t)_in0[i+1], (common##_t)_in1[i+1]);  \
        _out[i+2] = func((common##_t)_in0[i+2], (common##_t)_in1[i+2]);  \
        _out[i+3] = func((common##_t)_in0[i+3], (common##_t)_in1[i+3]);  \
        _out[i+4] = func((common##_t)_in0[i+4], (common##_t)_in1[i+4]);  \
        _out[i+5] = func((common##_t)_in0[i+5], (common##_t)_in1[i+5]);  \
        _out[i+6] = func((common##_t)_in0[i+6], (common##_t)_in1[i+6]);  \
        _out[i+7] = func((common##_t)_in0[i+7], (common##_t)_in1[i+7]);  \
    }                                                                    \
    for (; i < N; i++) {                                                 \
        _out[i] = func((common##_t)_in0[i], (common##_t)_in1[i]);        \
    }                                                                    \
}                                                                        \
                                                                         \
extern "C" void                                                          \
gm_cpu_device_0D_##name##_##t0##_##t1##_##t2(                            \
    const char *in0, const char *in1, char *out)                         \
{                                                                        \
    const t0##_t x = *(const t0##_t *)in0;                               \
    const t1##_t y = *(const t1##_t *)in1;                               \
    *(t2##_t *)out = func((common##_t)x, (common##_t)y);                 \
}

#define CPU_DEVICE_BINARYC(name, func, t0, t1, t2, common) \
  CPU_DEVICE_BINARY(name, func, t0, t1, t2, common)

#define CPU_DEVICE_NOIMPL(name, func, t0, t1, t2, common)
#define CPU_DEVICE_NOKERN(name, func, t0, t1, t2, common)


/*****************************************************************************/
/*                                 Arithmetic                                */
/*****************************************************************************/

#define CPU_DEVICE_ALL_BINARY(name, func, hfunc) \
    CPU_DEVICE_NOIMPL(name, func, uint8, complex32, complex32, complex32)          \
    CPU_DEVICE_BINARYC(name, func, uint8, complex64, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, uint8, complex128, complex128, complex128)      \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, uint16, complex32, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, uint16, complex64, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, uint16, complex128, complex128, complex128)     \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, uint32, complex32, complex128, complex128)       \
    CPU_DEVICE_BINARYC(name, func, uint32, complex64, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, uint32, complex128, complex128, complex128)     \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, int8, complex32, complex32, complex32)           \
    CPU_DEVICE_BINARYC(name, func, int8, complex64, complex64, complex64)          \
    CPU_DEVICE_BINARYC(name, func, int8, complex128, complex128, complex128)       \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, int16, complex32, complex64, complex64)          \
    CPU_DEVICE_BINARYC(name, func, int16, complex64, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, int16, complex128, complex128, complex128)      \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, int32, complex32, complex128, complex128)        \
    CPU_DEVICE_BINARYC(name, func, int32, complex64, complex128, complex128)       \
    CPU_DEVICE_BINARYC(name, func, int32, complex128, complex128, complex128)      \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, float16, complex32, complex32, complex32)        \
    CPU_DEVICE_NOIMPL(name, func, float16, complex64, complex64, complex64)        \
    CPU_DEVICE_NOIMPL(name, func, float16, complex128, complex128, complex128)     \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, float32, complex32, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, float32, complex64, complex64, complex64)       \
    CPU_DEVICE_BINARYC(name, func, float32, complex128, complex128, complex128)    \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, float64, complex32, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, float64, complex64, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, float64, complex128, complex128, complex128)    \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, complex32, uint8, complex32, complex32)          \
    CPU_DEVICE_NOIMPL(name, func, complex32, uint16, complex64, complex64)         \
    CPU_DEVICE_NOIMPL(name, func, complex32, uint32, complex128, complex128)       \
    CPU_DEVICE_NOIMPL(name, func, complex32, int8, complex32, complex32)           \
    CPU_DEVICE_NOIMPL(name, func, complex32, int16, complex64, complex64)          \
    CPU_DEVICE_NOIMPL(name, func, complex32, int32, complex128, complex128)        \
    CPU_DEVICE_NOIMPL(name, func, complex32, float16, complex32, complex32)        \
    CPU_DEVICE_NOIMPL(name, func, complex32, float32, complex64, complex64)        \
    CPU_DEVICE_NOIMPL(name, func, complex32, float64, complex128, complex128)      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex32, complex32, complex32)      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex64, complex64, complex64)      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex128, complex128, complex128)   \
                                                                                   \
    CPU_DEVICE_BINARYC(name, func, complex64, uint8, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, complex64, uint16, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, complex64, uint32, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, complex64, int8, complex64, complex64)          \
    CPU_DEVICE_BINARYC(name, func, complex64, int16, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, complex64, int32, complex128, complex128)       \
    CPU_DEVICE_NOIMPL(name, func, complex64, float16, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, complex64, float32, complex64, complex64)       \
    CPU_DEVICE_BINARYC(name, func, complex64, float64, complex128, complex128)     \
    CPU_DEVICE_NOIMPL(name, func, complex64, complex32, complex64, complex64)      \
    CPU_DEVICE_BINARYC(name, func, complex64, complex64, complex64, complex64)     \
    CPU_DEVICE_BINARYC(name, func, complex64, complex128, complex128, complex128)  \
                                                                                   \
    CPU_DEVICE_BINARYC(name, func, complex128, uint8, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, complex128, uint16, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, complex128, uint32, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, complex128, int8, complex128, complex128)       \
    CPU_DEVICE_BINARYC(name, func, complex128, int16, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, complex128, int32, complex128, complex128)      \
    CPU_DEVICE_NOIMPL(name, func, complex128, float16, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, complex128, float32, complex128, complex128)    \
    CPU_DEVICE_BINARYC(name, func, complex128, float64, complex128, complex128)    \
    CPU_DEVICE_NOIMPL(name, func, complex128, complex32, complex128, complex128)   \
    CPU_DEVICE_BINARYC(name, func, complex128, complex64, complex128, complex128)  \
    CPU_DEVICE_BINARYC(name, func, complex128, complex128, complex128, complex128) \

#define CPU_DEVICE_ALL_BINARY_FLOAT_RETURN(name, func, hfunc) \
    CPU_DEVICE_NOIMPL(name, func, uint8, complex32, complex32, complex32)          \
    CPU_DEVICE_BINARYC(name, func, uint8, complex64, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, uint8, complex128, complex128, complex128)      \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, uint16, complex32, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, uint16, complex64, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, uint16, complex128, complex128, complex128)     \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, uint32, complex32, complex128, complex128)       \
    CPU_DEVICE_BINARYC(name, func, uint32, complex64, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, uint32, complex128, complex128, complex128)     \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, int8, complex32, complex32, complex32)           \
    CPU_DEVICE_BINARYC(name, func, int8, complex64, complex64, complex64)          \
    CPU_DEVICE_BINARYC(name, func, int8, complex128, complex128, complex128)       \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, int16, complex32, complex64, complex64)          \
    CPU_DEVICE_BINARYC(name, func, int16, complex64, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, int16, complex128, complex128, complex128)      \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, int32, complex32, complex128, complex128)        \
    CPU_DEVICE_BINARYC(name, func, int32, complex64, complex128, complex128)       \
    CPU_DEVICE_BINARYC(name, func, int32, complex128, complex128, complex128)      \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, float16, complex32, complex32, complex32)        \
    CPU_DEVICE_NOIMPL(name, func, float16, complex64, complex64, complex64)        \
    CPU_DEVICE_NOIMPL(name, func, float16, complex128, complex128, complex128)     \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, float32, complex32, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, float32, complex64, complex64, complex64)       \
    CPU_DEVICE_BINARYC(name, func, float32, complex128, complex128, complex128)    \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, float64, complex32, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, float64, complex64, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, float64, complex128, complex128, complex128)    \
                                                                                   \
    CPU_DEVICE_NOIMPL(name, func, complex32, uint8, complex32, complex32)          \
    CPU_DEVICE_NOIMPL(name, func, complex32, uint16, complex64, complex64)         \
    CPU_DEVICE_NOIMPL(name, func, complex32, uint32, complex128, complex128)       \
    CPU_DEVICE_NOIMPL(name, func, complex32, int8, complex32, complex32)           \
    CPU_DEVICE_NOIMPL(name, func, complex32, int16, complex64, complex64)          \
    CPU_DEVICE_NOIMPL(name, func, complex32, int32, complex128, complex128)        \
    CPU_DEVICE_NOIMPL(name, func, complex32, float16, complex32, complex32)        \
    CPU_DEVICE_NOIMPL(name, func, complex32, float32, complex64, complex64)        \
    CPU_DEVICE_NOIMPL(name, func, complex32, float64, complex128, complex128)      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex32, complex32, complex32)      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex64, complex64, complex64)      \
    CPU_DEVICE_NOIMPL(name, func, complex32, complex128, complex128, complex128)   \
                                                                                   \
    CPU_DEVICE_BINARYC(name, func, complex64, uint8, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, complex64, uint16, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, complex64, uint32, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, complex64, int8, complex64, complex64)          \
    CPU_DEVICE_BINARYC(name, func, complex64, int16, complex64, complex64)         \
    CPU_DEVICE_BINARYC(name, func, complex64, int32, complex128, complex128)       \
    CPU_DEVICE_NOIMPL(name, func, complex64, float16, complex64, complex64)        \
    CPU_DEVICE_BINARYC(name, func, complex64, float32, complex64, complex64)       \
    CPU_DEVICE_BINARYC(name, func, complex64, float64, complex128, complex128)     \
    CPU_DEVICE_NOIMPL(name, func, complex64, complex32, complex64, complex64)      \
    CPU_DEVICE_BINARYC(name, func, complex64, complex64, complex64, complex64)     \
    CPU_DEVICE_BINARYC(name, func, complex64, complex128, complex128, complex128)  \
                                                                                   \
    CPU_DEVICE_BINARYC(name, func, complex128, uint8, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, complex128, uint16, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, complex128, uint32, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, complex128, int8, complex128, complex128)       \
    CPU_DEVICE_BINARYC(name, func, complex128, int16, complex128, complex128)      \
    CPU_DEVICE_BINARYC(name, func, complex128, int32, complex128, complex128)      \
    CPU_DEVICE_NOIMPL(name, func, complex128, float16, complex128, complex128)     \
    CPU_DEVICE_BINARYC(name, func, complex128, float32, complex128, complex128)    \
    CPU_DEVICE_BINARYC(name, func, complex128, float64, complex128, complex128)    \
    CPU_DEVICE_NOIMPL(name, func, complex128, complex32, complex128, complex128)   \
    CPU_DEVICE_BINARYC(name, func, complex128, complex64, complex128, complex128)  \
    CPU_DEVICE_BINARYC(name, func, complex128, complex128, complex128, complex128) \

#define add(x, y) x + y
CPU_DEVICE_ALL_BINARY(add, add, add)

#define subtract(x, y) x - y
CPU_DEVICE_ALL_BINARY(subtract, subtract, sub)

#define multiply(x, y) x * y
CPU_DEVICE_ALL_BINARY(multiply, multiply, multiply)

#define divide(x, y) x / y
CPU_DEVICE_ALL_BINARY_FLOAT_RETURN(divide, divide, divide)


/*****************************************************************************/
/*                                 Comparison                                */
/*****************************************************************************/

#define CPU_DEVICE_ALL_COMPARISON(name, func, hfunc, cfunc) \
    CPU_DEVICE_NOIMPL(name, cfunc, uint8, complex32, bool, complex32)         \
    CPU_DEVICE_BINARYC(name, cfunc, uint8, complex64, bool, complex64)        \
    CPU_DEVICE_BINARYC(name, cfunc, uint8, complex128, bool, complex128)      \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, uint16, complex32, bool, complex64)        \
    CPU_DEVICE_BINARYC(name, cfunc, uint16, complex64, bool, complex64)       \
    CPU_DEVICE_BINARYC(name, cfunc, uint16, complex128, bool, complex128)     \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, uint32, complex32, bool, complex128)       \
    CPU_DEVICE_BINARYC(name, cfunc, uint32, complex64, bool, complex128)      \
    CPU_DEVICE_BINARYC(name, cfunc, uint32, complex128, bool, complex128)     \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, int8, complex32, bool, complex32)          \
    CPU_DEVICE_BINARYC(name, cfunc, int8, complex64, bool, complex64)         \
    CPU_DEVICE_BINARYC(name, cfunc, int8, complex128, bool, complex128)       \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, int16, complex32, bool, complex64)         \
    CPU_DEVICE_BINARYC(name, cfunc, int16, complex64, bool, complex64)        \
    CPU_DEVICE_BINARYC(name, cfunc, int16, complex128, bool, complex128)      \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, int32, complex32, bool, complex128)        \
    CPU_DEVICE_BINARYC(name, cfunc, int32, complex64, bool, complex128)       \
    CPU_DEVICE_BINARYC(name, cfunc, int32, complex128, bool, complex128)      \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, float16, complex32, bool, complex32)       \
    CPU_DEVICE_NOIMPL(name, cfunc, float16, complex64, bool, complex64)       \
    CPU_DEVICE_NOIMPL(name, cfunc, float16, complex128, bool, complex128)     \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, float32, complex32, bool, complex64)       \
    CPU_DEVICE_BINARYC(name, cfunc, float32, complex64, bool, complex64)      \
    CPU_DEVICE_BINARYC(name, cfunc, float32, complex128, bool, complex128)    \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, float64, complex32, bool, complex128)      \
    CPU_DEVICE_BINARYC(name, cfunc, float64, complex64, bool, complex128)     \
    CPU_DEVICE_BINARYC(name, cfunc, float64, complex128, bool, complex128)    \
                                                                              \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, uint8, bool, complex32)         \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, uint16, bool, complex64)        \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, uint32, bool, complex128)       \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, int8, bool, complex32)          \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, int16, bool, complex64)         \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, int32, bool, complex128)        \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, float16, bool, complex32)       \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, float32, bool, complex64)       \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, float64, bool, complex128)      \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, complex32, bool, complex32)     \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, complex64, bool, complex64)     \
    CPU_DEVICE_NOIMPL(name, cfunc, complex32, complex128, bool, complex128)   \
                                                                              \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, uint8, bool, complex64)        \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, uint16, bool, complex64)       \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, uint32, bool, complex128)      \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, int8, bool, complex64)         \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, int16, bool, complex64)        \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, int32, bool, complex128)       \
    CPU_DEVICE_NOIMPL(name, cfunc, complex64, float16, bool, complex64)       \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, float32, bool, complex64)      \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, float64, bool, complex128)     \
    CPU_DEVICE_NOIMPL(name, cfunc, complex64, complex32, bool, complex64)     \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, complex64, bool, complex64)    \
    CPU_DEVICE_BINARYC(name, cfunc, complex64, complex128, bool, complex128)  \
                                                                              \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, uint8, bool, complex128)      \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, uint16, bool, complex128)     \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, uint32, bool, complex128)     \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, int8, bool, complex128)       \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, int16, bool, complex128)      \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, int32, bool, complex128)      \
    CPU_DEVICE_NOIMPL(name, cfunc, complex128, float16, bool, complex128)     \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, float32, bool, complex128)    \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, float64, bool, complex128)    \
    CPU_DEVICE_NOIMPL(name, cfunc, complex128, complex32, bool, complex128)   \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, complex64, bool, complex128)  \
    CPU_DEVICE_BINARYC(name, cfunc, complex128, complex128, bool, complex128) \


#define less(x, y) x < y
CPU_DEVICE_ALL_COMPARISON(less, less, less, lexorder_lt)

#define less_equal(x, y) x <= y
CPU_DEVICE_ALL_COMPARISON(less_equal, less_equal, less_equal, lexorder_le)

#define greater_equal(x, y) x >= y
CPU_DEVICE_ALL_COMPARISON(greater_equal, greater_equal, greater_equal, lexorder_ge)

#define greater(x, y) x > y
CPU_DEVICE_ALL_COMPARISON(greater, greater, greater, lexorder_gt)

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


#ifndef DEVICE_HH
#define DEVICE_HH


#include <cstdint>
#include <cinttypes>
#include "contrib/bfloat16.h"


#ifdef __CUDACC__
#define DEVICE __device__
#define ISNAN(x) (isnan(x))
#else
#define DEVICE
#define ISNAN(x) (std::isnan(x))
#endif


/*****************************************************************************/
/*                                   Divmod                                  */
/*****************************************************************************/

/* Python: floatobject.c */
static inline DEVICE void
_divmod(double *q, double *r, double vx, double wx)
{
    double div, mod, floordiv;

    mod = fmod(vx, wx);
    /* fmod is typically exact, so vx-mod is *mathematically* an
       exact multiple of wx.  But this is fp arithmetic, and fp
       vx - mod is an approximation; the result is that div may
       not be an exact integral value after the division, although
       it will always be very close to one.
    */
    div = (vx - mod) / wx;
    if (mod) {
        /* ensure the remainder has the same sign as the denominator */
        if ((wx < 0) != (mod < 0)) {
            mod += wx;
            div -= 1.0;
        }
    }
    else {
        /* the remainder is zero, and in the presence of signed zeroes
           fmod returns different results across platforms; ensure
           it has the same sign as the denominator. */
        mod = copysign(0.0, wx);
    }
    /* snap quotient to nearest integral value */
    if (div) {
        floordiv = floor(div);
        if (div - floordiv > 0.5)
            floordiv += 1.0;
    }
    else {
        /* div is zero - get the same sign as the true quotient */
        floordiv = copysign(0.0, vx / wx); /* zero w/ sign of vx/wx */
    }

    *q = floordiv;
    *r = mod;
}

static inline DEVICE void
_divmod(float *q, float *r, float vx, float wx)
{
    float div, mod, floordiv;

    mod = fmodf(vx, wx);
    /* fmod is typically exact, so vx-mod is *mathematically* an
       exact multiple of wx.  But this is fp arithmetic, and fp
       vx - mod is an approximation; the result is that div may
       not be an exact integral value after the division, although
       it will always be very close to one.
    */
    div = (vx - mod) / wx;
    if (mod) {
        /* ensure the remainder has the same sign as the denominator */
        if ((wx < 0) != (mod < 0)) {
            mod += wx;
            div -= 1.0;
        }
    }
    else {
        /* the remainder is zero, and in the presence of signed zeroes
           fmod returns different results across platforms; ensure
           it has the same sign as the denominator. */
        mod = copysignf(0.0, wx);
    }
    /* snap quotient to nearest integral value */
    if (div) {
        floordiv = floorf(div);
        if (div - floordiv > 0.5)
            floordiv += 1.0;
    }
    else {
        /* div is zero - get the same sign as the true quotient */
        floordiv = copysignf(0.0, vx / wx); /* zero w/ sign of vx/wx */
    }

    *q = floordiv;
    *r = mod;
}

static inline DEVICE void
_divmod(bfloat16_t *q, bfloat16_t *r, bfloat16_t a, bfloat16_t b)
{
    float qq;
    float rr;

    _divmod(&qq, &rr, (float)a, (float)b);

    *q = (bfloat16_t)qq;
    *r = (bfloat16_t)rr;
}

#define divmod_unsigned(T) \
static inline DEVICE void     \
_divmod(T *q, T *r, T a, T b) \
{                             \
   if (b == 0) {              \
        *q = 0;               \
        *r = 0;               \
   }                          \
   else {                     \
       *q = a / b;            \
       *r = a % b;            \
   }                          \
}

divmod_unsigned(uint8_t)
divmod_unsigned(uint16_t)
divmod_unsigned(uint32_t)
divmod_unsigned(uint64_t)

#define divmod_signed(T, MIN) \
static inline DEVICE void                          \
_divmod(T *q, T *r, T a, T b)                      \
{                                                  \
    if (b == 0 || (a == MIN && b == -1)) {         \
        *q = 0;                                    \
        *r = 0;                                    \
    }                                              \
    else {                                         \
        T qq = a / b;                              \
        T rr = a % b;                              \
                                                   \
        *q = rr ? (qq - ((a < 0) ^ (b < 0))) : qq; \
        *r = a - *q * b;                           \
    }                                              \
}

divmod_signed(int8_t, INT8_MIN)
divmod_signed(int16_t, INT16_MIN)
divmod_signed(int32_t, INT32_MIN)
divmod_signed(int64_t, INT64_MIN)

template <class T>
static inline DEVICE T
_floor_divide(T a, T b)
{
    T q;
    T r;

    _divmod(&q, &r, a, b);

    return q;
}

template <class T>
static inline DEVICE T
_remainder(T a, T b)
{
    T q;
    T r;

    _divmod(&q, &r, a, b);

    return r;
}


/*****************************************************************************/
/*                Lexicographic comparison for complex numbers               */
/*****************************************************************************/

template <class T>
static inline DEVICE bool
_isnan(T a)
{
    return ISNAN(a.real()) || ISNAN(a.imag());
}

template <class T, class U>
static inline DEVICE bool
lexorder_lt(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
}

template <class T, class U>
static inline DEVICE bool
lexorder_le(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() < b.real() || (a.real() == b.real() && a.imag() <= b.imag());
}

template <class T, class U>
static inline DEVICE bool
lexorder_ge(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() > b.real() || (a.real() == b.real() && a.imag() >= b.imag());
}

template <class T, class U>
static inline DEVICE bool
lexorder_gt(T a, U b)
{
    if (_isnan(a) || _isnan(b)) {
        return false;
    }

    return a.real() > b.real() || (a.real() == b.real() && a.imag() > b.imag());
}


#endif /* DEVICE_HH */

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
#include <complex>
#include "contrib/bfloat16.h"


#ifdef __CUDACC__
#include <cuda_fp16.h>
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
/*                                      Abs                                  */
/*****************************************************************************/

#ifdef __CUDACC__
#define abs_unsigned(T) \
static inline DEVICE T  \
_abs(T x)               \
{                       \
    return x;           \
}

abs_unsigned(uint8_t)
abs_unsigned(uint16_t)
abs_unsigned(uint32_t)
abs_unsigned(uint64_t)

#define abs_signed(T) \
static inline DEVICE T     \
_abs(T x)                  \
{                          \
    return x < 0 ? -x : x; \
}

abs_signed(int8_t)
abs_signed(int16_t)
abs_signed(int32_t)
abs_signed(int64_t)

static inline DEVICE float32_t
_abs(float32_t x)
{
    return fabsf(x);
}

static inline DEVICE float64_t
_abs(float64_t x)
{
    return fabs(x);
}

static inline DEVICE complex64_t
_abs(complex64_t x)
{
    return thrust::abs(x);
}

static inline DEVICE complex128_t
_abs(complex128_t x)
{
    return thrust::abs(x);
}
#endif


/*****************************************************************************/
/*                                    Pow                                    */
/*****************************************************************************/

#define pow_unsigned(name, T, mask) \
static inline DEVICE T               \
name(T base, T exp)                  \
{                                    \
    uint64_t r = 1;                  \
                                     \
    while (exp > 0) {                \
        if (exp & 1) {               \
            r = (r * base) & mask;   \
        }                            \
        base = (base * base) & mask; \
        exp >>= 1;                   \
    }                                \
                                     \
    return r;                        \
}

pow_unsigned(_pow, uint8_t, UINT8_MAX)
pow_unsigned(_pow, uint16_t, UINT16_MAX)
pow_unsigned(_pow, uint32_t, UINT32_MAX)
pow_unsigned(_pow, uint64_t, UINT64_MAX)

pow_unsigned(_pow_int8_t, uint8_t, INT8_MAX)
pow_unsigned(_pow_int16_t, uint16_t, INT16_MAX)
pow_unsigned(_pow_int32_t, uint32_t, INT32_MAX)
pow_unsigned(_pow_int64_t, uint64_t, INT64_MAX)

#define pow_signed(T, U, MIN, MAX)      \
static inline DEVICE T                  \
_pow(T ibase, T exp)                    \
{                                       \
    U base;                             \
    U r;                                \
                                        \
    if (ibase < 0) {                    \
        base = (U)(-ibase);             \
        r = _pow_##T(base, exp);        \
        return (exp % 2 == 0) ? r : -r; \
    }                                   \
    else {                              \
        base = (U)ibase;                \
        return _pow_##T(base, exp);     \
    }                                   \
}

pow_signed(int8_t, uint8_t, INT8_MIN, INT8_MAX)
pow_signed(int16_t, uint16_t, INT16_MIN, INT16_MAX)
pow_signed(int32_t, uint32_t, INT32_MIN, INT32_MAX)
pow_signed(int64_t, uint64_t, INT64_MIN, INT64_MAX)

static inline DEVICE bfloat16_t
_pow(bfloat16_t x, bfloat16_t y)
{
    return (bfloat16_t)powf((float)x, (float)y);
}

static inline DEVICE float32_t
_pow(float32_t x, float32_t y)
{
    return powf(x, y);
}

static inline DEVICE float64_t
_pow(float64_t x, float64_t y)
{
    return pow(x, y);
}

#ifdef __CUDACC__
static inline DEVICE half
_pow(half x, half y)
{
    return __float2half(pow(__half2float(x), __half2float(y)));
}
#endif


/*****************************************************************************/
/*                                 Complex pow                               */
/*****************************************************************************/

#ifdef __CUDACC__
template <class T>
using Complex = thrust::complex<T>;

template <class T>
static inline DEVICE Complex<T>
_cpow(Complex<T> x, Complex<T> y)
{
    return thrust::pow<T>(x, y);
}
#else
template <class T>
using Complex = std::complex<T>;

template <class T>
static inline DEVICE Complex<T>
_cpow(Complex<T> x, Complex<T> y)
{
    return std::pow<T>(x, y);
}
#endif

static inline DEVICE double xhypot(double x, double y) { return hypot(x, y); }
static inline DEVICE double xpow(double x, double y) { return pow(x, y); }
static inline DEVICE double xatan2(double x, double y) { return atan2(x, y); }
static inline DEVICE double xexp(double x) { return exp(x); }
static inline DEVICE double xlog(double x) { return log(x); }
static inline DEVICE float xhypot(float x, float y) { return hypotf(x, y); }
static inline DEVICE float xpow(float x, float y) { return powf(x, y); }
static inline DEVICE float xatan2(float x, float y) { return atan2f(x, y); }
static inline DEVICE float xexp(float x) { return expf(x); }
static inline DEVICE float xlog(float x) { return logf(x); }


/* Python: complexobject.c */
template <class T>
static inline DEVICE Complex<T>
c_quot(const Complex<T> a, const Complex<T> b)
{
    /* This algorithm is better, and is pretty obvious:  first divide the
     * numerators and denominator by whichever of {b.real, b.imag} has
     * larger magnitude.  The earliest reference I found was to CACM
     * Algorithm 116 (Complex Division, Robert L. Smith, Stanford
     * University).  As usual, though, we're still ignoring all IEEE
     * endcases.
     */
    const T abs_breal = b.real() < 0 ? -b.real() : b.real();
    const T abs_bimag = b.imag() < 0 ? -b.imag() : b.imag();
    T real, imag;

    if (abs_breal >= abs_bimag) {
        /* divide tops and bottom by b.real */
        if (abs_breal == 0.0) {
            // errno = EDOM;
            real = imag = 0.0;
        }
        else {
            const T ratio = b.imag() / b.real();
            const T denom = b.real() + b.imag() * ratio;
            real = (a.real() + a.imag() * ratio) / denom;
            imag = (a.imag() - a.real() * ratio) / denom;
        }
    }
    else if (abs_bimag >= abs_breal) {
        /* divide tops and bottom by b.imag */
        const T ratio = b.real() / b.imag();
        const T denom = b.real() * ratio + b.imag();
        real = (a.real() * ratio + a.imag()) / denom;
        imag = (a.imag() * ratio - a.real()) / denom;
    }
    else {
        /* At least one of b.real or b.imag is a NaN */
        real = imag = NAN;
    }

    return Complex<T>{real, imag};
}

template <class T>
static inline DEVICE Complex<T>
c_pow(const Complex<T> a, const Complex<T> b)
{
    if (b.real() == 0 && b.imag() == 0) {
        return Complex<T>{1, 0};
    }
    else if (a.real() == 0 && a.imag() == 0) {
        // if (b.imag() != 0 || b.real() < 0)
        //    errno = EDOM;
        return Complex<T>{0, 0};
    }
    else {
        T vabs = xhypot(a.real(), a.imag());
        T len = xpow(vabs, b.real());
        T at = xatan2(a.imag(), a.real());
        T phase = at * b.real();

        if (b.imag() != 0) {
            len /= xexp(at * b.imag());
            phase += b.imag() * xlog(vabs);
        }

        T real = len*cos(phase);
        T imag = len*sin(phase);

        return Complex<T>{real, imag};
    }
}

template <class T>
static inline DEVICE Complex<T>
c_powu(Complex<T> base, uint64_t exp)
{
    Complex<T> r{1, 0};

    while (exp > 0) {
        if (exp & 1) {
            r = r * base;
        }
        base = base * base;
        exp >>= 1;
    }

    return r;
}

template <class T>
static inline DEVICE Complex<T>
c_powi(Complex<T> x, int64_t n)
{
    if (n > 99 || n < -99) {
        Complex<T> y{(T)n, 0};
        return c_pow(x, y);
    }
    else if (n > 0) {
        return c_powu(x, (uint64_t)n);
    }
    else {
        Complex<T> one{1, 0};
        return c_quot(one, c_powu(x, (T)(-n)));
    }
}

template <class T>
static inline DEVICE Complex<T>
complex_pow(Complex<T> a, Complex<T> exponent)
{
    int64_t int_exponent;

    int_exponent = (int64_t)exponent.real();
    if (exponent.imag() == 0 && exponent.real() == int_exponent) {
        return c_powi(a, int_exponent);
    }
    else {
        return c_pow(a, exponent);
    }
}

template <class T>
static inline DEVICE Complex<T>
_pow(Complex<T> x, Complex<T> y)
{
    Complex<double> a = x;
    Complex<double> b = y;
    Complex<double> r = complex_pow(a, b);
    return (Complex<T>)r;
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

template <class T, class U>
static inline DEVICE bool
lexorder_eqn(T a, U b)
{
    bool real_equal = a.real() == b.real() || (ISNAN(a.real()) && ISNAN(b.real()));
    bool imag_equal = a.imag() == b.imag() || (ISNAN(a.imag()) && ISNAN(b.imag()));

    return real_equal && imag_equal;
}


/*****************************************************************************/
/*                                Half equality                              */
/*****************************************************************************/

#ifdef __CUDACC__
static inline DEVICE bool
half_ne(half a, half b)
{
    return !__heq(a, b);
}

static inline DEVICE bool
half_eqn(half a, half b)
{
    return __heq(a, b) || (__hisnan(a) && __hisnan(b));
}
#endif


#endif /* DEVICE_HH */

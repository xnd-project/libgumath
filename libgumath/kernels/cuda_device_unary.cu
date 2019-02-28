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
#include <thrust/device_vector.h>
#include "cuda_device_unary.h"
#include "contrib/bfloat16.h"


/*****************************************************************************/
/*                                 Half float                                */
/*****************************************************************************/

static inline __device__ half
half_abs(half a)
{
    return __hlt(a, 0) ? __hneg(a) : a;
}


/*****************************************************************************/
/*                         CUDA device unary kernels                         */
/*****************************************************************************/

#define CUDA_DEVICE_UNARY(name, func, t0, t1, common) \
static __global__ void                                                             \
_1D_C_##name##_##t0##_##t1(const t0##_t *x0, t1##_t *x1,                           \
                           const int64_t N)                                        \
{                                                                                  \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;                         \
    int64_t stride = blockDim.x * gridDim.x;                                       \
                                                                                   \
    for (int64_t i = index; i < N; i += stride) {                                  \
        x1[i] = func((common##_t)x0[i]);                                           \
    }                                                                              \
}                                                                                  \
                                                                                   \
extern "C" void                                                                    \
gm_cuda_device_fixed_1D_C_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                               const int64_t N)                    \
{                                                                                  \
    const t0##_t *x0 = (const t0##_t *)a0;                                         \
    t1##_t *x1 = (t1##_t *)a1;                                                     \
    const int blockSize = 256;                                                     \
    const int64_t numBlocks = (N + blockSize - 1) / blockSize;                     \
                                                                                   \
    _1D_C_##name##_##t0##_##t1<<<numBlocks, blockSize>>>(x0, x1, N);               \
}                                                                                  \
                                                                                   \
static __global__ void                                                             \
_1D_S_##name##_##t0##_##t1(const t0##_t *x0, t1##_t *x1,                           \
                           const int64_t s0, const int64_t s1,                     \
                           const int64_t N)                                        \
{                                                                                  \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;                         \
    int64_t stride = blockDim.x * gridDim.x;                                       \
                                                                                   \
    for (int64_t i = index; i < N; i += stride) {                                  \
        const int64_t k0 = i * s0;                                                 \
        const int64_t k1 = i * s1;                                                 \
        x1[k1] = func((common##_t)x0[k0]);                                         \
    }                                                                              \
}                                                                                  \
                                                                                   \
extern "C" void                                                                    \
gm_cuda_device_fixed_1D_S_##name##_##t0##_##t1(const char *a0, char *a1,           \
                                               const int64_t s0, const int64_t s1, \
                                               const int64_t N)                    \
{                                                                                  \
    const t0##_t *x0 = (const t0##_t *)a0;                                         \
    t1##_t *x1 = (t1##_t *)a1;                                                     \
    const int blockSize = 256;                                                     \
    const int64_t numBlocks = (N + blockSize - 1) / blockSize;                     \
                                                                                   \
    _1D_S_##name##_##t0##_##t1<<<numBlocks, blockSize>>>(x0, x1, s0, s1, N);       \
}                                                                                  \
                                                                                   \
static __global__ void                                                             \
_0D_##name##_##t0##_##t1(const t0##_t *x0, t1##_t *x1)                             \
{                                                                                  \
    *x1 = func((common##_t)*x0);                                                   \
}                                                                                  \
                                                                                   \
extern "C" void                                                                    \
gm_cuda_device_0D_##name##_##t0##_##t1(const char *a0, char *a1)                   \
{                                                                                  \
    const t0##_t *x0 = (const t0##_t *)a0;                                         \
    t1##_t *x1 = (t1##_t *)a1;                                                     \
                                                                                   \
    _0D_##name##_##t0##_##t1<<<1, 1>>>(x0, x1);                                    \
}

#define CUDA_DEVICE_UNARY_REDUCE(name, func, t0, t1) \
extern "C" void                                                                \
gm_cuda_device_1D_C_reduce_##name##_##t0##_##t1(const char *a0, char *a1,      \
                                                const int64_t N)               \
{                                                                              \
    thrust::device_ptr<t0##_t> x0 = thrust::device_pointer_cast((t0##_t *)a0); \
    t1##_t *x1 = (t1##_t *)a1;                                                 \
                                                                               \
    *x1 = thrust::reduce(x0, x0+N, (t1##_t)0, thrust::func<t1##_t>());         \
}

#define CUDA_DEVICE_NOIMPL(name, func, t0, t1, common)


/*****************************************************************************/
/*                                   Copy                                    */
/*****************************************************************************/

#define CUDA_DEVICE_ALL_UNARY_COPY(name, func, hfunc) \
    CUDA_DEVICE_UNARY(name, func, bool, bool, bool)                   \
    CUDA_DEVICE_UNARY(name, func, bool, uint8, uint8)                 \
    CUDA_DEVICE_UNARY(name, func, bool, uint16, uint16)               \
    CUDA_DEVICE_UNARY(name, func, bool, uint32, uint32)               \
    CUDA_DEVICE_UNARY(name, func, bool, uint64, uint64)               \
    CUDA_DEVICE_UNARY(name, func, bool, int8, int8)                   \
    CUDA_DEVICE_UNARY(name, func, bool, int16, int16)                 \
    CUDA_DEVICE_UNARY(name, func, bool, int32, int32)                 \
    CUDA_DEVICE_UNARY(name, func, bool, int64, int64)                 \
    CUDA_DEVICE_UNARY(name, func, bool, bfloat16, bfloat16)           \
    CUDA_DEVICE_UNARY(name, hfunc, bool, float16, float16)            \
    CUDA_DEVICE_UNARY(name, func, bool, float32, float32)             \
    CUDA_DEVICE_UNARY(name, func, bool, float64, float64)             \
    CUDA_DEVICE_NOIMPL(name, func, bool, complex32, complex32)        \
    CUDA_DEVICE_UNARY(name, func, bool, complex64, complex64)         \
    CUDA_DEVICE_UNARY(name, func, bool, complex128, complex128)       \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, uint8, uint8, uint8)                \
    CUDA_DEVICE_UNARY(name, func, uint8, uint16, uint16)              \
    CUDA_DEVICE_UNARY(name, func, uint8, uint32, uint32)              \
    CUDA_DEVICE_UNARY(name, func, uint8, uint64, uint64)              \
    CUDA_DEVICE_UNARY(name, func, uint8, int16, int16)                \
    CUDA_DEVICE_UNARY(name, func, uint8, int32, int32)                \
    CUDA_DEVICE_UNARY(name, func, uint8, int64, int64)                \
    CUDA_DEVICE_UNARY(name, func, uint8, bfloat16, bfloat16)          \
    CUDA_DEVICE_UNARY(name, hfunc, uint8, float16, float16)           \
    CUDA_DEVICE_UNARY(name, func, uint8, float32, float32)            \
    CUDA_DEVICE_UNARY(name, func, uint8, float64, float64)            \
    CUDA_DEVICE_NOIMPL(name, func, uint8, complex32, complex32)       \
    CUDA_DEVICE_UNARY(name, func, uint8, complex64, complex64)        \
    CUDA_DEVICE_UNARY(name, func, uint8, complex128, complex128)      \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, uint16, uint16, uint16)             \
    CUDA_DEVICE_UNARY(name, func, uint16, uint32, uint32)             \
    CUDA_DEVICE_UNARY(name, func, uint16, uint64, uint64)             \
    CUDA_DEVICE_UNARY(name, func, uint16, int32, int32)               \
    CUDA_DEVICE_UNARY(name, func, uint16, int64, int64)               \
    CUDA_DEVICE_UNARY(name, func, uint16, float32, float32)           \
    CUDA_DEVICE_UNARY(name, func, uint16, float64, float64)           \
    CUDA_DEVICE_UNARY(name, func, uint16, complex64, complex64)       \
    CUDA_DEVICE_UNARY(name, func, uint16, complex128, complex128)     \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, uint32, uint32, uint32)             \
    CUDA_DEVICE_UNARY(name, func, uint32, uint64, uint64)             \
    CUDA_DEVICE_UNARY(name, func, uint32, int64, int64)               \
    CUDA_DEVICE_UNARY(name, func, uint32, float64, float64)           \
    CUDA_DEVICE_UNARY(name, func, uint32, complex128, complex128)     \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, uint64, uint64, uint64)             \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, int8, int8, int8)                   \
    CUDA_DEVICE_UNARY(name, func, int8, int16, int16)                 \
    CUDA_DEVICE_UNARY(name, func, int8, int32, int32)                 \
    CUDA_DEVICE_UNARY(name, func, int8, int64, int64)                 \
    CUDA_DEVICE_UNARY(name, func, int8, bfloat16, bfloat16)           \
    CUDA_DEVICE_UNARY(name, hfunc, int8, float16, float16)            \
    CUDA_DEVICE_UNARY(name, func, int8, float32, float32)             \
    CUDA_DEVICE_UNARY(name, func, int8, float64, float64)             \
    CUDA_DEVICE_NOIMPL(name, func, int8, complex32, complex32)        \
    CUDA_DEVICE_UNARY(name, func, int8, complex64, complex64)         \
    CUDA_DEVICE_UNARY(name, func, int8, complex128, complex128)       \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, int16, int16, int16)                \
    CUDA_DEVICE_UNARY(name, func, int16, int32, int32)                \
    CUDA_DEVICE_UNARY(name, func, int16, int64, int64)                \
    CUDA_DEVICE_UNARY(name, func, int16, float32, float32)            \
    CUDA_DEVICE_UNARY(name, func, int16, float64, float64)            \
    CUDA_DEVICE_UNARY(name, func, int16, complex64, complex64)        \
    CUDA_DEVICE_UNARY(name, func, int16, complex128, complex128)      \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, int32, int32, int32)                \
    CUDA_DEVICE_UNARY(name, func, int32, int64, int64)                \
    CUDA_DEVICE_UNARY(name, func, int32, float64, float64)            \
    CUDA_DEVICE_UNARY(name, func, int32, complex128, complex128)      \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, int64, int64, int64)                \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, bfloat16, bfloat16, bfloat16)       \
    CUDA_DEVICE_UNARY(name, func, bfloat16, float32, float32)         \
    CUDA_DEVICE_UNARY(name, func, bfloat16, float64, float64)         \
    CUDA_DEVICE_UNARY(name, func, bfloat16, complex64, complex64)     \
    CUDA_DEVICE_UNARY(name, func, bfloat16, complex128, complex128)   \
                                                                      \
    CUDA_DEVICE_UNARY(name, hfunc, float16, float16, float16)         \
    CUDA_DEVICE_UNARY(name, func, float16, float32, float32)          \
    CUDA_DEVICE_UNARY(name, func, float16, float64, float64)          \
    CUDA_DEVICE_NOIMPL(name, func, float16, complex32, complex32)     \
    CUDA_DEVICE_UNARY(name, func, float16, complex64, complex64)      \
    CUDA_DEVICE_UNARY(name, func, float16, complex128, complex128)    \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, float32, float32, float32)          \
    CUDA_DEVICE_UNARY(name, func, float32, float64, float64)          \
    CUDA_DEVICE_UNARY(name, func, float32, complex64, complex64)      \
    CUDA_DEVICE_UNARY(name, func, float32, complex128, complex128)    \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, float64, float64, float64)          \
    CUDA_DEVICE_UNARY(name, func, float64, complex128, complex128)    \
                                                                      \
    CUDA_DEVICE_NOIMPL(name, func, complex32, complex32, complex32)   \
    CUDA_DEVICE_NOIMPL(name, func, complex32, complex64, complex64)   \
    CUDA_DEVICE_NOIMPL(name, func, complex32, complex128, complex128) \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, complex64, complex64, complex64)    \
    CUDA_DEVICE_UNARY(name, func, complex64, complex128, complex128)  \
                                                                      \
    CUDA_DEVICE_UNARY(name, func, complex128, complex128, complex128)


#define copy(x) x
CUDA_DEVICE_ALL_UNARY_COPY(copy, copy, copy)


/*****************************************************************************/
/*                                  Reduce                                   */
/*****************************************************************************/

#define CUDA_DEVICE_ALL_UNARY_REDUCE(name, func, hfunc) \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, bool)                   \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, uint8)                  \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, uint16)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, uint32)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, uint64)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, int8)                   \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, int16)                  \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, int32)                  \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, int64)                  \
    CUDA_DEVICE_NOIMPL(name, func, bool, bfloat16, bfloat16)           \
    CUDA_DEVICE_UNARY_REDUCE(name, hfunc, bool, float16)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, float32)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, bool, float64)                \
    CUDA_DEVICE_NOIMPL(name, func, bool, complex32, complex32)         \
    CUDA_DEVICE_NOIMPL(name, func, bool, complex64, complex64)         \
    CUDA_DEVICE_NOIMPL(name, func, bool, complex128, complex128)       \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, uint8)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, uint16)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, uint32)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, uint64)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, int16)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, int32)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, int64)                 \
    CUDA_DEVICE_NOIMPL(name, func, uint8, bfloat16, bfloat16)          \
    CUDA_DEVICE_UNARY_REDUCE(name, hfunc, uint8, float16)              \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, float32)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint8, float64)               \
    CUDA_DEVICE_NOIMPL(name, func, uint8, complex32, complex32)        \
    CUDA_DEVICE_NOIMPL(name, func, uint8, complex64, complex64)        \
    CUDA_DEVICE_NOIMPL(name, func, uint8, complex128, complex128)      \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, uint16)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, uint32)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, uint64)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, int32)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, int64)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, float32)              \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint16, float64)              \
    CUDA_DEVICE_NOIMPL(name, func, uint16, complex64, complex64)       \
    CUDA_DEVICE_NOIMPL(name, func, uint16, complex128, complex128)     \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint32, uint32)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint32, uint64)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint32, int64)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint32, float64)              \
    CUDA_DEVICE_NOIMPL(name, func, uint32, complex128, complex128)     \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, uint64, uint64)               \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int8, int8)                   \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int8, int16)                  \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int8, int32)                  \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int8, int64)                  \
    CUDA_DEVICE_NOIMPL(name, func, int8, bfloat16, bfloat16)           \
    CUDA_DEVICE_UNARY_REDUCE(name, hfunc, int8, float16)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int8, float32)                \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int8, float64)                \
    CUDA_DEVICE_NOIMPL(name, func, int8, complex32, complex32)         \
    CUDA_DEVICE_NOIMPL(name, func, int8, complex64, complex64)         \
    CUDA_DEVICE_NOIMPL(name, func, int8, complex128, complex128)       \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int16, int16)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int16, int32)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int16, int64)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int16, float32)               \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int16, float64)               \
    CUDA_DEVICE_NOIMPL(name, func, int16, complex64, complex64)        \
    CUDA_DEVICE_NOIMPL(name, func, int16, complex128, complex128)      \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int32, int32)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int32, int64)                 \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int32, float64)               \
    CUDA_DEVICE_NOIMPL(name, func, int32, complex128, complex128)      \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, int64, int64)                 \
                                                                       \
    CUDA_DEVICE_NOIMPL(name, func, bfloat16, bfloat16, bfloat16)       \
    CUDA_DEVICE_NOIMPL(name, func, bfloat16, float32, float32)         \
    CUDA_DEVICE_NOIMPL(name, func, bfloat16, float64, float64)         \
    CUDA_DEVICE_NOIMPL(name, func, bfloat16, complex64, complex64)     \
    CUDA_DEVICE_NOIMPL(name, func, bfloat16, complex128, complex128)   \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, hfunc, float16, float16)            \
    CUDA_DEVICE_NOIMPL(name, func, float16, float32, float32)          \
    CUDA_DEVICE_NOIMPL(name, func, float16, float64, float64)          \
    CUDA_DEVICE_NOIMPL(name, func, float16, complex32, complex32)      \
    CUDA_DEVICE_NOIMPL(name, func, float16, complex64, complex64)      \
    CUDA_DEVICE_NOIMPL(name, func, float16, complex128, complex128)    \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, float32, float32)             \
    CUDA_DEVICE_UNARY_REDUCE(name, func, float32, float64)             \
    CUDA_DEVICE_NOIMPL(name, func, float32, complex64, complex64)      \
    CUDA_DEVICE_NOIMPL(name, func, float32, complex128, complex128)    \
                                                                       \
    CUDA_DEVICE_UNARY_REDUCE(name, func, float64, float64)             \
    CUDA_DEVICE_NOIMPL(name, func, float64, complex128, complex128)    \
                                                                       \
    CUDA_DEVICE_NOIMPL(name, func, complex32, complex32, complex32)    \
    CUDA_DEVICE_NOIMPL(name, func, complex32, complex64, complex64)    \
    CUDA_DEVICE_NOIMPL(name, func, complex32, complex128, complex128)  \
                                                                       \
    CUDA_DEVICE_NOIMPL(name, func, complex64, complex64, complex64)    \
    CUDA_DEVICE_NOIMPL(name, func, complex64, complex128, complex128)  \
                                                                       \
    CUDA_DEVICE_NOIMPL(name, func, complex128, complex128, complex128)


CUDA_DEVICE_ALL_UNARY_REDUCE(add, plus, plus)
CUDA_DEVICE_ALL_UNARY_REDUCE(multiply, multiplies, multiplies)


/*****************************************************************************/
/*                               Bitwise NOT                                 */
/*****************************************************************************/

#define invert(x) !x
CUDA_DEVICE_UNARY(invert, invert, bool, bool, bool)
#undef invert

#define invert(x) ~x
CUDA_DEVICE_UNARY(invert, invert, uint8, uint8, uint8)
CUDA_DEVICE_UNARY(invert, invert, uint16, uint16, uint16)
CUDA_DEVICE_UNARY(invert, invert, uint32, uint32, uint32)
CUDA_DEVICE_UNARY(invert, invert, uint64, uint64, uint64)

CUDA_DEVICE_UNARY(invert, invert, int8, int8, int8)
CUDA_DEVICE_UNARY(invert, invert, int16, int16, int16)
CUDA_DEVICE_UNARY(invert, invert, int32, int32, int32)
CUDA_DEVICE_UNARY(invert, invert, int64, int64, int64)


/*****************************************************************************/
/*                                 Negative                                  */
/*****************************************************************************/

#define negative(x) -x
CUDA_DEVICE_UNARY(negative, negative, uint8, int16, int16)
CUDA_DEVICE_UNARY(negative, negative, uint16, int32, int32)
CUDA_DEVICE_UNARY(negative, negative, uint32, int64, int64)

CUDA_DEVICE_UNARY(negative, negative, int8, int8, int8)
CUDA_DEVICE_UNARY(negative, negative, int16, int16, int16)
CUDA_DEVICE_UNARY(negative, negative, int32, int32, int32)
CUDA_DEVICE_UNARY(negative, negative, int64, int64, int64)

CUDA_DEVICE_UNARY(negative, negative, bfloat16, bfloat16, bfloat16)
CUDA_DEVICE_UNARY(negative, __hneg, float16, float16, float16)
CUDA_DEVICE_UNARY(negative, negative, float32, float32, float32)
CUDA_DEVICE_UNARY(negative, negative, float64, float64, float64)

CUDA_DEVICE_NOIMPL(negative, negative, complex32, complex32, complex32)
CUDA_DEVICE_UNARY(negative, negative, complex64, complex64, complex64)
CUDA_DEVICE_UNARY(negative, negative, complex128, complex128, complex128)


/*****************************************************************************/
/*                                   Math                                    */
/*****************************************************************************/

#define CUDA_DEVICE_UNARY_ALL_REAL_MATH(name) \
    CUDA_DEVICE_UNARY(name##f, name##f, uint16, float32, float32)        \
    CUDA_DEVICE_UNARY(name##f, name##f, int16, float32, float32)         \
    CUDA_DEVICE_UNARY(name##b16, tf::name, bfloat16, bfloat16, bfloat16) \
    CUDA_DEVICE_UNARY(name##f, name##f, float32, float32, float32)       \
    CUDA_DEVICE_UNARY(name, name, uint32, float64, float64)              \
    CUDA_DEVICE_UNARY(name, name, int32, float64, float64)               \
    CUDA_DEVICE_UNARY(name, name, float64, float64, float64)

#define CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(name) \
    CUDA_DEVICE_UNARY_ALL_REAL_MATH(name)                             \
    CUDA_DEVICE_NOIMPL(name, name, complex32, complex32, complex32)   \
    CUDA_DEVICE_UNARY(name, name, complex64, complex64, complex64)    \
    CUDA_DEVICE_UNARY(name, name, complex128, complex128, complex128)

#define CUDA_DEVICE_UNARY_ALL_HALF_MATH(name, hfunc) \
    CUDA_DEVICE_UNARY(name##f16, hfunc, uint8, float16, float16)   \
    CUDA_DEVICE_UNARY(name##f16, hfunc, int8, float16, float16)    \
    CUDA_DEVICE_UNARY(name##f16, hfunc, float16, float16, float16)

#define CUDA_DEVICE_UNARY_ALL_REAL_MATH_WITH_HALF(name, hfunc) \
    CUDA_DEVICE_UNARY_ALL_HALF_MATH(name, hfunc)               \
    CUDA_DEVICE_UNARY_ALL_REAL_MATH(name)                      \

#define CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(name, hfunc) \
    CUDA_DEVICE_UNARY_ALL_HALF_MATH(name, hfunc)                  \
    CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(name)                      \


/*****************************************************************************/
/*                                Abs functions                              */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_REAL_MATH_WITH_HALF(fabs, half_abs)


/*****************************************************************************/
/*                             Exponential functions                         */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(exp, hexp)
CUDA_DEVICE_UNARY_ALL_REAL_MATH_WITH_HALF(exp2, hexp2)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(expm1)


/*****************************************************************************/
/*                              Logarithm functions                          */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(log, hlog)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(log10, hlog10)
CUDA_DEVICE_UNARY_ALL_REAL_MATH_WITH_HALF(log2, hlog2)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(log1p)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(logb)


/*****************************************************************************/
/*                              Power functions                              */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(sqrt, hsqrt)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(cbrt)


/*****************************************************************************/
/*                           Trigonometric functions                         */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(sin, hsin)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH_WITH_HALF(cos, hcos)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(tan)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(asin)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(acos)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(atan)


/*****************************************************************************/
/*                             Hyperbolic functions                          */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(sinh)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(cosh)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(tanh)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(asinh)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(acosh)
CUDA_DEVICE_UNARY_ALL_COMPLEX_MATH(atanh)


/*****************************************************************************/
/*                            Error and gamma functions                      */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_REAL_MATH(erf)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(erfc)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(lgamma)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(tgamma)


/*****************************************************************************/
/*                              Ceiling, floor, trunc                        */
/*****************************************************************************/

CUDA_DEVICE_UNARY_ALL_REAL_MATH(ceil)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(floor)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(trunc)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(round)
CUDA_DEVICE_UNARY_ALL_REAL_MATH(nearbyint)

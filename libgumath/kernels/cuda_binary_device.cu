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
#include "cuda_binary_device.h"


/*****************************************************************************/
/*                Lexicographic comparison for complex numbers               */
/*****************************************************************************/

template <class T, class U>
static inline __device__ bool
lexorder_le(T a, U b)
{
    return a.real() < b.real() || (a.real() == b.real() && a.imag() <= b.imag());
}

template <class T, class U>
static inline __device__ bool
lexorder_lt(T a, U b)
{
    return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
}

template <class T, class U>
static inline __device__ bool
lexorder_ge(T a, U b)
{
    return a.real() > b.real() || (a.real() == b.real() && a.imag() >= b.imag());
}

template <class T, class U>
static inline __device__ bool
lexorder_gt(T a, U b)
{
    return a.real() > b.real() || (a.real() == b.real() && a.imag() > b.imag());
}


/*****************************************************************************/
/*                         Cuda binary device kernels                        */
/*****************************************************************************/

#define CUDA_BINARY(name, func, t0, t1, t2, common) \
static __global__ void                                                       \
_##name##_##t0##_##t1##_##t2(                                                \
    const t0##_t* in0, const t1##_t* in1, t2##_t* out, int64_t N,            \
    enum cuda_binary tag)                                                    \
{                                                                            \
    int64_t index = threadIdx.x + blockIdx.x * blockDim.x;                   \
    int64_t stride = blockDim.x * gridDim.x;                                 \
                                                                             \
    switch (tag) {                                                           \
    case ZeroStepNone:                                                       \
        for (int64_t i = index; i < N; i += stride) {                        \
            out[i] = func((common##_t)in0[i], (common##_t)in1[i]);           \
        }                                                                    \
        break;                                                               \
    case ZeroStepIn0:                                                        \
        for (int64_t i = index; i < N; i += stride) {                        \
            out[i] = func((common##_t)in0[0], (common##_t)in1[i]);           \
        }                                                                    \
        break;                                                               \
    case ZeroStepIn1:                                                        \
        for (int64_t i = index; i < N; i += stride) {                        \
            out[i] = func((common##_t)in0[i], (common##_t)in1[0]);           \
        }                                                                    \
        break;                                                               \
    case ZeroStepIn0In1:                                                     \
        for (int64_t i = index; i < N; i += stride) {                        \
            out[i] = func((common##_t)in0[0], (common##_t)in1[0]);           \
        }                                                                    \
        break;                                                               \
    }                                                                        \
}                                                                            \
                                                                             \
extern "C" void                                                              \
gm_cuda_binary_device_fixed_##name##_1D_C_##t0##_##t1##_##t2(                \
    const char *in0, const char *in1, char *out, int64_t N,                  \
    enum cuda_binary tag)                                                    \
{                                                                            \
    const t0##_t *_in0 = (const t0##_t *)in0;                                \
    const t1##_t *_in1 = (const t1##_t *)in1;                                \
    t2##_t *_out = (t2##_t *)out;                                            \
    int blockSize = 256;                                                     \
    int64_t numBlocks = (N + blockSize - 1) / blockSize;                     \
                                                                             \
    _##name##_##t0##_##t1##_##t2<<<numBlocks, blockSize>>>(_in0, _in1, _out, \
                                                           N, tag);          \
}

#define CUDA_NOIMPL(name, func, t0, t1, t2, common)


/*****************************************************************************/
/*                                 Arithmetic                                */
/*****************************************************************************/

#define CUDA_ALL_BINARY(name, func, hfunc) \
    CUDA_BINARY(name, func, uint8, uint8, uint8, uint8)                     \
    CUDA_BINARY(name, func, uint8, uint16, uint16, uint16)                  \
    CUDA_BINARY(name, func, uint8, uint32, uint32, uint32)                  \
    CUDA_BINARY(name, func, uint8, uint64, uint64, uint64)                  \
    CUDA_BINARY(name, func, uint8, int8, int16, int16)                      \
    CUDA_BINARY(name, func, uint8, int16, int16, int16)                     \
    CUDA_BINARY(name, func, uint8, int32, int32, int32)                     \
    CUDA_BINARY(name, func, uint8, int64, int64, int64)                     \
    CUDA_BINARY(name, hfunc, uint8, float16, float16, float16)              \
    CUDA_BINARY(name, func, uint8, float32, float32, float32)               \
    CUDA_BINARY(name, func, uint8, float64, float64, float64)               \
    CUDA_NOIMPL(name, func, uint8, complex32, complex32, complex32)         \
    CUDA_BINARY(name, func, uint8, complex64, complex64, complex64)         \
    CUDA_BINARY(name, func, uint8, complex128, complex128, complex128)      \
                                                                            \
    CUDA_BINARY(name, func, uint16, uint8, uint16, uint16)                  \
    CUDA_BINARY(name, func, uint16, uint16, uint16, uint16)                 \
    CUDA_BINARY(name, func, uint16, uint32, uint32, uint32)                 \
    CUDA_BINARY(name, func, uint16, uint64, uint64, uint64)                 \
    CUDA_BINARY(name, func, uint16, int8, int32, int32)                     \
    CUDA_BINARY(name, func, uint16, int16, int32, int32)                    \
    CUDA_BINARY(name, func, uint16, int32, int32, int32)                    \
    CUDA_BINARY(name, func, uint16, int64, int64, int64)                    \
    CUDA_BINARY(name, func, uint16, float16, float32, float32)              \
    CUDA_BINARY(name, func, uint16, float32, float32, float32)              \
    CUDA_BINARY(name, func, uint16, float64, float64, float64)              \
    CUDA_NOIMPL(name, func, uint16, complex32, complex64, complex64)        \
    CUDA_BINARY(name, func, uint16, complex64, complex64, complex64)        \
    CUDA_BINARY(name, func, uint16, complex128, complex128, complex128)     \
                                                                            \
    CUDA_BINARY(name, func, uint32, uint8, uint32, uint32)                  \
    CUDA_BINARY(name, func, uint32, uint16, uint32, uint32)                 \
    CUDA_BINARY(name, func, uint32, uint32, uint32, uint32)                 \
    CUDA_BINARY(name, func, uint32, uint64, uint64, uint64)                 \
    CUDA_BINARY(name, func, uint32, int8, int64, int64)                     \
    CUDA_BINARY(name, func, uint32, int16, int64, int64)                    \
    CUDA_BINARY(name, func, uint32, int32, int64, int64)                    \
    CUDA_BINARY(name, func, uint32, int64, int64, int64)                    \
    CUDA_BINARY(name, func, uint32, float16, float64, float64)              \
    CUDA_BINARY(name, func, uint32, float32, float64, float64)              \
    CUDA_BINARY(name, func, uint32, float64, float64, float64)              \
    CUDA_NOIMPL(name, func, uint32, complex32, complex128, complex128)      \
    CUDA_BINARY(name, func, uint32, complex64, complex128, complex128)      \
    CUDA_BINARY(name, func, uint32, complex128, complex128, complex128)     \
                                                                            \
    CUDA_BINARY(name, func, uint64, uint8, uint64, uint64)                  \
    CUDA_BINARY(name, func, uint64, uint16, uint64, uint64)                 \
    CUDA_BINARY(name, func, uint64, uint32, uint64, uint64)                 \
    CUDA_BINARY(name, func, uint64, uint64, uint64, uint64)                 \
                                                                            \
    CUDA_BINARY(name, func, int8, uint8, int16, int16)                      \
    CUDA_BINARY(name, func, int8, uint16, int32, int32)                     \
    CUDA_BINARY(name, func, int8, uint32, int64, int64)                     \
    CUDA_BINARY(name, func, int8, int8, int8, int8)                         \
    CUDA_BINARY(name, func, int8, int16, int16, int16)                      \
    CUDA_BINARY(name, func, int8, int32, int32, int32)                      \
    CUDA_BINARY(name, func, int8, int64, int64, int64)                      \
    CUDA_BINARY(name, hfunc, int8, float16, float16, float16)               \
    CUDA_BINARY(name, func, int8, float32, float32, float32)                \
    CUDA_BINARY(name, func, int8, float64, float64, float64)                \
    CUDA_NOIMPL(name, func, int8, complex32, complex32, complex32)          \
    CUDA_BINARY(name, func, int8, complex64, complex64, complex64)          \
    CUDA_BINARY(name, func, int8, complex128, complex128, complex128)       \
                                                                            \
    CUDA_BINARY(name, func, int16, uint8, int16, int16)                     \
    CUDA_BINARY(name, func, int16, uint16, int32, int32)                    \
    CUDA_BINARY(name, func, int16, uint32, int64, int64)                    \
    CUDA_BINARY(name, func, int16, int8, int16, int16)                      \
    CUDA_BINARY(name, func, int16, int16, int16, int16)                     \
    CUDA_BINARY(name, func, int16, int32, int32, int32)                     \
    CUDA_BINARY(name, func, int16, int64, int64, int64)                     \
    CUDA_BINARY(name, func, int16, float16, float32, float32)               \
    CUDA_BINARY(name, func, int16, float32, float32, float32)               \
    CUDA_BINARY(name, func, int16, float64, float64, float64)               \
    CUDA_NOIMPL(name, func, int16, complex32, complex64, complex64)         \
    CUDA_BINARY(name, func, int16, complex64, complex64, complex64)         \
    CUDA_BINARY(name, func, int16, complex128, complex128, complex128)      \
                                                                            \
    CUDA_BINARY(name, func, int32, uint8, int32, int32)                     \
    CUDA_BINARY(name, func, int32, uint16, int32, int32)                    \
    CUDA_BINARY(name, func, int32, uint32, int64, int64)                    \
    CUDA_BINARY(name, func, int32, int8, int32, int32)                      \
    CUDA_BINARY(name, func, int32, int16, int32, int32)                     \
    CUDA_BINARY(name, func, int32, int32, int32, int32)                     \
    CUDA_BINARY(name, func, int32, int64, int64, int64)                     \
    CUDA_BINARY(name, func, int32, float16, float64, float64)               \
    CUDA_BINARY(name, func, int32, float32, float64, float64)               \
    CUDA_BINARY(name, func, int32, float64, float64, float64)               \
    CUDA_NOIMPL(name, func, int32, complex32, complex128, complex128)       \
    CUDA_BINARY(name, func, int32, complex64, complex128, complex128)       \
    CUDA_BINARY(name, func, int32, complex128, complex128, complex128)      \
                                                                            \
    CUDA_BINARY(name, func, int64, uint8, int64, int64)                     \
    CUDA_BINARY(name, func, int64, uint16, int64, int64)                    \
    CUDA_BINARY(name, func, int64, uint32, int64, int64)                    \
    CUDA_BINARY(name, func, int64, int8, int64, int64)                      \
    CUDA_BINARY(name, func, int64, int16, int64, int64)                     \
    CUDA_BINARY(name, func, int64, int32, int64, int64)                     \
    CUDA_BINARY(name, func, int64, int64, int64, int64)                     \
                                                                            \
    CUDA_BINARY(name, hfunc, float16, uint8, float16, float16)              \
    CUDA_BINARY(name, func, float16, uint16, float32, float32)              \
    CUDA_BINARY(name, func, float16, uint32, float64, float64)              \
    CUDA_BINARY(name, hfunc, float16, int8, float16, float16)               \
    CUDA_BINARY(name, func, float16, int16, float32, float32)               \
    CUDA_BINARY(name, func, float16, int32, float64, float64)               \
    CUDA_BINARY(name, hfunc, float16, float16, float16, float16)            \
    CUDA_BINARY(name, func, float16, float32, float32, float32)             \
    CUDA_BINARY(name, func, float16, float64, float64, float64)             \
    CUDA_NOIMPL(name, func, float16, complex32, complex32, complex32)       \
    CUDA_BINARY(name, func, float16, complex64, complex64, complex64)       \
    CUDA_BINARY(name, func, float16, complex128, complex128, complex128)    \
                                                                            \
    CUDA_BINARY(name, func, float32, uint8, float32, float32)               \
    CUDA_BINARY(name, func, float32, uint16, float32, float32)              \
    CUDA_BINARY(name, func, float32, uint32, float64, float64)              \
    CUDA_BINARY(name, func, float32, int8, float32, float32)                \
    CUDA_BINARY(name, func, float32, int16, float32, float32)               \
    CUDA_BINARY(name, func, float32, int32, float64, float64)               \
    CUDA_BINARY(name, func, float32, float16, float32, float32)             \
    CUDA_BINARY(name, func, float32, float32, float32, float32)             \
    CUDA_BINARY(name, func, float32, float64, float64, float64)             \
    CUDA_NOIMPL(name, func, float32, complex32, complex64, complex64)       \
    CUDA_BINARY(name, func, float32, complex64, complex64, complex64)       \
    CUDA_BINARY(name, func, float32, complex128, complex128, complex128)    \
                                                                            \
    CUDA_BINARY(name, func, float64, uint8, float64, float64)               \
    CUDA_BINARY(name, func, float64, uint16, float64, float64)              \
    CUDA_BINARY(name, func, float64, uint32, float64, float64)              \
    CUDA_BINARY(name, func, float64, int8, float64, float64)                \
    CUDA_BINARY(name, func, float64, int16, float64, float64)               \
    CUDA_BINARY(name, func, float64, int32, float64, float64)               \
    CUDA_BINARY(name, func, float64, float16, float64, float64)             \
    CUDA_BINARY(name, func, float64, float32, float64, float64)             \
    CUDA_BINARY(name, func, float64, float64, float64, float64)             \
    CUDA_NOIMPL(name, func, float64, complex32, complex128, complex128)     \
    CUDA_BINARY(name, func, float64, complex64, complex128, complex128)     \
    CUDA_BINARY(name, func, float64, complex128, complex128, complex128)    \
                                                                            \
    CUDA_NOIMPL(name, func, complex32, uint8, complex32, complex32)         \
    CUDA_NOIMPL(name, func, complex32, uint16, complex64, complex64)        \
    CUDA_NOIMPL(name, func, complex32, uint32, complex128, complex128)      \
    CUDA_NOIMPL(name, func, complex32, int8, complex32, complex32)          \
    CUDA_NOIMPL(name, func, complex32, int16, complex64, complex64)         \
    CUDA_NOIMPL(name, func, complex32, int32, complex128, complex128)       \
    CUDA_NOIMPL(name, func, complex32, float16, complex32, complex32)       \
    CUDA_NOIMPL(name, func, complex32, float32, complex64, complex64)       \
    CUDA_NOIMPL(name, func, complex32, float64, complex128, complex128)     \
    CUDA_NOIMPL(name, func, complex32, complex32, complex32, complex32)     \
    CUDA_NOIMPL(name, func, complex32, complex64, complex64, complex64)     \
    CUDA_NOIMPL(name, func, complex32, complex128, complex128, complex128)  \
                                                                            \
    CUDA_BINARY(name, func, complex64, uint8, complex64, complex64)         \
    CUDA_BINARY(name, func, complex64, uint16, complex64, complex64)        \
    CUDA_BINARY(name, func, complex64, uint32, complex128, complex128)      \
    CUDA_BINARY(name, func, complex64, int8, complex64, complex64)          \
    CUDA_BINARY(name, func, complex64, int16, complex64, complex64)         \
    CUDA_BINARY(name, func, complex64, int32, complex128, complex128)       \
    CUDA_BINARY(name, func, complex64, float16, complex64, complex64)       \
    CUDA_BINARY(name, func, complex64, float32, complex64, complex64)       \
    CUDA_BINARY(name, func, complex64, float64, complex128, complex128)     \
    CUDA_NOIMPL(name, func, complex64, complex32, complex64, complex64)     \
    CUDA_BINARY(name, func, complex64, complex64, complex64, complex64)     \
    CUDA_BINARY(name, func, complex64, complex128, complex128, complex128)  \
                                                                            \
    CUDA_BINARY(name, func, complex128, uint8, complex128, complex128)      \
    CUDA_BINARY(name, func, complex128, uint16, complex128, complex128)     \
    CUDA_BINARY(name, func, complex128, uint32, complex128, complex128)     \
    CUDA_BINARY(name, func, complex128, int8, complex128, complex128)       \
    CUDA_BINARY(name, func, complex128, int16, complex128, complex128)      \
    CUDA_BINARY(name, func, complex128, int32, complex128, complex128)      \
    CUDA_BINARY(name, func, complex128, float16, complex128, complex128)    \
    CUDA_BINARY(name, func, complex128, float32, complex128, complex128)    \
    CUDA_BINARY(name, func, complex128, float64, complex128, complex128)    \
    CUDA_NOIMPL(name, func, complex128, complex32, complex128, complex128)  \
    CUDA_BINARY(name, func, complex128, complex64, complex128, complex128)  \
    CUDA_BINARY(name, func, complex128, complex128, complex128, complex128) \


#define add(x, y) x + y
CUDA_ALL_BINARY(add, add, __hadd)

#define subtract(x, y) x - y
CUDA_ALL_BINARY(subtract, subtract, __hsub)

#define multiply(x, y) x * y
CUDA_ALL_BINARY(multiply, multiply, __hmul)

#define divide(x, y) x / y
CUDA_ALL_BINARY(divide, divide, __hdiv)


/*****************************************************************************/
/*                                 Comparison                                */
/*****************************************************************************/

#define CUDA_ALL_COMPARISON(name, func, hfunc, cfunc) \
    CUDA_BINARY(name, func, uint8, uint8, bool, uint8)                 \
    CUDA_BINARY(name, func, uint8, uint16, bool, uint16)               \
    CUDA_BINARY(name, func, uint8, uint32, bool, uint32)               \
    CUDA_BINARY(name, func, uint8, uint64, bool, uint64)               \
    CUDA_BINARY(name, func, uint8, int8, bool, int16)                  \
    CUDA_BINARY(name, func, uint8, int16, bool, int16)                 \
    CUDA_BINARY(name, func, uint8, int32, bool, int32)                 \
    CUDA_BINARY(name, func, uint8, int64, bool, int64)                 \
    CUDA_BINARY(name, func, uint8, float16, bool, float16)             \
    CUDA_BINARY(name, func, uint8, float32, bool, float32)             \
    CUDA_BINARY(name, func, uint8, float64, bool, float64)             \
    CUDA_NOIMPL(name, cfunc, uint8, complex32, bool, complex32)        \
    CUDA_BINARY(name, cfunc, uint8, complex64, bool, complex64)        \
    CUDA_BINARY(name, cfunc, uint8, complex128, bool, complex128)      \
                                                                       \
    CUDA_BINARY(name, func, uint16, uint8, bool, uint16)               \
    CUDA_BINARY(name, func, uint16, uint16, bool, uint16)              \
    CUDA_BINARY(name, func, uint16, uint32, bool, uint32)              \
    CUDA_BINARY(name, func, uint16, uint64, bool, uint64)              \
    CUDA_BINARY(name, func, uint16, int8, bool, int32)                 \
    CUDA_BINARY(name, func, uint16, int16, bool, int32)                \
    CUDA_BINARY(name, func, uint16, int32, bool, int32)                \
    CUDA_BINARY(name, func, uint16, int64, bool, int64)                \
    CUDA_BINARY(name, func, uint16, float16, bool, float32)            \
    CUDA_BINARY(name, func, uint16, float32, bool, float32)            \
    CUDA_BINARY(name, func, uint16, float64, bool, float64)            \
    CUDA_NOIMPL(name, cfunc, uint16, complex32, bool, complex64)       \
    CUDA_BINARY(name, cfunc, uint16, complex64, bool, complex64)       \
    CUDA_BINARY(name, cfunc, uint16, complex128, bool, complex128)     \
                                                                       \
    CUDA_BINARY(name, func, uint32, uint8, bool, uint32)               \
    CUDA_BINARY(name, func, uint32, uint16, bool, uint32)              \
    CUDA_BINARY(name, func, uint32, uint32, bool, uint32)              \
    CUDA_BINARY(name, func, uint32, uint64, bool, uint64)              \
    CUDA_BINARY(name, func, uint32, int8, bool, int64)                 \
    CUDA_BINARY(name, func, uint32, int16, bool, int64)                \
    CUDA_BINARY(name, func, uint32, int32, bool, int64)                \
    CUDA_BINARY(name, func, uint32, int64, bool, int64)                \
    CUDA_BINARY(name, func, uint32, float16, bool, float64)            \
    CUDA_BINARY(name, func, uint32, float32, bool, float64)            \
    CUDA_BINARY(name, func, uint32, float64, bool, float64)            \
    CUDA_NOIMPL(name, cfunc, uint32, complex32, bool, complex128)      \
    CUDA_BINARY(name, cfunc, uint32, complex64, bool, complex128)      \
    CUDA_BINARY(name, cfunc, uint32, complex128, bool, complex128)     \
                                                                       \
    CUDA_BINARY(name, func, uint64, uint8, bool, uint64)               \
    CUDA_BINARY(name, func, uint64, uint16, bool, uint64)              \
    CUDA_BINARY(name, func, uint64, uint32, bool, uint64)              \
    CUDA_BINARY(name, func, uint64, uint64, bool, uint64)              \
                                                                       \
    CUDA_BINARY(name, func, int8, uint8, bool, int16)                  \
    CUDA_BINARY(name, func, int8, uint16, bool, int32)                 \
    CUDA_BINARY(name, func, int8, uint32, bool, int64)                 \
    CUDA_BINARY(name, func, int8, int8, bool, int8)                    \
    CUDA_BINARY(name, func, int8, int16, bool, int16)                  \
    CUDA_BINARY(name, func, int8, int32, bool, int32)                  \
    CUDA_BINARY(name, func, int8, int64, bool, int64)                  \
    CUDA_BINARY(name, func, int8, float16, bool, float16)              \
    CUDA_BINARY(name, func, int8, float32, bool, float32)              \
    CUDA_BINARY(name, func, int8, float64, bool, float64)              \
    CUDA_NOIMPL(name, cfunc, int8, complex32, bool, complex32)         \
    CUDA_BINARY(name, cfunc, int8, complex64, bool, complex64)         \
    CUDA_BINARY(name, cfunc, int8, complex128, bool, complex128)       \
                                                                       \
    CUDA_BINARY(name, func, int16, uint8, bool, int16)                 \
    CUDA_BINARY(name, func, int16, uint16, bool, int32)                \
    CUDA_BINARY(name, func, int16, uint32, bool, int64)                \
    CUDA_BINARY(name, func, int16, int8, bool, int16)                  \
    CUDA_BINARY(name, func, int16, int16, bool, int16)                 \
    CUDA_BINARY(name, func, int16, int32, bool, int32)                 \
    CUDA_BINARY(name, func, int16, int64, bool, int64)                 \
    CUDA_BINARY(name, func, int16, float16, bool, float32)             \
    CUDA_BINARY(name, func, int16, float32, bool, float32)             \
    CUDA_BINARY(name, func, int16, float64, bool, float64)             \
    CUDA_NOIMPL(name, cfunc, int16, complex32, bool, complex64)        \
    CUDA_BINARY(name, cfunc, int16, complex64, bool, complex64)        \
    CUDA_BINARY(name, cfunc, int16, complex128, bool, complex128)      \
                                                                       \
    CUDA_BINARY(name, func, int32, uint8, bool, int32)                 \
    CUDA_BINARY(name, func, int32, uint16, bool, int32)                \
    CUDA_BINARY(name, func, int32, uint32, bool, int64)                \
    CUDA_BINARY(name, func, int32, int8, bool, int32)                  \
    CUDA_BINARY(name, func, int32, int16, bool, int32)                 \
    CUDA_BINARY(name, func, int32, int32, bool, int32)                 \
    CUDA_BINARY(name, func, int32, int64, bool, int64)                 \
    CUDA_BINARY(name, func, int32, float16, bool, float64)             \
    CUDA_BINARY(name, func, int32, float32, bool, float64)             \
    CUDA_BINARY(name, func, int32, float64, bool, float64)             \
    CUDA_NOIMPL(name, cfunc, int32, complex32, bool, complex128)       \
    CUDA_BINARY(name, cfunc, int32, complex64, bool, complex128)       \
    CUDA_BINARY(name, cfunc, int32, complex128, bool, complex128)      \
                                                                       \
    CUDA_BINARY(name, func, int64, uint8, bool, int64)                 \
    CUDA_BINARY(name, func, int64, uint16, bool, int64)                \
    CUDA_BINARY(name, func, int64, uint32, bool, int64)                \
    CUDA_BINARY(name, func, int64, int8, bool, int64)                  \
    CUDA_BINARY(name, func, int64, int16, bool, int64)                 \
    CUDA_BINARY(name, func, int64, int32, bool, int64)                 \
    CUDA_BINARY(name, func, int64, int64, bool, int64)                 \
                                                                       \
    CUDA_BINARY(name, func, float16, uint8, bool, float16)             \
    CUDA_BINARY(name, func, float16, uint16, bool, float32)            \
    CUDA_BINARY(name, func, float16, uint32, bool, float64)            \
    CUDA_BINARY(name, func, float16, int8, bool, float16)              \
    CUDA_BINARY(name, func, float16, int16, bool, float32)             \
    CUDA_BINARY(name, func, float16, int32, bool, float64)             \
    CUDA_BINARY(name, func, float16, float16, bool, float16)           \
    CUDA_BINARY(name, func, float16, float32, bool, float32)           \
    CUDA_BINARY(name, func, float16, float64, bool, float64)           \
    CUDA_NOIMPL(name, cfunc, float16, complex32, bool, complex32)      \
    CUDA_BINARY(name, cfunc, float16, complex64, bool, complex64)      \
    CUDA_BINARY(name, cfunc, float16, complex128, bool, complex128)    \
                                                                       \
    CUDA_BINARY(name, func, float32, uint8, bool, float32)             \
    CUDA_BINARY(name, func, float32, uint16, bool, float32)            \
    CUDA_BINARY(name, func, float32, uint32, bool, float64)            \
    CUDA_BINARY(name, func, float32, int8, bool, float32)              \
    CUDA_BINARY(name, func, float32, int16, bool, float32)             \
    CUDA_BINARY(name, func, float32, int32, bool, float64)             \
    CUDA_BINARY(name, func, float32, float16, bool, float32)           \
    CUDA_BINARY(name, func, float32, float32, bool, float32)           \
    CUDA_BINARY(name, func, float32, float64, bool, float64)           \
    CUDA_NOIMPL(name, cfunc, float32, complex32, bool, complex64)      \
    CUDA_BINARY(name, cfunc, float32, complex64, bool, complex64)      \
    CUDA_BINARY(name, cfunc, float32, complex128, bool, complex128)    \
                                                                       \
    CUDA_BINARY(name, func, float64, uint8, bool, float64)             \
    CUDA_BINARY(name, func, float64, uint16, bool, float64)            \
    CUDA_BINARY(name, func, float64, uint32, bool, float64)            \
    CUDA_BINARY(name, func, float64, int8, bool, float64)              \
    CUDA_BINARY(name, func, float64, int16, bool, float64)             \
    CUDA_BINARY(name, func, float64, int32, bool, float64)             \
    CUDA_BINARY(name, func, float64, float16, bool, float64)           \
    CUDA_BINARY(name, func, float64, float32, bool, float64)           \
    CUDA_BINARY(name, func, float64, float64, bool, float64)           \
    CUDA_NOIMPL(name, cfunc, float64, complex32, bool, complex128)     \
    CUDA_BINARY(name, cfunc, float64, complex64, bool, complex128)     \
    CUDA_BINARY(name, cfunc, float64, complex128, bool, complex128)    \
                                                                       \
    CUDA_NOIMPL(name, cfunc, complex32, uint8, bool, complex32)        \
    CUDA_NOIMPL(name, cfunc, complex32, uint16, bool, complex64)       \
    CUDA_NOIMPL(name, cfunc, complex32, uint32, bool, complex128)      \
    CUDA_NOIMPL(name, cfunc, complex32, int8, bool, complex32)         \
    CUDA_NOIMPL(name, cfunc, complex32, int16, bool, complex64)        \
    CUDA_NOIMPL(name, cfunc, complex32, int32, bool, complex128)       \
    CUDA_NOIMPL(name, cfunc, complex32, float16, bool, complex32)      \
    CUDA_NOIMPL(name, cfunc, complex32, float32, bool, complex64)      \
    CUDA_NOIMPL(name, cfunc, complex32, float64, bool, complex128)     \
    CUDA_NOIMPL(name, cfunc, complex32, complex32, bool, complex32)    \
    CUDA_NOIMPL(name, cfunc, complex32, complex64, bool, complex64)    \
    CUDA_NOIMPL(name, cfunc, complex32, complex128, bool, complex128)  \
                                                                       \
    CUDA_BINARY(name, cfunc, complex64, uint8, bool, complex64)        \
    CUDA_BINARY(name, cfunc, complex64, uint16, bool, complex64)       \
    CUDA_BINARY(name, cfunc, complex64, uint32, bool, complex128)      \
    CUDA_BINARY(name, cfunc, complex64, int8, bool, complex64)         \
    CUDA_BINARY(name, cfunc, complex64, int16, bool, complex64)        \
    CUDA_BINARY(name, cfunc, complex64, int32, bool, complex128)       \
    CUDA_BINARY(name, cfunc, complex64, float16, bool, complex64)      \
    CUDA_BINARY(name, cfunc, complex64, float32, bool, complex64)      \
    CUDA_BINARY(name, cfunc, complex64, float64, bool, complex128)     \
    CUDA_NOIMPL(name, cfunc, complex64, complex32, bool, complex64)    \
    CUDA_BINARY(name, cfunc, complex64, complex64, bool, complex64)    \
    CUDA_BINARY(name, cfunc, complex64, complex128, bool, complex128)  \
                                                                       \
    CUDA_BINARY(name, cfunc, complex128, uint8, bool, complex128)      \
    CUDA_BINARY(name, cfunc, complex128, uint16, bool, complex128)     \
    CUDA_BINARY(name, cfunc, complex128, uint32, bool, complex128)     \
    CUDA_BINARY(name, cfunc, complex128, int8, bool, complex128)       \
    CUDA_BINARY(name, cfunc, complex128, int16, bool, complex128)      \
    CUDA_BINARY(name, cfunc, complex128, int32, bool, complex128)      \
    CUDA_BINARY(name, cfunc, complex128, float16, bool, complex128)    \
    CUDA_BINARY(name, cfunc, complex128, float32, bool, complex128)    \
    CUDA_BINARY(name, cfunc, complex128, float64, bool, complex128)    \
    CUDA_NOIMPL(name, cfunc, complex128, complex32, bool, complex128)  \
    CUDA_BINARY(name, cfunc, complex128, complex64, bool, complex128)  \
    CUDA_BINARY(name, cfunc, complex128, complex128, bool, complex128)


#define less(x, y) x < y
CUDA_ALL_COMPARISON(less, less, __hlt, lexorder_lt)

#define less_equal(x, y) x <= y
CUDA_ALL_COMPARISON(less_equal, less_equal, __hle, lexorder_le)

#define greater_equal(x, y) x >= y
CUDA_ALL_COMPARISON(greater_equal, greater_equal, __hge, lexorder_ge)

#define greater(x, y) x > y
CUDA_ALL_COMPARISON(greater, greater, __hgt, lexorder_gt)

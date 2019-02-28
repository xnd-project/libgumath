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


#ifndef CPU_DEVICE_BINARY_H
#define CPU_DEVICE_BINARY_H


#ifdef __cplusplus
#include <cinttypes>
#include <complex>
#include "contrib/bfloat16.h"
 
typedef tf::bfloat16 bfloat16_t;
typedef std::complex<float> complex64_t;
typedef std::complex<double> complex128_t;
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
  #define CPU_DEVICE_BINARY_DECL(name, t0, t1, t2) \
  extern "C" void gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2( \
                 const char *a0, const char *a1, char *a2,              \
                 const int64_t N);                                      \
  extern "C" void gm_cpu_device_fixed_1D_S_##name##_##t0##_##t1##_##t2( \
                 const char *a0, const char *a1, char *a2,              \
                 const int64_t s0, const int64_t s1, const int64_t s2,  \
                 const int64_t N);                                      \
  extern "C" void gm_cpu_device_0D_##name##_##t0##_##t1##_##t2(         \
                 const char *a0, const char *a1, char *a2);

  #define CPU_DEVICE_BINARY_MV_DECL(name, t0, t1, t2, t3) \
  extern "C" void gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2##_##t3( \
                 const char *a0, const char *a1, char *a2, char *a3,           \
                 const int64_t N);                                             \
  extern "C" void gm_cpu_device_0D_##name##_##t0##_##t1##_##t2##_##t3(         \
                 const char *a0, const char *a1, char *a2, char *a3);
#else
  #define CPU_DEVICE_BINARY_DECL(name, t0, t1, t2) \
  void gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2( \
      const char *a0, const char *a1, char *a2,              \
      const int64_t N);                                      \
  void gm_cpu_device_fixed_1D_S_##name##_##t0##_##t1##_##t2( \
      const char *a0, const char *a1, char *a2,              \
      const int64_t s0, const int64_t s1, const int64_t s2,  \
      const int64_t N);                                      \
  void gm_cpu_device_0D_##name##_##t0##_##t1##_##t2(         \
      const char *a0, const char *a1, char *a2);

  #define CPU_DEVICE_BINARY_MV_DECL(name, t0, t1, t2, t3) \
  void gm_cpu_device_fixed_1D_C_##name##_##t0##_##t1##_##t2##_##t3( \
      const char *a0, const char *a1, char *a2, char *a3,           \
      const int64_t N);                                             \
  void gm_cpu_device_0D_##name##_##t0##_##t1##_##t2##_##t3(         \
      const char *a0, const char *a1, char *a2, char *a3);
#endif

#define CPU_DEVICE_NOKERN_DECL(name, t0, t1, t2)
#define CPU_DEVICE_BINARY_NOIMPL_DECL(name, t0, t1, t2)


/*****************************************************************************/
/*                                 Arithmetic                                */
/*****************************************************************************/

#define CPU_DEVICE_BINARY_ARITHMETIC_DECL(name) \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint8, uint8)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint16, uint16)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint32, uint32)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint64, uint64)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, int8, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, uint8, int16, int16)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, int32, int32)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, bfloat16, bfloat16)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, float16, float16)            \
    CPU_DEVICE_BINARY_DECL(name, uint8, float32, float32)            \
    CPU_DEVICE_BINARY_DECL(name, uint8, float64, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, complex32, complex32)        \
    CPU_DEVICE_BINARY_DECL(name, uint8, complex64, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, uint8, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint8, uint16)              \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint16, uint16)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint32, uint32)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint64, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, int8, int32)                \
    CPU_DEVICE_BINARY_DECL(name, uint16, int16, int32)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, int32, int32)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, int64, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, bfloat16, float32)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, float16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, uint16, float32, float32)           \
    CPU_DEVICE_BINARY_DECL(name, uint16, float64, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, complex32, complex64)       \
    CPU_DEVICE_BINARY_DECL(name, uint16, complex64, complex64)       \
    CPU_DEVICE_BINARY_DECL(name, uint16, complex128, complex128)     \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint8, uint32)              \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint16, uint32)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint32, uint32)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint64, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, int8, int64)                \
    CPU_DEVICE_BINARY_DECL(name, uint32, int16, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, int32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, int64, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, bfloat16, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, float16, float64)           \
    CPU_DEVICE_BINARY_DECL(name, uint32, float32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, uint32, float64, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, complex32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, uint32, complex64, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, uint32, complex128, complex128)     \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint8, uint64)              \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint16, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint32, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint64, uint64)             \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int8, uint8, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, uint16, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int8, uint32, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int8, int8, int8)                   \
    CPU_DEVICE_BINARY_DECL(name, int8, int16, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, int32, int32)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, int64, int64)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, bfloat16, bfloat16)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, float16, float16)             \
    CPU_DEVICE_BINARY_DECL(name, int8, float32, float32)             \
    CPU_DEVICE_BINARY_DECL(name, int8, float64, float64)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, complex32, complex32)         \
    CPU_DEVICE_BINARY_DECL(name, int8, complex64, complex64)         \
    CPU_DEVICE_BINARY_DECL(name, int8, complex128, complex128)       \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int16, uint8, int16)                \
    CPU_DEVICE_BINARY_DECL(name, int16, uint16, int32)               \
    CPU_DEVICE_BINARY_DECL(name, int16, uint32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int16, int8, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, int16, int16, int16)                \
    CPU_DEVICE_BINARY_DECL(name, int16, int32, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int16, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int16, bfloat16, float32)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, float16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, int16, float32, float32)            \
    CPU_DEVICE_BINARY_DECL(name, int16, float64, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, complex32, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, int16, complex64, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, int16, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int32, uint8, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int32, uint16, int32)               \
    CPU_DEVICE_BINARY_DECL(name, int32, uint32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int32, int8, int32)                 \
    CPU_DEVICE_BINARY_DECL(name, int32, int16, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int32, int32, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int32, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int32, bfloat16, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, float16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, int32, float32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, int32, float64, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, complex32, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, int32, complex64, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, int32, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int64, uint8, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int64, uint16, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int64, uint32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int64, int8, int64)                 \
    CPU_DEVICE_BINARY_DECL(name, int64, int16, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int64, int32, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int64, int64, int64)                \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint8, bfloat16)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint16, float32)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint32, float64)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int8, bfloat16)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, bfloat16, bfloat16)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, bfloat16, float16, float32)         \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float32, float32)         \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float64, float64)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, bfloat16, complex32, complex64)     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, complex64, complex64)     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, complex128, complex128)   \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint8, float16)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint16, float32)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint32, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int8, float16)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int16, float32)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int32, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, bfloat16, float32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float16, float16)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float32, float32)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float64, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex32, complex32)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex64, complex64)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float32, uint8, float32)            \
    CPU_DEVICE_BINARY_DECL(name, float32, uint16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, float32, uint32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float32, int8, float32)             \
    CPU_DEVICE_BINARY_DECL(name, float32, int16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, float32, int32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float32, bfloat16, float32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, float16, float32)          \
    CPU_DEVICE_BINARY_DECL(name, float32, float32, float32)          \
    CPU_DEVICE_BINARY_DECL(name, float32, float64, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, complex32, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, float32, complex64, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, float32, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float64, uint8, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, uint16, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float64, uint32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float64, int8, float64)             \
    CPU_DEVICE_BINARY_DECL(name, float64, int16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, int32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, bfloat16, float64)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, float16, float64)          \
    CPU_DEVICE_BINARY_DECL(name, float64, float32, float64)          \
    CPU_DEVICE_BINARY_DECL(name, float64, float64, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, complex32, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, float64, complex64, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, float64, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint8, complex32)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint16, complex64)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint32, complex128)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int8, complex32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int16, complex64)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int32, complex128)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, bfloat16, complex64)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float16, complex32)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float32, complex64)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float64, complex128)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex32, complex32)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex64, complex64)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex128, complex128)  \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint8, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint16, complex64)       \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex64, int8, complex64)         \
    CPU_DEVICE_BINARY_DECL(name, complex64, int16, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, complex64, int32, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, complex64, bfloat16, complex64)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex64, float16, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, complex64, float32, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, complex64, float64, complex128)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex64, complex32, complex64)    \
    CPU_DEVICE_BINARY_DECL(name, complex64, complex64, complex64)    \
    CPU_DEVICE_BINARY_DECL(name, complex64, complex128, complex128)  \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint8, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint16, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint32, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, complex128, int8, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, complex128, int16, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, int32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, bfloat16, complex128)   \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex128, float16, complex128)    \
    CPU_DEVICE_BINARY_DECL(name, complex128, float32, complex128)    \
    CPU_DEVICE_BINARY_DECL(name, complex128, float64, complex128)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex128, complex32, complex128)  \
    CPU_DEVICE_BINARY_DECL(name, complex128, complex64, complex128)  \
    CPU_DEVICE_BINARY_DECL(name, complex128, complex128, complex128)

#define CPU_DEVICE_BINARY_ARITHMETIC_NO_COMPLEX_DECL(name) \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint8, uint8)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint16, uint16)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint32, uint32)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint64, uint64)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, int8, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, uint8, int16, int16)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, int32, int32)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, bfloat16, bfloat16)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, float16, float16)            \
    CPU_DEVICE_BINARY_DECL(name, uint8, float32, float32)            \
    CPU_DEVICE_BINARY_DECL(name, uint8, float64, float64)            \
    CPU_DEVICE_NOKERN_DECL(name, uint8, complex32, complex32)        \
    CPU_DEVICE_NOKERN_DECL(name, uint8, complex64, complex64)        \
    CPU_DEVICE_NOKERN_DECL(name, uint8, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint8, uint16)              \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint16, uint16)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint32, uint32)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint64, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, int8, int32)                \
    CPU_DEVICE_BINARY_DECL(name, uint16, int16, int32)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, int32, int32)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, int64, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, bfloat16, float32)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, float16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, uint16, float32, float32)           \
    CPU_DEVICE_BINARY_DECL(name, uint16, float64, float64)           \
    CPU_DEVICE_NOKERN_DECL(name, uint16, complex32, complex64)       \
    CPU_DEVICE_NOKERN_DECL(name, uint16, complex64, complex64)       \
    CPU_DEVICE_NOKERN_DECL(name, uint16, complex128, complex128)     \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint8, uint32)              \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint16, uint32)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint32, uint32)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint64, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, int8, int64)                \
    CPU_DEVICE_BINARY_DECL(name, uint32, int16, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, int32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, int64, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, bfloat16, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, float16, float64)           \
    CPU_DEVICE_BINARY_DECL(name, uint32, float32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, uint32, float64, float64)           \
    CPU_DEVICE_NOKERN_DECL(name, uint32, complex32, complex128)      \
    CPU_DEVICE_NOKERN_DECL(name, uint32, complex64, complex128)      \
    CPU_DEVICE_NOKERN_DECL(name, uint32, complex128, complex128)     \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint8, uint64)              \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint16, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint32, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint64, uint64)             \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int8, uint8, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, uint16, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int8, uint32, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int8, int8, int8)                   \
    CPU_DEVICE_BINARY_DECL(name, int8, int16, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, int32, int32)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, int64, int64)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, bfloat16, bfloat16)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, float16, float16)             \
    CPU_DEVICE_BINARY_DECL(name, int8, float32, float32)             \
    CPU_DEVICE_BINARY_DECL(name, int8, float64, float64)             \
    CPU_DEVICE_NOKERN_DECL(name, int8, complex32, complex32)         \
    CPU_DEVICE_NOKERN_DECL(name, int8, complex64, complex64)         \
    CPU_DEVICE_NOKERN_DECL(name, int8, complex128, complex128)       \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int16, uint8, int16)                \
    CPU_DEVICE_BINARY_DECL(name, int16, uint16, int32)               \
    CPU_DEVICE_BINARY_DECL(name, int16, uint32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int16, int8, int16)                 \
    CPU_DEVICE_BINARY_DECL(name, int16, int16, int16)                \
    CPU_DEVICE_BINARY_DECL(name, int16, int32, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int16, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int16, bfloat16, float32)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, float16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, int16, float32, float32)            \
    CPU_DEVICE_BINARY_DECL(name, int16, float64, float64)            \
    CPU_DEVICE_NOKERN_DECL(name, int16, complex32, complex64)        \
    CPU_DEVICE_NOKERN_DECL(name, int16, complex64, complex64)        \
    CPU_DEVICE_NOKERN_DECL(name, int16, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int32, uint8, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int32, uint16, int32)               \
    CPU_DEVICE_BINARY_DECL(name, int32, uint32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int32, int8, int32)                 \
    CPU_DEVICE_BINARY_DECL(name, int32, int16, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int32, int32, int32)                \
    CPU_DEVICE_BINARY_DECL(name, int32, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int32, bfloat16, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, float16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, int32, float32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, int32, float64, float64)            \
    CPU_DEVICE_NOKERN_DECL(name, int32, complex32, complex128)       \
    CPU_DEVICE_NOKERN_DECL(name, int32, complex64, complex128)       \
    CPU_DEVICE_NOKERN_DECL(name, int32, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int64, uint8, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int64, uint16, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int64, uint32, int64)               \
    CPU_DEVICE_BINARY_DECL(name, int64, int8, int64)                 \
    CPU_DEVICE_BINARY_DECL(name, int64, int16, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int64, int32, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int64, int64, int64)                \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint8, bfloat16)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint16, float32)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint32, float64)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int8, bfloat16)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, bfloat16, bfloat16)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, bfloat16, float16, float32)         \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float32, float32)         \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float64, float64)         \
    CPU_DEVICE_NOKERN_DECL(name, bfloat16, complex32, complex64)     \
    CPU_DEVICE_NOKERN_DECL(name, bfloat16, complex64, complex64)     \
    CPU_DEVICE_NOKERN_DECL(name, bfloat16, complex128, complex128)   \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint8, float16)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint16, float32)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint32, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int8, float16)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int16, float32)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int32, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, bfloat16, float32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float16, float16)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float32, float32)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float64, float64)          \
    CPU_DEVICE_NOKERN_DECL(name, float16, complex32, complex32)      \
    CPU_DEVICE_NOKERN_DECL(name, float16, complex64, complex64)      \
    CPU_DEVICE_NOKERN_DECL(name, float16, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float32, uint8, float32)            \
    CPU_DEVICE_BINARY_DECL(name, float32, uint16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, float32, uint32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float32, int8, float32)             \
    CPU_DEVICE_BINARY_DECL(name, float32, int16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, float32, int32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float32, bfloat16, float32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, float16, float32)          \
    CPU_DEVICE_BINARY_DECL(name, float32, float32, float32)          \
    CPU_DEVICE_BINARY_DECL(name, float32, float64, float64)          \
    CPU_DEVICE_NOKERN_DECL(name, float32, complex32, complex64)      \
    CPU_DEVICE_NOKERN_DECL(name, float32, complex64, complex64)      \
    CPU_DEVICE_NOKERN_DECL(name, float32, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float64, uint8, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, uint16, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float64, uint32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float64, int8, float64)             \
    CPU_DEVICE_BINARY_DECL(name, float64, int16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, int32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, bfloat16, float64)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, float16, float64)          \
    CPU_DEVICE_BINARY_DECL(name, float64, float32, float64)          \
    CPU_DEVICE_BINARY_DECL(name, float64, float64, float64)          \
    CPU_DEVICE_NOKERN_DECL(name, float64, complex32, complex128)     \
    CPU_DEVICE_NOKERN_DECL(name, float64, complex64, complex128)     \
    CPU_DEVICE_NOKERN_DECL(name, float64, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_NOKERN_DECL(name, complex32, uint8, complex32)        \
    CPU_DEVICE_NOKERN_DECL(name, complex32, uint16, complex64)       \
    CPU_DEVICE_NOKERN_DECL(name, complex32, uint32, complex128)      \
    CPU_DEVICE_NOKERN_DECL(name, complex32, int8, complex32)         \
    CPU_DEVICE_NOKERN_DECL(name, complex32, int16, complex64)        \
    CPU_DEVICE_NOKERN_DECL(name, complex32, int32, complex128)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, bfloat16, complex64)     \
    CPU_DEVICE_NOKERN_DECL(name, complex32, float16, complex32)      \
    CPU_DEVICE_NOKERN_DECL(name, complex32, float32, complex64)      \
    CPU_DEVICE_NOKERN_DECL(name, complex32, float64, complex128)     \
    CPU_DEVICE_NOKERN_DECL(name, complex32, complex32, complex32)    \
    CPU_DEVICE_NOKERN_DECL(name, complex32, complex64, complex64)    \
    CPU_DEVICE_NOKERN_DECL(name, complex32, complex128, complex128)  \
                                                                     \
    CPU_DEVICE_NOKERN_DECL(name, complex64, uint8, complex64)        \
    CPU_DEVICE_NOKERN_DECL(name, complex64, uint16, complex64)       \
    CPU_DEVICE_NOKERN_DECL(name, complex64, uint32, complex128)      \
    CPU_DEVICE_NOKERN_DECL(name, complex64, int8, complex64)         \
    CPU_DEVICE_NOKERN_DECL(name, complex64, int16, complex64)        \
    CPU_DEVICE_NOKERN_DECL(name, complex64, int32, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, complex64, bfloat16, complex64)     \
    CPU_DEVICE_NOKERN_DECL(name, complex64, float16, complex64)      \
    CPU_DEVICE_NOKERN_DECL(name, complex64, float32, complex64)      \
    CPU_DEVICE_NOKERN_DECL(name, complex64, float64, complex128)     \
    CPU_DEVICE_NOKERN_DECL(name, complex64, complex32, complex64)    \
    CPU_DEVICE_NOKERN_DECL(name, complex64, complex64, complex64)    \
    CPU_DEVICE_NOKERN_DECL(name, complex64, complex128, complex128)  \
                                                                     \
    CPU_DEVICE_NOKERN_DECL(name, complex128, uint8, complex128)      \
    CPU_DEVICE_NOKERN_DECL(name, complex128, uint16, complex128)     \
    CPU_DEVICE_NOKERN_DECL(name, complex128, uint32, complex128)     \
    CPU_DEVICE_NOKERN_DECL(name, complex128, int8, complex128)       \
    CPU_DEVICE_NOKERN_DECL(name, complex128, int16, complex128)      \
    CPU_DEVICE_NOKERN_DECL(name, complex128, int32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, bfloat16, complex128)   \
    CPU_DEVICE_NOKERN_DECL(name, complex128, float16, complex128)    \
    CPU_DEVICE_NOKERN_DECL(name, complex128, float32, complex128)    \
    CPU_DEVICE_NOKERN_DECL(name, complex128, float64, complex128)    \
    CPU_DEVICE_NOKERN_DECL(name, complex128, complex32, complex128)  \
    CPU_DEVICE_NOKERN_DECL(name, complex128, complex64, complex128)  \
    CPU_DEVICE_NOKERN_DECL(name, complex128, complex128, complex128)

#define CPU_DEVICE_BINARY_ARITHMETIC_FLOAT_RETURN_DECL(name) \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, uint8, float16)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint16, float32)             \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint32, float64)             \
    CPU_DEVICE_NOKERN_DECL(name, uint8, uint64, uint64)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, int8, float16)               \
    CPU_DEVICE_BINARY_DECL(name, uint8, int16, float32)              \
    CPU_DEVICE_BINARY_DECL(name, uint8, int32, float64)              \
    CPU_DEVICE_NOKERN_DECL(name, uint8, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, bfloat16, bfloat16)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, float16, float16)            \
    CPU_DEVICE_BINARY_DECL(name, uint8, float32, float32)            \
    CPU_DEVICE_BINARY_DECL(name, uint8, float64, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, complex32, complex32)        \
    CPU_DEVICE_BINARY_DECL(name, uint8, complex64, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, uint8, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint8, float32)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint32, float64)            \
    CPU_DEVICE_NOKERN_DECL(name, uint16, uint64, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, int8, float32)              \
    CPU_DEVICE_BINARY_DECL(name, uint16, int16, float32)             \
    CPU_DEVICE_BINARY_DECL(name, uint16, int32, float64)             \
    CPU_DEVICE_NOKERN_DECL(name, uint16, int64, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, bfloat16, float32)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, float16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, uint16, float32, float32)           \
    CPU_DEVICE_BINARY_DECL(name, uint16, float64, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, complex32, complex64)       \
    CPU_DEVICE_BINARY_DECL(name, uint16, complex64, complex64)       \
    CPU_DEVICE_BINARY_DECL(name, uint16, complex128, complex128)     \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint8, float64)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint32, float64)            \
    CPU_DEVICE_NOKERN_DECL(name, uint32, uint64, uint64)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, int8, float64)              \
    CPU_DEVICE_BINARY_DECL(name, uint32, int16, float64)             \
    CPU_DEVICE_BINARY_DECL(name, uint32, int32, float64)             \
    CPU_DEVICE_NOKERN_DECL(name, uint32, int64, int64)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, bfloat16, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, float16, float64)           \
    CPU_DEVICE_BINARY_DECL(name, uint32, float32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, uint32, float64, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, complex32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, uint32, complex64, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, uint32, complex128, complex128)     \
                                                                     \
    CPU_DEVICE_NOKERN_DECL(name, uint64, uint8, uint64)              \
    CPU_DEVICE_NOKERN_DECL(name, uint64, uint16, uint64)             \
    CPU_DEVICE_NOKERN_DECL(name, uint64, uint32, uint64)             \
    CPU_DEVICE_NOKERN_DECL(name, uint64, uint64, uint64)             \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, uint8, float16)               \
    CPU_DEVICE_BINARY_DECL(name, int8, uint16, float32)              \
    CPU_DEVICE_BINARY_DECL(name, int8, uint32, float64)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, int8, float16)                \
    CPU_DEVICE_BINARY_DECL(name, int8, int16, float32)               \
    CPU_DEVICE_BINARY_DECL(name, int8, int32, float64)               \
    CPU_DEVICE_NOKERN_DECL(name, int8, int64, int64)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, bfloat16, bfloat16)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, float16, float16)             \
    CPU_DEVICE_BINARY_DECL(name, int8, float32, float32)             \
    CPU_DEVICE_BINARY_DECL(name, int8, float64, float64)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, complex32, complex32)         \
    CPU_DEVICE_BINARY_DECL(name, int8, complex64, complex64)         \
    CPU_DEVICE_BINARY_DECL(name, int8, complex128, complex128)       \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int16, uint8, float32)              \
    CPU_DEVICE_BINARY_DECL(name, int16, uint16, float32)             \
    CPU_DEVICE_BINARY_DECL(name, int16, uint32, float64)             \
    CPU_DEVICE_BINARY_DECL(name, int16, int8, float32)               \
    CPU_DEVICE_BINARY_DECL(name, int16, int16, float32)              \
    CPU_DEVICE_BINARY_DECL(name, int16, int32, float64)              \
    CPU_DEVICE_NOKERN_DECL(name, int16, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int16, bfloat16, float32)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, float16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, int16, float32, float32)            \
    CPU_DEVICE_BINARY_DECL(name, int16, float64, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, complex32, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, int16, complex64, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, int16, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int32, uint8, float64)              \
    CPU_DEVICE_BINARY_DECL(name, int32, uint16, float64)             \
    CPU_DEVICE_BINARY_DECL(name, int32, uint32, float64)             \
    CPU_DEVICE_BINARY_DECL(name, int32, int8, float64)               \
    CPU_DEVICE_BINARY_DECL(name, int32, int16, float64)              \
    CPU_DEVICE_BINARY_DECL(name, int32, int32, float64)              \
    CPU_DEVICE_NOKERN_DECL(name, int32, int64, int64)                \
    CPU_DEVICE_BINARY_DECL(name, int32, bfloat16, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, float16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, int32, float32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, int32, float64, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, complex32, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, int32, complex64, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, int32, complex128, complex128)      \
                                                                     \
    CPU_DEVICE_NOKERN_DECL(name, int64, uint8, int64)                \
    CPU_DEVICE_NOKERN_DECL(name, int64, uint16, int64)               \
    CPU_DEVICE_NOKERN_DECL(name, int64, uint32, int64)               \
    CPU_DEVICE_NOKERN_DECL(name, int64, int8, int64)                 \
    CPU_DEVICE_NOKERN_DECL(name, int64, int16, int64)                \
    CPU_DEVICE_NOKERN_DECL(name, int64, int32, int64)                \
    CPU_DEVICE_NOKERN_DECL(name, int64, int64, int64)                \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint8, bfloat16)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint16, float32)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint32, float64)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int8, bfloat16)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, bfloat16, bfloat16)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, bfloat16, float16, float32)         \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float32, float32)         \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float64, float64)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, bfloat16, complex32, complex64)     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, complex64, complex64)     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, complex128, complex128)   \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint8, float16)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint16, float32)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint32, float64)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int8, float16)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int16, float32)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int32, float64)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, bfloat16, float32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float16, float16)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float32, float32)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float64, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex32, complex32)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex64, complex64)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float32, uint8, float32)            \
    CPU_DEVICE_BINARY_DECL(name, float32, uint16, float32)           \
    CPU_DEVICE_BINARY_DECL(name, float32, uint32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float32, int8, float32)             \
    CPU_DEVICE_BINARY_DECL(name, float32, int16, float32)            \
    CPU_DEVICE_BINARY_DECL(name, float32, int32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float32, bfloat16, float32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, float16, float32)          \
    CPU_DEVICE_BINARY_DECL(name, float32, float32, float32)          \
    CPU_DEVICE_BINARY_DECL(name, float32, float64, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, complex32, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, float32, complex64, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, float32, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float64, uint8, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, uint16, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float64, uint32, float64)           \
    CPU_DEVICE_BINARY_DECL(name, float64, int8, float64)             \
    CPU_DEVICE_BINARY_DECL(name, float64, int16, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, int32, float64)            \
    CPU_DEVICE_BINARY_DECL(name, float64, bfloat16, float64)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, float16, float64)          \
    CPU_DEVICE_BINARY_DECL(name, float64, float32, float64)          \
    CPU_DEVICE_BINARY_DECL(name, float64, float64, float64)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, complex32, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, float64, complex64, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, float64, complex128, complex128)    \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint8, complex32)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint16, complex64)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint32, complex128)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int8, complex32)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int16, complex64)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int32, complex128)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, bfloat16, complex64)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float16, complex32)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float32, complex64)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float64, complex128)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex32, complex32)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex64, complex64)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex128, complex128)  \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint8, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint16, complex64)       \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex64, int8, complex64)         \
    CPU_DEVICE_BINARY_DECL(name, complex64, int16, complex64)        \
    CPU_DEVICE_BINARY_DECL(name, complex64, int32, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, complex64, bfloat16, complex64)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex64, float16, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, complex64, float32, complex64)      \
    CPU_DEVICE_BINARY_DECL(name, complex64, float64, complex128)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex64, complex32, complex64)    \
    CPU_DEVICE_BINARY_DECL(name, complex64, complex64, complex64)    \
    CPU_DEVICE_BINARY_DECL(name, complex64, complex128, complex128)  \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint8, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint16, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint32, complex128)     \
    CPU_DEVICE_BINARY_DECL(name, complex128, int8, complex128)       \
    CPU_DEVICE_BINARY_DECL(name, complex128, int16, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, int32, complex128)      \
    CPU_DEVICE_BINARY_DECL(name, complex128, bfloat16, complex128)   \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex128, float16, complex128)    \
    CPU_DEVICE_BINARY_DECL(name, complex128, float32, complex128)    \
    CPU_DEVICE_BINARY_DECL(name, complex128, float64, complex128)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex128, complex32, complex128)  \
    CPU_DEVICE_BINARY_DECL(name, complex128, complex64, complex128)  \
    CPU_DEVICE_BINARY_DECL(name, complex128, complex128, complex128)


CPU_DEVICE_BINARY_ARITHMETIC_DECL(add)
CPU_DEVICE_BINARY_ARITHMETIC_DECL(subtract)
CPU_DEVICE_BINARY_ARITHMETIC_DECL(multiply)
CPU_DEVICE_BINARY_ARITHMETIC_NO_COMPLEX_DECL(floor_divide)
CPU_DEVICE_BINARY_ARITHMETIC_NO_COMPLEX_DECL(remainder)
CPU_DEVICE_BINARY_ARITHMETIC_FLOAT_RETURN_DECL(divide)


/*****************************************************************************/
/*                                 Comparison                                */
/*****************************************************************************/

#define CPU_DEVICE_ALL_COMPARISON_DECL(name) \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint8, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint16, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint64, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint8, int8, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, uint8, int16, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, uint8, int32, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, uint8, int64, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, uint8, bfloat16, bool)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, float16, bool)        \
    CPU_DEVICE_BINARY_DECL(name, uint8, float32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint8, float64, bool)               \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint8, complex32, bool)      \
    CPU_DEVICE_BINARY_DECL(name, uint8, complex64, bool)             \
    CPU_DEVICE_BINARY_DECL(name, uint8, complex128, bool)            \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint8, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint16, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint64, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint16, int8, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, uint16, int16, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint16, int32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint16, int64, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint16, bfloat16, bool)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, float16, bool)       \
    CPU_DEVICE_BINARY_DECL(name, uint16, float32, bool)              \
    CPU_DEVICE_BINARY_DECL(name, uint16, float64, bool)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint16, complex32, bool)     \
    CPU_DEVICE_BINARY_DECL(name, uint16, complex64, bool)            \
    CPU_DEVICE_BINARY_DECL(name, uint16, complex128, bool)           \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint8, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint16, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint64, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint32, int8, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, uint32, int16, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint32, int32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint32, int64, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint32, bfloat16, bool)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, float16, bool)       \
    CPU_DEVICE_BINARY_DECL(name, uint32, float32, bool)              \
    CPU_DEVICE_BINARY_DECL(name, uint32, float64, bool)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, uint32, complex32, bool)     \
    CPU_DEVICE_BINARY_DECL(name, uint32, complex64, bool)            \
    CPU_DEVICE_BINARY_DECL(name, uint32, complex128, bool)           \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint8, bool)                \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint16, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint64, bool)               \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int8, uint8, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int8, uint16, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, uint32, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int8, int8, bool)                   \
    CPU_DEVICE_BINARY_DECL(name, int8, int16, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int8, int32, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int8, int64, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int8, bfloat16, bool)               \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, float16, bool)         \
    CPU_DEVICE_BINARY_DECL(name, int8, float32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int8, float64, bool)                \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int8, complex32, bool)       \
    CPU_DEVICE_BINARY_DECL(name, int8, complex64, bool)              \
    CPU_DEVICE_BINARY_DECL(name, int8, complex128, bool)             \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int16, uint8, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int16, uint16, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int16, uint32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int16, int8, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int16, int16, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int16, int32, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int16, int64, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int16, bfloat16, bool)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, float16, bool)        \
    CPU_DEVICE_BINARY_DECL(name, int16, float32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, int16, float64, bool)               \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int16, complex32, bool)      \
    CPU_DEVICE_BINARY_DECL(name, int16, complex64, bool)             \
    CPU_DEVICE_BINARY_DECL(name, int16, complex128, bool)            \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int32, uint8, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int32, uint16, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int32, uint32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int32, int8, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int32, int16, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int32, int32, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int32, int64, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int32, bfloat16, bool)              \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, float16, bool)        \
    CPU_DEVICE_BINARY_DECL(name, int32, float32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, int32, float64, bool)               \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, int32, complex32, bool)      \
    CPU_DEVICE_BINARY_DECL(name, int32, complex64, bool)             \
    CPU_DEVICE_BINARY_DECL(name, int32, complex128, bool)            \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, int64, uint8, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int64, uint16, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int64, uint32, bool)                \
    CPU_DEVICE_BINARY_DECL(name, int64, int8, bool)                  \
    CPU_DEVICE_BINARY_DECL(name, int64, int16, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int64, int32, bool)                 \
    CPU_DEVICE_BINARY_DECL(name, int64, int64, bool)                 \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint8, bool)              \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint16, bool)             \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, uint32, bool)             \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int8, bool)               \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int16, bool)              \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, int32, bool)              \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, bfloat16, bool)           \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float16, bool)            \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float32, bool)            \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, float64, bool)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, bfloat16, complex32, bool)   \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, complex64, bool)          \
    CPU_DEVICE_BINARY_DECL(name, bfloat16, complex128, bool)         \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint8, bool)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint16, bool)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, uint32, bool)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int8, bool)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int16, bool)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, int32, bool)        \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, bfloat16, bool)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float16, bool)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float32, bool)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, float64, bool)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex32, bool)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex64, bool)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float16, complex128, bool)   \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float32, uint8, bool)               \
    CPU_DEVICE_BINARY_DECL(name, float32, uint16, bool)              \
    CPU_DEVICE_BINARY_DECL(name, float32, uint32, bool)              \
    CPU_DEVICE_BINARY_DECL(name, float32, int8, bool)                \
    CPU_DEVICE_BINARY_DECL(name, float32, int16, bool)               \
    CPU_DEVICE_BINARY_DECL(name, float32, int32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, float32, bfloat16, bool)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, float16, bool)      \
    CPU_DEVICE_BINARY_DECL(name, float32, float32, bool)             \
    CPU_DEVICE_BINARY_DECL(name, float32, float64, bool)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float32, complex32, bool)    \
    CPU_DEVICE_BINARY_DECL(name, float32, complex64, bool)           \
    CPU_DEVICE_BINARY_DECL(name, float32, complex128, bool)          \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, float64, uint8, bool)               \
    CPU_DEVICE_BINARY_DECL(name, float64, uint16, bool)              \
    CPU_DEVICE_BINARY_DECL(name, float64, uint32, bool)              \
    CPU_DEVICE_BINARY_DECL(name, float64, int8, bool)                \
    CPU_DEVICE_BINARY_DECL(name, float64, int16, bool)               \
    CPU_DEVICE_BINARY_DECL(name, float64, int32, bool)               \
    CPU_DEVICE_BINARY_DECL(name, float64, bfloat16, bool)            \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, float16, bool)      \
    CPU_DEVICE_BINARY_DECL(name, float64, float32, bool)             \
    CPU_DEVICE_BINARY_DECL(name, float64, float64, bool)             \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, float64, complex32, bool)    \
    CPU_DEVICE_BINARY_DECL(name, float64, complex64, bool)           \
    CPU_DEVICE_BINARY_DECL(name, float64, complex128, bool)          \
                                                                     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint8, bool)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint16, bool)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, uint32, bool)     \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int8, bool)       \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int16, bool)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, int32, bool)      \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, bfloat16, bool)   \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float16, bool)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float32, bool)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, float64, bool)    \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex32, bool)  \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex64, bool)  \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex32, complex128, bool) \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint8, bool)             \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint16, bool)            \
    CPU_DEVICE_BINARY_DECL(name, complex64, uint32, bool)            \
    CPU_DEVICE_BINARY_DECL(name, complex64, int8, bool)              \
    CPU_DEVICE_BINARY_DECL(name, complex64, int16, bool)             \
    CPU_DEVICE_BINARY_DECL(name, complex64, int32, bool)             \
    CPU_DEVICE_BINARY_DECL(name, complex64, bfloat16, bool)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex64, float16, bool)    \
    CPU_DEVICE_BINARY_DECL(name, complex64, float32, bool)           \
    CPU_DEVICE_BINARY_DECL(name, complex64, float64, bool)           \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex64, complex32, bool)  \
    CPU_DEVICE_BINARY_DECL(name, complex64, complex64, bool)         \
    CPU_DEVICE_BINARY_DECL(name, complex64, complex128, bool)        \
                                                                     \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint8, bool)            \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint16, bool)           \
    CPU_DEVICE_BINARY_DECL(name, complex128, uint32, bool)           \
    CPU_DEVICE_BINARY_DECL(name, complex128, int8, bool)             \
    CPU_DEVICE_BINARY_DECL(name, complex128, int16, bool)            \
    CPU_DEVICE_BINARY_DECL(name, complex128, int32, bool)            \
    CPU_DEVICE_BINARY_DECL(name, complex128, bfloat16, bool)         \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex128, float16, bool)   \
    CPU_DEVICE_BINARY_DECL(name, complex128, float32, bool)          \
    CPU_DEVICE_BINARY_DECL(name, complex128, float64, bool)          \
    CPU_DEVICE_BINARY_NOIMPL_DECL(name, complex128, complex32, bool) \
    CPU_DEVICE_BINARY_DECL(name, complex128, complex64, bool)        \
    CPU_DEVICE_BINARY_DECL(name, complex128, complex128, bool)


CPU_DEVICE_ALL_COMPARISON_DECL(less)
CPU_DEVICE_ALL_COMPARISON_DECL(less_equal)
CPU_DEVICE_ALL_COMPARISON_DECL(greater_equal)
CPU_DEVICE_ALL_COMPARISON_DECL(greater)
CPU_DEVICE_ALL_COMPARISON_DECL(equal)
CPU_DEVICE_ALL_COMPARISON_DECL(not_equal)


/*****************************************************************************/
/*                                  Bitwise                                  */
/*****************************************************************************/

#define CPU_DEVICE_ALL_BITWISE_DECL(name) \
    CPU_DEVICE_BINARY_DECL(name, bool, bool, bool)       \
    CPU_DEVICE_BINARY_DECL(name, bool, uint8, uint8)     \
    CPU_DEVICE_BINARY_DECL(name, bool, uint16, uint16)   \
    CPU_DEVICE_BINARY_DECL(name, bool, uint32, uint32)   \
    CPU_DEVICE_BINARY_DECL(name, bool, uint64, uint64)   \
    CPU_DEVICE_BINARY_DECL(name, bool, int8, int8)       \
    CPU_DEVICE_BINARY_DECL(name, bool, int16, int16)     \
    CPU_DEVICE_BINARY_DECL(name, bool, int32, int32)     \
    CPU_DEVICE_BINARY_DECL(name, bool, int64, int64)     \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, uint8, bool, uint8)     \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint8, uint8)    \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint16, uint16)  \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint32, uint32)  \
    CPU_DEVICE_BINARY_DECL(name, uint8, uint64, uint64)  \
    CPU_DEVICE_BINARY_DECL(name, uint8, int8, int16)     \
    CPU_DEVICE_BINARY_DECL(name, uint8, int16, int16)    \
    CPU_DEVICE_BINARY_DECL(name, uint8, int32, int32)    \
    CPU_DEVICE_BINARY_DECL(name, uint8, int64, int64)    \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, uint16, bool, uint16)   \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint8, uint16)  \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint16, uint16) \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint32, uint32) \
    CPU_DEVICE_BINARY_DECL(name, uint16, uint64, uint64) \
    CPU_DEVICE_BINARY_DECL(name, uint16, int8, int32)    \
    CPU_DEVICE_BINARY_DECL(name, uint16, int16, int32)   \
    CPU_DEVICE_BINARY_DECL(name, uint16, int32, int32)   \
    CPU_DEVICE_BINARY_DECL(name, uint16, int64, int64)   \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, uint32, bool, uint32)   \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint8, uint32)  \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint16, uint32) \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint32, uint32) \
    CPU_DEVICE_BINARY_DECL(name, uint32, uint64, uint64) \
    CPU_DEVICE_BINARY_DECL(name, uint32, int8, int64)    \
    CPU_DEVICE_BINARY_DECL(name, uint32, int16, int64)   \
    CPU_DEVICE_BINARY_DECL(name, uint32, int32, int64)   \
    CPU_DEVICE_BINARY_DECL(name, uint32, int64, int64)   \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, uint64, bool, uint64)   \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint8, uint64)  \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint16, uint64) \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint32, uint64) \
    CPU_DEVICE_BINARY_DECL(name, uint64, uint64, uint64) \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, int8, bool, int8)       \
    CPU_DEVICE_BINARY_DECL(name, int8, uint8, int16)     \
    CPU_DEVICE_BINARY_DECL(name, int8, uint16, int32)    \
    CPU_DEVICE_BINARY_DECL(name, int8, uint32, int64)    \
    CPU_DEVICE_BINARY_DECL(name, int8, int8, int8)       \
    CPU_DEVICE_BINARY_DECL(name, int8, int16, int16)     \
    CPU_DEVICE_BINARY_DECL(name, int8, int32, int32)     \
    CPU_DEVICE_BINARY_DECL(name, int8, int64, int64)     \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, int16, bool, int16)     \
    CPU_DEVICE_BINARY_DECL(name, int16, uint8, int16)    \
    CPU_DEVICE_BINARY_DECL(name, int16, uint16, int32)   \
    CPU_DEVICE_BINARY_DECL(name, int16, uint32, int64)   \
    CPU_DEVICE_BINARY_DECL(name, int16, int8, int16)     \
    CPU_DEVICE_BINARY_DECL(name, int16, int16, int16)    \
    CPU_DEVICE_BINARY_DECL(name, int16, int32, int32)    \
    CPU_DEVICE_BINARY_DECL(name, int16, int64, int64)    \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, int32, bool, int32)     \
    CPU_DEVICE_BINARY_DECL(name, int32, uint8, int32)    \
    CPU_DEVICE_BINARY_DECL(name, int32, uint16, int32)   \
    CPU_DEVICE_BINARY_DECL(name, int32, uint32, int64)   \
    CPU_DEVICE_BINARY_DECL(name, int32, int8, int32)     \
    CPU_DEVICE_BINARY_DECL(name, int32, int16, int32)    \
    CPU_DEVICE_BINARY_DECL(name, int32, int32, int32)    \
    CPU_DEVICE_BINARY_DECL(name, int32, int64, int64)    \
                                                         \
    CPU_DEVICE_BINARY_DECL(name, int64, bool, int64)     \
    CPU_DEVICE_BINARY_DECL(name, int64, uint8, int64)    \
    CPU_DEVICE_BINARY_DECL(name, int64, uint16, int64)   \
    CPU_DEVICE_BINARY_DECL(name, int64, uint32, int64)   \
    CPU_DEVICE_BINARY_DECL(name, int64, int8, int64)     \
    CPU_DEVICE_BINARY_DECL(name, int64, int16, int64)    \
    CPU_DEVICE_BINARY_DECL(name, int64, int32, int64)    \
    CPU_DEVICE_BINARY_DECL(name, int64, int64, int64)

CPU_DEVICE_ALL_BITWISE_DECL(bitwise_and)
CPU_DEVICE_ALL_BITWISE_DECL(bitwise_or)
CPU_DEVICE_ALL_BITWISE_DECL(bitwise_xor)


/*****************************************************************************/
/*                             Two return values                             */
/*****************************************************************************/

#define CPU_DEVICE_ALL_BINARY_MV_DECL(name) \
    CPU_DEVICE_BINARY_MV_DECL(name, uint8, uint8, uint8, uint8)             \
    CPU_DEVICE_BINARY_MV_DECL(name, uint16, uint16, uint16, uint16)         \
    CPU_DEVICE_BINARY_MV_DECL(name, uint32, uint32, uint32, uint32)         \
    CPU_DEVICE_BINARY_MV_DECL(name, uint64, uint64, uint64, uint64)         \
    CPU_DEVICE_BINARY_MV_DECL(name, int8, int8, int8, int8)                 \
    CPU_DEVICE_BINARY_MV_DECL(name, int16, int16, int16, int16)             \
    CPU_DEVICE_BINARY_MV_DECL(name, int32, int32, int32, int32)             \
    CPU_DEVICE_BINARY_MV_DECL(name, int64, int64, int64, int64)             \
    CPU_DEVICE_BINARY_MV_DECL(name, bfloat16, bfloat16, bfloat16, bfloat16) \
    CPU_DEVICE_BINARY_MV_DECL(name, float32, float32, float32, float32)     \
    CPU_DEVICE_BINARY_MV_DECL(name, float64, float64, float64, float64)

CPU_DEVICE_ALL_BINARY_MV_DECL(divmod)


#endif /* CPU_DEVICE_BINARY_H */

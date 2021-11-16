#ifndef SSE2MSA_H
#define SSE2MSA_H

/* 
 * This header file provides a simple API translation layer between
 * SSE intrinsics to their corresponding MIPS/MIPS64 MSA versions.
 *
 * This header file does not yet translate all of the SSE intrinsics.
 * 
 * This project may only work with GCC since it has some GCC builtin functions.
 */

/*
 * This project is a fork from sse2neon(https://github.com/DLTcollab/sse2neon).
 * Contributors to this work are:
 *   John W. Ratcliff <jratcliffscarab@gmail.com>
 *   Brandon Rowlett <browlett@nvidia.com>
 *   Ken Fast <kfast@gdeb.com>
 *   Eric van Beurden <evanbeurden@nvidia.com>
 *   Alexander Potylitsin <apotylitsin@nvidia.com>
 *   Hasindu Gamaarachchi <hasindu2008@gmail.com>
 *   Jim Huang <jserv@biilabs.io>
 *   Mark Cheng <marktwtn@biilabs.io>
 *   Malcolm James MacLeod <malcolm@gulden.com>
 *   Devin Hussey (easyaspi314) <husseydevin@gmail.com>
 *   Sebastian Pop <spop@amazon.com>
 *   Developer Ecosystem Engineering <DeveloperEcosystemEngineering@apple.com>
 *   Danila Kutenin <danilak@google.com>
 *   François Turban (JishinMaster) <francois.turban@gmail.com>
 *   Pei-Hsuan Hung <afcidk@gmail.com>
 *   Yang-Hao Yuan <yanghau@biilabs.io>
 *   Syoyo Fujita <syoyo@lighttransport.com>
 *   Brecht Van Lommel <brecht@blender.org>
 *   Evidence John <mail@evi.fun>
 */

/*
 * sse2msa is freely redistributable under the MIT License.
 *
 * Copyright (c) 2015-2021, The sse2neon project.
 * Copyright (c) 2021 CIP United Co. Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#if defined(__GNUC__)
#pragma push_macro("FORCE_INLINE")
#pragma push_macro("ALIGN_STRUCT")
#define FORCE_INLINE static inline __attribute__((always_inline))
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#else
#error Unsupported compiler
#endif

#include <msa.h>

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define SSE2MSA_NO_IMPL 0

#define _MM_SHUFFLE(fp3, fp2, fp1, fp0) \
	(((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | ((fp0)))

#define _MM_FROUND_TO_NEAREST_INT 0x00
#define _MM_FROUND_TO_NEG_INF 0x01
#define _MM_FROUND_TO_POS_INF 0x02
#define _MM_FROUND_TO_ZERO 0x03
#define _MM_FROUND_CUR_DIRECTION 0x04
#define _MM_FROUND_NO_EXC 0x08
#define _MM_ROUND_NEAREST 0x0000
#define _MM_ROUND_DOWN 0x2000
#define _MM_ROUND_UP 0x4000
#define _MM_ROUND_TOWARD_ZERO 0x6000

typedef int32_t v2i32 __attribute__((vector_size(8)));

/* Ref: mmintrin.h emmintrin.h */
typedef v2i32 __m64;    /* int       */
typedef v4f32 __m128;   /* float     */
typedef v2f64 __m128d;  /* double    */
typedef v2i64 __m128i;  /* long long */

#if defined(__mips64)
typedef          __int128 __i128_t;
typedef unsigned __int128 __u128_t;
#endif

#define v_msa_setzero(type) ((type)__builtin_msa_ldi_b(0))

/* To access the contents of a MSA register */
typedef union ALIGN_STRUCT(16) VREG128 {
/*---------------------------------------------------*/
	__m64   m64[2];
	__m128  m128;              /* MMX, SSE types */
	__m128d m128d;
	__m128i m128i;
/*---------------------------------------------------*/
#if defined(__mips64)
	__i128_t i128;             /* GCC extensions */
	__u128_t u128;
#endif
/*---------------------------------------------------*/
	int8_t   i8[16];
	int16_t  i16[8];
	int32_t  i32[4];
	int64_t  i64[2];
	uint8_t  u8[16];                  /* C types */
	uint16_t u16[8];
	uint32_t u32[4];
	uint64_t u64[2];
	float    f32[4];
	double   f64[2];
/*---------------------------------------------------*/
	v16i8 msa_v16i8;
	v16u8 msa_v16u8;
	v8i16 msa_v8i16;
	v8u16 msa_v8u16;
	v4i32 msa_v4i32;         /* MSA vector types */
	v4u32 msa_v4u32;
	v2i64 msa_v2i64;
	v2u64 msa_v2u64;
	v4f32 msa_v4f32;
	v2f64 msa_v2f64;
/*---------------------------------------------------*/
} VREG128;

#define reinterpret_i32(x) \
__extension__(({union {int32_t i; typeof(x) v;} $ = {.v = x}; $.i;}))
#define reinterpret_i64(x) \
__extension__(({union {int64_t i; typeof(x) v;} $ = {.v = x}; $.i;}))

#define vreinterpret_m64(x) ((__m64)(x))
#define vreinterpret_m128(x) ((__m128)(x))
#define vreinterpret_m128d(x) ((__m128d)(x))
#define vreinterpret_m128i(x) ((__m128i)(x))
#define vreinterpret_v16i8(x) ((v16i8)(x))
#define vreinterpret_v8i16(x) ((v8i16)(x))
#define vreinterpret_v4i32(x) ((v4i32)(x))
#define vreinterpret_v2i64(x) ((v2i64)(x))
#define vreinterpret_v16u8(x) ((v16u8)(x))
#define vreinterpret_v8u16(x) ((v8u16)(x))
#define vreinterpret_v4u32(x) ((v4u32)(x))
#define vreinterpret_v2u64(x) ((v2u64)(x))
#define vreinterpret_v4f32(x) ((v4f32)(x))
#define vreinterpret_v2f64(x) ((v2f64)(x))

#define vreinterpret_nth_f32_m128(x, n) (((VREG128*)&x)->f32[n])
#define vreinterpret_nth_f64_m128d(x, n) (((VREG128*)&x)->f64[n])

#define vreinterpret_nth_i16_m128(x, n) (((VREG128*)&x)->i16[n])
#define vreinterpret_nth_i16_m128i(x, n) (((VREG128*)&x)->i16[n])
#define vpreinterpret_nth_i16_m128d(p, n) (((VREG128*)p)->i16[n])

#define vreinterpret_nth_u16_m128(x, n) (((VREG128*)&x)->u16[n])
#define vreinterpret_nth_u16_m128i(x, n) (((VREG128*)&x)->u16[n])
#define vpreinterpret_nth_u16_m128d(p, n) (((VREG128*)p)->u16[n])

#define vreinterpret_nth_i32_m128(x, n) (((VREG128*)&x)->i32[n])
#define vreinterpret_nth_i32_m128i(x, n) (((VREG128*)&x)->i32[n])
#define vpreinterpret_nth_i32_m128d(p, n) (((VREG128*)p)->i32[n])

#define vreinterpret_nth_i64_m128(x, n) (((VREG128*)&x)->i64[n])
#define vreinterpret_nth_i64_m128i(x, n) (((VREG128*)&x)->i64[n])
#define vpreinterpret_nth_i64_m128d(p, n) (((VREG128*)p)->i64[n])

#define vreinterpret_nth_u32_m128(x, n) (((VREG128*)&x)->u32[n])
#define vreinterpret_nth_u32_m128i(x, n) (((VREG128*)&x)->u32[n])
#define vpreinterpret_nth_u32_m128d(p, n) (((VREG128*)p)->u32[n])

#define vreinterpret_nth_u64_m128(x, n) (((VREG128*)&x)->u64[n])
#define vreinterpret_nth_u64_m128i(x, n) (((VREG128*)&x)->u64[n])
#define vpreinterpret_nth_u64_m128d(p, n) (((VREG128*)p)->u64[n])

FORCE_INLINE void _mm_prefetch(const void *p, int i)
{
	(void) i;
	__builtin_prefetch(p);
}

FORCE_INLINE __m128i _mm_setzero_si128(void)
{
	return v_msa_setzero(__m128i);
}

FORCE_INLINE __m128 _mm_setzero_ps(void)
{
	return v_msa_setzero(__m128);
}

FORCE_INLINE __m128d _mm_setzero_pd(void)
{
	return v_msa_setzero(__m128d);
}

FORCE_INLINE __m128 _mm_set1_ps(float a)
{
	return vreinterpret_m128(__builtin_msa_fill_w(reinterpret_i32(a)));
}

FORCE_INLINE __m128 _mm_set_ps1(float a)
{
	return vreinterpret_m128(__builtin_msa_fill_w(reinterpret_i32(a)));
}

FORCE_INLINE __m128 _mm_set_ps(float e3, float e2, float e1, float e0)
{
	VREG128 v = {
		.f32 = {e0, e1, e2, e3}
	};
	return v.m128;
}

FORCE_INLINE __m128 _mm_set_ss(float a)
{
	VREG128 v = {
		.f32 = {a, 0, 0, 0}
	};
	return v.m128;
}

FORCE_INLINE __m128 _mm_setr_ps(float e3, float e2, float e1, float e0)
{
	VREG128 v = {
		.f32 = {e3, e2, e1, e0}
	};
	return v.m128;
}

FORCE_INLINE __m128d _mm_setr_pd(double e1, double e0)
{
	VREG128 v = {
		.f64 = {e1, e0}
	};
	return v.m128d;
}

FORCE_INLINE __m128i _mm_setr_epi16(
		short e7, short e6, short e5, short e4,
		short e3, short e2, short e1, short e0)
{
	VREG128 v = {
		.i16 = {e7, e6, e5, e4, e3, e2, e1, e0}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_setr_epi32(int e3, int e2, int e1, int e0)
{
	VREG128 v = {
		.i32 = {e3, e2, e1, e0}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_setr_epi64(__m64 e1, __m64 e0)
{
	VREG128 v = {
		.m64 = {e1, e0}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set1_epi8(char a)
{
	VREG128 v = {
		.msa_v16i8 = __builtin_msa_fill_b(a)
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set1_epi16(short a)
{
	VREG128 v = {
		.msa_v8i16 = __builtin_msa_fill_h(a)
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set_epi8(char e15, char e14, char e13, 
		char e12, char e11, char e10, char e9, char e8, char e7, char e6,
		char e5, char e4, char e3, char e2, char e1, char e0)
{
	VREG128 v = {
		.i8 = {e0, e1, e2, e3, e4, e5,
			e6, e7, e8, e9, e10, e11, e12, e13, e14, e15}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set_epi16(short e7, short e6, short e5,
		short e4, short e3, short e2, short e1, short e0)
{
	VREG128 v = {
		.i16 = {e0, e1, e2, e3, e4, e5, e6, e7}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_setr_epi8(char e15, char e14, char e13, 
		char e12, char e11, char e10, char e9, char e8, char e7, char e6,
		char e5, char e4, char e3, char e2, char e1, char e0)
{
	VREG128 v = {
		.i8 = {e15, e14, e13, e12, e11, e10,
			e9, e8, e7, e6, e5, e4, e3, e2, e1, e0}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set1_epi32(int a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_fill_w(a)
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set1_epi64(__m64 a)
{
	VREG128 v = {
		.m64 = {a, a}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set1_epi64x(int64_t a)
{
	VREG128 v = {
		.msa_v2i64 = __builtin_msa_fill_d(a)
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set_epi32(int e3, int e2, int e1, int e0)
{
	VREG128 v = {
		.msa_v4i32 = {e0, e1, e2, e3}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set_epi64x(int64_t e1, int64_t e0)
{
	VREG128 v = {
		.msa_v2i64 = {e0, e1}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_set_epi64(__m64 e1, __m64 e0)
{
	VREG128 v = {
		.m64 = {e0, e1}
	};
	return v.m128i;
}

FORCE_INLINE __m128d _mm_set_pd(double e1, double e0)
{
	VREG128 v = {
		.f64 = {e0, e1}
	};
	return v.m128d;
}

FORCE_INLINE __m128d _mm_set_sd(double a)
{
	VREG128 v = {
		.f64 = {a, 0}
	};
	return v.m128d;
}

FORCE_INLINE __m128d _mm_set1_pd(double a)
{
	return vreinterpret_m128d(
		__builtin_msa_fill_d(reinterpret_i64(a)));
}

#define _mm_set_pd1 _mm_set1_pd

FORCE_INLINE void _mm_store_ps(float *p, __m128 a)
{
	__builtin_msa_st_w(vreinterpret_v4i32(a), p, 0);
}

FORCE_INLINE void _mm_storer_ps(float *p, __m128 a)
{
	__builtin_msa_st_w(__builtin_msa_shf_w(
		vreinterpret_v4i32(a), 0x1b), p, 0);
}

FORCE_INLINE void _mm_storer_pd(double *p, __m128d a)
{
	p[0] = vreinterpret_nth_f64_m128d(a, 1);
	p[1] = vreinterpret_nth_f64_m128d(a, 0);
}

FORCE_INLINE void _mm_store_ps1(float *p, __m128 a)
{
	v4i32 v = __builtin_msa_fill_w(vreinterpret_nth_i32_m128(a, 0));
	__builtin_msa_st_w(v, p, 0);
}

#define _mm_store1_ps _mm_store_ps1

FORCE_INLINE void _mm_storeu_ps(float *p, __m128 a)
{
	__builtin_msa_st_w(vreinterpret_v4i32(a), p, 0);
}

FORCE_INLINE void _mm_store_si128(__m128i *p, __m128i a)
{
	__builtin_msa_st_d(vreinterpret_v2i64(a), p, 0);
}

FORCE_INLINE void _mm_storeu_si128(__m128i *p, __m128i a)
{
	__builtin_msa_st_d(vreinterpret_v2i64(a), p, 0);
}

FORCE_INLINE void _mm_store_ss(float *p, __m128 a)
{
	*p = vreinterpret_nth_f32_m128(a, 0);
}

FORCE_INLINE void _mm_store_pd(double *p, __m128d a)
{
	__builtin_msa_st_d(vreinterpret_v2i64(a), p, 0);
}

FORCE_INLINE void _mm_store_pd1(double *p, __m128d a)
{
	p[0] = vreinterpret_nth_f64_m128d(a, 0);
	p[1] = vreinterpret_nth_f64_m128d(a, 0);
}

#define _mm_store1_pd _mm_store_pd1

FORCE_INLINE void _mm_store_sd(double *p, __m128d a)
{
	p[0] = vreinterpret_nth_f64_m128d(a, 0);
}

FORCE_INLINE void _mm_storeh_pd(double *p, __m128d a)
{
	p[0] = vreinterpret_nth_f64_m128d(a, 1);
}

FORCE_INLINE void _mm_storel_pd(double *p, __m128d a)
{
	p[0] = vreinterpret_nth_f64_m128d(a, 0);
}

FORCE_INLINE void _mm_storeu_pd(double *p, __m128d a)
{
	__builtin_msa_st_d(vreinterpret_v2i64(a), p, 0);
}

FORCE_INLINE void _mm_storeu_si16(void *p, __m128i a)
{
	*((int16_t*)p) = vreinterpret_nth_i16_m128(a, 0);
}

FORCE_INLINE void _mm_storeu_si32(void *p, __m128i a)
{
	*((int32_t*)p) = vreinterpret_nth_i32_m128(a, 0);
}

FORCE_INLINE void _mm_storeu_si64(void *p, __m128i a)
{
	*((int64_t*)p) = vreinterpret_nth_i64_m128(a, 0);
}

FORCE_INLINE void _mm_storel_epi64(__m128i *p, __m128i a)
{
	*((int64_t*)p) = vreinterpret_nth_i64_m128(a, 0);
}

FORCE_INLINE void _mm_storel_pi(__m64 *p, __m128 a)
{
	*(((float*)p) + 0) = vreinterpret_nth_f32_m128(a, 0);
	*(((float*)p) + 1) = vreinterpret_nth_f32_m128(a, 1);
}

FORCE_INLINE void _mm_storeh_pi(__m64 *p, __m128 a)
{
	*(((float*)p) + 0) = vreinterpret_nth_f32_m128(a, 2);
	*(((float*)p) + 1) = vreinterpret_nth_f32_m128(a, 3);
}

FORCE_INLINE __m128i _mm_stream_load_si128(__m128i *p)
{
	return vreinterpret_m128i(__builtin_msa_ld_d(p, 0));
}

FORCE_INLINE void _mm_stream_pd(double *p, __m128d a)
{
	__builtin_msa_st_d(vreinterpret_v2i64(a), p, 0);
}

FORCE_INLINE void _mm_stream_pi(__m64 *p, __m64 a) { *p = a; }

FORCE_INLINE void _mm_stream_ps(float *p, __m128 a)
{
	__builtin_msa_st_w(vreinterpret_v4i32(a), p, 0);	
}

FORCE_INLINE void _mm_stream_si128(__m128i *p, __m128i a)
{
	__builtin_msa_st_d(vreinterpret_v2i64(a), p, 0);
}

FORCE_INLINE void _mm_stream_si32(int *p, int a) { *p = a; }

FORCE_INLINE void _mm_stream_si64(int64_t *p, int64_t a) { *p = a; }

FORCE_INLINE __m128 _mm_load1_ps(const float *p)
{
	return vreinterpret_m128(__builtin_msa_fill_w(reinterpret_i32(*p)));
}

#define _mm_load_ps1 _mm_load1_ps

FORCE_INLINE __m128d _mm_load1_pd(const double *p)
{
	return vreinterpret_m128d(__builtin_msa_fill_d(reinterpret_i64(*p)));
}

#define _mm_load_pd1 _mm_load1_pd

FORCE_INLINE __m128 _mm_loadl_pi(__m128 a, __m64 const *p)
{
	VREG128 v = {.m128 = a};
	v.m64[0] = *p;
	return v.m128;
}

FORCE_INLINE __m128 _mm_loadh_pi(__m128 a, __m64 const *p)
{
	VREG128 v = {.m128 = a};
	v.m64[1] = *p;
	return v.m128;
}

FORCE_INLINE __m128 _mm_load_ps(const float *p)
{
	return vreinterpret_m128(__builtin_msa_ld_w(p, 0));
}

FORCE_INLINE __m128 _mm_loadr_ps(float const* p)
{
	return vreinterpret_m128(__builtin_msa_shf_w(
		__builtin_msa_ld_w(p, 0), 0x1b));
}

FORCE_INLINE __m128d _mm_loadr_pd(const double *p)
{
	VREG128 v = {
		.f64 = {p[1], p[0]}
	};
	return v.m128d;
}

FORCE_INLINE __m128 _mm_loadu_ps(const float *p)
{
	return vreinterpret_m128(__builtin_msa_ld_w(p, 0));
}

FORCE_INLINE __m128d _mm_load_pd(const double *p)
{
	return vreinterpret_m128d(__builtin_msa_ld_d(p, 0));
}

FORCE_INLINE __m128d _mm_loadu_pd(const double *p)
{
	return vreinterpret_m128d(__builtin_msa_ld_d(p, 0));
}

FORCE_INLINE __m128d _mm_loadh_pd(__m128d a, const double *p)
{
	VREG128 v = {.m128d = a};
	v.f64[1] = *p;
	return v.m128d;
}

FORCE_INLINE __m128d _mm_loadl_pd(__m128d a, const double *p)
{
	VREG128 v = {.m128d = a};
	v.f64[0] = *p;
	return v.m128d;
}

FORCE_INLINE __m128d _mm_loaddup_pd(const double *p)
{
	VREG128 v = {.f64 = {*p, *p}};
	return v.m128d;
}

FORCE_INLINE __m128 _mm_load_ss(const float *p)
{
	VREG128 v = {
		.f32 = {*p, 0, 0, 0}
	};
	return v.m128;
}

FORCE_INLINE __m128d _mm_load_sd(double const *p)
{
	VREG128 v = {
		.f64 = {*p, 0}
	};
	return v.m128d;
}

FORCE_INLINE __m128i _mm_loadu_si64(const void *p)
{
	VREG128 v = {
		.i64 = {*(int64_t*)p, 0}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_loadl_epi64(__m128i const *p)
{
	VREG128 v = {
		.i64 = {vpreinterpret_nth_i64_m128d(p, 0), 0}
	};
	return v.m128i;
}

FORCE_INLINE void *_mm_malloc(size_t size, size_t align)
{
	void *ptr;
	if (align == 1)
		return malloc(size);
	if (align == 2 || (sizeof(void *) == 8 && align == 4))
		align = sizeof(void *);
	if (!posix_memalign(&ptr, align, size))
		return ptr;
	return NULL;
}

FORCE_INLINE void _mm_free(void *addr)
{
	free(addr);
}

FORCE_INLINE __m128 _mm_move_ss(__m128 a, __m128 b)
{
	VREG128 v = {.m128 = a};
	v.f32[0] = vreinterpret_nth_f32_m128(b, 0);
	return v.m128;
}

FORCE_INLINE __m128d _mm_move_sd(__m128d a, __m128d b)
{
	VREG128 v = {.m128d = a};
	v.f64[0] = vreinterpret_nth_f64_m128d(b, 0);
	return v.m128d;
}

FORCE_INLINE __m128i _mm_move_epi64(__m128i a)
{
	VREG128 v = {
		.i64 = {vreinterpret_nth_i64_m128i(a, 0), 0}
	};
	return v.m128i;
}

FORCE_INLINE __m128 _mm_undefined_ps(void)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
	__m128 a;
	return a;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

FORCE_INLINE __m128d _mm_undefined_pd(void)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
	__m128d a;
	return a;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

FORCE_INLINE __m128i _mm_undefined_si128(void)
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
	__m128i a;
	return a;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

FORCE_INLINE __m128 _mm_andnot_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(
		__builtin_msa_and_v((__builtin_msa_nor_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(a))),
		vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128d _mm_andnot_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(
		__builtin_msa_and_v((__builtin_msa_nor_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(a))),
		vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_and_v((__builtin_msa_nor_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(a))),
		vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128i _mm_and_si128(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_and_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128 _mm_and_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_and_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128d _mm_and_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_and_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128 _mm_or_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_or_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128d _mm_or_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_or_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128 _mm_xor_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_xor_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128d _mm_xor_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_xor_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128i _mm_or_si128(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_or_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128i _mm_xor_si128(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_xor_v(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128d _mm_movedup_pd(__m128d a)
{
	VREG128 v = {
		.f64 = {
			vreinterpret_nth_f64_m128d(a, 0),
			vreinterpret_nth_f64_m128d(a, 0),
		}
	};
	return v.m128d;
}

FORCE_INLINE __m128 _mm_movehdup_ps(__m128 a)
{
	return vreinterpret_m128(__builtin_msa_shf_w(
		vreinterpret_v4i32(a), 0xf5));
}

FORCE_INLINE __m128 _mm_moveldup_ps(__m128 a)
{
	return vreinterpret_m128(__builtin_msa_shf_w(
		vreinterpret_v4i32(a), 0xa0));
}

FORCE_INLINE __m128 _mm_movehl_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_vshf_w(
		({v4i32 mask = {6, 7, 2, 3}; mask;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128 _mm_movelh_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_vshf_w(
		({v4i32 mask = {0, 1, 4, 5}; mask;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128i _mm_abs_epi32(__m128i a)
{
	__m128i v = _mm_setzero_si128();
	return vreinterpret_m128i(__builtin_msa_add_a_w(
		vreinterpret_v4i32(v), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128i _mm_abs_epi16(__m128i a)
{
	__m128i v = _mm_setzero_si128();
	return vreinterpret_m128i(__builtin_msa_add_a_h(
		vreinterpret_v8i16(v), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128i _mm_abs_epi8(__m128i a)
{
	__m128i v = _mm_setzero_si128();
	return vreinterpret_m128i(__builtin_msa_add_a_b(
		vreinterpret_v16i8(v), vreinterpret_v16i8(a)));
}

FORCE_INLINE __m64 _mm_abs_pi32(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.m128i = _mm_abs_epi32(v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_abs_pi16(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.m128i = _mm_abs_epi16(v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_abs_pi8(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.m128i = _mm_abs_epi8(v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m128i _mm_sad_epu8(__m128i a, __m128i b)
{
	VREG128 v = {
		.msa_v16u8 = __builtin_msa_asub_u_b(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b))
	};
	v.msa_v8u16 = __builtin_msa_hadd_u_h(v.msa_v16u8, v.msa_v16u8);
	v.msa_v4u32 = __builtin_msa_hadd_u_w(v.msa_v8u16, v.msa_v8u16);
	v.msa_v2u64 = __builtin_msa_hadd_u_d(v.msa_v4u32, v.msa_v4u32);
	return v.m128i;
}

FORCE_INLINE __m64 _mm_sad_pu8(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v = {
		.m128i = _mm_sad_epu8(va.m128i, vb.m128i)
	};
	return v.m64[0];
}

#define _m_psadbw(a, b) _mm_sad_pu8(a, b)

FORCE_INLINE __m128 _mm_shuffle_ps(__m128 a, __m128 b, int imm8)
{
	return vreinterpret_m128(__builtin_msa_vshf_w(
		({v4i32 mask = {imm8 & 0x3, (imm8 >> 2) & 0x3,
		((imm8 >> 4) & 0x3) + 4, ((imm8 >> 6) & 0x3) + 4}; mask;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128i _mm_shuffle_epi32(__m128i a, int imm8)
{
	return vreinterpret_m128i(__builtin_msa_vshf_w(
		({v4i32 mask = {imm8 & 0x3, (imm8 >> 2) & 0x3,
		((imm8 >> 4) & 0x3) + 4, ((imm8 >> 6) & 0x3) + 4}; mask;}),
		vreinterpret_v4i32(a), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128i _mm_shuffle_epi8(__m128i a, __m128i b)
{
	v16i8 mask = vreinterpret_v16i8((vreinterpret_v16u8(b) << 4) >> 4);
	mask = mask | (vreinterpret_v16i8(b) & __builtin_msa_fill_b(0x80));
	return vreinterpret_m128i(
		__builtin_msa_vshf_b(mask,
		vreinterpret_v16i8(a), vreinterpret_v16i8(a)
	));
}

FORCE_INLINE __m64 _mm_shuffle_pi8(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	v16i8 mask = vreinterpret_v16i8((vb.msa_v16u8 << 5) >> 5);
	mask = mask | (vb.msa_v16i8 & __builtin_msa_fill_b(0x80));
	VREG128 v = {
		.msa_v16i8 = __builtin_msa_vshf_b(
			mask, va.msa_v16i8, va.msa_v16i8)
	};
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_shuffle_pi16(__m64 a, int imm8)
{
	VREG128 v = {.m64 = {a, a}};
	v.msa_v8i16 = __builtin_msa_vshf_h(
		({v8i16 mask = {(imm8) & (0x3), ((imm8) >> 2) & 0x3,
		(((imm8) >> 4) & 0x3) + 4, (((imm8) >> 6) & 0x3) + 4,
		0, 0, 0, 0}; mask;}), v.msa_v8i16, v.msa_v8i16);
	return v.m64[0];
}

#define _m_pshufw(a, imm) _mm_shuffle_pi16(a, imm)

FORCE_INLINE __m128i _mm_shufflehi_epi16(__m128i a, int imm8)
{
	v8i16 mask = {
		0, 1, 2, 3,
		((imm8     ) & 0x3) + 4,
		((imm8 >> 2) & 0x3) + 4,
		((imm8 >> 4) & 0x3) + 4,
		((imm8 >> 6) & 0x3) + 4

	};
	return vreinterpret_m128i(__builtin_msa_vshf_h(mask,
		vreinterpret_v8i16(a), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128i _mm_shufflelo_epi16(__m128i a, int imm8)
{
	v8i16 mask = {
		(imm8     ) & 0x3,
		(imm8 >> 2) & 0x3,
		(imm8 >> 4) & 0x3,
		(imm8 >> 6) & 0x3,
		4, 5, 6, 7

	};
	return vreinterpret_m128i(__builtin_msa_vshf_h(mask,
		vreinterpret_v8i16(a), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128d _mm_shuffle_pd(__m128d a, __m128d b, int imm8)
{
	imm8 = imm8 & 3;
	switch (imm8) {
	case 0:
		return vreinterpret_m128d(
			__builtin_msa_vshf_d(({v2i64 mask = {0, 2}; mask;}),
			vreinterpret_v2i64(b), vreinterpret_v2i64(a)));
	case 1:
		return vreinterpret_m128d(
			__builtin_msa_vshf_d(({v2i64 mask = {1, 2}; mask;}),
			vreinterpret_v2i64(b), vreinterpret_v2i64(a)));
	case 2:
		return vreinterpret_m128d(
			__builtin_msa_vshf_d(({v2i64 mask = {0, 3}; mask;}),
			vreinterpret_v2i64(b), vreinterpret_v2i64(a)));
	case 3:
		return vreinterpret_m128d(
			__builtin_msa_vshf_d(({v2i64 mask = {1, 3}; mask;}),
			vreinterpret_v2i64(b), vreinterpret_v2i64(a)));
	}
	return v_msa_setzero(__m128d);
}

FORCE_INLINE __m128i _mm_blend_epi16(__m128i a, __m128i b, const int imm8)
{
	v8i16 mask = {
		(imm8     ) & 0x1 ?  8 : 0, (imm8 >> 1) & 0x1 ?  9 : 1,
		(imm8 >> 2) & 0x1 ? 10 : 2, (imm8 >> 3) & 0x1 ? 11 : 3,
		(imm8 >> 4) & 0x1 ? 12 : 4, (imm8 >> 5) & 0x1 ? 13 : 5,
		(imm8 >> 6) & 0x1 ? 14 : 6, (imm8 >> 7) & 0x1 ? 15 : 7
	};
	return vreinterpret_m128i(__builtin_msa_vshf_h(mask,
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128d _mm_blend_pd(__m128d a, __m128d b, const int imm8)
{
	v2i64 mask = {
		(imm8     ) & 0x1 ? 2 : 0,
		(imm8 >> 1) & 0x1 ? 3 : 1
	};
	return vreinterpret_m128d(__builtin_msa_vshf_d(mask,
		vreinterpret_v2i64(b), vreinterpret_v2i64(a)));
}

FORCE_INLINE __m128 _mm_blend_ps(__m128 a, __m128 b, const int imm8)
{
	v4i32 mask = {
		(imm8     ) & 0x1 ? 4 : 0, (imm8 >> 1) & 0x1 ? 5 : 1,
		(imm8 >> 2) & 0x1 ? 6 : 2, (imm8 >> 3) & 0x1 ? 7 : 3
	};
	return vreinterpret_m128(__builtin_msa_vshf_w(mask,
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128i _mm_blendv_epi8(__m128i a, __m128i b, __m128i mask)
{
	VREG128 u = {
		.i8 = {0, 1, 2, 3, 4, 5, 6, 7, 8,
			9, 10, 11, 12, 13, 14, 15}
	};
	VREG128 v = {
		.msa_v16i8 = u.msa_v16i8 + 16
	};
	v16i8 masklo, maskhi;
	masklo = __builtin_msa_srai_b(vreinterpret_v16i8(mask), 7);
	maskhi = ~masklo;
	masklo &= u.msa_v16i8;
	maskhi &= v.msa_v16i8;
	masklo |= maskhi;
	return vreinterpret_m128i(__builtin_msa_vshf_b(masklo,
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128d _mm_blendv_pd(__m128d a, __m128d b, __m128d mask)
{
	VREG128 u = {
		.i64 = {0, 1}
	};
	VREG128 v = {
		.i64 = {2, 3}
	};
	v2i64 masklo, maskhi;
	masklo = __builtin_msa_srai_d(vreinterpret_v2i64(mask), 63);
	maskhi = ~masklo;
	masklo &= u.msa_v2i64;
	maskhi &= v.msa_v2i64;
	masklo |= maskhi;
	return vreinterpret_m128d(__builtin_msa_vshf_d(masklo,
		vreinterpret_v2i64(a), vreinterpret_v2i64(b)));
}

FORCE_INLINE __m128 _mm_blendv_ps(__m128 a, __m128 b, __m128 mask)
{
	VREG128 u = {
		.i32 = {0, 1, 2, 3}
	};
	VREG128 v = {
		.i32 = {4, 5, 6, 7}
	};
	v4i32 masklo, maskhi;
	masklo = __builtin_msa_srai_w(vreinterpret_v4i32(mask), 31);
	maskhi = ~masklo;
	masklo &= u.msa_v4i32;
	maskhi &= v.msa_v4i32;
	masklo |= maskhi;
	return vreinterpret_m128(__builtin_msa_vshf_w(masklo,
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

FORCE_INLINE __m128i _mm_srli_si128(__m128i a, int imm8)
{
	if (imm8 <= 0)
		return a;
	else if (imm8 >= 16)
		return v_msa_setzero(__m128i);
	int n = imm8 << 3;
	VREG128 v = {.m128i = a};
#if defined(__mips64)
	v.u128 = v.u128 >> n;
#else
	if (n < 64) {
		v.u64[0]  = v.u64[0] >> n;
		v.u64[0] |= v.u64[1] << 64 - n;
		v.u64[1]  = v.u64[1] >> n;
	} else if (n >= 64 && n < 128) {
		v.u64[0] = v.u64[1];
		v.u64[1] = 0;
		v.u64[0] = v.u64[0] >> (n - 64);
	}
#endif
	return v.m128i;
}

FORCE_INLINE __m128i _mm_slli_si128(__m128i a, int imm8)
{
	if (imm8 <= 0)
		return a;
	else if (imm8 >= 16)
		return v_msa_setzero(__m128i);
	int n = imm8 << 3;
	VREG128 v = {.m128i = a};
#if defined(__mips64)
	v.u128 = v.u128 << n;
#else
	if (n < 64) {
		v.u64[1]  = v.u64[1] << n;
		v.u64[1] |= v.u64[0] >> 64 - n;
		v.u64[0]  = v.u64[0] << n;
	} else if (n >= 64 && n < 128) {
		v.u64[1] = v.u64[0];
		v.u64[0] = 0;
		v.u64[1] = v.u64[1] << (n - 64);
	}
#endif
	return v.m128i;
}

#define CASE_RANK(prefix, r) \
prefix ## _case(0x ## r ## 0) prefix ## _case(0x ## r ## 1) \
prefix ## _case(0x ## r ## 2) prefix ## _case(0x ## r ## 3) \
prefix ## _case(0x ## r ## 4) prefix ## _case(0x ## r ## 5) \
prefix ## _case(0x ## r ## 6) prefix ## _case(0x ## r ## 7) \
prefix ## _case(0x ## r ## 8) prefix ## _case(0x ## r ## 9) \
prefix ## _case(0x ## r ## a) prefix ## _case(0x ## r ## b) \
prefix ## _case(0x ## r ## c) prefix ## _case(0x ## r ## d) \
prefix ## _case(0x ## r ## e) prefix ## _case(0x ## r ## f)

#define _mm_srai_epi16_case(n) \
	case n:                                                     \
		v = __builtin_msa_srai_h(vreinterpret_v8i16(a), n); \
		break;
FORCE_INLINE __m128i _mm_srai_epi16(__m128i a, int imm8)
{
	v8i16 v;
	switch (imm8) {
	CASE_RANK(_mm_srai_epi16, 0)
	default:                  /* imm8 > 15 */
		v = __builtin_msa_srai_h(vreinterpret_v8i16(a), 0xf);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_srai_epi32_case(n) \
	case n:                                                     \
		v = __builtin_msa_srai_w(vreinterpret_v4i32(a), n); \
		break;
FORCE_INLINE __m128i _mm_srai_epi32(__m128i a, int imm8)
{
	v4i32 v;
	switch (imm8) {
	CASE_RANK(_mm_srai_epi32, 0)
	CASE_RANK(_mm_srai_epi32, 1)
	default:                  /* imm8 > 31 */
		v = __builtin_msa_srai_w(vreinterpret_v4i32(a), 0x1f);
		break;
	}
	return vreinterpret_m128i(v);
}

/* `_mm_srai_epi64`: AVX512 */
#define _mm_srai_epi64_case(n) \
	case n:                                                     \
		v = __builtin_msa_srai_d(vreinterpret_v2i64(a), n); \
		break;
FORCE_INLINE __m128i _mm_srai_epi64(__m128i a, int imm8)
{
	v2i64 v;
	switch (imm8) {
	CASE_RANK(_mm_srai_epi64, 0)
	CASE_RANK(_mm_srai_epi64, 1)
	CASE_RANK(_mm_srai_epi64, 2)
	CASE_RANK(_mm_srai_epi64, 3)
	default:                  /* imm8 > 63 */
		v = __builtin_msa_srai_d(vreinterpret_v2i64(a), 0x3f);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_slli_epi16_case(n) \
	case n:                                                     \
		v = __builtin_msa_slli_h(vreinterpret_v8i16(a), n); \
		break;
FORCE_INLINE __m128i _mm_slli_epi16(__m128i a, int imm8)
{
	v8i16 v;
	switch (imm8) {
	CASE_RANK(_mm_slli_epi16, 0)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_slli_epi32_case(n) \
	case n:                                                     \
		v = __builtin_msa_slli_w(vreinterpret_v4i32(a), n); \
		break;
FORCE_INLINE __m128i _mm_slli_epi32(__m128i a, int imm8)
{
	v4i32 v;
	switch (imm8) {
	CASE_RANK(_mm_slli_epi32, 0)
	CASE_RANK(_mm_slli_epi32, 1)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_slli_epi64_case(n) \
	case n:                                                     \
		v = __builtin_msa_slli_d(vreinterpret_v2i64(a), n); \
		break;
FORCE_INLINE __m128i _mm_slli_epi64(__m128i a, int imm8)
{
	v2i64 v;
	switch (imm8) {
	CASE_RANK(_mm_slli_epi64, 0)
	CASE_RANK(_mm_slli_epi64, 1)
	CASE_RANK(_mm_slli_epi64, 2)
	CASE_RANK(_mm_slli_epi64, 3)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_srli_epi16_case(n) \
	case n:                                                     \
		v = __builtin_msa_srli_h(vreinterpret_v8i16(a), n); \
		break;
FORCE_INLINE __m128i _mm_srli_epi16(__m128i a, int imm8)
{
	v8i16 v;
	switch (imm8) {
	CASE_RANK(_mm_srli_epi16, 0)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_srli_epi32_case(n) \
	case n:                                                     \
		v = __builtin_msa_srli_w(vreinterpret_v4i32(a), n); \
		break;
FORCE_INLINE __m128i _mm_srli_epi32(__m128i a, int imm8)
{
	v4i32 v;
	switch (imm8) {
	CASE_RANK(_mm_srli_epi32, 0)
	CASE_RANK(_mm_srli_epi32, 1)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

#define _mm_srli_epi64_case(n) \
	case n:                                                     \
		v = __builtin_msa_srli_d(vreinterpret_v2i64(a), n); \
		break;
FORCE_INLINE __m128i _mm_srli_epi64(__m128i a, int imm8)
{
	v2i64 v;
	switch (imm8) {
	CASE_RANK(_mm_srli_epi64, 0)
	CASE_RANK(_mm_srli_epi64, 1)
	CASE_RANK(_mm_srli_epi64, 2)
	CASE_RANK(_mm_srli_epi64, 3)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE __m128i _mm_sll_epi16(__m128i a, __m128i count)
{
	v8i16 v;
	VREG128 i = {.m128i = count};
	switch (i.i64[0]) {
	CASE_RANK(_mm_slli_epi16, 0)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE __m128i _mm_sll_epi32(__m128i a, __m128i count)
{
	v4i32 v;
	VREG128 i = {.m128i = count};
	switch (i.i64[0]) {
	CASE_RANK(_mm_slli_epi32, 0)
	CASE_RANK(_mm_slli_epi32, 1)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE __m128i _mm_sll_epi64(__m128i a, __m128i count)
{
	v2i64 v;
	VREG128 i = {.m128i = count};
	switch (i.i64[0]) {
	CASE_RANK(_mm_slli_epi64, 0)
	CASE_RANK(_mm_slli_epi64, 1)
	CASE_RANK(_mm_slli_epi64, 2)
	CASE_RANK(_mm_slli_epi64, 3)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE __m128i _mm_srl_epi16(__m128i a, __m128i count)
{
	v8i16 v;
	VREG128 i = {.m128i = count};
	switch (i.i64[0]) {
	CASE_RANK(_mm_srli_epi16, 0)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE __m128i _mm_srl_epi32(__m128i a, __m128i count)
{
	v4i32 v;
	VREG128 i = {.m128i = count};
	switch (i.i64[0]) {
	CASE_RANK(_mm_srli_epi32, 0)
	CASE_RANK(_mm_srli_epi32, 1)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE __m128i _mm_srl_epi64(__m128i a, __m128i count)
{
	v2i64 v;
	VREG128 i = {.m128i = count};
	switch (i.i64[0]) {
	CASE_RANK(_mm_srli_epi64, 0)
	CASE_RANK(_mm_srli_epi64, 1)
	CASE_RANK(_mm_srli_epi64, 2)
	CASE_RANK(_mm_srli_epi64, 3)
	default:
		return v_msa_setzero(__m128i);
		break;
	}
	return vreinterpret_m128i(v);
}

FORCE_INLINE int _mm_movemask_epi8(__m128i a)
{
	VREG128 v;
	v8u16 s = vreinterpret_v8u16(
		__builtin_msa_srli_b(vreinterpret_v16i8(a), 7));
	v4u32 p16 = vreinterpret_v4u32((vreinterpret_v8u16(
		__builtin_msa_srli_h(vreinterpret_v8i16(s), 7))) + s);
	v2u64 p32 = vreinterpret_v2u64((vreinterpret_v4u32(
		__builtin_msa_srli_w(vreinterpret_v4i32(p16), 14))) + p16);
	v.msa_v2u64 = (vreinterpret_v2u64(
		__builtin_msa_srli_d((v2i64)p32, 28))) + p32;
	return v.u8[0] | ((int)v.u8[8] << 8);
}

FORCE_INLINE int _mm_movemask_pi8(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v8u16 s = vreinterpret_v8u16(__builtin_msa_srli_b(v.msa_v16i8, 7));
	v4u32 p16 = vreinterpret_v4u32((vreinterpret_v8u16(
		__builtin_msa_srli_h(vreinterpret_v8i16(s), 7))) + s);
	v2u64 p32 = vreinterpret_v2u64((vreinterpret_v4u32(
		__builtin_msa_srli_w(vreinterpret_v4i32(p16), 14))) + p16);
	v.msa_v2u64 = (vreinterpret_v2u64(
		__builtin_msa_srli_d((v2i64)p32, 28))) + p32;
	return v.u8[0];
}

#define _m_pmovmskb(a) _mm_movemask_pi8(a)

FORCE_INLINE int _mm_movemask_ps(__m128 a)
{
	VREG128 v = {.m128 = a};
	v2u64 s = vreinterpret_v2u64(__builtin_msa_srli_w(v.msa_v4i32, 31));
	v.msa_v2u64 = (vreinterpret_v2u64(
		__builtin_msa_srli_d((v2i64)s, 31))) + s;
	return v.u8[0] | ((int)v.u8[8] << 2);
}

FORCE_INLINE int _mm_movemask_pd(__m128d a)
{
	VREG128 v = {.m128d = a};
	v.i32[0]  = v.i64[0] >> 63 ? 1 : 0;
	v.i32[0] |= v.i64[1] >> 63 ? 2 : 0;
	return v.i32[0];
}

FORCE_INLINE __m128i _mm_movpi64_epi64(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	return v.m128i;
}

FORCE_INLINE __m64 _mm_movepi64_pi64(__m128i a)
{
	VREG128 v = {.m128i = a};
	return v.m64[0];
}

FORCE_INLINE int _mm_test_all_ones(__m128i a)
{
	return (vreinterpret_nth_i64_m128i(a, 0) &
		vreinterpret_nth_i64_m128i(a, 1)) == ~(int64_t)0;
}

FORCE_INLINE int _mm_test_all_zeros(__m128i a, __m128i mask)
{
	VREG128 v = {
		.msa_v16u8 = __builtin_msa_and_v(
			vreinterpret_v16u8(a), vreinterpret_v16u8(mask))
	};
	return (v.i64[0] | v.i64[1]) ? 0 : 1;
}

FORCE_INLINE int _mm_testc_si128(__m128i a, __m128i b)
{
	VREG128 v = {
		.msa_v16u8 = __builtin_msa_and_v(
			~vreinterpret_v16u8(a), vreinterpret_v16u8(b))
	};
	return (v.i64[0] | v.i64[1]) ? 0 : 1;
}

FORCE_INLINE int _mm_testz_si128(__m128i a, __m128i b)
{
	VREG128 v = {
		.msa_v16u8 = __builtin_msa_and_v(
			vreinterpret_v16u8(a), vreinterpret_v16u8(b))
	};
	return (v.i64[0] | v.i64[1]) ? 0 : 1;
}

FORCE_INLINE int _mm_testnzc_si128(__m128i a, __m128i b)
{
	VREG128 vz = {
		.msa_v16u8 = __builtin_msa_and_v(
			vreinterpret_v16u8(a), vreinterpret_v16u8(b))
	};
	VREG128 vc = {
		.msa_v16u8 = __builtin_msa_and_v(
			~vreinterpret_v16u8(a), vreinterpret_v16u8(b))
	};
	return (vz.i64[0] | vz.i64[1]) && (vc.i64[0] | vc.i64[1]) ? 1 : 0;
}

FORCE_INLINE int _mm_test_mix_ones_zeros(__m128i a, __m128i mask)
{
	VREG128 vz = {
		.msa_v16u8 = __builtin_msa_and_v(
			vreinterpret_v16u8(a), vreinterpret_v16u8(mask))
	};
	VREG128 vc = {
		.msa_v16u8 = __builtin_msa_and_v(
			~vreinterpret_v16u8(a), vreinterpret_v16u8(mask))
	};
	return (vz.i64[0] | vz.i64[1]) && (vc.i64[0] | vc.i64[1]) ? 1 : 0;
}

FORCE_INLINE __m128 _mm_sub_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fsub_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_sub_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_sub_ps(a, b));
}

FORCE_INLINE __m128d _mm_sub_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fsub_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_sub_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_sub_pd(a, b));
}

FORCE_INLINE __m128i _mm_sub_epi64(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subv_d(
		vreinterpret_v2i64(a), vreinterpret_v2i64(b)));
}

FORCE_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subv_w(
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

FORCE_INLINE __m128i _mm_sub_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subv_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m128i _mm_sub_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subv_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m64 _mm_sub_si64(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.i64[0] -= v.i64[1]; 
	return v.m64[0];
}

FORCE_INLINE __m128i _mm_subs_epu16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subs_u_h(
		vreinterpret_v8u16(a), vreinterpret_v8u16(b)));
}

FORCE_INLINE __m128i _mm_subs_epu8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subs_u_b(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128i _mm_subs_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subs_s_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128i _mm_subs_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_subs_s_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m128i _mm_adds_epu16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_adds_u_h(
		vreinterpret_v8u16(a), vreinterpret_v8u16(b)));
}

FORCE_INLINE __m128i _mm_adds_epu8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_adds_u_b(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128i _mm_adds_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_adds_s_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128i _mm_adds_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_adds_s_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m64 _mm_avg_pu16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v  = {
		.msa_v8u16 = __builtin_msa_aver_u_h(
			va.msa_v8u16, vb.msa_v8u16)
	};
	return v.m64[0];
}

#define _m_pavgw(a, b) _mm_avg_pu16(a, b)

FORCE_INLINE __m64 _mm_avg_pu8(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v  = {
		.msa_v16u8 = __builtin_msa_aver_u_b(
			va.msa_v16u8, vb.msa_v16u8)
	};
	return v.m64[0];
}

#define _m_pavgb(a, b) _mm_avg_pu8(a, b)

FORCE_INLINE __m128i _mm_avg_epu16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_aver_u_h(
		vreinterpret_v8u16(a), vreinterpret_v8u16(b)));
}

FORCE_INLINE __m128i _mm_avg_epu8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_aver_u_b(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fadd_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128d _mm_add_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fadd_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_add_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_add_pd(a, b));
}

FORCE_INLINE __m64 _mm_add_si64(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.i64[0] += v.i64[1]; 
	return v.m64[0];
}

FORCE_INLINE __m128 _mm_add_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_add_ps(a, b));
}

FORCE_INLINE __m128i _mm_add_epi64(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_addv_d(
		vreinterpret_v2i64(a), vreinterpret_v2i64(b)));
}

FORCE_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_addv_w(
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

FORCE_INLINE __m128i _mm_add_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_addv_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m128i _mm_add_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_addv_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128 _mm_hadd_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(vreinterpret_v4f32(
		__builtin_msa_vshf_w(({v4i32 m = {0, 2, 4, 6}; m;}),
		vreinterpret_v4i32(b),vreinterpret_v4i32(a))) +
		vreinterpret_v4f32(
		__builtin_msa_vshf_w(({v4i32 m = {1, 3, 5, 7}; m;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a))));
}

FORCE_INLINE __m128d _mm_hadd_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(vreinterpret_v2f64(
		__builtin_msa_vshf_d(({v2i64 m = {0, 2}; m;}),
		vreinterpret_v2i64(b), vreinterpret_v2i64(a))) +
		vreinterpret_v2f64(
		__builtin_msa_vshf_d(({v2i64 m = {1, 3}; m;}),
		vreinterpret_v2i64(b), vreinterpret_v2i64(a))));
}

FORCE_INLINE __m128i _mm_hadd_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_vshf_h(
		({v8i16 m = {0, 2, 4, 6, 8, 10, 12, 14}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)) +
		__builtin_msa_vshf_h(
		({v8i16 m = {1, 3, 5, 7, 9, 11, 13, 15}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128i _mm_hadd_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_vshf_w(({v4i32 m = {0, 2, 4, 6}; m;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)) +
		__builtin_msa_vshf_w(({v4i32 m = {1, 3, 5, 7}; m;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m64 _mm_hadd_pi16(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.m128i = _mm_hadd_epi16(v.m128i, v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_hadd_pi32(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.m128i = _mm_hadd_epi32(v.m128i, v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m128i _mm_hadds_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_adds_s_h(__builtin_msa_vshf_h(
		({v8i16 m = {0, 2, 4, 6, 8, 10, 12, 14}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)),
		__builtin_msa_vshf_h(
		({v8i16 m = {1, 3, 5, 7, 9, 11, 13, 15}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a))));
}

FORCE_INLINE __m64 _mm_hadds_pi16(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.m128i = _mm_hadds_epi16(v.m128i, v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m128 _mm_hsub_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(vreinterpret_v4f32(
		__builtin_msa_vshf_w(({v4i32 m = {0, 2, 4, 6}; m;}),
		vreinterpret_v4i32(b),vreinterpret_v4i32(a))) -
		vreinterpret_v4f32(
		__builtin_msa_vshf_w(({v4i32 m = {1, 3, 5, 7}; m;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a))));
}

FORCE_INLINE __m128d _mm_hsub_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(vreinterpret_v2f64(
		__builtin_msa_vshf_d(({v2i64 m = {0, 2}; m;}),
		vreinterpret_v2i64(b), vreinterpret_v2i64(a))) -
		vreinterpret_v2f64(
		__builtin_msa_vshf_d(({v2i64 m = {1, 3}; m;}),
		vreinterpret_v2i64(b), vreinterpret_v2i64(a))));
}

FORCE_INLINE __m128i _mm_hsub_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_vshf_h(
		({v8i16 m = {0, 2, 4, 6, 8, 10, 12, 14}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)) -
		__builtin_msa_vshf_h(
		({v8i16 m = {1, 3, 5, 7, 9, 11, 13, 15}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128i _mm_hsub_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_vshf_w(({v4i32 m = {0, 2, 4, 6}; m;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)) -
		__builtin_msa_vshf_w(({v4i32 m = {1, 3, 5, 7}; m;}),
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m64 _mm_hsub_pi16(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.m128i = _mm_hsub_epi16(v.m128i, v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_hsub_pi32(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.m128i = _mm_hsub_epi32(v.m128i, v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m128i _mm_hsubs_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_subs_s_h(__builtin_msa_vshf_h(
		({v8i16 m = {0, 2, 4, 6, 8, 10, 12, 14}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)),
		__builtin_msa_vshf_h(
		({v8i16 m = {1, 3, 5, 7, 9, 11, 13, 15}; m;}),
		vreinterpret_v8i16(b), vreinterpret_v8i16(a))));
}

FORCE_INLINE __m64 _mm_hsubs_pi16(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.m128i = _mm_hsubs_epi16(v.m128i, v.m128i);
	return v.m64[0];
}

FORCE_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fmul_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_mul_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_mul_ps(a, b));
}

FORCE_INLINE __m128d _mm_mul_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fmul_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_mul_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_mul_pd(a, b));
}

FORCE_INLINE __m128i _mm_mul_epu32(__m128i a, __m128i b)
{
	VREG128 v = {
		.u64 = {
			(uint64_t)vreinterpret_nth_u32_m128i(a, 0) *
			(uint64_t)vreinterpret_nth_u32_m128i(b, 0),
			(uint64_t)vreinterpret_nth_u32_m128i(a, 2) *
			(uint64_t)vreinterpret_nth_u32_m128i(b, 2)
		}
	};
	return v.m128i;
}

FORCE_INLINE __m64 _mm_mul_su32(__m64 a, __m64 b)
{
	uint32_t *pa = (uint32_t*)&(a);
	uint32_t *pb = (uint32_t*)&(b);
	uint64_t r = (uint64_t)pa[0] * (uint64_t)pb[0];
	return vreinterpret_m64(r);
}

FORCE_INLINE __m128i _mm_mul_epi32(__m128i a, __m128i b)
{
	VREG128 v = {
		.i64 = {
			(int64_t)vreinterpret_nth_i32_m128i(a, 0) *
			(int64_t)vreinterpret_nth_i32_m128i(b, 0),
			(int64_t)vreinterpret_nth_i32_m128i(a, 2) *
			(int64_t)vreinterpret_nth_i32_m128i(b, 2)
		}
	};
	return v.m128i;
}

FORCE_INLINE __m128i _mm_mullo_epi16(__m128i a, __m128i b)
{
	v8i16 z = v_msa_setzero(v8i16);
	v4i32 eao = __builtin_msa_hadd_s_w(vreinterpret_v8i16(a), z);
	v4i32 eae = __builtin_msa_hadd_s_w(z, vreinterpret_v8i16(a));
	v4i32 ebo = __builtin_msa_hadd_s_w(vreinterpret_v8i16(b), z);
	v4i32 ebe = __builtin_msa_hadd_s_w(z, vreinterpret_v8i16(b));
	v4i32 vo = __builtin_msa_mulv_w(eao, ebo);
	v4i32 ve = __builtin_msa_mulv_w(eae, ebe);
	return vreinterpret_m128i(__builtin_msa_ilvev_h(
		vreinterpret_v8i16(vo), vreinterpret_v8i16(ve)));
}

FORCE_INLINE __m128i _mm_mullo_epi32(__m128i a, __m128i b)
{
	v4i32 z = v_msa_setzero(v4i32);
	v2i64 eao = __builtin_msa_hadd_s_d(vreinterpret_v4i32(a), z);
	v2i64 eae = __builtin_msa_hadd_s_d(z, vreinterpret_v4i32(a));
	v2i64 ebo = __builtin_msa_hadd_s_d(vreinterpret_v4i32(b), z);
	v2i64 ebe = __builtin_msa_hadd_s_d(z, vreinterpret_v4i32(b));
	v2i64 vo = __builtin_msa_mulv_d(eao, ebo);
	v2i64 ve = __builtin_msa_mulv_d(eae, ebe);
	return vreinterpret_m128i(__builtin_msa_ilvev_w(
		vreinterpret_v4i32(vo), vreinterpret_v4i32(ve)));
}

FORCE_INLINE __m128i _mm_mulhi_epi16(__m128i a, __m128i b)
{
	v8i16 z = v_msa_setzero(v8i16);
	v4i32 eao = __builtin_msa_hadd_s_w(vreinterpret_v8i16(a), z);
	v4i32 eae = __builtin_msa_hadd_s_w(z, vreinterpret_v8i16(a));
	v4i32 ebo = __builtin_msa_hadd_s_w(vreinterpret_v8i16(b), z);
	v4i32 ebe = __builtin_msa_hadd_s_w(z, vreinterpret_v8i16(b));
	v4i32 vo = __builtin_msa_mulv_w(eao, ebo);
	v4i32 ve = __builtin_msa_mulv_w(eae, ebe);
	return vreinterpret_m128i(__builtin_msa_ilvod_h(
		vreinterpret_v8i16(vo), vreinterpret_v8i16(ve)));
}

FORCE_INLINE __m128i _mm_mulhi_epu16(__m128i a, __m128i b)
{
	v8u16 z = v_msa_setzero(v8u16);
	v4u32 eao = __builtin_msa_hadd_u_w(vreinterpret_v8u16(a), z);
	v4u32 eae = __builtin_msa_hadd_u_w(z, vreinterpret_v8u16(a));
	v4u32 ebo = __builtin_msa_hadd_u_w(vreinterpret_v8u16(b), z);
	v4u32 ebe = __builtin_msa_hadd_u_w(z, vreinterpret_v8u16(b));
	v4i32 vo = __builtin_msa_mulv_w((v4i32)eao, (v4i32)ebo);
	v4i32 ve = __builtin_msa_mulv_w((v4i32)eae, (v4i32)ebe);
	return vreinterpret_m128i(__builtin_msa_ilvod_h(
		vreinterpret_v8i16(vo), vreinterpret_v8i16(ve)));
}

FORCE_INLINE __m64 _mm_mullo_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v = {
		.m128i = _mm_mullo_epi16(va.m128i, vb.m128i)
	};
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_mulhi_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v = {
		.m128i = _mm_mulhi_epi16(va.m128i, vb.m128i)
	};
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_mulhi_pu16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v = {
		.m128i = _mm_mulhi_epu16(va.m128i, vb.m128i)
	};
	return v.m64[0];
}

#define _m_pmulhuw(a, b) _mm_mulhi_pu16(a, b)

FORCE_INLINE __m128i _mm_mulhrs_epi16(__m128i a, __m128i b)
{
	v8i16 z = v_msa_setzero(v8i16);
	v4i32 eao = __builtin_msa_hadd_s_w(vreinterpret_v8i16(a), z);
	v4i32 eae = __builtin_msa_hadd_s_w(z, vreinterpret_v8i16(a));
	v4i32 ebo = __builtin_msa_hadd_s_w(vreinterpret_v8i16(b), z);
	v4i32 ebe = __builtin_msa_hadd_s_w(z, vreinterpret_v8i16(b));
	v4i32 vo = __builtin_msa_srai_w(__builtin_msa_mulv_w(eao, ebo), 14) + 1;
	v4i32 ve = __builtin_msa_srai_w(__builtin_msa_mulv_w(eae, ebe), 14) + 1;
	vo = __builtin_msa_srli_w(vo, 1);
	ve = __builtin_msa_srli_w(ve, 1);
	return vreinterpret_m128i(__builtin_msa_ilvev_h(
		vreinterpret_v8i16(vo), vreinterpret_v8i16(ve)));
}

FORCE_INLINE __m64 _mm_mulhrs_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v = {
		.m128i = _mm_mulhrs_epi16(va.m128i, vb.m128i)
	};
	return v.m64[0];
}

FORCE_INLINE __m128i _mm_maddubs_epi16(__m128i a, __m128i b)
{
	v16u8 z = v_msa_setzero(v16u8);
	v16i8 i = v_msa_setzero(v16i8);
	v8u16 eao = __builtin_msa_hadd_u_h(vreinterpret_v16u8(a), z);
	v8u16 eae = __builtin_msa_hadd_u_h(z, vreinterpret_v16u8(a));
	v8i16 ebo = __builtin_msa_hadd_s_h(vreinterpret_v16i8(b), i);
	v8i16 ebe = __builtin_msa_hadd_s_h(i, vreinterpret_v16i8(b));
	v8i16 vo = __builtin_msa_mulv_h((v8i16)eao, ebo);
	v8i16 ve = __builtin_msa_mulv_h((v8i16)eae, ebe);
	return vreinterpret_m128i(__builtin_msa_adds_s_h(vo, ve));
}

FORCE_INLINE __m64 _mm_maddubs_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	VREG128 v = {
		.m128i = _mm_maddubs_epi16(va.m128i, vb.m128i)
	};
	return v.m64[0];
}

FORCE_INLINE __m128i _mm_madd_epi16(__m128i a, __m128i b)
{
	VREG128 v = {
		.i32 = {
			(int32_t)vreinterpret_nth_i16_m128i(a, 0) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 0) +
			(int32_t)vreinterpret_nth_i16_m128i(a, 1) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 1),
			(int32_t)vreinterpret_nth_i16_m128i(a, 2) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 2) +
			(int32_t)vreinterpret_nth_i16_m128i(a, 3) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 3),
			(int32_t)vreinterpret_nth_i16_m128i(a, 4) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 4) +
			(int32_t)vreinterpret_nth_i16_m128i(a, 5) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 5),
			(int32_t)vreinterpret_nth_i16_m128i(a, 6) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 6) +
			(int32_t)vreinterpret_nth_i16_m128i(a, 7) *
			(int32_t)vreinterpret_nth_i16_m128i(b, 7),
		}
	};
	return v.m128i;
}

FORCE_INLINE __m128 _mm_addsub_ps(__m128 a, __m128 b)
{
	v4f32 mask = {-1.0f, 1.0f, -1.0f, 1.0f};
	return vreinterpret_m128(__builtin_msa_fmadd_w(
		vreinterpret_v4f32(b), mask, vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128d _mm_addsub_pd(__m128d a, __m128d b)
{
	v2f64 mask = {-1.0f, 1.0f};
	return vreinterpret_m128d(__builtin_msa_fmadd_d(
		vreinterpret_v2f64(b), mask, vreinterpret_v2f64(a)));
}

FORCE_INLINE __m128 _mm_div_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fdiv_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_div_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_div_ps(a, b));
}

FORCE_INLINE __m128d _mm_div_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fdiv_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_div_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_div_pd(a, b));
}

FORCE_INLINE __m128 _mm_rcp_ps(__m128 a)
{
	return vreinterpret_m128(
		__builtin_msa_frcp_w(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128 _mm_rcp_ss(__m128 a)
{
	return _mm_move_ss(a, _mm_rcp_ps(a));
}

FORCE_INLINE __m128 _mm_sqrt_ps(__m128 a)
{
	return vreinterpret_m128(
		__builtin_msa_fsqrt_w(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128d _mm_sqrt_pd(__m128d a)
{
	return vreinterpret_m128d(
		__builtin_msa_fsqrt_d(vreinterpret_v2f64(a)));
}

FORCE_INLINE __m128d _mm_sqrt_sd(__m128d a, __m128d b)
{
    return _mm_move_sd(a, _mm_sqrt_pd(b));
}

FORCE_INLINE __m128 _mm_sqrt_ss(__m128 a)
{
	return _mm_move_ss(a, _mm_sqrt_ps(a));
}

FORCE_INLINE __m128 _mm_rsqrt_ps(__m128 a)
{
	return vreinterpret_m128(
		__builtin_msa_frsqrt_w(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128 _mm_rsqrt_ss(__m128 a)
{
	return _mm_move_ss(a, _mm_rsqrt_ps(a));
}

int _mm_popcnt_u32(unsigned int a)
{
	VREG128 v = {.u32 = {a, 0, 0, 0}};
	v.msa_v4i32 = __builtin_msa_pcnt_w(v.msa_v4i32);
	return v.i32[0];
}

FORCE_INLINE int64_t _mm_popcnt_u64(uint64_t a)
{
	VREG128 v = {.u64 = {a, 0}};
	v.msa_v2i64 = __builtin_msa_pcnt_d(v.msa_v2i64);
	return v.i64[0];
}

FORCE_INLINE __m128 _mm_max_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fmax_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128d _mm_max_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fmax_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128 _mm_max_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_max_ps(a, b));
}

FORCE_INLINE __m128d _mm_max_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_max_pd(a, b));
}

FORCE_INLINE __m128 _mm_min_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fmin_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128d _mm_min_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fmin_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128 _mm_min_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_min_ps(a, b));
}

FORCE_INLINE __m128d _mm_min_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_min_pd(a, b));
}

FORCE_INLINE __m128i _mm_max_epu8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_max_u_b(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m64 _mm_max_pu8(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.msa_v16u8 = __builtin_msa_max_u_b(va.msa_v16u8, vb.msa_v16u8);
	return va.m64[0];
}

#define _m_pmaxub(a, b) _mm_max_pu8(a, b)

FORCE_INLINE __m128i _mm_min_epu8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_min_u_b(
		vreinterpret_v16u8(a), vreinterpret_v16u8(b)));
}

FORCE_INLINE __m64 _mm_min_pu8(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.msa_v16u8 = __builtin_msa_min_u_b(va.msa_v16u8, vb.msa_v16u8);
	return va.m64[0];
}

#define _m_pminub(a, b) _mm_min_pu8(a, b)

FORCE_INLINE __m128i _mm_max_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_max_s_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128i _mm_min_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_min_s_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128i _mm_max_epu16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_max_u_h(
		vreinterpret_v8u16(a), vreinterpret_v8u16(b)));
}

FORCE_INLINE __m128i _mm_min_epu16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_min_u_h(
		vreinterpret_v8u16(a), vreinterpret_v8u16(b)));
}

FORCE_INLINE __m128i _mm_max_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_max_s_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m64 _mm_max_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.msa_v8i16 = __builtin_msa_max_s_h(va.msa_v8i16, vb.msa_v8i16);
	return va.m64[0];
}

#define _m_pmaxsw(a, b) _mm_max_pi16(a, b)

FORCE_INLINE __m128i _mm_min_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_min_s_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m64 _mm_min_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.msa_v8i16 = __builtin_msa_min_s_h(va.msa_v8i16, vb.msa_v8i16);
	return va.m64[0];
}

#define _m_pminsw(a, b) _mm_min_pi16(a, b)

FORCE_INLINE __m128i _mm_max_epu32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_max_u_w(
		vreinterpret_v4u32(a), vreinterpret_v4u32(b)));
}

FORCE_INLINE __m128i _mm_min_epu32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_min_u_w(
		vreinterpret_v4u32(a), vreinterpret_v4u32(b)));
}

FORCE_INLINE __m128i _mm_max_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_max_s_w(
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

FORCE_INLINE __m128i _mm_min_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_min_s_w(
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

/*
 * Kahan summation algorithm
 * https://en.wikipedia.org/wiki/Kahan_summation_algorithm
 */
FORCE_INLINE void sse2msa_kadd_f32(float *sum, float *c, float y)
{
    y -= *c;
    float t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

FORCE_INLINE float sse2msa_vadd_v4f32(v4f32 v)
{
	float s = 0, c = 0;
	float *p = (float*)&v;
	sse2msa_kadd_f32(&s, &c, p[0]);
	sse2msa_kadd_f32(&s, &c, p[1]);
	sse2msa_kadd_f32(&s, &c, p[2]);
	sse2msa_kadd_f32(&s, &c, p[3]);
	return s + c;
}

FORCE_INLINE __m128 _mm_dp_ps(__m128 a, __m128 b, const int imm8)
{
	if (imm8 == 0xff) {
		v4f32 v = _mm_mul_ps(a, b);
		return _mm_set1_ps(sse2msa_vadd_v4f32(v));
	}
	if (imm8 == 0x7f) {
		v4f32 v = _mm_mul_ps(a, b);
		v[3] = 0;
		return _mm_set1_ps(sse2msa_vadd_v4f32(v));
	}
	float s = 0, c = 0;
	v4f32 f32a = vreinterpret_v4f32(a);
	v4f32 f32b = vreinterpret_v4f32(b);
	if (imm8 & (1 << 4))
		sse2msa_kadd_f32(&s, &c, f32a[0] * f32b[0]);
	if (imm8 & (1 << 5))
		sse2msa_kadd_f32(&s, &c, f32a[1] * f32b[1]);
	if (imm8 & (1 << 6))
		sse2msa_kadd_f32(&s, &c, f32a[2] * f32b[2]);
	if (imm8 & (1 << 7))
		sse2msa_kadd_f32(&s, &c, f32a[3] * f32b[3]);
	s += c;
	v4f32 res = {
		(imm8 & 0x1) ? s : 0,
		(imm8 & 0x2) ? s : 0,
		(imm8 & 0x4) ? s : 0,
		(imm8 & 0x8) ? s : 0
	};
	return vreinterpret_m128(res);
}

FORCE_INLINE __m128 _mm_cmplt_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fclt_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_cmplt_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmplt_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmplt_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fclt_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_cmplt_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmplt_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fclt_w(
		vreinterpret_v4f32(b), vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128 _mm_cmpgt_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmpgt_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmpgt_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fclt_d(
		vreinterpret_v2f64(b), vreinterpret_v2f64(a)));
}

FORCE_INLINE __m128d _mm_cmpgt_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmpgt_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmpge_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fcle_w(
		vreinterpret_v4f32(b), vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128 _mm_cmpge_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmpge_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmpge_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fcle_d(
		vreinterpret_v2f64(b), vreinterpret_v2f64(a)));
}

FORCE_INLINE __m128d _mm_cmpge_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmpge_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmple_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fcle_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_cmple_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmple_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmple_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fcle_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_cmple_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmple_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmpeq_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fceq_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_cmpeq_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmpeq_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmpeq_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fceq_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_cmpeq_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmpeq_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmpneq_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fcne_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_cmpneq_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmpneq_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmpneq_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fcne_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_cmpneq_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmpneq_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmpnge_ps(__m128 a, __m128 b)
{
	return _mm_cmplt_ps(a, b);
}

FORCE_INLINE __m128 _mm_cmpnge_ss(__m128 a, __m128 b)
{
	return _mm_cmplt_ss(a, b);
}

FORCE_INLINE __m128d _mm_cmpnge_pd(__m128d a, __m128d b)
{
	return _mm_cmplt_pd(a, b);
}

FORCE_INLINE __m128d _mm_cmpnge_sd(__m128d a, __m128d b)
{
	return _mm_cmplt_sd(a, b);
}

FORCE_INLINE __m128 _mm_cmpngt_ps(__m128 a, __m128 b)
{
	return _mm_cmple_ps(a, b);
}

FORCE_INLINE __m128 _mm_cmpngt_ss(__m128 a, __m128 b)
{
	return _mm_cmple_ss(a, b);
}

FORCE_INLINE __m128d _mm_cmpngt_pd(__m128d a, __m128d b)
{
	return _mm_cmple_pd(a, b);
}

FORCE_INLINE __m128d _mm_cmpngt_sd(__m128d a, __m128d b)
{
	return _mm_cmple_sd(a, b);
}

FORCE_INLINE __m128 _mm_cmpnle_ps(__m128 a, __m128 b)
{
	return _mm_cmpgt_ps(a, b);
}

FORCE_INLINE __m128 _mm_cmpnle_ss(__m128 a, __m128 b)
{
	return _mm_cmpgt_ss(a, b);
}

FORCE_INLINE __m128d _mm_cmpnle_pd(__m128d a, __m128d b)
{
	return _mm_cmpgt_pd(a, b);
}

FORCE_INLINE __m128d _mm_cmpnle_sd(__m128d a, __m128d b)
{
	return _mm_cmpgt_sd(a, b);
}

FORCE_INLINE __m128 _mm_cmpnlt_ps(__m128 a, __m128 b)
{
	return _mm_cmpge_ps(a, b);
}

FORCE_INLINE __m128 _mm_cmpnlt_ss(__m128 a, __m128 b)
{
	return _mm_cmpge_ss(a, b);
}

FORCE_INLINE __m128d _mm_cmpnlt_pd(__m128d a, __m128d b)
{
	return _mm_cmpge_pd(a, b);
}

FORCE_INLINE __m128d _mm_cmpnlt_sd(__m128d a, __m128d b)
{
	return _mm_cmpge_sd(a, b);
}

FORCE_INLINE __m128i _mm_cmpeq_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_ceq_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128i _mm_cmpeq_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_ceq_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_ceq_w(
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

FORCE_INLINE __m128i _mm_cmpeq_epi64(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_ceq_d(
		vreinterpret_v2i64(a), vreinterpret_v2i64(b)));
}

FORCE_INLINE __m128i _mm_cmplt_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b)));
}

FORCE_INLINE __m128i _mm_cmpgt_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_b(
		vreinterpret_v16i8(b), vreinterpret_v16i8(a)));
}

FORCE_INLINE __m128i _mm_cmplt_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_h(
		vreinterpret_v8i16(a), vreinterpret_v8i16(b)));
}

FORCE_INLINE __m128i _mm_cmpgt_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_h(
		vreinterpret_v8i16(b), vreinterpret_v8i16(a)));
}

FORCE_INLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_w(
		vreinterpret_v4i32(a), vreinterpret_v4i32(b)));
}

FORCE_INLINE __m128i _mm_cmpgt_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_w(
		vreinterpret_v4i32(b), vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128i _mm_cmpgt_epi64(__m128i a, __m128i b)
{
	return vreinterpret_m128i(__builtin_msa_clt_s_d(
		vreinterpret_v2i64(b), vreinterpret_v2i64(a)));
}

FORCE_INLINE __m128 _mm_cmpord_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(~__builtin_msa_fcun_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_cmpord_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmpord_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmpord_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(~__builtin_msa_fcun_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_cmpord_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmpord_pd(a, b));
}

FORCE_INLINE __m128 _mm_cmpunord_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(__builtin_msa_fcun_w(
		vreinterpret_v4f32(a), vreinterpret_v4f32(b)));
}

FORCE_INLINE __m128 _mm_cmpunord_ss(__m128 a, __m128 b)
{
	return _mm_move_ss(a, _mm_cmpunord_ps(a, b));
}

FORCE_INLINE __m128d _mm_cmpunord_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(__builtin_msa_fcun_d(
		vreinterpret_v2f64(a), vreinterpret_v2f64(b)));
}

FORCE_INLINE __m128d _mm_cmpunord_sd(__m128d a, __m128d b)
{
	return _mm_move_sd(a, _mm_cmpunord_pd(a, b));
}

FORCE_INLINE int _mm_comilt_ss(__m128 a, __m128 b)
{
	return vreinterpret_nth_f32_m128(a, 0) <
		vreinterpret_nth_f32_m128(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comilt_sd(__m128d a, __m128d b)
{
	return vreinterpret_nth_f64_m128d(a, 0) <
		vreinterpret_nth_f64_m128d(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comigt_ss(__m128 a, __m128 b)
{
	return vreinterpret_nth_f32_m128(a, 0) >
		vreinterpret_nth_f32_m128(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comigt_sd(__m128d a, __m128d b)
{
	return vreinterpret_nth_f64_m128d(a, 0) >
		vreinterpret_nth_f64_m128d(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comile_ss(__m128 a, __m128 b)
{
	return vreinterpret_nth_f32_m128(a, 0) <=
		vreinterpret_nth_f32_m128(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comile_sd(__m128d a, __m128d b)
{
	return vreinterpret_nth_f64_m128d(a, 0) <=
		vreinterpret_nth_f64_m128d(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comige_ss(__m128 a, __m128 b)
{
	return vreinterpret_nth_f32_m128(a, 0) >=
		vreinterpret_nth_f32_m128(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comige_sd(__m128d a, __m128d b)
{
	return vreinterpret_nth_f64_m128d(a, 0) >=
		vreinterpret_nth_f64_m128d(b, 0) ? 1 : 0;
}


FORCE_INLINE int _mm_comieq_ss(__m128 a, __m128 b)
{
	return vreinterpret_nth_f32_m128(a, 0) ==
		vreinterpret_nth_f32_m128(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comieq_sd(__m128d a, __m128d b)
{
	return vreinterpret_nth_f64_m128d(a, 0) ==
		vreinterpret_nth_f64_m128d(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comineq_ss(__m128 a, __m128 b)
{
	return vreinterpret_nth_f32_m128(a, 0) !=
		vreinterpret_nth_f32_m128(b, 0) ? 1 : 0;
}

FORCE_INLINE int _mm_comineq_sd(__m128d a, __m128d b)
{
	return vreinterpret_nth_f64_m128d(a, 0) !=
		vreinterpret_nth_f64_m128d(b, 0) ? 1 : 0;
}

/* 
 * according to the documentation, these intrinsics behave the same as the
 * non-'u' versions.  We'll just alias them here.
 */
#define _mm_ucomilt_ss _mm_comilt_ss
#define _mm_ucomile_ss _mm_comile_ss
#define _mm_ucomigt_ss _mm_comigt_ss
#define _mm_ucomige_ss _mm_comige_ss
#define _mm_ucomieq_ss _mm_comieq_ss
#define _mm_ucomineq_ss _mm_comineq_ss
#define _mm_ucomieq_sd _mm_comieq_sd
#define _mm_ucomige_sd _mm_comige_sd
#define _mm_ucomigt_sd _mm_comigt_sd
#define _mm_ucomile_sd _mm_comile_sd
#define _mm_ucomilt_sd _mm_comilt_sd
#define _mm_ucomineq_sd _mm_comineq_sd

FORCE_INLINE __m128 _mm_round_ps(__m128 a, int rounding)
{
	switch (rounding) {
	case (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x0 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x3);
		return vreinterpret_m128(
			__builtin_msa_frint_w(vreinterpret_v4f32(a)));
	case (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x1 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x2);
		return vreinterpret_m128(
			__builtin_msa_frint_w(vreinterpret_v4f32(a)));
	case (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
		return vreinterpret_m128(
			__builtin_msa_frint_w(vreinterpret_v4f32(a)));
	case (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
			(__builtin_msa_cfcmsa(1) | 0x3));
		return vreinterpret_m128(
			__builtin_msa_frint_w(vreinterpret_v4f32(a)));
	default:
		return vreinterpret_m128(
			__builtin_msa_frint_w(vreinterpret_v4f32(a)));
	}
}

FORCE_INLINE __m128 _mm_round_ss(__m128 a, __m128 b, int rounding)
{
    return _mm_move_ss(a, _mm_round_ps(b, rounding));
}

FORCE_INLINE __m128d _mm_round_pd(__m128d a, int rounding)
{
	switch (rounding) {
	case (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x0 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x3);
		return vreinterpret_m128d(
			__builtin_msa_frint_d(vreinterpret_v2f64(a)));
	case (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x1 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x2);
		return vreinterpret_m128d(
			__builtin_msa_frint_d(vreinterpret_v2f64(a)));
	case (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
		return vreinterpret_m128d(
			__builtin_msa_frint_d(vreinterpret_v2f64(a)));
	case (_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC):
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
			(__builtin_msa_cfcmsa(1) | 0x3));
		return vreinterpret_m128d(
			__builtin_msa_frint_d(vreinterpret_v2f64(a)));
	default:
		return vreinterpret_m128d(
			__builtin_msa_frint_d(vreinterpret_v2f64(a)));
	}
}

FORCE_INLINE __m128d _mm_round_sd(__m128d a, __m128d b, int rounding)
{
	return _mm_move_sd(a, _mm_round_pd(b, rounding));
}

FORCE_INLINE void _MM_SET_ROUNDING_MODE(unsigned int a)
{
	switch (a) {
	case _MM_ROUND_NEAREST:
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x0 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x3);
		break;
	case _MM_ROUND_TOWARD_ZERO:
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x1 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x2);
		break;
	case _MM_ROUND_UP:
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
			(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
		break;
	case _MM_ROUND_DOWN:
		__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
			(__builtin_msa_cfcmsa(1) | 0x3));
		break;
	}
}

FORCE_INLINE unsigned int _MM_GET_ROUNDING_MODE()
{
	int m = __builtin_msa_cfcmsa(1) & 0x3; /* MSACSR[1:0] */
	switch (m) {
	case 0: return _MM_ROUND_NEAREST;
	case 1: return _MM_ROUND_TOWARD_ZERO;
	case 2: return _MM_ROUND_UP;
	case 3: return _MM_ROUND_DOWN;
	}
	return m;
}

FORCE_INLINE float _mm_cvtss_f32(__m128 a)
{
	return vreinterpret_nth_f32_m128(a, 0);
}

FORCE_INLINE int _mm_cvtss_si32(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftint_s_w(vreinterpret_v4f32(a))
	};
	return v.i32[0];
}

FORCE_INLINE int64_t _mm_cvtss_si64(__m128 a)
{
	VREG128 v = {
		.msa_v2f64 = __builtin_msa_fexupr_d(vreinterpret_v4f32(a))
	};
	v.msa_v2i64 = __builtin_msa_ftint_s_d(v.msa_v2f64);
	return v.i64[0];
}

FORCE_INLINE int64_t _mm_cvtsd_si64(__m128d a)
{
	VREG128 v = {
		.msa_v2i64 = __builtin_msa_ftint_s_d(vreinterpret_v2f64(a))
	};
	return v.i64[0];
}

FORCE_INLINE __m128 _mm_cvt_pi2ps(__m128 a, __m64 b)
{
	VREG128 vb = {.m64 = {b, {0}}};
	vb.msa_v4f32 = __builtin_msa_ffint_s_w(vb.msa_v4i32);
	VREG128 *v = (VREG128*)&a;
	v->m64[0] = vb.m64[0];
	return v->m128;
}

FORCE_INLINE __m64 _mm_cvt_ps2pi(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftint_s_w(vreinterpret_v4f32(a))
	};
	return v.m64[0];
}

FORCE_INLINE __m128 _mm_cvt_si2ss(__m128 a, int b)
{
	float f = (float)b;
	return vreinterpret_m128(__builtin_msa_insert_w(
		vreinterpret_v4i32(a), 0, reinterpret_i32(f)));
}

FORCE_INLINE int _mm_cvt_ss2si(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftint_s_w(vreinterpret_v4f32(a))
	};
	return v.i32[0];
}

#define _mm_cvtsi32_ss(a, b) _mm_cvt_si2ss(a, b)

FORCE_INLINE __m128 _mm_cvtsi64_ss(__m128 a, int64_t b)
{
	float f = (float)b;
	return vreinterpret_m128(__builtin_msa_insert_w(
		vreinterpret_v4i32(a), 0, reinterpret_i32(f)));
}

FORCE_INLINE __m128d _mm_cvtsi64_sd(__m128d a, int64_t b)
{
	double f = (float)b;
	return vreinterpret_m128d(__builtin_msa_insert_d(
		vreinterpret_v2i64(a), 0, reinterpret_i64(f)));
}

#define _mm_cvtsi64x_sd(a, b) _mm_cvtsi64_sd(a, b)

FORCE_INLINE __m128 _mm_cvtpi8_ps(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.f32[3] = (float)v.i8[3];
	v.f32[2] = (float)v.i8[2];
	v.f32[1] = (float)v.i8[1];
	v.f32[0] = (float)v.i8[0];
	return v.m128;
}

FORCE_INLINE __m128 _mm_cvtpi16_ps(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.f32[3] = (float)v.i16[3];
	v.f32[2] = (float)v.i16[2];
	v.f32[1] = (float)v.i16[1];
	v.f32[0] = (float)v.i16[0];
	return v.m128;
}

FORCE_INLINE __m128 _mm_cvtpi32_ps(__m128 a, __m64 b)
{
	VREG128 vb = {.m64 = {b, {0}}};
	vb.msa_v4f32 = __builtin_msa_ffint_s_w(vb.msa_v4i32);
	VREG128 *v = (VREG128*)&a;
	v->m64[0] = vb.m64[0];
	return v->m128;
}

FORCE_INLINE __m128d _mm_cvtpi32_pd(__m64 a)
{
	union {__m64 m64; int32_t i32[2];} u = {.m64 = a};
	VREG128 v = {.i64 = {u.i32[0], u.i32[1]}};
	v.msa_v2f64 = __builtin_msa_ffint_s_d(v.msa_v2i64);
	return v.m128d;
}

FORCE_INLINE __m128 _mm_cvtpi32x2_ps(__m64 a, __m64 b)
{
	VREG128 v = {
		.m64 = {a, b}
	};
	v.msa_v4f32 = __builtin_msa_ffint_s_w(v.msa_v4i32);
	return v.m128;
}

FORCE_INLINE __m128 _mm_cvtpu8_ps(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.f32[3] = (float)v.u8[3];
	v.f32[2] = (float)v.u8[2];
	v.f32[1] = (float)v.u8[1];
	v.f32[0] = (float)v.u8[0];
	return v.m128;
}

FORCE_INLINE __m128 _mm_cvtpu16_ps(__m64 a)
{
	VREG128 v = {.m64 = {a, {0}}};
	v.f32[3] = (float)v.u16[3];
	v.f32[2] = (float)v.u16[2];
	v.f32[1] = (float)v.u16[1];
	v.f32[0] = (float)v.u16[0];
	return v.m128;
}

FORCE_INLINE __m128i _mm_cvttps_epi32(__m128 a)
{
	return vreinterpret_m128i(
		__builtin_msa_ftrunc_s_w(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128i _mm_cvttpd_epi32(__m128d a)
{
	double a0 = ((double *)&a)[0];
	double a1 = ((double *)&a)[1];
	return _mm_set_epi32(0, 0, (int32_t)a1, (int32_t)a0);
}

FORCE_INLINE __m64 _mm_cvttpd_pi32(__m128d a)
{
	double a0 = ((double *)&a)[0];
	double a1 = ((double *)&a)[1];
	VREG128 v = {
		.i32 = {
			(int32_t)a0, (int32_t)a1, 0, 0
		}
	};
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_cvtt_ps2pi(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftrunc_s_w(vreinterpret_v4f32(a))
	};
	return v.m64[0];
}

#define _mm_cvttps_pi32(a) _mm_cvtt_ps2pi(a)

FORCE_INLINE int _mm_cvtt_ss2si(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftrunc_s_w(vreinterpret_v4f32(a))
	};
	return v.i32[0];
}

#define _mm_cvttss_si32(a) _mm_cvtt_ss2si(a)

FORCE_INLINE int _mm_cvttsd_si32(__m128d a)
{
	double ret = *((double*) &a);
	return (int32_t)ret;
}

FORCE_INLINE int64_t _mm_cvttsd_si64(__m128d a)
{
	VREG128 v = {
		.msa_v2i64 = __builtin_msa_ftrunc_s_d(vreinterpret_v2f64(a))
	};
	return v.i64[0];
}

#define _mm_cvttsd_si64x(a) _mm_cvttsd_si64(a)

FORCE_INLINE __m128 _mm_cvtepi32_ps(__m128i a)
{
	return vreinterpret_m128(
		__builtin_msa_ffint_s_w(vreinterpret_v4i32(a)));
}

FORCE_INLINE __m128d _mm_cvtepi32_pd(__m128i a)
{
	VREG128 v = {
		.f64 = {
			(double)vreinterpret_nth_i32_m128i(a, 0),
			(double)vreinterpret_nth_i32_m128i(a, 1)
		}
	};
	return v.m128d;
}

FORCE_INLINE int64_t _mm_cvttss_si64(__m128 a)
{
	VREG128 v = {
		.msa_v2f64 = __builtin_msa_fexupr_d(vreinterpret_v4f32(a))
	};
	v.msa_v2i64 = __builtin_msa_ftrunc_s_d(v.msa_v2f64);
	return v.i64[0];
}

FORCE_INLINE __m128i _mm_cvtepu8_epi16(__m128i a)
{
	VREG128 vl = {.m128i = a};
	VREG128 vr = {.m128i = a};
	v16u8 z = v_msa_setzero(v16u8);
	vl.msa_v8u16 = __builtin_msa_hadd_u_h(vl.msa_v16u8, z);
	vr.msa_v8u16 = __builtin_msa_hadd_u_h(z, vr.msa_v16u8);
	vl.msa_v8i16 = __builtin_msa_ilvr_h(vl.msa_v8i16, vr.msa_v8i16);
	return vl.m128i;
}

FORCE_INLINE __m128i _mm_cvtepu8_epi32(__m128i a)
{
	VREG128 v = {
		.msa_v16i8 = v_msa_setzero(v16i8)
	};
	VREG128 *s = (VREG128*)(&a);
	for (int i = 0, j = 0; i < 4; ++i, j += 4) {
		v.u8[j] = s->u8[i];
	}
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepu8_epi64(__m128i a)
{
	VREG128 v = {
		.msa_v16i8 = v_msa_setzero(v16i8)
	};
	VREG128 *s = (VREG128*)(&a);
	v.u8[0] = s->u8[0];
	v.u8[8] = s->u8[1];
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepi8_epi16(__m128i a)
{
	VREG128 v;
	VREG128 *s = (VREG128*)(&a);
	for (int i = 0; i < 8; ++i) {
		v.i16[i] = s->i8[i];
	}
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepi8_epi32(__m128i a)
{
	VREG128 v;
	VREG128 *s = (VREG128*)(&a);
	for (int i = 0; i < 4; ++i) {
		v.i32[i] = s->i8[i];
	}
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepi8_epi64(__m128i a)
{
	VREG128 v;
	VREG128 *s = (VREG128*)(&a);
	v.i64[0] = s->i8[0];
	v.i64[1] = s->i8[1];
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepi16_epi32(__m128i a)
{
	VREG128 v;
	VREG128 *s = (VREG128*)(&a);
	for (int i = 0; i < 4; ++i) {
		v.i32[i] = s->i16[i];
	}
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepi16_epi64(__m128i a)
{
	VREG128 v;
	VREG128 *s = (VREG128*)(&a);
	v.i64[0] = s->i16[0];
	v.i64[1] = s->i16[1];
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepu16_epi32(__m128i a)
{
	VREG128 v = {
		.msa_v16i8 = v_msa_setzero(v16i8)
	};
	VREG128 *s = (VREG128*)(&a);
	for (int i = 0, j = 0; i < 4; ++i, j += 2) {
		v.u16[j] = s->u16[i];
	}
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepu16_epi64(__m128i a)
{
	VREG128 v = {
		.msa_v16i8 = v_msa_setzero(v16i8)
	};
	VREG128 *s = (VREG128*)(&a);
	v.u16[0] = s->u16[0];
	v.u16[4] = s->u16[1];
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepu32_epi64(__m128i a)
{
	VREG128 v = {
		.msa_v16i8 = v_msa_setzero(v16i8)
	};
	VREG128 *s = (VREG128*)(&a);
	v.u32[0] = s->u32[0];
	v.u32[2] = s->u32[1];
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtepi32_epi64(__m128i a)
{
	VREG128 v;
	VREG128 *s = (VREG128*)(&a);
	v.i64[0] = s->i32[0];
	v.i64[1] = s->i32[1];
	return v.m128i;
}

FORCE_INLINE __m128i _mm_cvtps_epi32(__m128 a)
{
	return vreinterpret_m128i(
		__builtin_msa_ftint_s_w(vreinterpret_v4f32(a)));
}

#define _mm_cvtps_pi32(a) _mm_cvt_ps2pi(a)

FORCE_INLINE __m128i _mm_cvtpd_epi32(__m128d a)
{
	VREG128 v = {
		.msa_v2i64 = __builtin_msa_ftint_s_d(vreinterpret_v2f64(a))
	};
	v.msa_v2i64 = __builtin_msa_slli_d(
		__builtin_msa_sat_s_d(v.msa_v2i64, 31), 32);
	v.msa_v4i32 = __builtin_msa_pckod_w(v_msa_setzero(v4i32), v.msa_v4i32);
	return v.m128i;
}

FORCE_INLINE __m64 _mm_cvtpd_pi32(__m128d a)
{
	VREG128 v = {
		.m128i = _mm_cvtpd_epi32(a)
	};
	return v.m64[0];
}

FORCE_INLINE int _mm_cvtsd_si32(__m128d a)
{
	VREG128 v = {
		.m128i = _mm_cvtpd_epi32(a)
	};
	return v.i32[0];
}

FORCE_INLINE __m128 _mm_cvtsd_ss(__m128 a, __m128d b)
{
	VREG128 v = {
		.msa_v4f32 = __builtin_msa_fexdo_w(
			vreinterpret_v2f64(b), vreinterpret_v2f64(b))
	};
	return _mm_move_ss(a, v.m128);
}

FORCE_INLINE __m64 _mm_cvtps_pi16(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftint_s_w(vreinterpret_v4f32(a))
	};
	v.msa_v4i32 = __builtin_msa_slli_w(
		__builtin_msa_sat_s_w(v.msa_v4i32, 15), 16);
	v.msa_v8i16 = __builtin_msa_pckod_h(v.msa_v8i16, v.msa_v8i16);
	return v.m64[0];
}

FORCE_INLINE __m64 _mm_cvtps_pi8(__m128 a)
{
	VREG128 v = {
		.msa_v4i32 = __builtin_msa_ftint_s_w(vreinterpret_v4f32(a))
	};
	v.msa_v4i32 = __builtin_msa_slli_w(
		__builtin_msa_sat_s_w(v.msa_v4i32, 7), 24);
	v.msa_v8i16 = __builtin_msa_pckod_h(
		v_msa_setzero(v8i16), v.msa_v8i16);
	v.msa_v16i8 = __builtin_msa_pckod_b(v.msa_v16i8, v.msa_v16i8);
	return v.m64[0];
}

FORCE_INLINE int _mm_cvtsi128_si32(__m128i a)
{
	return vreinterpret_nth_i32_m128i(a, 0);
}

FORCE_INLINE int64_t _mm_cvtsi128_si64(__m128i a)
{
	return vreinterpret_nth_i64_m128i(a, 0);
}

#define _mm_cvtsi128_si64x(a) _mm_cvtsi128_si64(a)

FORCE_INLINE __m128i _mm_cvtsi32_si128(int a)
{
	v4i32 v = v_msa_setzero(v4i32);
	return vreinterpret_m128i(__builtin_msa_insert_w(v, 0, a));
}

FORCE_INLINE __m128d _mm_cvtsi32_sd(__m128d a, int b)
{
	VREG128 v = {.m128d = a};
	v.f64[0] = (double)b;
	return v.m128d;
}

FORCE_INLINE __m128i _mm_cvtsi64_si128(int64_t a)
{
	v2i64 v = v_msa_setzero(v2i64);
	return vreinterpret_m128i(__builtin_msa_insert_d(v, 0, a));
}

#define _mm_cvtsi64x_si128(a) _mm_cvtsi64_si128(a)

FORCE_INLINE __m128 _mm_castpd_ps(__m128d a)
{
	return vreinterpret_m128(a);
}

FORCE_INLINE __m128i _mm_castpd_si128(__m128d a)
{
	return vreinterpret_m128i(a);
}

FORCE_INLINE __m128d _mm_castps_pd(__m128 a)
{
	return vreinterpret_m128d(a);
}

FORCE_INLINE __m128i _mm_castps_si128(__m128 a)
{
	return vreinterpret_m128i(a);
}

FORCE_INLINE __m128d _mm_castsi128_pd(__m128i a)
{
	return vreinterpret_m128d(a);
}

FORCE_INLINE __m128 _mm_castsi128_ps(__m128i a)
{
	return vreinterpret_m128(a);
}

FORCE_INLINE __m128i _mm_load_si128(const __m128i *p)
{
	return vreinterpret_m128i(__builtin_msa_ld_w(p, 0));
}

FORCE_INLINE __m128i _mm_loadu_si128(const __m128i *p)
{
	return vreinterpret_m128i(__builtin_msa_ld_w(p, 0));
}

#define _mm_lddqu_si128 _mm_loadu_si128

FORCE_INLINE __m128 _mm_cvtpd_ps(__m128d a)
{
	return vreinterpret_m128(__builtin_msa_fexdo_w(
		vreinterpret_v2f64(a), v_msa_setzero(v2f64)));
}

FORCE_INLINE double _mm_cvtsd_f64(__m128d a)
{
	return vreinterpret_nth_f64_m128d(a, 0);
}

FORCE_INLINE __m128d _mm_cvtps_pd(__m128 a)
{
	return vreinterpret_m128d(
		__builtin_msa_fexupr_d(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128d _mm_cvtss_sd(__m128d a, __m128 b)
{
	return _mm_move_sd(a, _mm_cvtps_pd(b));
}

FORCE_INLINE __m128 _mm_ceil_ps(__m128 a)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
		(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
	return vreinterpret_m128(
		__builtin_msa_frint_w(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128d _mm_ceil_pd(__m128d a)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
		(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
	return vreinterpret_m128d(
		__builtin_msa_frint_d(vreinterpret_v2f64(a)));
}

FORCE_INLINE __m128 _mm_ceil_ss(__m128 a, __m128 b)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
		(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
	return _mm_move_ss(a, vreinterpret_m128(
		__builtin_msa_frint_w(vreinterpret_v4f32(b))));
}

FORCE_INLINE __m128d _mm_ceil_sd(__m128d a, __m128d b)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x2 */
		(__builtin_msa_cfcmsa(1) | 0x3) ^ 0x1);
	return _mm_move_sd(a, vreinterpret_m128d(
		__builtin_msa_frint_d(vreinterpret_v2f64(b))));
}

FORCE_INLINE __m128 _mm_floor_ps(__m128 a)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
		(__builtin_msa_cfcmsa(1) | 0x3));
	return vreinterpret_m128(
		__builtin_msa_frint_w(vreinterpret_v4f32(a)));
}

FORCE_INLINE __m128d _mm_floor_pd(__m128d a)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
		(__builtin_msa_cfcmsa(1) | 0x3));
	return vreinterpret_m128d(
		__builtin_msa_frint_d(vreinterpret_v2f64(a)));
}

FORCE_INLINE __m128 _mm_floor_ss(__m128 a, __m128 b)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
		(__builtin_msa_cfcmsa(1) | 0x3));
	return _mm_move_ss(a, vreinterpret_m128(
		__builtin_msa_frint_w(vreinterpret_v4f32(b))));
}

FORCE_INLINE __m128d _mm_floor_sd(__m128d a, __m128d b)
{
	__builtin_msa_ctcmsa(1,   /* MSACSR[1:0] = 0x3 */
		(__builtin_msa_cfcmsa(1) | 0x3));
	return _mm_move_sd(a, vreinterpret_m128d(
		__builtin_msa_frint_d(vreinterpret_v2f64(b))));
}

FORCE_INLINE __m128i _mm_sra_epi16(__m128i a, __m128i count)
{
	int64_t ci64 = vreinterpret_nth_i64_m128i(count, 0);
	int32_t ci32 = INT32_MAX < ci64 ? INT32_MAX : ci64;
	return _mm_srai_epi16(a, ci32);
}

FORCE_INLINE __m128i _mm_sra_epi32(__m128i a, __m128i count)
{
	int64_t ci64 = vreinterpret_nth_i64_m128i(count, 0);
	int32_t ci32 = INT32_MAX < ci64 ? INT32_MAX : ci64;
	return _mm_srai_epi32(a, ci32);
}

FORCE_INLINE __m128i _mm_packs_epi16(__m128i a, __m128i b)
{
	v16i8 mask = {
		0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32};
	return vreinterpret_m128i(__builtin_msa_vshf_b(mask,
		vreinterpret_v16i8(__builtin_msa_sat_s_h(vreinterpret_v8i16(b), 7)),
		vreinterpret_v16i8(__builtin_msa_sat_s_h(vreinterpret_v8i16(a), 7))));
}

FORCE_INLINE __m128i _mm_packus_epi16(__m128i a, __m128i b)
{
	v16i8 mask = {
		0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32};
	v8i16 u = __builtin_msa_maxi_s_h(vreinterpret_v8i16(a), 0);
	v8i16 v = __builtin_msa_maxi_s_h(vreinterpret_v8i16(b), 0);
	return vreinterpret_m128i(__builtin_msa_vshf_b(mask,
		vreinterpret_v16i8(__builtin_msa_sat_u_h(vreinterpret_v8u16(v), 7)),
		vreinterpret_v16i8(__builtin_msa_sat_u_h(vreinterpret_v8u16(u), 7))));
}

FORCE_INLINE __m128i _mm_packs_epi32(__m128i a, __m128i b)
{
	v8i16 mask = {0, 2, 4, 6, 8, 10, 12, 14};
	return vreinterpret_m128i(__builtin_msa_vshf_h(mask,
		vreinterpret_v8i16(__builtin_msa_sat_s_w(vreinterpret_v4i32(b), 15)),
		vreinterpret_v8i16(__builtin_msa_sat_s_w(vreinterpret_v4i32(a), 15))));
}

FORCE_INLINE __m128i _mm_packus_epi32(__m128i a, __m128i b)
{
	v8i16 mask = {0, 2, 4, 6, 8, 10, 12, 14};
	v4i32 u = __builtin_msa_maxi_s_w(vreinterpret_v4i32(a), 0);
	v4i32 v = __builtin_msa_maxi_s_w(vreinterpret_v4i32(b), 0);
	return vreinterpret_m128i(__builtin_msa_vshf_h(mask,
		vreinterpret_v8i16(__builtin_msa_sat_u_w(vreinterpret_v4u32(v), 15)),
		vreinterpret_v8i16(__builtin_msa_sat_u_w(vreinterpret_v4u32(u), 15))));
}

FORCE_INLINE __m128i _mm_unpacklo_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvr_b((v16i8)b, (v16i8)a));
}

FORCE_INLINE __m128i _mm_unpacklo_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvr_h((v8i16)b, (v8i16)a));
}

FORCE_INLINE __m128i _mm_unpacklo_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvr_w((v4i32)b, (v4i32)a));
}

FORCE_INLINE __m128i _mm_unpacklo_epi64(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvr_d((v2i64)b, (v2i64)a));
}

FORCE_INLINE __m128 _mm_unpacklo_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(
		__builtin_msa_ilvr_w((v4i32)b, (v4i32)a));
}

FORCE_INLINE __m128d _mm_unpacklo_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(
		__builtin_msa_ilvr_d((v2i64)b, (v2i64)a));
}

FORCE_INLINE __m128 _mm_unpackhi_ps(__m128 a, __m128 b)
{
	return vreinterpret_m128(
		__builtin_msa_ilvl_w((v4i32)b, (v4i32)a));
}

FORCE_INLINE __m128d _mm_unpackhi_pd(__m128d a, __m128d b)
{
	return vreinterpret_m128d(
		__builtin_msa_ilvl_d((v2i64)b, (v2i64)a));
}

FORCE_INLINE __m128i _mm_unpackhi_epi8(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvl_b((v16i8)b, (v16i8)a));
}

FORCE_INLINE __m128i _mm_unpackhi_epi16(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvl_h((v8i16)b, (v8i16)a));
}

FORCE_INLINE __m128i _mm_unpackhi_epi32(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvl_w((v4i32)b, (v4i32)a));
}

FORCE_INLINE __m128i _mm_unpackhi_epi64(__m128i a, __m128i b)
{
	return vreinterpret_m128i(
		__builtin_msa_ilvl_d((v2i64)b, (v2i64)a));
}

#define _MM_TRANSPOSE4_PS(row0, row1, row2, row3) \
do {                                              \
  __m128 __r0 = (row0), __r1 = (row1),            \
         __r2 = (row2), __r3 = (row3);            \
  __m128 __t0 = _mm_unpacklo_ps (__r0, __r1);     \
  __m128 __t1 = _mm_unpacklo_ps (__r2, __r3);     \
  __m128 __t2 = _mm_unpackhi_ps (__r0, __r1);     \
  __m128 __t3 = _mm_unpackhi_ps (__r2, __r3);     \
  (row0) = _mm_movelh_ps (__t0, __t1);            \
  (row1) = _mm_movehl_ps (__t1, __t0);            \
  (row2) = _mm_movelh_ps (__t2, __t3);            \
  (row3) = _mm_movehl_ps (__t3, __t2);            \
} while (0)

FORCE_INLINE __m128i _mm_minpos_epu16(__m128i a)
{
	VREG128 v = {
		.m128i = _mm_setzero_si128()
	};
	uint16_t i, min = 0xffff, idx = 0;
	for (i = 0; i < 8; ++i) {
		if (min > vreinterpret_nth_u16_m128i(a, i)) {
			min = vreinterpret_nth_u16_m128i(a, i);
			idx = i;
		}
	}
	v.u16[0] = min;
	v.u16[1] = idx;
	return v.m128i;
}

#define _mm_extract_epi8(a, imm8) \
(__builtin_msa_copy_u_b(vreinterpret_v16i8(a), imm8))

#define _mm_extract_epi16(a, imm8) \
(__builtin_msa_copy_u_h(vreinterpret_v8i16(a), imm8))

#define _mm_extract_pi16(a, imm8) \
__extension__({VREG128 v = {.m64 = {a, {0}}}; \
	__builtin_msa_copy_u_h(v.msa_v8i16, imm8);})

#define _mm_extract_epi32(a, imm8) \
(__builtin_msa_copy_s_w(vreinterpret_v4i32(a), imm8))

#define _mm_extract_epi64(a, imm8) \
(__builtin_msa_copy_s_d(vreinterpret_v2i64(a), imm8))

#define _mm_extract_ps(a, imm8) \
(__builtin_msa_copy_s_w(vreinterpret_v4i32(a), imm8))

#define _mm_insert_epi8(a, b, imm8) \
__extension__((__m128i)                                              \
	({__builtin_msa_insert_b(vreinterpret_v16i8(a), imm8, b);}))

#define _mm_insert_epi16(a, b, imm8) \
__extension__((__m128i)                                              \
	({__builtin_msa_insert_h(vreinterpret_v8i16(a), imm8, b);}))

#define _mm_insert_pi16(a, b, imm8) \
__extension__({                                                \
  VREG128 v = {.m64 = {a, {0}}};                               \
  v.msa_v8i16 = __builtin_msa_insert_h(v.msa_v8i16, imm8, b);  \
  v.m64[0];})

#define _mm_insert_epi32(a, b, imm8) \
__extension__((__m128i)                                              \
	({__builtin_msa_insert_w(vreinterpret_v4i32(a), imm8, b);}))

#define _mm_insert_epi64(a, b, imm8) \
__extension__((__m128i)                                              \
	({__builtin_msa_insert_d(vreinterpret_v2i64(a), imm8, b);}))

#define _m_pextrw(a, imm) _mm_extract_pi16(a, imm)

#define _m_pinsrw(a, i, imm) _mm_insert_pi16(a, i, imm)

FORCE_INLINE __m128i _mm_sign_epi8(__m128i a, __m128i b)
{
	v16i8 mask = __builtin_msa_maxi_s_b(vreinterpret_v16i8(b), -1);
	mask = __builtin_msa_mini_s_b(vreinterpret_v16i8(mask), 1);
	return vreinterpret_m128i(
		__builtin_msa_mulv_b(vreinterpret_v16i8(a), mask));
}

FORCE_INLINE __m128i _mm_sign_epi16(__m128i a, __m128i b)
{
	v8i16 mask = __builtin_msa_maxi_s_h(vreinterpret_v8i16(b), -1);
	mask = __builtin_msa_mini_s_h(vreinterpret_v8i16(mask), 1);
	return vreinterpret_m128i(
		__builtin_msa_mulv_h(vreinterpret_v8i16(a), mask));
}

FORCE_INLINE __m128i _mm_sign_epi32(__m128i a, __m128i b)
{
	v4i32 mask = __builtin_msa_maxi_s_w(vreinterpret_v4i32(b), -1);
	mask = __builtin_msa_mini_s_w(vreinterpret_v4i32(mask), 1);
	return vreinterpret_m128i(
		__builtin_msa_mulv_w(vreinterpret_v4i32(a), mask));
}

FORCE_INLINE __m64 _mm_sign_pi8(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.m128i = _mm_sign_epi8(va.m128i, vb.m128i);
	return va.m64[0];
}

FORCE_INLINE __m64 _mm_sign_pi16(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.m128i = _mm_sign_epi16(va.m128i, vb.m128i);
	return va.m64[0];
}

FORCE_INLINE __m64 _mm_sign_pi32(__m64 a, __m64 b)
{
	VREG128 va = {.m64 = {a, {0}}};
	VREG128 vb = {.m64 = {b, {0}}};
	va.m128i = _mm_sign_epi32(va.m128i, vb.m128i);
	return va.m64[0];
}

FORCE_INLINE __m128i _mm_alignr_epi8(__m128i a, __m128i b, int imm8)
{
	return vreinterpret_m128i(__builtin_msa_sld_b(
		vreinterpret_v16i8(a), vreinterpret_v16i8(b), imm8));
}

__m64 _mm_alignr_pi8(__m64 a, __m64 b, int imm8)
{
	VREG128 v = {
		.m64 = {b, a}
	};
	v16i8 z = {0};
	v.msa_v16i8 = __builtin_msa_sld_b(z, v.msa_v16i8, imm8);
	return v.m64[0];
}

FORCE_INLINE void _mm_pause()
{
	__asm__ __volatile__("pause\n");
}

FORCE_INLINE void _mm_sfence(void)
{
    __sync_synchronize();
}

#if defined(__GNUC__)
#pragma pop_macro("FORCE_INLINE")
#pragma pop_macro("ALIGN_STRUCT")
#endif
#endif /* SSE2MSA_H */

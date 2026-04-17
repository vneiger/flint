/*
    Copyright (C) 2026 Vincent Neiger

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include <stdlib.h>  /* for atoi */
#include "flint/flint.h"
#include "flint/longlong.h"
#include "flint/longlong_div_gnu.h"
#include "flint/profiler.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

/* FIXME might not work if not gcc! */
#define FLINT_NO_VECTORIZE __attribute__((optimize("no-tree-vectorize")))
/* FIXME comment out if machine does not have avx512 */
#define FLINT_HAVE_AVX512

#ifdef FLINT_HAVE_AVX512
#  include <immintrin.h>
#  include <flint/machine_vectors.h>
#endif // FLINT_HAVE_AVX512


// must be a multiple of 8
#define LEN 10000

typedef struct
{
    flint_bitcnt_t bits;
} info_t;

/*-----------------------*/
/* single word reduction */
/*-----------------------*/

// modular reduction with precomputation,
// using n_pr = floor(2**FLINT_BITS / n),
// e.g. computed via n_mulmod_precomp_shoup(1L, n)
static inline
ulong n_modred_barrett(ulong a, ulong n_pr, ulong n)
{
    return n_mulmod_shoup(1L, a, n_pr, n);
}

/* same but make it "vectorizable" by avoiding 64x64->128 mul */
static inline
ulong n_modred_barrett_simd(ulong a, ulong n_pr, ulong n)
{
    ulong p_mid = (((a >> 32) * n_pr) + (a * (n_pr >> 32))) >> 32;
    ulong p_hi = ((a >> 32) * (n_pr >> 32));
    a -= (p_mid + p_hi) * n;

    if (a >= n)
        a -= n;
    if (a >= n)
        a -= n;

    return a;
}

#ifdef FLINT_HAVE_AVX512
/* same with explicit vectorization */
/* a[i] >= n[i] ? a[i] - n[i] : a[i] */
FLINT_FORCE_INLINE __m512i _mm512_subtract_if_cmpge(__m512i a, __m512i n)
{
    return _mm512_min_epu64(a, _mm512_sub_epi64(a, n));
}

/* high word of widening 64x64 multiplication, lost carry */
FLINT_FORCE_INLINE __m512i _mm512_mulhi_lazy_epu64(__m512i a, __m512i b)
{
    __m512i ahi = _mm512_shuffle_epi32(a, 0xB1);
    __m512i bhi = _mm512_shuffle_epi32(b, 0xB1);

    __m512i alo_bhi = _mm512_mul_epu32(a, bhi);
    __m512i ahi_blo = _mm512_mul_epu32(ahi, b);

    __m512i mid = _mm512_add_epi64(_mm512_srli_epi64(alo_bhi, 32),
                                   _mm512_srli_epi64(ahi_blo, 32));

    __m512i ahi_bhi = _mm512_mul_epu32(ahi, bhi);
    return _mm512_add_epi64(ahi_bhi, mid);
}

FLINT_FORCE_INLINE __m512i
_mm512_mulmod_barrett(__m512i a, __m512i b, __m512i a_pr, __m512i n)
{
    __m512i mulhi = _mm512_mulhi_lazy_epu64(b, a_pr);
    __m512i mullo1 = _mm512_mullo_epi64(b, a);
    __m512i mullo2 = _mm512_mullo_epi64(mulhi, n);
    __m512i mul = _mm512_sub_epi64(mullo1, mullo2);
    mul = _mm512_subtract_if_cmpge(mul, n);
    return _mm512_subtract_if_cmpge(mul, n);
}

FLINT_FORCE_INLINE __m512i
_mm512_modred_barrett(__m512i a, __m512i n_pr, __m512i n)
{
    __m512i mulhi = _mm512_mulhi_lazy_epu64(a, n_pr);
    __m512i mullo = _mm512_mullo_epi64(mulhi, n);
    __m512i mul = _mm512_sub_epi64(a, mullo);
    mul = _mm512_subtract_if_cmpge(mul, n);
    return _mm512_subtract_if_cmpge(mul, n);
}
#endif  /* FLINT_HAVE_AVX512 */



// modular reduction with precomputation,
// gives the correct reduced remainder possibly +n,
// and always in the range [0,n+2)
// below we take aa = 64, tt = nbits(n) + 64 - 1
// sets shift to tt - aa = nbits(n) - 1,
// sets n_pr to floor(2**tt / n)
// TODO might not work for n == 1 or power of 2
static inline
void n_precomp_modred_barrett2(flint_bitcnt_t * shift, ulong * n_pr, ulong n)
{
    ulong tmp;

    *shift = FLINT_BIT_COUNT(n) - 1;
    udiv_qrnnd(*n_pr, tmp, UWORD(1) << *shift, UWORD(0), n);
}

static inline
ulong n_modred_barrett2(ulong a, ulong n_pr, flint_bitcnt_t shift, ulong n)
{
    ulong q, tmp;

    umul_ppmm(q, tmp, a, n_pr);
    a = a - (q >> shift) * n;
    if (a >= n)
        a -= n;

    return a;
}

static inline
ulong n_modred_barrett2_lazy(ulong a, ulong n_pr, flint_bitcnt_t shift, ulong n)
{
    ulong q, tmp;

    umul_ppmm(q, tmp, a, n_pr);
    return a - (q >> shift) * n;
}


/* NOTE !!experimental!! */
/* requires nbits >= 33 */
/* guarantees on output? */

static inline
void nmod_precomp_fast(ulong * redp, ulong * shift, ulong n)
{
    ulong nbits = FLINT_BITS - flint_clz(n);
    *shift = nbits + FLINT_BITS - 1;
    ulong FLINT_SET_BUT_UNUSED(rem);
    udiv_qrnnd(*redp, rem, (UWORD(1) << (nbits - 1)), UWORD(0), n);
}

static inline
ulong n_modred_fast_lazy(ulong a, ulong n, ulong redp, ulong shift)
{
    ulong q, a_red;
    q = ((a >> 32) * (redp >> 32)) >> (shift - 64);
    a_red = a - q * n;
    return a_red;
}

static inline
ulong n_modred_fast(ulong a, ulong n, ulong redp, ulong shift)
{
    ulong q, a_red;
    q = ((a >> 32) * (redp >> 32)) >> (shift - 64);
    a_red = a - q * n;
    if (a_red >= n)
        a_red -= n;
    return a_red;
}


/*--------------------------------------*/
/* reduce vector: res[i] = vec[i] % n   */
/*--------------------------------------*/

void prof_nmod_vec_reduce_nmod_red(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    for (slong i = 0 ; i < len; i++)
        NMOD_RED(res[i], vec[i], mod);
}

void prof_nmod_vec_reduce_nmod_red_unroll(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        NMOD_RED(res[i+0], vec[i+0], mod);
        NMOD_RED(res[i+1], vec[i+1], mod);
        NMOD_RED(res[i+2], vec[i+2], mod);
        NMOD_RED(res[i+3], vec[i+3], mod);
    }
    for ( ; i < len; i++)
        NMOD_RED(res[i], vec[i], mod);
}

void prof_nmod_vec_reduce_barrett(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
    for (slong i = 0 ; i < len; i++)
        res[i] = n_modred_barrett(vec[i], one_barrett, mod.n);
}

void prof_nmod_vec_reduce_barrett_unroll(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
    slong i;
    for (i = 0 ; i+3 < len; i+=4)
    {
        res[i+0] = n_modred_barrett(vec[i+0], one_barrett, mod.n);
        res[i+1] = n_modred_barrett(vec[i+1], one_barrett, mod.n);
        res[i+2] = n_modred_barrett(vec[i+2], one_barrett, mod.n);
        res[i+3] = n_modred_barrett(vec[i+3], one_barrett, mod.n);
    }
    for ( ; i < len; i++)
        res[i] = n_modred_barrett(vec[i], one_barrett, mod.n);
}

void prof_nmod_vec_reduce_barrett_simd(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
    for (slong i = 0 ; i < len; i++)
        res[i] = n_modred_barrett_simd(vec[i], one_barrett, mod.n);
}

#ifdef FLINT_HAVE_AVX512
void prof_nmod_vec_reduce_barrett_simdexplicit(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
    const __m512i vecn_pr = _mm512_set1_epi64(one_barrett);
    const __m512i vecn = _mm512_set1_epi64(mod.n);
    for (slong i = 0 ; i+7 < len; i+=8)
    {
        __m512i v = _mm512_loadu_si512(vec+i);
        v = _mm512_modred_barrett(v, vecn_pr, vecn);
        _mm512_storeu_si512(res+i, v);
    }
}
#endif /* FLINT_HAVE_AVX512 */

FLINT_NO_VECTORIZE
void prof_nmod_vec_reduce_barrett_simd_novec(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
    slong i;
    for (i = 0 ; i+3 < len; i+=4)
    {
        res[i+0] = n_modred_barrett_simd(vec[i+0], one_barrett, mod.n);
        res[i+1] = n_modred_barrett_simd(vec[i+1], one_barrett, mod.n);
        res[i+2] = n_modred_barrett_simd(vec[i+2], one_barrett, mod.n);
        res[i+3] = n_modred_barrett_simd(vec[i+3], one_barrett, mod.n);
    }
    for ( ; i < len; i++)
        res[i] = n_modred_barrett_simd(vec[i], one_barrett, mod.n);
}


void prof_nmod_vec_reduce_barrett2(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);
    for (slong i = 0 ; i < len; i++)
        res[i] = n_modred_barrett2(vec[i], n_pr, shift, mod.n);
}

void prof_nmod_vec_reduce_barrett2_unroll(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);
    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        res[i+0] = n_modred_barrett2(vec[i+0], n_pr, shift, mod.n);
        res[i+1] = n_modred_barrett2(vec[i+1], n_pr, shift, mod.n);
        res[i+2] = n_modred_barrett2(vec[i+2], n_pr, shift, mod.n);
        res[i+3] = n_modred_barrett2(vec[i+3], n_pr, shift, mod.n);
    }
    for ( ; i < len; i++)
        res[i] = n_modred_barrett2(vec[i], n_pr, shift, mod.n);
}

void prof_nmod_vec_reduce_barrett2_lazy(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);
    slong i;
    for (i = 0; i < len; i++)
        res[i+0] = n_modred_barrett2_lazy(vec[i+0], n_pr, shift, mod.n);
}

void prof_nmod_vec_reduce_barrett2_lazy_unroll(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);
    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        res[i+0] = n_modred_barrett2_lazy(vec[i+0], n_pr, shift, mod.n);
        res[i+1] = n_modred_barrett2_lazy(vec[i+1], n_pr, shift, mod.n);
        res[i+2] = n_modred_barrett2_lazy(vec[i+2], n_pr, shift, mod.n);
        res[i+3] = n_modred_barrett2_lazy(vec[i+3], n_pr, shift, mod.n);
    }
    for ( ; i < len; i++)
        res[i] = n_modred_barrett2_lazy(vec[i], n_pr, shift, mod.n);
}

void prof_nmod_vec_reduce_fast_lazy(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong redp;
    ulong shift;
    nmod_precomp_fast(&redp, &shift, mod.n);
    for (slong i = 0 ; i < len; i++)
        res[i] = n_modred_fast_lazy(vec[i], mod.n, redp, shift);
}

FLINT_NO_VECTORIZE
void prof_nmod_vec_reduce_fast_lazy_novec(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong redp;
    ulong shift;
    nmod_precomp_fast(&redp, &shift, mod.n);
    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        res[i+0] = n_modred_fast_lazy(vec[i+0], mod.n, redp, shift);
        res[i+1] = n_modred_fast_lazy(vec[i+1], mod.n, redp, shift);
        res[i+2] = n_modred_fast_lazy(vec[i+2], mod.n, redp, shift);
        res[i+3] = n_modred_fast_lazy(vec[i+3], mod.n, redp, shift);
    }
    for ( ; i < len; i++)
        res[i] = n_modred_fast_lazy(vec[i], mod.n, redp, shift);
}

void prof_nmod_vec_reduce_fast(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong redp;
    ulong shift;
    nmod_precomp_fast(&redp, &shift, mod.n);
    for (slong i = 0 ; i < len; i++)
        res[i] = n_modred_fast(vec[i], mod.n, redp, shift);
}

FLINT_NO_VECTORIZE
void prof_nmod_vec_reduce_fast_novec(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong redp;
    ulong shift;
    nmod_precomp_fast(&redp, &shift, mod.n);
    slong i;
    for (i = 0; i+3 < len; i+=4)
    {
        res[i+0] = n_modred_fast(vec[i+0], mod.n, redp, shift);
        res[i+1] = n_modred_fast(vec[i+1], mod.n, redp, shift);
        res[i+2] = n_modred_fast(vec[i+2], mod.n, redp, shift);
        res[i+3] = n_modred_fast(vec[i+3], mod.n, redp, shift);
    }
    for ( ; i < len; i++)
        res[i] = n_modred_fast(vec[i], mod.n, redp, shift);
}


/*----------------------------------------*/
/*        res[i] = reduce double word     */
/* (hi, lo) = (vec[i], vec[i+1]) modulo n */
/*----------------------------------------*/
// warning: using mulmod_shoup here restricts n to 63 bits

void prof_nmod_vec_reduce_nmod2_red2(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    for (slong i = 0 ; i+1 < len; i+=2)
        NMOD2_RED2(res[i], vec[i], vec[i+1], mod);
}

void prof_nmod_vec_reduce_nmod2_red2_unroll(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    slong i;
    for (i = 0 ; i+7 < len; i+=8)
    {
        NMOD2_RED2(res[i+0], vec[i+0], vec[i+1], mod);
        NMOD2_RED2(res[i+2], vec[i+2], vec[i+3], mod);
        NMOD2_RED2(res[i+4], vec[i+4], vec[i+5], mod);
        NMOD2_RED2(res[i+6], vec[i+6], vec[i+7], mod);
    }
    for ( ; i+1 < len; i+=2)
        NMOD2_RED2(res[i], vec[i], vec[i+1], mod);
}

void prof_nmod_vec_reduce_barrett22(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong one_barrett, W, W_pr;
    n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
    // --> W = remainder of division of 2**FLINT_BITS by n
    W_pr = n_mulmod_precomp_shoup(W, mod.n);

    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);

    // naive: res[i] = (vec[i] * W mod n + vec[i+1] mod n) mod n
    for (slong i = 0 ; i+1 < len; i+=2)
    {
        const ulong a = n_mulmod_shoup(W, vec[i], W_pr, mod.n);
        //const ulong b = n_modred_precomp(vec[i+1], one_precomp, mod.n);
        const ulong b = n_modred_barrett2_lazy(vec[i+1], n_pr, shift, mod.n);
        res[i] = _nmod_add(a, b, mod);
    }
}

void prof_nmod_vec_reduce_barrett22_unroll(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong one_barrett, W, W_pr;
    n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
    // --> W = remainder of division of 2**FLINT_BITS by n
    W_pr = n_mulmod_precomp_shoup(W, mod.n);

    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);

    // naive: res[i] = (vec[i] * W mod n + vec[i+1] mod n) mod n
    slong i;
    for (i = 0; i+7 < len; i+=8)
    {
        res[i+0] = _nmod_add(n_mulmod_shoup(W, vec[i+0], W_pr, mod.n),
                             //n_modred_precomp(vec[i+1], one_precomp, mod.n),
                             n_modred_barrett2(vec[i+1], n_pr, shift, mod.n),
                             mod);
        res[i+2] = _nmod_add(n_mulmod_shoup(W, vec[i+2], W_pr, mod.n),
                             //n_modred_barrett(vec[i+3], one_precomp, mod.n),
                             n_modred_barrett2(vec[i+3], n_pr, shift, mod.n),
                             mod);
        res[i+4] = _nmod_add(n_mulmod_shoup(W, vec[i+4], W_pr, mod.n),
                             //n_modred_barrett(vec[i+5], one_precomp, mod.n),
                             n_modred_barrett2(vec[i+5], n_pr, shift, mod.n),
                             mod);
        res[i+6] = _nmod_add(n_mulmod_shoup(W, vec[i+6], W_pr, mod.n),
                             //n_modred_precomp(vec[i+7], one_precomp, mod.n),
                             n_modred_barrett2(vec[i+7], n_pr, shift, mod.n),
                             mod);
    }

    for ( ; i+1 < len; i+=2)
    {
        res[i] = _nmod_add(n_mulmod_shoup(W, vec[i], W_pr, mod.n),
                           //n_modred_precomp(vec[i+1], one_precomp, mod.n),
                            n_modred_barrett2(vec[i+1], n_pr, shift, mod.n),
                           mod);
    }
}

// version without unrolling is slower
void prof_nmod_vec_reduce_barrett22_bis(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong one_barrett, W, W_pr;
    n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
    // --> W = remainder of division of 2**FLINT_BITS by n
    W_pr = n_mulmod_precomp_shoup(W, mod.n);

    // naive: res[i] = (vec[i] * W mod n + vec[i+1] mod n) mod n
    slong i;
    for (i = 0; i+7 < len; i+=8)
    {
        ulong a0, b0, a1, b1, a2, b2, a3, b3;
        a0 = n_mulmod_shoup(W, vec[i+0], W_pr, mod.n);
        a1 = n_mulmod_shoup(W, vec[i+2], W_pr, mod.n);
        a2 = n_mulmod_shoup(W, vec[i+4], W_pr, mod.n);
        a3 = n_mulmod_shoup(W, vec[i+6], W_pr, mod.n);

        b0 = vec[i+1];
        b1 = vec[i+3];
        b2 = vec[i+5];
        b3 = vec[i+7];
        if (b0 >= (mod.n << (mod.norm-1)))
            b0 -= (mod.n << (mod.norm-1));
        if (b1 >= (mod.n << (mod.norm-1)))
            b1 -= (mod.n << (mod.norm-1));
        if (b2 >= (mod.n << (mod.norm-1)))
            b2 -= (mod.n << (mod.norm-1));
        if (b3 >= (mod.n << (mod.norm-1)))
            b3 -= (mod.n << (mod.norm-1));

        res[i+0] = n_modred_barrett(a0+b0, one_barrett, mod.n);
        res[i+2] = n_modred_barrett(a1+b3, one_barrett, mod.n);
        res[i+4] = n_modred_barrett(a2+b3, one_barrett, mod.n);
        res[i+6] = n_modred_barrett(a3+b3, one_barrett, mod.n);
    }
    for ( ; i+1 < len; i+=2)
    {
        ulong a, b;
        a = n_mulmod_shoup(W, vec[i], W_pr, mod.n);
        b = vec[i+1];
        if (b >= (mod.n << (mod.norm-1)))
            b -= (mod.n << (mod.norm-1));
        res[i] = n_modred_barrett(a+b, one_barrett, mod.n);
    }
}

// version without unrolling is slower
void prof_nmod_vec_reduce_barrett22_ter(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    ulong one_barrett, W, W_pr;
    n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
    // --> W = remainder of division of 2**FLINT_BITS by n
    W_pr = n_mulmod_precomp_shoup(W, mod.n);

    flint_bitcnt_t shift;
    ulong n_pr;
    n_precomp_modred_barrett2(&shift, &n_pr, mod.n);

    // naive: res[i] = (vec[i] * W mod n + vec[i+1] mod n) mod n
    slong i;
    for (i = 0; i+7 < len; i+=8)
    {
        ulong a0 = n_mulmod_shoup(W, vec[i+0], W_pr, mod.n) + n_modred_barrett2_lazy(vec[i+1], n_pr, shift, mod.n);
        ulong a1 = n_mulmod_shoup(W, vec[i+2], W_pr, mod.n) + n_modred_barrett2_lazy(vec[i+3], n_pr, shift, mod.n);
        ulong a2 = n_mulmod_shoup(W, vec[i+4], W_pr, mod.n) + n_modred_barrett2_lazy(vec[i+5], n_pr, shift, mod.n);
        ulong a3 = n_mulmod_shoup(W, vec[i+6], W_pr, mod.n) + n_modred_barrett2_lazy(vec[i+7], n_pr, shift, mod.n);

        if (a0 >= mod.n) a0 -= mod.n;
        if (a1 >= mod.n) a1 -= mod.n;
        if (a2 >= mod.n) a2 -= mod.n;
        if (a3 >= mod.n) a3 -= mod.n;

        res[i+0] = a0;
        res[i+2] = a1;
        res[i+4] = a2;
        res[i+6] = a3;
    }
    for ( ; i+1 < len; i+=2)
    {
        ulong a, b;
        a = n_mulmod_shoup(W, vec[i], W_pr, mod.n);
        b = vec[i+1];
        if (b >= (mod.n << (mod.norm-1)))
            b -= (mod.n << (mod.norm-1));
        res[i] = n_modred_barrett(a+b, one_barrett, mod.n);
    }
}

int check(flint_bitcnt_t bits)
{
    ulong n;
    nmod_t mod;

    FLINT_TEST_INIT(state);

    for (ulong i = 0; i < UWORD(10000000); i++)
    {
        n = n_randbits(state, bits);
        nmod_init(&mod, n);

        ulong a_lo = n_randlimb(state);
        ulong a_me = n_randlimb(state);
        ulong a_hi = n_randlimb(state);
        ulong a_hi_red = n_randint(state, n);

        ulong witness;
        ulong candidate;
        { // 1 limb
            NMOD_RED(witness, a_lo, mod);

            { /* classical Barrett with precomp_shoup */
                ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
                candidate = n_modred_barrett(a_lo, one_barrett, mod.n);

                if (witness != candidate)
                {
                    flint_printf("\n\n\nFAILURE 1 limb modred_barrett!!\n\n\n"); return 0;
                }

                candidate = n_modred_barrett_simd(a_lo, one_barrett, mod.n);
                if (witness != candidate)
                {
                    flint_printf("\n\n\nFAILURE 1 limb modred_barrett_simd!!\n\n\n"); return 0;
                }

#ifdef FLINT_HAVE_AVX512
                __m512i vecn_pr = _mm512_set1_epi64(one_barrett);
                __m512i vecn = _mm512_set1_epi64(n);
                __m512i veca = _mm512_set1_epi64(a_lo);
                __m512i veccand = _mm512_modred_barrett(veca, vecn_pr, vecn);
                candidate = veccand[0];
                if (witness != candidate)
                {
                    flint_printf("\n\n\nFAILURE 1 limb _mm512_modred_barrett!!\n\n\n"); return 0;
                }
#endif // FLINT_HAVE_AVX512

            }

            { /* Barrett with another shift */
                flint_bitcnt_t shift;
                ulong n_pr;
                n_precomp_modred_barrett2(&shift, &n_pr, n);
                candidate = n_modred_barrett2(a_lo, n_pr, shift, n);

                if (witness != candidate)
                {
                    flint_printf("\n\n\nFAILURE 1 limb modred_barrett2!!\n\n\n"); return 0;
                }

                candidate = n_modred_barrett2_lazy(a_lo, n_pr, shift, n);

                if (witness != candidate)
                {
                    flint_printf("\n\n\nFAILURE 1 limb modred_barrett2_lazy!! ");
                    flint_printf("n = %wu, a = %wu, shift = %wu, n_pr = %wu witness = %wu, candidate = %wu\n\n\n", n, a_lo, shift, n_pr, witness, candidate);
                    return 0;
                }
            }

            if (bits >= 33)
            { /* approach aiming at allowing vectorization */
                ulong shift;
                ulong redp;
                nmod_precomp_fast(&redp, &shift, n);

                candidate = n_modred_fast_lazy(a_lo, n, redp, shift);
                if (witness != candidate && witness != candidate - n)
                {
                    flint_printf("\n\n\nFAILURE 1 limb modred_fast_lazy!! ");
                    flint_printf("n = %wu, a = %wu, shift = %wu, n_pr = %wu witness = %wu, candidate = %wu\n\n\n", n, a_lo, shift, redp, witness, candidate);
                    return 0;
                }

                candidate = n_modred_fast(a_lo, n, redp, shift);
                if (witness != candidate)
                {
                    flint_printf("\n\n\nFAILURE 1 limb modred_fast!! ");
                    flint_printf("n = %wu, a = %wu, shift = %wu, n_pr = %wu witness = %wu, candidate = %wu\n\n\n", n, a_lo, shift, redp, witness, candidate);
                    return 0;
                }
            }
        }

        if (bits < FLINT_BITS)
        { // 2 limbs, direct
            NMOD2_RED2(witness, a_hi, a_lo, mod);

            ulong one_barrett, W, W_pr;
            n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
            W_pr = n_mulmod_precomp_shoup(W, mod.n);

            ulong candidate =
                _nmod_add(n_mulmod_shoup(W, a_hi, W_pr, mod.n),
                           n_modred_barrett(a_lo, one_barrett, mod.n),
                           mod);

            if (witness != candidate)
            { flint_printf("\n\n\nFAILURE 2 limbs direct!!\n\n\n"); return 0; }
        }

        if (bits < FLINT_BITS)
        { // 2 limbs, less direct
            NMOD2_RED2(witness, a_hi, a_lo, mod);

            ulong one_barrett, W, W_pr;
            n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
            W_pr = n_mulmod_precomp_shoup(W, mod.n);

            ulong candidate = n_mulmod_shoup(W, a_hi, W_pr, mod.n);
            ulong correct = 0;
            if (a_lo >= (mod.n << (mod.norm-1)))
                correct = (mod.n << (mod.norm-1));
            candidate = n_modred_barrett(candidate+a_lo-correct, one_barrett, mod.n);

            if (witness != candidate)
            //{ flint_printf("\n\n\nFAILURE 2 limbs!!\n\n\n"); flint_printf("%lu, %lu, %lu, %lu, %lu\n", mod.n, a_lo, a_hi, candidate, witness); return 0; }
            { flint_printf("\n\n\nFAILURE 2 limbs!!\n\n\n"); return 0; }
        }

        if (bits < FLINT_BITS-1)
        { // 2 limbs, less direct, bis
          // <= 62 bits allows to remove one conditional (see below), but
          // this does not seem to gain anything in the vector reduction
          // context at least (tried within barrett22)
            NMOD2_RED2(witness, a_hi, a_lo, mod);

            ulong one_barrett, W, W_pr;
            n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
            W_pr = n_mulmod_precomp_shoup(W, mod.n);

            ulong candidate, tmp, p_hi;
            umul_ppmm(candidate, tmp, W_pr, a_hi);
            candidate = W * a_hi - candidate * mod.n;
            //if (candidate >= mod.n)
                //candidate -= mod.n;
            ulong correct = 0;
            if (a_lo >= (mod.n << (mod.norm-1)))
                correct = (mod.n << (mod.norm-1));
            candidate += a_lo - correct;
            umul_ppmm(p_hi, tmp, one_barrett, candidate);
            candidate -= p_hi * mod.n;
            if (candidate >= mod.n)
                candidate -= mod.n;

            if (witness != candidate)
            //{ flint_printf("\n\n\nFAILURE 2 limbs bis!!\n\n\n"); flint_printf("%lu, %lu, %lu, %lu, %lu\n", mod.n, a_lo, a_hi, candidate, witness); return 0; }
            { flint_printf("\n\n\nFAILURE 2 limbs!!\n\n\n"); return 0; }
        }

        if (bits < FLINT_BITS)
        { // 3 limbs, direct
            NMOD_RED3(witness, a_hi_red, a_me, a_lo, mod);

            ulong one_barrett, W, W_pr_quo, W_pr_rem, W2, W2_pr;
            n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
            // --> W = remainder of division of 2**FLINT_BITS by n

            n_mulmod_precomp_shoup_quo_rem(&W_pr_quo, &W_pr_rem, W, mod.n);
            n_mulmod_and_precomp_shoup(&W2, &W2_pr, W, W, W_pr_quo, W_pr_rem, W_pr_quo, mod.n);
            // --> W2 = remainder of division of 2**(2*FLINT_BITS) by n

            // naive: res[i] = (vec[i] * W2 mod n + vec[i+1] * W mod n + vec[i+2] mod n) mod n
            ulong candidate = _nmod_add(n_mulmod_shoup(W2, a_hi_red, W2_pr, mod.n),
                               _nmod_add(n_mulmod_shoup(W, a_me, W_pr_quo, mod.n),
                                         n_modred_barrett(a_lo, one_barrett, mod.n),
                                         mod),
                               mod);

            if (witness != candidate)
            { flint_printf("\n\n\nFAILURE 3 limbs!!\n\n\n"); return 0; }
        }

        if (bits < FLINT_BITS)
        { // 3 limbs, less direct
            NMOD_RED3(witness, a_hi_red, a_me, a_lo, mod);

            ulong one_barrett, W, W_pr_quo, W_pr_rem, W2, W2_pr;
            n_mulmod_precomp_shoup_quo_rem(&one_barrett, &W, 1L, mod.n);
            // --> W = remainder of division of 2**FLINT_BITS by n

            n_mulmod_precomp_shoup_quo_rem(&W_pr_quo, &W_pr_rem, W, mod.n);
            n_mulmod_and_precomp_shoup(&W2, &W2_pr, W, W, W_pr_quo, W_pr_rem, W_pr_quo, mod.n);
            // --> W2 = remainder of division of 2**(2*FLINT_BITS) by n

            // naive: res[i] = (vec[i] * W2 mod n + vec[i+1] * W mod n + vec[i+2] mod n) mod n
            ulong c0 = n_mulmod_shoup(W2, a_hi_red, W2_pr, mod.n);
            ulong c1 = n_mulmod_shoup(W, a_me, W_pr_quo, mod.n);
            ulong candidate = _nmod_add(c0, c1, mod);
            ulong correct = 0;
            if (a_lo >= (mod.n << (mod.norm-1)))
                correct = (mod.n << (mod.norm-1));
            candidate += a_lo - correct;
            umul_ppmm(c1, c0, one_barrett, candidate);
            candidate -= c1 * mod.n;
            if (candidate >= mod.n)
                candidate -= mod.n;

            if (witness != candidate)
            { flint_printf("\n\n\nFAILURE 3 limbs less direct!!\n\n\n"); return 0; }
        }
    }

    FLINT_TEST_CLEAR(state);
    return 1;
}


/***************
*  FFT_SMALL  *
***************/

FLINT_FORCE_INLINE void vec8n_store_unaligned(ulong* z, vec8n a) {
    vec4n_store_unaligned(z+0, a.e1);
    vec4n_store_unaligned(z+4, a.e2);
}

FLINT_FORCE_INLINE vec8n vec8d_convert_limited_vec8n(vec8d a) {
    vec8n z = {vec4d_convert_limited_vec4n(a.e1), vec4d_convert_limited_vec4n(a.e2)};
    return z;
}

FLINT_FORCE_INLINE vec4n vec4n_bit_shift_left(vec4n a, ulong b) {
    return _mm256_slli_epi64(a, b);
}

FLINT_FORCE_INLINE vec8n vec8n_bit_shift_left(vec8n a, ulong b) {
    vec8n z = {vec4n_bit_shift_left(a.e1, b), vec4n_bit_shift_left(a.e2, b)};
    return z;
}

void
prof_nmod_vec_reduce_fft_small_256(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    slong i;

    vec4d p = vec4d_set_d((double) mod.n);
    vec4d pinv = vec4d_set_d(1.0 / mod.n);

    for (i = 0; i + 3 < len; i += 4)
    {
        vec4n t = vec4n_load_unaligned(vec + i);
        vec4d d = vec4n_convert_limited_vec4d(t);
        vec4d r = vec4d_reduce_to_0n(d, p, pinv);
        vec4n u = vec4d_convert_limited_vec4n(r);
        vec4n_store_unaligned(res + i, u);
    }
}

void
prof_nmod_vec_reduce_fft_small_512(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    slong i;

    vec8d p = vec8d_set_d((double) mod.n);
    vec8d pinv = vec8d_set_d(1.0 / mod.n);

    for (i = 0; i + 7 < len; i += 8)
    {
        vec8n t = vec8n_load_unaligned(vec + i);
        vec8d d = vec8n_convert_limited_vec8d(t);
        vec8d r = vec8d_reduce_to_0n(d, p, pinv);
        vec8n u = vec8d_convert_limited_vec8n(r);
        vec8n_store_unaligned(res + i, u);
    }
}


/* the three ones are unused/unmodified yet */
void
_nmod_vec_reduce_simd2(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    slong i;

    vec8d n = vec8d_set_d(mod.n);
    vec8d ninv = vec8d_set_d(1.0 / mod.n);

    ulong n32 = UWORD(1) << 32;

    for (i = 0; i + 7 < len; i += 8)
    {
        vec8n t = vec8n_load_unaligned(vec + i);
        vec8d tlo = vec8n_convert_limited_vec8d(vec8n_bit_and(t, vec8n_set_n(n32-1)));
        vec8d thi = vec8n_convert_limited_vec8d(vec8n_bit_shift_right_32(t));
        vec8d d = vec8d_add(tlo, vec8d_mulmod(thi, vec8d_set_d(n32), n, ninv));
        vec8d r = vec8d_reduce_to_0n(d, n, ninv);
        vec8n u = vec8d_convert_limited_vec8n(r);
        vec8n_store_unaligned(res + i, u);
    }
}

void
_nmod_vec_reduce_ll_simd2(nn_ptr res, nn_srcptr vec, nn_srcptr vec2, slong len, nmod_t mod)
{
    slong i;

    vec8d n = vec8d_set_d(mod.n);
    vec8d ninv = vec8d_set_d(1.0 / mod.n);

    // xxx: distinguish mod / not mod
    ulong n32 = UWORD(1) << 32;
    ulong n64 = nmod_mul(n32, n32, mod);
    ulong n96 = nmod_mul(n64, n32, mod);

    for (i = 0; i + 7 < len; i += 8)
    {
        vec8n t = vec8n_load_unaligned(vec + i);
        vec8d tlo = vec8n_convert_limited_vec8d(vec8n_bit_and(t, vec8n_set_n(n32-1)));
        vec8d thi = vec8n_convert_limited_vec8d(vec8n_bit_shift_right_32(t));

        vec8n v = vec8n_load_unaligned(vec2 + i);
        vec8d vlo = vec8n_convert_limited_vec8d(vec8n_bit_and(v, vec8n_set_n(n32-1)));
        vec8d vhi = vec8n_convert_limited_vec8d(vec8n_bit_shift_right_32(v));

        vec8d d = vec8d_add(tlo, vec8d_mulmod(thi, vec8d_set_d(n32), n, ninv));
              d = vec8d_add(d, vec8d_mulmod(vlo, vec8d_set_d(n64), n, ninv));
              d = vec8d_add(d, vec8d_mulmod(vhi, vec8d_set_d(n96), n, ninv));

        vec8d r = vec8d_reduce_to_0n(d, n, ninv);
        vec8n u = vec8d_convert_limited_vec8n(r);
        vec8n_store_unaligned(res + i, u);
    }
}

void
_nmod_vec_reduce_ll_simd3(nn_ptr res, nn_srcptr vec, nn_srcptr vec2, slong len, nmod_t mod)
{
    slong i;

    vec8d n = vec8d_set_d(mod.n);
    vec8d ninv = vec8d_set_d(1.0 / mod.n);

    ulong n32 = UWORD(1) << 32;
    ulong n48 = UWORD(1) << 48; /* FIXME modified, was << 32 */
    ulong n64 = nmod_mul(n32, n32, mod);
    ulong n96 = nmod_mul(n64, n32, mod);

    for (i = 0; i + 7 < len; i += 8)
    {
        vec8n tlo = vec8n_load_unaligned(vec + i);
        vec8n thi = vec8n_load_unaligned(vec2 + i);

        vec8n u0 = vec8n_bit_and(tlo, vec8n_set_n(n48-1));

        vec8n u1lo = vec8n_bit_shift_right(tlo, 48);
        vec8n u1hi = vec8n_bit_and(thi, vec8n_set_n(n32-1));
              u1hi = vec8n_bit_shift_left(thi, 16);
        vec8n u1 = vec8n_bit_and(u1lo, u1hi);

        vec8n u2 = vec8n_bit_shift_right(thi, 32);

        vec8d c0 = vec8n_convert_limited_vec8d(u0);
        vec8d c1 = vec8n_convert_limited_vec8d(u1);
        vec8d c2 = vec8n_convert_limited_vec8d(u2);

        vec8d d = vec8d_add(c0, vec8d_mulmod(c1, vec8d_set_d(n48), n, ninv));
              d = vec8d_add(d,  vec8d_mulmod(c2, vec8d_set_d(n96), n, ninv));

        vec8d r = vec8d_reduce_to_0n(d, n, ninv);
        vec8n u = vec8d_convert_limited_vec8n(r);
        vec8n_store_unaligned(res + i, u);
    }
}

/*******************
*  END FFT_SMALL  *
*******************/


#define SAMPLE(fun)                                       \
void sample_##fun(void * arg, ulong count)                \
{                                                         \
    ulong n;                                              \
    nmod_t mod;                                           \
    info_t * info = (info_t *) arg;                       \
    flint_bitcnt_t bits = info->bits;                     \
    nn_ptr vec = _nmod_vec_init(LEN);                     \
    nn_ptr res = _nmod_vec_init(LEN);                     \
    FLINT_TEST_INIT(state);                               \
                                                          \
    for (ulong j = 0; j < LEN; j++)                       \
        vec[j] = n_randlimb(state);                       \
                                                          \
    prof_start();                                         \
    for (ulong i = 0; i < count; i++)                     \
    {                                                     \
        n = n_randbits(state, bits);                      \
        nmod_init(&mod, n);                               \
                                                          \
        prof_nmod_vec_reduce_##fun(res, vec, LEN, mod);   \
    }                                                     \
    prof_stop();                                          \
                                                          \
    _nmod_vec_clear(vec);                                 \
    _nmod_vec_clear(res);                                 \
    FLINT_TEST_CLEAR(state);                              \
}

SAMPLE(nmod_red)
SAMPLE(nmod_red_unroll)
SAMPLE(barrett)
SAMPLE(barrett_unroll)
SAMPLE(barrett_simd)
SAMPLE(barrett_simd_novec)
#ifdef FLINT_HAVE_AVX512
    SAMPLE(barrett_simdexplicit)
#endif // FLINT_HAVE_AVX512

SAMPLE(barrett2)
SAMPLE(barrett2_unroll)
SAMPLE(barrett2_lazy)
SAMPLE(barrett2_lazy_unroll)
SAMPLE(fast_lazy)
SAMPLE(fast_lazy_novec)
SAMPLE(fast)
SAMPLE(fast_novec)

/* fft_small */
SAMPLE(fft_small_256)
SAMPLE(fft_small_512)

SAMPLE(nmod2_red2)
SAMPLE(nmod2_red2_unroll)
SAMPLE(barrett22)
SAMPLE(barrett22_unroll)
SAMPLE(barrett22_bis)
SAMPLE(barrett22_ter)

int main(int argc, char ** argv)
{
    const slong nb = 23;
    double min[nb], max[nb];
    char * description[] =
    {
        "red1: nmod_red",
        "red1: nmod_red unroll",
        "red1: barrett",
        "red1: barrett unroll",
        "red1: barrett_simd",
        "red1: barrett_simd novec",
#ifdef FLINT_HAVE_AVX512
        "red1: barrett_simdexplicit",
#endif // FLINT_HAVE_AVX512
        "red1: barrett2",
        "red1: barrett2 unroll",
        "red1: barrett2_lazy",
        "red1: barrett2_lazy unroll",
        "red1: fast_lazy (!lazy!)",
        "red1: fast_lazy novec (!lazy!)",
        "red1: fast",
        "red1: fast novec",
#ifdef FLINT_HAVE_AVX512
        "red1: fft_small_256",
        "red1: fft_small_512",
#endif // FLINT_HAVE_AVX512
        "red2: nmod2_red2",
        "red2: nmod2_red2 unroll",
        "red2: barrett22 (<64b)",
        "red2: barrett22_bis (<64b)",
        "red2: barrett22_ter (<64b)",
        "red2: barrett22 unroll (<64b)",
    };

    info_t info;
    flint_bitcnt_t nbits = atoi(argv[1]);
    info.bits = nbits;

    int correct = check(nbits);
    flint_printf("correct ? %s\n", correct ? "pass" : "fail");

    slong i = 0;
    prof_repeat(min+i, max+i, sample_nmod_red, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_nmod_red_unroll, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett_unroll, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett_simd, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett_simd_novec, (void *) &info);
    i++;
#ifdef FLINT_HAVE_AVX512
    prof_repeat(min+i, max+i, sample_barrett_simdexplicit, (void *) &info);
    i++;
#endif // FLINT_HAVE_AVX512

    prof_repeat(min+i, max+i, sample_barrett2, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett2_unroll, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett2_lazy, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett2_lazy_unroll, (void *) &info);
    i++;

    prof_repeat(min+i, max+i, sample_fast_lazy, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_fast_lazy_novec, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_fast, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_fast_novec, (void *) &info);
    i++;

#ifdef FLINT_HAVE_AVX512
    prof_repeat(min+i, max+i, sample_fft_small_256, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_fft_small_512, (void *) &info);
    i++;
#endif // FLINT_HAVE_AVX512

    prof_repeat(min+i, max+i, sample_nmod2_red2, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_nmod2_red2_unroll, (void *) &info);
    i++;

    if (nbits == FLINT_BITS)
    {
        min[i] = -1.;
        i++;
        min[i] = -1.;
        i++;
        min[i] = -1.;
        i++;
        min[i] = -1.;
        i++;
    }
    else
    {
        prof_repeat(min+i, max+i, sample_barrett22, (void *) &info);
        i++;
        prof_repeat(min+i, max+i, sample_barrett22_bis, (void *) &info);
        i++;
        prof_repeat(min+i, max+i, sample_barrett22_ter, (void *) &info);
        i++;
        prof_repeat(min+i, max+i, sample_barrett22_unroll, (void *) &info);
        i++;
    }

    for (i = 0; i < nb; i++)
    {
        flint_printf("%-30s : %10f\n", description[i], min[i]);
    }



    return 0;
}


#include <stdlib.h>  /* for atoi */

#include "flint/flint.h"
#include "flint/longlong.h"
#include "flint/longlong_div_gnu.h"
#include "flint/profiler.h"
#include "flint/nmod_vec.h"
#include "flint/ulong_extras.h"

// must be a multiple of 6
#define LEN 10002

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
    // q <- floor(a * n_pr / 2**tt)
    ulong q, tmp;

    umul_ppmm(q, tmp, a, n_pr);
    q >>= shift;

    // r <- a - q * n
    a -= q * n;
    if (a >= n)
        a -= n;

    return a;
}

static inline
ulong n_modred_barrett2_lazy(ulong a, ulong n_pr, flint_bitcnt_t shift, ulong n)
{
    // q <- floor(a * n_pr / 2**tt)
    ulong q, tmp;

    umul_ppmm(q, tmp, a, n_pr);
    q >>= shift;

    // r <- a - q * n
    a -= q * n;

    return a;
}

/*-----------------------*/
/* double word reduction */
/*-----------------------*/

#define SAMPLE(fun)                                       \
void sample_##fun(void * arg, ulong count)                \
{                                                         \
    ulong n;                                              \
    nmod_t mod;                                           \
    info_t * info = (info_t *) arg;                       \
    flint_bitcnt_t bits = info->bits;                     \
    nn_ptr vec = _nmod_vec_init(LEN);                     \
    nn_ptr vec2 = _nmod_vec_init(LEN);                    \
    FLINT_TEST_INIT(state);                               \
                                                          \
    for (ulong j = 0; j+2 < LEN; j+=3)                    \
    {                                                     \
        /* already reduced, for NMOD_RED3 */              \
        vec[j+0] = n_randbits(state, bits-1);             \
        vec[j+1] = n_randlimb(state);                     \
        vec[j+2] = n_randlimb(state);                     \
    }                                                     \
                                                          \
    prof_start();                                         \
    for (ulong i = 0; i < count; i++)                     \
    {                                                     \
        n = n_randbits(state, bits);                      \
        nmod_init(&mod, n);                               \
                                                          \
        prof_nmod_vec_reduce_##fun(vec2, vec, LEN, mod);  \
    }                                                     \
    prof_stop();                                          \
                                                          \
    _nmod_vec_clear(vec);                                 \
    _nmod_vec_clear(vec2);                                \
    FLINT_TEST_CLEAR(state);                              \
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

// note: in this particular instance of n_mulmod_shoup, no restriction on n
void prof_nmod_vec_reduce_barrett(nn_ptr res, nn_srcptr vec, slong len, nmod_t mod)
{
    const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
    for (slong i = 0 ; i < len; i++)
        res[i] = n_modred_barrett(vec[i], one_barrett, mod.n);
}

// note: in this particular instance of n_mulmod_shoup, no restriction on n
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

    for (ulong i = 0; i < LEN; i++)
    {
        n = n_randbits(state, bits);
        nmod_init(&mod, n);

        ulong a_lo = n_randlimb(state);
        ulong a_me = n_randlimb(state);
        ulong a_hi = n_randlimb(state);
        ulong a_hi_red = n_randint(state, n);

        ulong witness;
        { // 1 limb
            NMOD_RED(witness, a_lo, mod);

            const ulong one_barrett = n_mulmod_precomp_shoup(1L, mod.n);
            ulong candidate = n_modred_barrett(a_lo, one_barrett, mod.n);

            if (witness != candidate)
            { printf("\n\n\nFAILURE 1 limb modred_barrett!!\n\n\n"); return 0; }

            flint_bitcnt_t shift;
            ulong n_pr;
            n_precomp_modred_barrett2(&shift, &n_pr, n);
            candidate = n_modred_barrett2(a_lo, n_pr, shift, n);

            if (witness != candidate)
            { printf("\n\n\nFAILURE 1 limb modred_barrett2!!\n\n\n"); return 0; }
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
            { printf("\n\n\nFAILURE 2 limbs direct!!\n\n\n"); return 0; }
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
            //{ printf("\n\n\nFAILURE 2 limbs!!\n\n\n"); printf("%lu, %lu, %lu, %lu, %lu\n", mod.n, a_lo, a_hi, candidate, witness); return 0; }
            { printf("\n\n\nFAILURE 2 limbs!!\n\n\n"); return 0; }
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
            //{ printf("\n\n\nFAILURE 2 limbs bis!!\n\n\n"); printf("%lu, %lu, %lu, %lu, %lu\n", mod.n, a_lo, a_hi, candidate, witness); return 0; }
            { printf("\n\n\nFAILURE 2 limbs!!\n\n\n"); return 0; }
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
            { printf("\n\n\nFAILURE 3 limbs!!\n\n\n"); return 0; }
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
            { printf("\n\n\nFAILURE 3 limbs less direct!!\n\n\n"); return 0; }
        }
    }

    FLINT_TEST_CLEAR(state);
    return 1;
}

SAMPLE(nmod_red)
SAMPLE(nmod_red_unroll)
SAMPLE(barrett)
SAMPLE(barrett_unroll)
SAMPLE(barrett2)
SAMPLE(barrett2_unroll)

SAMPLE(nmod2_red2)
SAMPLE(nmod2_red2_unroll)
SAMPLE(barrett22)
SAMPLE(barrett22_unroll)
SAMPLE(barrett22_bis)
SAMPLE(barrett22_ter)

int main(int argc, char ** argv)
{
    const slong nb = 12;
    double min[nb], max[nb];
    char * description[] =
    {
        "red1: nmod_red",
        "red1: nmod_red unroll",
        "red1: barrett",
        "red1: barrett unroll",
        "red1: barrett2",
        "red1: barrett2 unroll",
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
    prof_repeat(min+i, max+i, sample_barrett2, (void *) &info);
    i++;
    prof_repeat(min+i, max+i, sample_barrett2_unroll, (void *) &info);
    i++;

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


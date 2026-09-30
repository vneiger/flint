/*
    Copyright (C) 2026 Vincent Neiger

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

/*
   TL;DR a good way to run measurements that help for tuning crossovers:
    ./build/nmod_poly/profile/p-mulmid -q -n 3 -F 0,1,2,3,5,6 -b 0,20,30,50,60,64 -f 16:1024:x1.25 -r 0.1,1,10 > mulmid-$(hostname).txt
 */

/*
    The implementations of the middle product over Z/nZ, side by side,
    and against the multiplication that it is the transpose of.

    For `1 <= fn` and `outlen >= 1`, let `gn = fn + outlen - 1`. The
    middle product of `(f, fn)` and `(g, gn)` is the range `[fn - 1, gn)`
    of `f*g`, the `outlen` coefficients that are sums of the full `fn`
    terms. Every timed middle product computes exactly that, so they are
    directly comparable; they are all called with the longer operand
    first, as the fft_small entry points document.

      #0 mulmid       _nmod_poly_mulmid, the dispatcher: what a caller
                      actually gets;
      #1 classical    _nmod_poly_mulmid_classical, outlen dot products
                      of length fn;
      #2 KS           _nmod_poly_mulmid_KS, Kronecker substitution;

    and, where fft_small is available,

      #3 fft_small    _nmod_poly_mulmid_fft_small, i.e. the 16-bit
                      repacking when it applies and #4 otherwise;
      #4 window       _nmod_poly_mul_mid_default_mpn_ctx, the window of
                      the full fn by gn product;
      #5 fft mul      the same entry point as #4, asked for the full
                      product of lengths fn and outlen.

    Finally, always,

      #6 mul          _nmod_poly_mul, the multiplication dispatcher, on
                      lengths fn and outlen.

    #5 and #6 are not middle products. By the transposition principle,
    the middle product of this shape is the transpose of the product of
    lengths fn and outlen, and costs the same: #6 is the target for #0,
    and #5 is the target for #4.

    Ratio columns:

      best            which of #1, #2, #3 is fastest: the three the
                      dispatcher chooses from (#4 is what #3 calls when
                      it does not repack, so it is not a separate choice);
      #0/best         the dispatcher against the best of those: 1.00 means
                      it chose right, above 1 is what the wrong choice costs;
      #0/#6           the dispatcher against multiplication: what the
                      middle product costs above its floor;
      #4/#5           the same for fft_small alone.

    and, where fft_small is available, the plan fft_small uses for #4:
    `np` the number of primes (`1d` when it transforms directly modulo
    mod.n) and `ztrunc` the (possibly truncated) transform length. Its
    cost is mostly a function of (np, ztrunc), and ztrunc is a staircase
    in fn and outlen (a power of two below 512, a multiple of 256
    above), which is what a length threshold does not see.

    Every middle product is checked against a reference (#4, or #2 without
    fft_small) on each shape before anything is timed, and #6 against #5.
    Shapes where classical would do more than 2e7 coefficient products
    skip it. Unless -a is given, #1 and #2 also drop out of the rest of
    the run once they are hopeless (25x slower than the reference and
    above 50 ms).

    Usage:
        p-mulmid [options]

        -b B1,B2,...    bit lengths of the modulus, one table each
                        (default 60). The modulus is the first prime
                        >= 2^(B-1); B = 0 selects one of the fft_small
                        context's own fft primes (single-prime plan, no
                        CRT). An entry of the form "=N" uses the modulus
                        N itself (any N >= 2, prime or not).
        -f LIST         the lengths fn (default 16:4096, see below);
        -o LIST         absolute values of outlen;
        -r R1,R2,...    outlen = round(R*fn), for each real R.
                        -o and -r can be combined: each fn is timed with
                        the union of both. Neither: -r 1 (balanced).
        -F I1,I2,...    time only these functions (default: all). The
                        reference is still computed for the checks.
        -t T            number of threads (default 1).
        -n K            each time is the minimum over K timings (default
                        1); useful on noisy machines.
        -q              quick: aim at 20 ms per timing rather than 100 ms.
        -a              never drop #1 and #2 from the run.

        A LIST is comma separated, each item one of
            a           the single value a;
            a:b         a to b by steps x -> x + 1 + x/2;
            a:b:xR      a to b by factors R > 1 (at least +1 each step);
            a:b:+S      a to b by steps of S.
        For instance -f 100:200:+10 -r 1, or -f 2:200:x1.3 -o 1024,4096.

    Examples:

        p-mulmid -b 60 -f 100:220:+8 -r 1
            where the balanced crossovers are, at 60 bits;
        p-mulmid -b 60 -f 227 -o 16:2048:x2
            one fn, outlen from short to long;
        p-mulmid -b 20,30,50,60,64 -f 2:256:x1.25 -o 4096
            short f against many outputs, several moduli;
        p-mulmid -f 127,128,129,255,256,257 -r 1 -F 0,1,3,6
            around transform length boundaries, only a few columns.

    Single measurements, for scripting one point at a time (nbits as in
    -b, without the "=N" form; the output is one line):

        p-mulmid nbits fun fn outlen
            function #fun on the middle product above, with
            gn = fn + outlen - 1;
        p-mulmid nbits fun fn gn nlo nhi
            function #fun on a general window: the coefficients
            [nlo, nhi) of the product of f of length fn and g of length
            gn, where 0 <= nlo < nhi <= fn + gn - 1 and the lengths are
            not tied to the window. The operands are passed longer first;
            #5 and #6 then time the full product of f and g (fft_small,
            _nmod_poly_mul), for comparison. The result of #fun is
            checked against the reference first.
*/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ulong_extras.h"
#include "nmod.h"
#include "nmod_vec.h"
#include "nmod_poly.h"
#if FLINT_HAVE_FFT_SMALL
# include "fft_small.h"
#endif
#include "profiler.h"

/* ------------------------------------------------------------------ */
/* the timed functions, under one signature                            */
/* ------------------------------------------------------------------ */

/* For the middle products: writes to z the outlen = gn - fn + 1
   coefficients of the range [fn - 1, gn) of the product of (f, fn) and
   (g, gn). For the multiplications (#5, #6): writes to z the gn
   coefficients of the product of (f, fn) and (g, outlen), where
   outlen = gn - fn + 1. */
typedef void (*mulmid_fun) (nn_ptr z, nn_srcptr f, slong fn,
                            nn_srcptr g, slong gn, nmod_t mod);

static void
mid_dispatch(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    _nmod_poly_mulmid(z, g, gn, f, fn, fn - 1, gn, mod);
}

static void
mid_classical(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    _nmod_poly_mulmid_classical(z, g, gn, f, fn, fn - 1, gn, mod);
}

static void
mid_KS(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    _nmod_poly_mulmid_KS(z, g, gn, f, fn, fn - 1, gn, mod);
}

/* the multiplication dispatcher on lengths fn and outlen, longer first
   as _nmod_poly_mul requires */
static void
mul_dispatch(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    slong outlen = gn - fn + 1;

    if (fn >= outlen)
        _nmod_poly_mul(z, f, fn, g, outlen, mod);
    else
        _nmod_poly_mul(z, g, outlen, f, fn, mod);
}

#if FLINT_HAVE_FFT_SMALL

static void
mid_fft_small(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    _nmod_poly_mulmid_fft_small(z, g, gn, f, fn, fn - 1, gn, mod);
}

static void
mid_window(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    _nmod_poly_mul_mid_default_mpn_ctx(z, fn - 1, gn, g, gn, f, fn, mod);
}

/* the fft_small product of lengths fn and outlen, through the same
   entry point as #4 so that the two differ in the shape alone */
static void
mul_fft(nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn, nmod_t mod)
{
    slong outlen = gn - fn + 1;

    if (fn >= outlen)
        _nmod_poly_mul_mid_default_mpn_ctx(z, 0, gn, f, fn, g, outlen, mod);
    else
        _nmod_poly_mul_mid_default_mpn_ctx(z, 0, gn, g, outlen, f, fn, mod);
}

#endif

#define NFUNS 7
#define IFUN_MUL_FFT 5
#define IFUN_MUL 6
#define IS_MID(j) ((j) < IFUN_MUL_FFT)

#if FLINT_HAVE_FFT_SMALL
# define IFUN_REF 4     /* middle products are checked against the window */
#else
# define IFUN_REF 2     /* no fft_small: KS is the only fast reference */
#endif

static const mulmid_fun funs[NFUNS] = {
    mid_dispatch,       /* 0 */
    mid_classical,      /* 1 */
    mid_KS,             /* 2 */
#if FLINT_HAVE_FFT_SMALL
    mid_fft_small,      /* 3 */
    mid_window,         /* 4 */
    mul_fft,            /* 5 */
#else
    NULL, NULL, NULL,
#endif
    mul_dispatch,       /* 6 */
};

static const char * const shortname[NFUNS] = {
    "mulmid", "cl", "KS", "fft", "window", "fftmul", "mul"
};

static const char * const collabel[NFUNS] = {
    "#0", "#1", "#2", "#3", "#4", "#5", "#6"
};

static const char * const description[NFUNS] = {
    "#0  --> _nmod_poly_mulmid                    (the dispatcher)",
    "#1  --> _nmod_poly_mulmid_classical          (dot products)",
    "#2  --> _nmod_poly_mulmid_KS                 (Kronecker substitution)",
    "#3  --> _nmod_poly_mulmid_fft_small          (repacking, else #4)",
    "#4  --> _nmod_poly_mul_mid_default_mpn_ctx   (window of the full product)",
    "#5  --> _nmod_poly_mul_mid_default_mpn_ctx   (product of lengths fn, outlen)",
    "#6  --> _nmod_poly_mul                       (product of lengths fn, outlen)",
};

static int
fun_available(int j)
{
    return funs[j] != NULL;
}

/* the classical one is quadratic; above this many coefficient products
   its timing is a foregone conclusion and only costs wall time */
#define CLASSICAL_MAX_WORK 2.0e7

/* #1 and #2 are dropped from the rest of the run once they are this
   much slower than the reference and slow in absolute terms */
#define SKIP_FACTOR 25.0
#define SKIP_MIN_TIME 5.0e-2

/* ------------------------------------------------------------------ */
/* options                                                             */
/* ------------------------------------------------------------------ */

typedef struct
{
    slong * v;
    slong len;
    slong alloc;
}
slong_list_struct;

typedef struct
{
    /* moduli: either a bit length (value > 0 with is_n = 0, 0 for the
       fft prime) or an explicit modulus (is_n = 1) */
    ulong mods[64];
    int mod_is_n[64];
    slong nmods;

    slong_list_struct fns;
    slong_list_struct outlens;
    double ratios[64];
    slong nratios;

    int timed[NFUNS];
    slong threads;
    slong nrep;
    slong target_ms;
    int no_skip;
}
options_struct;

static void
list_push(slong_list_struct * L, slong x)
{
    if (L->len == L->alloc)
    {
        L->alloc = FLINT_MAX(16, 2 * L->alloc);
        L->v = flint_realloc(L->v, L->alloc * sizeof(slong));
    }

    L->v[L->len++] = x;
}

static int
cmp_slong(const void * a, const void * b)
{
    slong x = *(const slong *) a, y = *(const slong *) b;
    return (x > y) - (x < y);
}

/* splits a comma separated list in place: returns the next item, or
   NULL at the end (strtok_r is not portable) */
static char *
next_item(char ** rest)
{
    char * item = *rest;
    char * c;

    if (item == NULL || *item == '\0')
        return NULL;

    c = strchr(item, ',');

    if (c != NULL)
    {
        *c = '\0';
        *rest = c + 1;
    }
    else
        *rest = NULL;

    return item;
}

/* sorts and removes duplicates */
static void
list_normalise(slong_list_struct * L)
{
    slong i, j;

    if (L->len == 0)
        return;

    qsort(L->v, L->len, sizeof(slong), cmp_slong);

    for (i = j = 1; i < L->len; i++)
        if (L->v[i] != L->v[j - 1])
            L->v[j++] = L->v[i];

    L->len = j;
}

/* parses one LIST argument into L; returns 0 on a syntax error */
static int
parse_list(slong_list_struct * L, const char * s)
{
    char * buf = flint_malloc(strlen(s) + 1);
    char * item, * rest;
    int ok = 1;

    strcpy(buf, s);

    for (rest = buf; (item = next_item(&rest)) != NULL; )
    {
        char * end;
        slong a, b, x;

        a = strtol(item, &end, 10);

        if (end == item || a < 1)
        {
            ok = 0;
            break;
        }

        if (*end == '\0')
        {
            list_push(L, a);
            continue;
        }

        if (*end != ':')
        {
            ok = 0;
            break;
        }

        item = end + 1;
        b = strtol(item, &end, 10);

        if (end == item || b < a)
        {
            ok = 0;
            break;
        }

        if (*end == '\0')
        {
            for (x = a; x <= b; x += 1 + x / 2)
                list_push(L, x);
        }
        else if (end[0] == ':' && end[1] == '+')
        {
            slong step = strtol(end + 2, &end, 10);

            if (step < 1 || *end != '\0')
            {
                ok = 0;
                break;
            }

            for (x = a; x <= b; x += step)
                list_push(L, x);
        }
        else if (end[0] == ':' && end[1] == 'x')
        {
            double r = strtod(end + 2, &end);
            double y;

            if (!(r > 1.0) || *end != '\0')
            {
                ok = 0;
                break;
            }

            /* the grid is a*r^k rounded, with no repetition */
            for (x = a, y = a; x <= b; )
            {
                slong xn;

                list_push(L, x);
                y *= r;
                xn = (slong) floor(y + 0.5);
                x = FLINT_MAX(x + 1, xn);
            }
        }
        else
        {
            ok = 0;
            break;
        }
    }

    flint_free(buf);
    return ok;
}

static int
parse_int_list(slong * v, slong * n, slong nmax, const char * s)
{
    char * buf = flint_malloc(strlen(s) + 1);
    char * item, * rest;
    int ok = 1;

    strcpy(buf, s);
    *n = 0;

    for (rest = buf; (item = next_item(&rest)) != NULL; )
    {
        char * end;
        slong x = strtol(item, &end, 10);

        if (end == item || *end != '\0' || *n >= nmax)
        {
            ok = 0;
            break;
        }

        v[(*n)++] = x;
    }

    flint_free(buf);
    return ok;
}

static int
parse_mods(options_struct * O, const char * s)
{
    char * buf = flint_malloc(strlen(s) + 1);
    char * item, * rest;
    int ok = 1;

    strcpy(buf, s);
    O->nmods = 0;

    for (rest = buf; (item = next_item(&rest)) != NULL; )
    {
        char * end;
        int is_n = (item[0] == '=');
        ulong x = strtoul(item + is_n, &end, 10);

        if (end == item + is_n || *end != '\0' || O->nmods >= 64
            || (is_n && x < 2) || (!is_n && x > FLINT_BITS))
        {
            ok = 0;
            break;
        }

        O->mods[O->nmods] = x;
        O->mod_is_n[O->nmods] = is_n;
        O->nmods++;
    }

    flint_free(buf);
    return ok;
}

static int
parse_ratios(options_struct * O, const char * s)
{
    char * buf = flint_malloc(strlen(s) + 1);
    char * item, * rest;
    int ok = 1;

    strcpy(buf, s);
    O->nratios = 0;

    for (rest = buf; (item = next_item(&rest)) != NULL; )
    {
        char * end;
        double r = strtod(item, &end);

        if (end == item || *end != '\0' || !(r > 0.0) || O->nratios >= 64)
        {
            ok = 0;
            break;
        }

        O->ratios[O->nratios++] = r;
    }

    flint_free(buf);
    return ok;
}

/* ------------------------------------------------------------------ */
/* timing                                                              */
/* ------------------------------------------------------------------ */

/* a wall clock in seconds: clock_gettime where there is one, otherwise
   the millisecond clock of profiler.h (then keep the target >= 20 ms) */
static double
wall_clock(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
#else
    timeit_t T;
    timeit_start(T);
    return -1e-3 * (double) T->wall;
#endif
}

/* wall clock seconds for one call, on already-allocated operands: the
   repetition count is grown until one batch takes target_ms, then the
   minimum over nrep batches is returned */
static double
time_fun(int ifun, nn_ptr z, nn_srcptr f, slong fn, nn_srcptr g, slong gn,
         nmod_t mod, const options_struct * O)
{
    double target = 1e-3 * O->target_ms;
    double t0, t, best;
    slong reps, k, r;

    for (reps = 1; ; )
    {
        t0 = wall_clock();
        for (k = 0; k < reps; k++)
            funs[ifun](z, f, fn, g, gn, mod);
        t = wall_clock() - t0;

        if (t >= target)
            break;

        /* aim straight at the target once the batch is measurable */
        if (t >= 0.05 * target)
            reps = (slong) (reps * (1.05 * target / t)) + 1;
        else
            reps *= 10;
    }

    best = t / reps;

    for (r = 1; r < O->nrep; r++)
    {
        t0 = wall_clock();
        for (k = 0; k < reps; k++)
            funs[ifun](z, f, fn, g, gn, mod);
        t = wall_clock() - t0;

        best = FLINT_MIN(best, t / reps);
    }

    return best;
}

/* the first prime at least 2^(nbits-1), or an fft prime of the
   fft_small default context for nbits = 0, or n itself */
static void
select_modulus(nmod_t * mod, ulong x, int is_n)
{
    if (is_n)
    {
        nmod_init(mod, x);
        return;
    }

#if FLINT_HAVE_FFT_SMALL
    if (x == 0)
    {
        mpn_ctx_struct * R = get_default_mpn_ctx();

        nmod_init(mod, R->ffts[1].mod.n);
        return;
    }
#endif

    if (x < 2)
        x = 60;

    if (x == FLINT_BITS)
        nmod_init(mod, n_nextprime(UWORD_MAX - 1000, 1));
    else
        nmod_init(mod, n_nextprime(UWORD(1) << (x - 1), 1));
}

#if FLINT_HAVE_FFT_SMALL

/* as in fft_small/nmod_poly_mul.c */
#define _len_trunc(x) \
    ((n_clog2(x) < LG_BLK_SZ) ? n_pow2(n_max((ulong) 4, n_clog2(x))) \
                              : n_round_up((x), BLK_SZ))

/* the plan that #4 uses: the same arguments as in
   _nmod_poly_mul_mid_mpn_ctx for a = g, b = f */
static void
window_plan(ulong * np, int * direct, ulong * ztrunc,
            slong fn, slong gn, nmod_t mod)
{
    fft_small_plan_t P;
    ulong modbits = FLINT_BITS - mod.norm;
    ulong zn = fn + gn - 1;
    ulong xt = n_max(_len_trunc((ulong) gn), _len_trunc((ulong) fn));

    if (!fft_small_plan_init_nmod(P, get_default_mpn_ctx(), fn - 1, gn, zn,
                                  xt, fn, 2 * modbits, mod, fn))
    {
        *np = 0;
        *direct = 0;
        *ztrunc = 0;
        return;
    }

    *np = P->np;
    *direct = P->use_direct_fft;
    *ztrunc = P->ztrunc;

    fft_small_plan_clear(P);
}

#endif

/* ------------------------------------------------------------------ */
/* the table                                                           */
/* ------------------------------------------------------------------ */

static void
print_header(nmod_t mod, const options_struct * O)
{
    int j;

    flint_printf("# middle product: the range [fn - 1, gn) of f*g, with\n"
                 "# fn = len(f), gn = len(g) = fn + outlen - 1, "
                 "mod.n = %wu (%wu bits), %wd thread(s)\n",
                 mod.n, FLINT_BIT_COUNT(mod.n), flint_get_num_threads());

    for (j = 0; j < NFUNS; j++)
        if (fun_available(j))
            flint_printf("# %s\n", description[j]);

    flint_printf("# times are wall clock seconds for one call (min of %wd); "
                 "'-' is not timed\n", O->nrep);
    flint_printf("# best = fastest of #1..#3; np, ztrunc: fft_small plan "
                 "of #4 (d = direct fft mod n)\n");

    flint_printf("%7s %7s |", "fn", "outlen");

    for (j = 0; j < NFUNS; j++)
    {
        if (!fun_available(j))
            continue;

        if (j == IFUN_MUL_FFT || (j == IFUN_MUL && !fun_available(IFUN_MUL_FFT)))
            flint_printf(" |");

        flint_printf("  %8s", collabel[j]);
    }

    flint_printf(" | %4s %7s %7s", "best", "#0/best", "#0/#6");
#if FLINT_HAVE_FFT_SMALL
    flint_printf(" %7s | %3s %6s", "#4/#5", "np", "ztrunc");
#endif
    flint_printf("\n");
}

static void
print_ratio(double num, double den)
{
    if (num > 0.0 && den > 0.0)
        flint_printf(" %7.2f", num / den);
    else
        flint_printf(" %7s", "-");
}

static void
run_shape(slong fn, slong outlen, nmod_t mod, flint_rand_t state,
          nn_ptr f, nn_ptr g, nn_ptr z, nn_ptr zref,
          int * enabled, const options_struct * O)
{
    slong gn = fn + outlen - 1;
    double t[NFUNS];
    int run[NFUNS];
    int j, jbest;

    _nmod_vec_randtest(f, state, fn, mod);
    _nmod_vec_randtest(g, state, gn, mod);

    for (j = 0; j < NFUNS; j++)
        run[j] = enabled[j] && O->timed[j] && fun_available(j);

    if ((double) fn * (double) outlen > CLASSICAL_MAX_WORK)
        run[1] = 0;

    /* a wrong answer computed quickly is not a data point */
    funs[IFUN_REF](zref, f, fn, g, gn, mod);

    for (j = 0; j < IFUN_MUL_FFT; j++)
    {
        if (!run[j] || j == IFUN_REF)
            continue;

        _nmod_vec_zero(z, outlen);
        funs[j](z, f, fn, g, gn, mod);

        if (!_nmod_vec_equal(z, zref, outlen))
        {
            flint_printf("\nFAIL: %s disagrees at fn = %wd, outlen = %wd, "
                         "mod.n = %wu\n", shortname[j], fn, outlen, mod.n);
            flint_abort();
        }
    }

#if FLINT_HAVE_FFT_SMALL
    if (run[IFUN_MUL])
    {
        funs[IFUN_MUL_FFT](zref, f, fn, g, gn, mod);
        _nmod_vec_zero(z, gn);
        funs[IFUN_MUL](z, f, fn, g, gn, mod);

        if (!_nmod_vec_equal(z, zref, gn))
        {
            flint_printf("\nFAIL: mul disagrees at fn = %wd, outlen = %wd, "
                         "mod.n = %wu\n", fn, outlen, mod.n);
            flint_abort();
        }
    }
#endif

    for (j = 0; j < NFUNS; j++)
        t[j] = run[j] ? time_fun(j, z, f, fn, g, gn, mod, O) : 0.0;

    flint_printf("%7wd %7wd |", fn, outlen);

    for (j = 0; j < NFUNS; j++)
    {
        if (!fun_available(j))
            continue;

        if (j == IFUN_MUL_FFT || (j == IFUN_MUL && !fun_available(IFUN_MUL_FFT)))
            flint_printf(" |");

        if (run[j])
            flint_printf("  %8.2e", t[j]);
        else
            flint_printf("  %8s", "-");
    }

    /* the best of the algorithms the dispatcher picks from */
    jbest = -1;
    for (j = 1; j <= 3; j++)
        if (run[j] && (jbest == -1 || t[j] < t[jbest]))
            jbest = j;

    flint_printf(" | %4s", jbest == -1 ? "-" : shortname[jbest]);
    print_ratio(run[0] ? t[0] : 0.0, jbest == -1 ? 0.0 : t[jbest]);
    print_ratio(run[0] ? t[0] : 0.0, run[IFUN_MUL] ? t[IFUN_MUL] : 0.0);

#if FLINT_HAVE_FFT_SMALL
    print_ratio(run[4] ? t[4] : 0.0, run[5] ? t[5] : 0.0);
    {
        ulong np, ztrunc;
        int direct;

        window_plan(&np, &direct, &ztrunc, fn, gn, mod);
        flint_printf(" | %2wu%s %6wu", np, direct ? "d" : " ", ztrunc);
    }
#endif

    flint_printf("\n");
    fflush(stdout);

    /* the run grows, and so does the gap: once one of the two
       sub-quadratic-at-best implementations is hopeless it is dropped
       for good rather than spending the rest of the run confirming it */
    if (!O->no_skip && run[IFUN_REF])
        for (j = 1; j <= 2; j++)
            if (j != IFUN_REF && run[j] && t[j] > SKIP_FACTOR * t[IFUN_REF]
                              && t[j] > SKIP_MIN_TIME)
                enabled[j] = 0;
}

/* the outlen values for one fn: the absolute ones, and the ratios */
static void
outlens_for(slong_list_struct * L, slong fn, const options_struct * O)
{
    slong i;

    L->len = 0;

    for (i = 0; i < O->outlens.len; i++)
        list_push(L, O->outlens.v[i]);

    for (i = 0; i < O->nratios; i++)
        list_push(L, FLINT_MAX(WORD(1), (slong) floor(O->ratios[i] * fn + 0.5)));

    list_normalise(L);
}

static void
run_table(ulong modx, int is_n, const options_struct * O, flint_rand_t state)
{
    nmod_t mod;
    nn_ptr f, g, z, zref;
    slong maxfn, maxgn, i, k;
    slong_list_struct L = {NULL, 0, 0};
    int enabled[NFUNS];
    int j;

    for (j = 0; j < NFUNS; j++)
        enabled[j] = 1;

    select_modulus(&mod, modx, is_n);
    print_header(mod, O);

    maxfn = 1;
    maxgn = 1;

    for (i = 0; i < O->fns.len; i++)
    {
        slong fn = O->fns.v[i];

        outlens_for(&L, fn, O);

        maxfn = FLINT_MAX(maxfn, fn);
        for (k = 0; k < L.len; k++)
            maxgn = FLINT_MAX(maxgn, fn + L.v[k] - 1);
    }

    f    = _nmod_vec_init(maxfn);
    g    = _nmod_vec_init(maxgn);
    z    = _nmod_vec_init(maxgn);
    zref = _nmod_vec_init(maxgn);

    for (i = 0; i < O->fns.len; i++)
    {
        slong fn = O->fns.v[i];

        outlens_for(&L, fn, O);

        for (k = 0; k < L.len; k++)
            run_shape(fn, L.v[k], mod, state, f, g, z, zref, enabled, O);
    }

    _nmod_vec_clear(f);
    _nmod_vec_clear(g);
    _nmod_vec_clear(z);
    _nmod_vec_clear(zref);
    flint_free(L.v);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static void
usage(const char * name)
{
    int j;

    flint_printf("Usage: %s [options]\n\n", name);
    flint_printf(
"  -b B1,B2,...  modulus bit lengths, one table each (default 60); the modulus\n"
"                is the first prime >= 2^(B-1); B = 0: an fft_small prime\n"
"                (single-prime plan); =N: the modulus N itself\n"
"  -f LIST       the lengths fn = len(f) (default 16:4096)\n"
"  -o LIST       absolute values of outlen (number of output coefficients)\n"
"  -r R1,R2,...  outlen = round(R*fn); -o and -r combine (union);\n"
"                neither: -r 1\n"
"  -F I1,I2,...  time only these functions (default all)\n"
"  -t T          threads (default 1)\n"
"  -n K          report the min over K timings (default 1)\n"
"  -q            quick: 20 ms per timing instead of 100 ms\n"
"  -a            never drop #1 and #2 from the run once hopeless\n"
"\n"
"  LIST: comma separated items a | a:b (steps x -> x+1+x/2) | a:b:xR\n"
"        (factor R) | a:b:+S (step S)\n"
"\n"
"  e.g. %s -b 60 -f 100:220:+8 -r 1\n"
"       %s -b 20,60 -f 2:256:x1.25 -o 4096 -F 0,1,2,3,6\n"
"\n"
"  single timings: %s nbits fun fn outlen          (middle product)\n"
"                  %s nbits fun fn gn nlo nhi      (coefficients [nlo, nhi)\n"
"                                                  of f*g, len(f) = fn,\n"
"                                                  len(g) = gn; #5, #6: f*g)\n"
"\nFunctions:\n", name, name, name, name);

    for (j = 0; j < NFUNS; j++)
        if (fun_available(j))
            flint_printf("   %s\n", description[j]);

    flint_printf("\n"
"  Ratio columns: best = fastest of #1..#3, the dispatcher's candidates (#4\n"
"  is what #3 calls when it does not repack); #0/best; #0/#6 (mulmid against\n"
"  the multiplication it is the transpose of); #4/#5.\n");

#if !FLINT_HAVE_FFT_SMALL
    flint_printf("\n  (built without fft_small: #3, #4, #5 are absent)\n");
#endif
}

static void
warm_up(flint_rand_t state)
{
    /* the first fft_small call of the process builds the default
       context, which is not what is being measured */
    nmod_t mod;
    nn_ptr f, g, z;
    int j;

    nmod_init(&mod, n_nextprime(UWORD(1) << 59, 1));

    f = _nmod_vec_init(200);
    g = _nmod_vec_init(399);
    z = _nmod_vec_init(399);
    _nmod_vec_randtest(f, state, 200, mod);
    _nmod_vec_randtest(g, state, 399, mod);

    for (j = 0; j < NFUNS; j++)
        if (fun_available(j))
            funs[j](z, f, 200, g, 399, mod);

    _nmod_vec_clear(f);
    _nmod_vec_clear(g);
    _nmod_vec_clear(z);
}

/* ------------------------------------------------------------------ */
/* single timing on a general window                                   */
/* ------------------------------------------------------------------ */

/* writes to z the coefficients [nlo, nhi) of the product of (a, an) and
   (b, bn), an >= bn; #5 and #6 write the full product instead */
typedef void (*window_fun) (nn_ptr z, nn_srcptr a, slong an,
                            nn_srcptr b, slong bn, slong nlo, slong nhi,
                            nmod_t mod);

static void
win_dispatch(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
             slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mulmid(z, a, an, b, bn, nlo, nhi, mod);
}

static void
win_classical(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
              slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mulmid_classical(z, a, an, b, bn, nlo, nhi, mod);
}

static void
win_KS(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
       slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mulmid_KS(z, a, an, b, bn, nlo, nhi, mod);
}

static void
win_mul(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
        slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mul(z, a, an, b, bn, mod);
}

#if FLINT_HAVE_FFT_SMALL

static void
win_fft_small(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
              slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mulmid_fft_small(z, a, an, b, bn, nlo, nhi, mod);
}

static void
win_window(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
           slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mul_mid_default_mpn_ctx(z, nlo, nhi, a, an, b, bn, mod);
}

static void
win_mul_fft(nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
            slong nlo, slong nhi, nmod_t mod)
{
    _nmod_poly_mul_mid_default_mpn_ctx(z, 0, an + bn - 1, a, an, b, bn, mod);
}

#endif

static const window_fun win_funs[NFUNS] = {
    win_dispatch, win_classical, win_KS,
#if FLINT_HAVE_FFT_SMALL
    win_fft_small, win_window, win_mul_fft,
#else
    NULL, NULL, NULL,
#endif
    win_mul,
};

static double
time_window(int ifun, nn_ptr z, nn_srcptr a, slong an, nn_srcptr b, slong bn,
            slong nlo, slong nhi, nmod_t mod, double target)
{
    double t0, t;
    slong reps, k;

    for (reps = 1; ; )
    {
        t0 = wall_clock();
        for (k = 0; k < reps; k++)
            win_funs[ifun](z, a, an, b, bn, nlo, nhi, mod);
        t = wall_clock() - t0;

        if (t >= target)
            break;

        if (t >= 0.05 * target)
            reps = (slong) (reps * (1.05 * target / t)) + 1;
        else
            reps *= 10;
    }

    return t / reps;
}

/* p-mulmid nbits fun fn gn nlo nhi */
static int
single_window(char ** argv, flint_rand_t state)
{
    const ulong nbits = (ulong) atoi(argv[1]);
    const int ifun = atoi(argv[2]);
    const slong fn = (slong) atol(argv[3]);
    const slong gn = (slong) atol(argv[4]);
    const slong nlo = (slong) atol(argv[5]);
    const slong nhi = (slong) atol(argv[6]);
    slong an, bn;
    nn_ptr f, g, z, zref;
    nn_srcptr a, b;
    nmod_t mod;
    double t;

    if (ifun < 0 || ifun >= NFUNS || !fun_available(ifun)
                 || fn < 1 || gn < 1 || nbits > FLINT_BITS
                 || nlo < 0 || nhi <= nlo || nhi > fn + gn - 1)
    {
        flint_printf("bad arguments; run with no argument for help\n");
        flint_rand_clear(state);
        return 1;
    }

    warm_up(state);
    select_modulus(&mod, nbits, 0);

    f = _nmod_vec_init(fn);
    g = _nmod_vec_init(gn);
    z = _nmod_vec_init(fn + gn - 1);
    zref = _nmod_vec_init(fn + gn - 1);
    _nmod_vec_randtest(f, state, fn, mod);
    _nmod_vec_randtest(g, state, gn, mod);

    /* longer operand first */
    if (fn >= gn)
        a = f, an = fn, b = g, bn = gn;
    else
        a = g, an = gn, b = f, bn = fn;

    /* check #fun against the reference: the window of the full product
       for the middle products, the full product for #5 and #6 */
    _nmod_poly_mul(zref, a, an, b, bn, mod);
    win_funs[ifun](z, a, an, b, bn, nlo, nhi, mod);

    if (ifun < IFUN_MUL_FFT ? !_nmod_vec_equal(z, zref + nlo, nhi - nlo)
                            : !_nmod_vec_equal(z, zref, an + bn - 1))
    {
        flint_printf("FAIL: %s disagrees at fn = %wd, gn = %wd, nlo = %wd, "
                     "nhi = %wd, mod.n = %wu\n", shortname[ifun], fn, gn,
                     nlo, nhi, mod.n);
        flint_abort();
    }

    t = time_window(ifun, z, a, an, b, bn, nlo, nhi, mod, 0.1);

    flint_printf("bits fun fn        gn        nlo       nhi       \n");
    flint_printf("%-4wu %-3d %-10wd%-10wd%-10wd%-10wd%.2e\n",
                 FLINT_BIT_COUNT(mod.n), ifun, fn, gn, nlo, nhi, t);

    _nmod_vec_clear(f);
    _nmod_vec_clear(g);
    _nmod_vec_clear(z);
    _nmod_vec_clear(zref);
    flint_rand_clear(state);
    return 0;
}

int main(int argc, char ** argv)
{
    flint_rand_t state;
    options_struct O;
    int i, j, ok = 1;

    memset(&O, 0, sizeof(O));
    O.mods[0] = 60;
    O.nmods = 1;
    O.threads = 1;
    O.nrep = 1;
    O.target_ms = 100;
    for (j = 0; j < NFUNS; j++)
        O.timed[j] = 1;

    flint_rand_init(state);

    if (argc == 1 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))
    {
        usage(argv[0]);
        flint_rand_clear(state);
        return 0;
    }

    if (argc == 7 && argv[1][0] != '-')
        return single_window(argv, state);

    if (argc == 5 && argv[1][0] != '-')
    {
        /* one timing: nbits fun fn outlen */
        const ulong nbits = (ulong) atoi(argv[1]);
        const int ifun = atoi(argv[2]);
        const slong fn = (slong) atol(argv[3]);
        const slong outlen = (slong) atol(argv[4]);
        const slong gn = fn + outlen - 1;
        nmod_t mod;
        nn_ptr f, g, z;

        if (ifun < 0 || ifun >= NFUNS || !fun_available(ifun)
                     || fn < 1 || outlen < 1 || nbits > FLINT_BITS)
        {
            flint_printf("bad arguments; run with no argument for help\n");
            flint_rand_clear(state);
            return 1;
        }

        warm_up(state);
        select_modulus(&mod, nbits, 0);

        f = _nmod_vec_init(fn);
        g = _nmod_vec_init(gn);
        z = _nmod_vec_init(gn);
        _nmod_vec_randtest(f, state, fn, mod);
        _nmod_vec_randtest(g, state, gn, mod);

        flint_printf("bits fun fn        outlen    \n");
        flint_printf("%-4wu %-3d %-10wd%-10wd", FLINT_BIT_COUNT(mod.n), ifun,
                     fn, outlen);
        flint_printf("%.2e", time_fun(ifun, z, f, fn, g, gn, mod, &O));
        flint_printf("\n");

        _nmod_vec_clear(f);
        _nmod_vec_clear(g);
        _nmod_vec_clear(z);
        flint_rand_clear(state);
        return 0;
    }

    for (i = 1; i < argc && ok; i++)
    {
        const char * a = argv[i];
        const char * v = (i + 1 < argc) ? argv[i + 1] : NULL;

        if (!strcmp(a, "-q"))
            O.target_ms = 20;
        else if (!strcmp(a, "-a"))
            O.no_skip = 1;
        else if (v == NULL)
            ok = 0;
        else if (!strcmp(a, "-b"))
            ok = parse_mods(&O, v), i++;
        else if (!strcmp(a, "-f"))
            ok = parse_list(&O.fns, v), i++;
        else if (!strcmp(a, "-o"))
            ok = parse_list(&O.outlens, v), i++;
        else if (!strcmp(a, "-r"))
            ok = parse_ratios(&O, v), i++;
        else if (!strcmp(a, "-t"))
            O.threads = atol(v), ok = (O.threads >= 1), i++;
        else if (!strcmp(a, "-n"))
            O.nrep = atol(v), ok = (O.nrep >= 1), i++;
        else if (!strcmp(a, "-F"))
        {
            slong ids[NFUNS], nids, k;

            ok = parse_int_list(ids, &nids, NFUNS, v), i++;

            for (j = 0; j < NFUNS; j++)
                O.timed[j] = 0;

            for (k = 0; k < nids && ok; k++)
            {
                if (ids[k] < 0 || ids[k] >= NFUNS)
                    ok = 0;
                else
                    O.timed[ids[k]] = 1;
            }
        }
        else
            ok = 0;
    }

    if (ok && O.fns.len == 0)
        ok = parse_list(&O.fns, "16:4096");

    if (ok && O.outlens.len == 0 && O.nratios == 0)
        ok = parse_ratios(&O, "1");

    if (!ok)
    {
        flint_printf("bad arguments; run with -h for help\n");
        flint_free(O.fns.v);
        flint_free(O.outlens.v);
        flint_rand_clear(state);
        return 1;
    }

    list_normalise(&O.fns);
    flint_set_num_threads(O.threads);
    warm_up(state);

    for (i = 0; i < O.nmods; i++)
    {
        if (i > 0)
            flint_printf("\n");
        run_table(O.mods[i], O.mod_is_n[i], &O, state);
    }

    flint_free(O.fns.v);
    flint_free(O.outlens.v);
    flint_cleanup_master();
    flint_rand_clear(state);
    return 0;
}

/*
    Copyright (C) 2026 Vincent Neiger

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include <string.h>
#include "nmod.h"
#include "nmod_vec.h"
#include "nmod_mat.h"
#include "nmod_mat/impl.h"

/*
    Rows of A (all its columns) permuted in place so that the new row i is
    the old row perm[i], one cycle of perm at a time with one temporary
    row; rows fixed by perm are not touched.
*/
void
_nmod_mat_ple_permute_rows(nmod_mat_t A, const slong * perm)
{
    slong m = A->r, n = A->c, i, j, start;
    ulong * tmp;
    char * done;
    TMP_INIT;

    if (n == 0)
        return;

    for (start = 0; start < m && perm[start] == start; start++)
        ;

    if (start == m)
        return;

    TMP_START;
    tmp = TMP_ALLOC(n * sizeof(ulong));
    done = TMP_ALLOC(m * sizeof(char));
    memset(done, 0, m);

    for (i = start; i < m; i++)
    {
        if (done[i] || perm[i] == i)
            continue;

        memcpy(tmp, nmod_mat_entry_ptr(A, i, 0), n * sizeof(ulong));

        for (j = i; perm[j] != i; j = perm[j])
        {
            memcpy(nmod_mat_entry_ptr(A, j, 0),
                   nmod_mat_entry_ptr(A, perm[j], 0), n * sizeof(ulong));
            done[j] = 1;
        }

        memcpy(nmod_mat_entry_ptr(A, j, 0), tmp, n * sizeof(ulong));
        done[j] = 1;
    }

    TMP_END;
}

/*
    End of the base cases, which work with the rows in place and the
    logical order in P: moves row P[i] to row i, and writes the columns
    without pivot in increasing order to Q[rank..n), after the pivot
    columns Q[0..rank) (increasing).
*/
void
_nmod_mat_ple_finish(slong * P, slong * Q, nmod_mat_t A, slong rank)
{
    slong n = A->c, j, k, l;

    for (j = k = 0, l = rank; j < n; j++)
    {
        if (k < rank && Q[k] == j)
            k++;
        else
            Q[l++] = j;
    }

    _nmod_mat_ple_permute_rows(A, P);
}

void
_nmod_mat_ple_rows_finish(slong * P, slong * Q, nmod_mat_t A, slong * orig,
                          slong rank)
{
    slong m = A->r, n = A->c, i, j, k;

    if (rank < m)
    {
        slong * pos;
        nmod_mat_t W;
        TMP_INIT;

        TMP_START;
        pos = TMP_ALLOC(m * sizeof(slong));

        /* P[rank..m): the rows that are not pivot rows, increasing */
        for (i = 0; i < m; i++)
            pos[i] = -1;
        for (k = 0; k < rank; k++)
            pos[P[k]] = k;
        for (i = 0, k = rank; i < m; i++)
            if (pos[i] < 0)
                P[k++] = i;

        /* their rows in A, in this order */
        for (i = rank; i < m; i++)
            pos[orig[i]] = i;
        for (k = rank; k < m; k++)
            orig[k - rank] = pos[P[k]] - rank;
        nmod_mat_window_init(W, A, rank, 0, m, n);
        _nmod_mat_ple_permute_rows(W, orig);
        nmod_mat_window_clear(W);

        TMP_END;
    }

    for (j = k = 0, i = rank; j < n; j++)
    {
        if (k < rank && Q[k] == j)
            k++;
        else
            Q[i++] = j;
    }
}

/*
    The choice of the algorithm, from measurements on a Xeon Cascade Lake
    (AVX-512 without IFMA, one thread, no external BLAS) on uniformly
    random matrices; the profile program p-ple gives the timings to tune it
    on other machines.

    - min(m, n) <= 3, or <= 7 for moduli of more than 28 bits: the
      classical algorithm, as in nmod_mat_lu.
    - The recursive algorithm from min(m, n) = c = ple_cutoff_tab[bits - 1],
      and for m >= 4 n from n = c / 2 (the products of the recursion are
      then larger; not for the moduli 2 and 3, whose c is already small):
      it pays earlier where nmod_mat_mul is fast compared with the base
      cases (moduli of 30 to 52 bits, for which these need two words per
      entry, and moduli 2 and 3), later above 2^52, where nmod_mat_mul has
      no SIMD kernel.
    - Below, the Crout algorithm when the delayed one would need two or
      three words per entry: for moduli of more than 50 bits from
      m n = 2^15 (1.2-1.6x faster for squares from 192 to 768), and for
      matrices with 32 rows or more and at least twice as many columns
      (four times for moduli of 32 bits or less), where its vector-matrix
      products are most of the work (1.2-2x faster). Otherwise the delayed
      algorithm.

    The classical and delayed algorithms skip the rows whose multiplier is
    zero, the Crout algorithm only part of them and the recursion none: on
    matrices with many zero multipliers (structured or sparse matrices,
    non-generic rank profiles), the delayed algorithm can be faster than
    the others beyond these crossovers, as for nmod_mat_lu.
*/
static const short ple_cutoff_tab[64] = {
    64, 64, 244, 280, 296, 320, 332,                /*  1 to  7 bits */
    320, 320, 320, 320, 320, 320, 320, 320, 320,    /*  8 to 16 bits */
    320, 320, 320, 320, 320, 320, 320, 320, 320,    /* 17 to 25 bits */
    320, 320, 320, 160, 64, 64, 64,                 /* 26 to 32 bits */
    160, 160, 160, 160, 160, 160, 160, 160, 160,    /* 33 to 41 bits */
    160, 160, 160, 160, 160, 160, 160, 160, 160,    /* 42 to 50 bits */
    256, 256,                                       /* 51 to 52 bits */
    768, 768, 768, 768, 768, 768,                   /* 53 to 58 bits */
    768, 768, 768, 768, 768, 768 };                 /* 59 to 64 bits */

slong
_nmod_mat_ple(slong * P, slong * Q, nmod_mat_t A, slong max_nonpivots)
{
    slong m = A->r, n = A->c, s = FLINT_MIN(m, n), cutoff;
    flint_bitcnt_t bits = NMOD_BITS(A->mod);

    if (s <= 3 || (bits > 28 && s <= 7))
        return _nmod_mat_ple_classical(P, Q, A, max_nonpivots);

    cutoff = ple_cutoff_tab[bits - 1];
    if (s >= cutoff || (m >= 4 * n && 2 * n >= cutoff && bits > 2))
        return _nmod_mat_ple_recursive(P, Q, A, max_nonpivots);

    if (_nmod_vec_dot_bound_limbs(s, A->mod) >= 2
            && ((bits > 50 && (ulong) m * (ulong) n >= 32768)
                || (m >= 32 && (n >= 4 * m || (bits > 32 && n >= 2 * m)))))
        return _nmod_mat_ple_crout(P, Q, A, max_nonpivots);

    return _nmod_mat_ple_classical_delayed(P, Q, A, max_nonpivots);
}

slong
_nmod_mat_ple_wrap(slong * P, slong * pivots_nonpivots, nmod_mat_t A,
        int flags, slong (* ple)(slong *, slong *, nmod_mat_t, slong))
{
    slong m = A->r, n = A->c, rank, max_nonpivots;
    slong * Q = pivots_nonpivots;

    /* with the rank check, stop as soon as more than n - min(m, n)
       columns without pivot are met: the rank is then less than min(m, n) */
    max_nonpivots = (flags & NMOD_MAT_PLE_RANK_CHECK) ? n - FLINT_MIN(m, n) : -1;

    if (Q == NULL)
        Q = flint_malloc((n + 1) * sizeof(slong));

    rank = ple(P, Q, A, max_nonpivots);

    if (pivots_nonpivots == NULL)
        flint_free(Q);

    return FLINT_MAX(rank, 0);
}

slong
nmod_mat_ple(slong * P, slong * pivots_nonpivots, nmod_mat_t A, int flags)
{
    return _nmod_mat_ple_wrap(P, pivots_nonpivots, A, flags, _nmod_mat_ple);
}

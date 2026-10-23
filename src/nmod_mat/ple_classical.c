/*
    Copyright (C) 2026 Vincent Neiger

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "nmod.h"
#include "nmod_vec.h"
#include "nmod_mat.h"
#include "nmod_mat/impl.h"

/*
    Gaussian elimination column by column, with a reduction after each
    update. The rows stay in place: the row in logical position i is row
    P[i] of A, and bringing the pivot row to position rank is an upward rotation
    of P[rank..r] (the rows passed over keep their relative order). Rows with
    a zero multiplier are not updated.
*/
slong
_nmod_mat_ple_classical(slong * P, slong * Q, nmod_mat_t A, slong max_nonpivots)
{
    const slong m = A->r, n = A->c;
    const nmod_t mod = A->mod;
    slong rank = 0, nonpivots = 0, col, i, r;

    for (i = 0; i < m; i++)
        P[i] = i;

    for (col = 0; col < n && rank < m; col++)
    {
        ulong * prow;
        ulong d;
        slong len;

        /* pivot: first row in logical order with a nonzero entry */
        for (r = rank; r < m && nmod_mat_entry(A, P[r], col) == 0; r++)
            ;

        if (r == m)
        {
            nonpivots++;
            if (max_nonpivots >= 0 && nonpivots > max_nonpivots)
                return -1;
            continue;
        }

        _nmod_mat_ple_rotate(P, rank, r);

        prow = nmod_mat_entry_ptr(A, P[rank], 0);
        d = nmod_inv(prow[col], mod);
        len = n - col - 1;

        for (i = rank + 1; i < m; i++)
        {
            ulong * row = nmod_mat_entry_ptr(A, P[i], 0);
            ulong e = row[col];

            /* nothing to do: row[col] and row[rank] are already zero */
            if (e == 0)
                continue;

            e = nmod_mul(e, d, mod);
            if (len > 0)
                _nmod_vec_scalar_addmul_nmod(row + col + 1, prow + col + 1,
                                             len, nmod_neg(e, mod), mod);
            row[col] = 0;
            row[rank] = e;
        }

        Q[rank] = col;
        rank++;
    }

    _nmod_mat_ple_finish(P, Q, A, rank);

    return rank;
}

slong
nmod_mat_ple_classical(slong * P, slong * pivots_nonpivots, nmod_mat_t A, int flags)
{
    return _nmod_mat_ple_wrap(P, pivots_nonpivots, A, flags,
                              _nmod_mat_ple_classical);
}

/*
    Copyright (C) 2014 Abhinav Baid

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include <gmp.h>
#include "fmpz_mat.h"
#include "fmpq.h"
#include "fmpq_vec.h"
#include "fmpq_mat.h"

int
fmpz_mat_is_reduced_gram_with_removal(const fmpz_mat_t A, double delta,
                                      double eta, const fmpz_t gs_B, int newd)
{
    slong i, j, k, d = A->r;
    fmpq_mat_t r, mu;
    fmpq *s;
    mpq_t deltax, etax;
    fmpq_t deltaq, etaq, tmp, gs_Bq;

    if (d == 0 || d == 1)
        return 1;

    fmpq_mat_init(r, d, d);
    fmpq_mat_init(mu, d, d);

    s = _fmpq_vec_init(d);

    mpq_init(deltax);
    mpq_init(etax);

    fmpq_init(deltaq);
    fmpq_init(etaq);
    fmpq_init(tmp);
    fmpq_init(gs_Bq);

    mpq_set_d(deltax, delta);
    mpq_set_d(etax, eta);
    fmpq_set_mpq(deltaq, deltax);
    fmpq_set_mpq(etaq, etax);
    mpq_clears(deltax, etax, NULL);

    fmpz_set(fmpq_mat_entry_num(r, 0, 0), fmpz_mat_entry(A, 0, 0));
    if (newd == 0 && fmpz_cmp(fmpz_mat_entry(A, 0, 0), gs_B) < 0)
    {
        fmpq_mat_clear(r);
        fmpq_mat_clear(mu);
        fmpq_clear(deltaq);
        fmpq_clear(etaq);
        fmpq_clear(tmp);
        fmpq_clear(gs_Bq);
        _fmpq_vec_clear(s, d);
        return 0;
    }

    fmpz_set(fmpq_numref(gs_Bq), gs_B);
    fmpz_one(fmpq_denref(gs_Bq));
    for (i = 1; i < d; i++)
    {
        fmpz_set(fmpq_numref(s), fmpz_mat_entry(A, i, i));
        fmpz_one(fmpq_denref(s));
        for (j = 0; j <= i - 1; j++)
        {
            fmpz_set(fmpq_mat_entry_num(r, i, j), fmpz_mat_entry(A, i, j));
            for (k = 0; k <= j - 1; k++)
            {
                fmpq_submul(fmpq_mat_entry(r, i, j), fmpq_mat_entry(mu, j, k),
                            fmpq_mat_entry(r, i, k));
            }
            fmpq_div(fmpq_mat_entry(mu, i, j), fmpq_mat_entry(r, i, j),
                     fmpq_mat_entry(r, j, j));
            if (i < newd)
            {
                fmpq_abs(tmp, fmpq_mat_entry(mu, i, j));
                if (fmpq_cmp(tmp, etaq) > 0)    /* check size reduction */
                {
                    fmpq_mat_clear(r);
                    fmpq_mat_clear(mu);
                    fmpq_clear(deltaq);
                    fmpq_clear(etaq);
                    fmpq_clear(tmp);
                    fmpq_clear(gs_Bq);
                    _fmpq_vec_clear(s, d);
                    return 0;
                }
            }
            fmpq_set(s + j + 1, s + j);
            fmpq_submul(s + j + 1, fmpq_mat_entry(mu, i, j),
                        fmpq_mat_entry(r, i, j));
        }
        fmpq_set(fmpq_mat_entry(r, i, i), s + i);
        if (i >= newd && fmpq_cmp(fmpq_mat_entry(r, i, i), gs_Bq) < 0)  /* check removals */
        {
            fmpq_mat_clear(r);
            fmpq_mat_clear(mu);
            fmpq_clear(deltaq);
            fmpq_clear(etaq);
            fmpq_clear(tmp);
            fmpq_clear(gs_Bq);
            _fmpq_vec_clear(s, d);
            return 0;
        }
        if (i < newd)
        {
            fmpq_mul(tmp, deltaq, fmpq_mat_entry(r, i - 1, i - 1));
            if (fmpq_cmp(tmp, s + i - 1) > 0)   /* check Lovasz condition */
            {
                fmpq_mat_clear(r);
                fmpq_mat_clear(mu);
                fmpq_clear(deltaq);
                fmpq_clear(etaq);
                fmpq_clear(tmp);
                fmpq_clear(gs_Bq);
                _fmpq_vec_clear(s, d);
                return 0;
            }
        }
    }

    fmpq_mat_clear(r);
    fmpq_mat_clear(mu);
    fmpq_clear(deltaq);
    fmpq_clear(etaq);
    fmpq_clear(tmp);
    fmpq_clear(gs_Bq);
    _fmpq_vec_clear(s, d);
    return 1;
}

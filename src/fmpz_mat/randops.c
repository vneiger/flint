/*
    Copyright (C) 2010 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz.h"
#include "fmpz_mat.h"

void
fmpz_mat_randops(fmpz_mat_t mat, flint_rand_t state, slong count)
{
    slong c, i, j, k;
    slong m = mat->r;
    slong n = mat->c;

    if (mat->r == 0 || mat->c == 0)
        return;

    for (c = 0; c < count; c++)
    {
        if (n_randint(state, 2))
        {
            if ((i = n_randint(state, m)) == (j = n_randint(state, m)))
                continue;
            if (n_randint(state, 2))
                for (k = 0; k < n; k++)
                    fmpz_add(fmpz_mat_entry(mat, j, k), fmpz_mat_entry(mat, j, k),
                        fmpz_mat_entry(mat, i, k));
            else
                for (k = 0; k < n; k++)
                    fmpz_sub(fmpz_mat_entry(mat, j, k), fmpz_mat_entry(mat, j, k),
                        fmpz_mat_entry(mat, i, k));
        }
        else
        {
            if ((i = n_randint(state, n)) == (j = n_randint(state, n)))
                continue;
            if (n_randint(state, 2))
                for (k = 0; k < m; k++)
                    fmpz_add(fmpz_mat_entry(mat, k, j), fmpz_mat_entry(mat, k, j),
                        fmpz_mat_entry(mat, k, i));
            else
                for (k = 0; k < m; k++)
                    fmpz_sub(fmpz_mat_entry(mat, k, j), fmpz_mat_entry(mat, k, j),
                        fmpz_mat_entry(mat, k, i));
        }
    }
}

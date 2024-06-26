/*
    Copyright (C) 2011 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz.h"
#include "fmpz_vec.h"

void
_fmpz_vec_sum(fmpz_t res, const fmpz * vec, slong len)
{
    if (len <= 1)
    {
        if (len == 1)
            fmpz_set(res, vec);
        else
            fmpz_zero(res);
    }
    else
    {
        slong i;

        fmpz_add(res, vec, vec + 1);

        for (i = 2; i < len; i++)
            fmpz_add(res, res, vec + i);
    }
}

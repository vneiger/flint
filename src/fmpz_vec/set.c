/*
    Copyright (C) 2010 William Hart

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz.h"
#include "fmpz_vec.h"

void
_fmpz_vec_set(fmpz * vec1, const fmpz * vec2, slong len2)
{
    slong i;
    if (vec1 != vec2)
    {
        for (i = 0; i < len2; i++)
            fmpz_set(vec1 + i, vec2 + i);
    }
}

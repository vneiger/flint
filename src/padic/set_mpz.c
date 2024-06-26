/*
    Copyright (C) 2011, 2012 Sebastian Pancratz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include <gmp.h>
#include "padic.h"

void padic_set_mpz(padic_t rop, const mpz_t op, const padic_ctx_t ctx)
{
    fmpz_t t;

    fmpz_init(t);
    fmpz_set_mpz(t, op);
    padic_set_fmpz(rop, t, ctx);
    fmpz_clear(t);
}

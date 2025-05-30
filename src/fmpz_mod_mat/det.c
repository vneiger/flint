/*
    Copyright (C) 2024 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "fmpz_mod.h"
#include "fmpz_mod_mat.h"
#include "gr.h"
#include "gr_mat.h"

void fmpz_mod_mat_det(fmpz_t res, const fmpz_mod_mat_t mat, const fmpz_mod_ctx_t ctx)
{
    gr_ctx_t gr_ctx;

    if (!fmpz_mod_mat_is_square(mat, ctx))
        flint_throw(FLINT_ERROR, "Exception (fmpz_mod_mat_charpoly_berkowitz). Non-square matrix.\n");

    _gr_ctx_init_fmpz_mod_from_ref(gr_ctx, ctx);
    GR_MUST_SUCCEED(gr_mat_det(res, (const gr_mat_struct *) mat, gr_ctx));
}

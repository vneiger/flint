/*
    Copyright (C) 2018 Daniel Schultz

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "mpn_extras.h"
#include "nmod_vec.h"
#include "mpoly.h"
#include "nmod_mpoly.h"

void nmod_mpoly_resize(
    nmod_mpoly_t A,
    slong new_length,
    const nmod_mpoly_ctx_t ctx)
{
    slong old_length = A->length;
    slong N;

    new_length = FLINT_MAX(WORD(0), new_length);

    N = mpoly_words_per_exp(A->bits, ctx->minfo);

    if (new_length > old_length)
    {
        nmod_mpoly_fit_length(A, new_length, ctx);

        /* must zero out the new coeffs/exps past the old end */
        N = mpoly_words_per_exp(A->bits, ctx->minfo);
        flint_mpn_zero(A->exps + N*old_length, N*(new_length - old_length));
        _nmod_vec_zero(A->coeffs + old_length, new_length - old_length);
    }

    A->length = new_length;
}

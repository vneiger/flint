/*
    Copyright (C) 2023 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "gr_vec.h"
#include "gr_mat.h"

int
gr_mat_concat_vertical(gr_mat_t res, const gr_mat_t mat1, const gr_mat_t mat2, gr_ctx_t ctx)
{
    int status = GR_SUCCESS;
    slong i;
    slong r1 = mat1->r;
    slong c1 = mat1->c;
    slong r2 = mat2->r;
    slong sz = ctx->sizeof_elem;

    if (mat1->c != mat2->c || res->c != mat1->c || res->r != mat1->r + mat2->r)
        return GR_DOMAIN;

    if (c1 > 0)
    {
        for (i = 0; i < r1; i++)
            status |= _gr_vec_set(GR_MAT_ENTRY(res, i, 0, sz), GR_MAT_ENTRY(mat1, i, 0, sz), c1, ctx);
        for (i = 0; i < r2; i++)
            status |= _gr_vec_set(GR_MAT_ENTRY(res, i + r1, 0, sz), GR_MAT_ENTRY(mat2, i, 0, sz), c1, ctx);
    }

    return status;
}

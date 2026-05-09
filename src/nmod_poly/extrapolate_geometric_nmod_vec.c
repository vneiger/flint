/*
    Copyright (C) 2026 Vincent Neiger, Kevin Tran

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "nmod.h"
#include "nmod_poly.h"

void nmod_poly_extrapolate_geometric_precomp(nn_ptr oval, slong olen,
                                             nn_srcptr ival, slong ilen,
                                             slong offset,
                                             const nmod_geometric_progression_t G)
{
    /* precomputation has been done */
    FLINT_ASSERT((G->function >> 2) & 1);
    /* input/output points are disjoint, and stay within precomputed data length */
    FLINT_ASSERT((offset >= ilen && G->len >= offset+olen)
                 || (offset <= -olen && G->len >= ilen-offset));

    if (ilen == 0)
    {
        for (slong i = 0; i < olen; i++)
            oval[i] = 0;
        return;
    }

    if (ilen == 1)
    {
        for (slong i = 0; i < olen; i++)
            oval[i] = ival[0];
        return;
    }

    /* forward extrapolation */
    if (offset > 0)
    {
        /* TODO TMP ALLOC? */
        nn_ptr tmp = FLINT_ARRAY_ALLOC(ilen, ulong);
        /* first scaling */
        //    svals = [ext_s2[i] * ext_s3[m-1-i] * vals[i] for i in range(m)]
        for (slong i = 0; i < ilen; i++)
        {
            /* TODO use some NMOD_RED3?? */
            tmp[i] = nmod_mul(G->ext_s3[ilen-1-i], ival[i], G->mod);
            tmp[i] = nmod_mul(G->ext_s2[i], tmp[i], G->mod);
        }

        /* middle product */
        //    g = xring(svals)
        //    f = ext_ff.shift(-(lmk - m)).truncate(m+n-1)
        //    mp = (f * g).shift(-(m-1)).truncate(n)
        _nmod_poly_mulmid(oval, G->ext_ff->coeffs + offset-ilen, ilen+olen-1, tmp, ilen, ilen-1, ilen+olen-1, G->mod);

        /* second scaling */
        //    w = [ext_s1f[lmk+j] * ext_s2[lmk-m+j] * mp[j] for j in range(n)]
        for (slong j = 0; j < olen; j++)
        {
            oval[j] = nmod_mul(G->ext_s2[offset-ilen+j], oval[j], G->mod);
            oval[j] = nmod_mul(G->ext_s1f[offset+j], oval[j], G->mod);
        }
        flint_free(tmp);
    }

    /* backward extrapolation */
    else
    {
        /* TODO TMP ALLOC? */
        nn_ptr tmp = FLINT_ARRAY_ALLOC(FLINT_MAX(ilen, olen), ulong);
        /* first scaling */
        //    svals = [ext_s2[m-1-i] * ext_s3[i] * vals[m-1-i] for i in range(m)]
        for (slong i = 0; i < ilen; i++)
        {
            /* TODO use some NMOD_RED3?? */
            tmp[i] = nmod_mul(G->ext_s2[ilen-1-i], ival[ilen-1-i], G->mod);
            tmp[i] = nmod_mul(G->ext_s3[i], tmp[i], G->mod);
        }

        /* middle product */
        //    g = xring(svals)
        //    f = ext_fb.shift(lmk + n).truncate(m+n-1)
        //    mp = (f * g).shift(-(m-1)).truncate(n)
        _nmod_poly_mulmid(oval, G->ext_fb->coeffs - (offset+olen), ilen+olen-1, tmp, ilen, ilen-1, ilen+olen-1, G->mod);

        /* second scaling */
        //    w = [ext_s1b[m-1-lmk-j] * ext_s3[-lmk-j-1] * mp[n-1-j] for j in range(n)]
        for (slong j = 0; j < olen; j++)
            tmp[j] = oval[olen - 1 - j];
        for (slong j = 0; j < olen; j++)
        {
            oval[j] = nmod_mul(G->ext_s3[-offset-1-j], tmp[j], G->mod);
            oval[j] = nmod_mul(G->ext_s1b[ilen-1-offset-j], oval[j], G->mod);
        }

        flint_free(tmp);
    }
}

void nmod_poly_extrapolate_geometric(nn_ptr oval, slong olen,
                                     nn_srcptr ival, slong ilen,
                                     slong offset, ulong r, nmod_t mod)
{
    /* TODO handle overlap */
    if (ilen == 0)
    {
        for (slong i = 0; i < olen; i++)
            oval[i] = 0;
        return;
    }

    if (ilen == 1)
    {
        for (slong i = 0; i < olen; i++)
            oval[i] = ival[0];
        return;
    }

    nmod_geometric_progression_t G;
    slong len = (offset > 0) ? offset+olen : ilen-offset;
    _nmod_geometric_progression_init_function(G, r, len, mod, UWORD(4));
    nmod_poly_extrapolate_geometric_precomp(oval, olen, ival, ilen, offset, G);
    nmod_geometric_progression_clear(G);
}

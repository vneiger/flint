/*
    Copyright (C) 2011, 2012 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "nmod.h"
#include "nmod_vec.h"
#include "nmod_poly.h"

void
_nmod_poly_evaluate_nmod_vec(nn_ptr ys, nn_srcptr coeffs, slong len,
    nn_srcptr xs, slong n, nmod_t mod)
{
    if (len < 32)
        _nmod_poly_evaluate_nmod_vec_iter(ys, coeffs, len, xs, n, mod);
    else
        _nmod_poly_evaluate_nmod_vec_fast(ys, coeffs, len, xs, n, mod);
}

void
nmod_poly_evaluate_nmod_vec(nn_ptr ys,
        const nmod_poly_t poly, nn_srcptr xs, slong n)
{
    _nmod_poly_evaluate_nmod_vec(ys, poly->coeffs,
                                        poly->length, xs, n, poly->mod);
}

/* This gives some speedup for small lengths. */
static inline void _nmod_poly_rem_2(nn_ptr r, nn_srcptr a, slong al,
    nn_srcptr b, slong bl, nmod_t mod)
{
    if (al == 2)
        r[0] = nmod_sub(a[0], nmod_mul(a[1], b[0], mod), mod);
    else
        _nmod_poly_rem(r, a, al, b, bl, mod);
}

void
_nmod_poly_evaluate_nmod_vec_fast_precomp(nn_ptr vs, nn_srcptr poly,
    slong plen, const nn_ptr * tree, slong len, nmod_t mod)
{
    slong height, i, j, pow, left;
    slong tree_height;
    slong tlen;
    nn_ptr t, u, swap, pa, pb, pc;

    /* avoid worrying about some degenerate cases */
    if (len < 2 || plen < 2)
    {
        if (len == 1)
            vs[0] = _nmod_poly_evaluate_nmod(poly, plen,
                nmod_neg(tree[0][0], mod), mod);
        else if (len != 0 && plen == 0)
            _nmod_vec_zero(vs, len);
        else if (len != 0 && plen == 1)
            for (i = 0; i < len; i++)
                vs[i] = poly[0];
        return;
    }

    t = _nmod_vec_init(len);
    u = _nmod_vec_init(len);

    left = len;

    /* Initial reduction. We allow the polynomial to be larger
       or smaller than the number of points. */
    height = FLINT_BIT_COUNT(plen - 1) - 1;
    tree_height = FLINT_CLOG2(len);
    while (height >= tree_height)
        height--;
    pow = WORD(1) << height;

    for (i = j = 0; i < len; i += pow, j += (pow + 1))
    {
        tlen = ((i + pow) <= len) ? pow : len % pow;
        _nmod_poly_rem(t + i, poly, plen, tree[height] + j, tlen + 1, mod);
    }

    for (i = height - 1; i >= 0; i--)
    {
        pow = WORD(1) << i;
        left = len;
        pa = tree[i];
        pb = t;
        pc = u;

        while (left >= 2 * pow)
        {
            _nmod_poly_rem_2(pc, pb, 2 * pow, pa, pow + 1, mod);
            _nmod_poly_rem_2(pc + pow, pb, 2 * pow, pa + pow + 1, pow + 1, mod);

            pa += 2 * pow + 2;
            pb += 2 * pow;
            pc += 2 * pow;
            left -= 2 * pow;
        }

        if (left > pow)
        {
            _nmod_poly_rem(pc, pb, left, pa, pow + 1, mod);
            _nmod_poly_rem(pc + pow, pb, left, pa + pow + 1, left - pow + 1, mod);
        }
        else if (left > 0)
            _nmod_vec_set(pc, pb, left);

        swap = t;
        t = u;
        u = swap;
    }

    _nmod_vec_set(vs, t, len);
    _nmod_vec_clear(t);
    _nmod_vec_clear(u);
}

void _nmod_poly_evaluate_nmod_vec_fast(nn_ptr ys, nn_srcptr poly, slong plen,
    nn_srcptr xs, slong n, nmod_t mod)
{
    nn_ptr * tree;

    tree = _nmod_poly_tree_alloc(n);
    _nmod_poly_tree_build(tree, xs, n, mod);
    _nmod_poly_evaluate_nmod_vec_fast_precomp(ys, poly, plen, tree, n, mod);
    _nmod_poly_tree_free(tree, n);
}

void
nmod_poly_evaluate_nmod_vec_fast(nn_ptr ys,
        const nmod_poly_t poly, nn_srcptr xs, slong n)
{
    _nmod_poly_evaluate_nmod_vec_fast(ys, poly->coeffs,
                                        poly->length, xs, n, poly->mod);
}

void
_nmod_poly_evaluate_nmod_vec_iter(nn_ptr ys, nn_srcptr coeffs, slong len,
    nn_srcptr xs, slong n, nmod_t mod)
{
    slong i;
    for (i = 0; i < n; i++)
        ys[i] = _nmod_poly_evaluate_nmod(coeffs, len, xs[i], mod);
}

void
nmod_poly_evaluate_nmod_vec_iter(nn_ptr ys,
        const nmod_poly_t poly, nn_srcptr xs, slong n)
{
    _nmod_poly_evaluate_nmod_vec_iter(ys, poly->coeffs,
                                        poly->length, xs, n, poly->mod);
}

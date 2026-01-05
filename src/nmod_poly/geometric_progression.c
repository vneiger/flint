/*
    Copyright (C) 2025, Vincent Neiger, Éric Schost
    Copyright (C) 2025, Mael Hostettler

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include "nmod_vec.h"
#include "nmod_poly.h"

void
nmod_geometric_progression_init(nmod_geometric_progression_t G, ulong r, slong len, nmod_t mod)
{
    ulong tmp, qk, inv_qk, qq, s;
    nn_ptr diff, inv_diff, prod_diff;
    slong i;

    const slong q = nmod_mul(r, r, mod);
    const slong inv_r = nmod_inv(r, mod);
    const slong inv_q = nmod_mul(inv_r, inv_r, mod);

    G->len = len;
    G->mod = mod;

    /* for evaluation */
    nmod_poly_init2_preinv(G->f, mod.n, mod.ninv, 2*len - 1);
    G->x = _nmod_vec_init(len);

    /* for interpolation */
    nmod_poly_init2_preinv(G->g1, mod.n, mod.ninv, len);
    nmod_poly_init2_preinv(G->g2, mod.n, mod.ninv, len);
    G->y = _nmod_vec_init(len);
    G->z = _nmod_vec_init(len);
    G->w = _nmod_vec_init(len);

    /* G->f = sum_{0 <= i < 2*len - 1} r**(i*i) * x**i */
    G->f->length = 2*len - 1;
    G->f->coeffs[0] = 1;
    tmp = r;
    for (i = 1; i < 2*len - 1; i++)
    {
        G->f->coeffs[i] = nmod_mul(G->f->coeffs[i - 1], tmp, mod);
        tmp = nmod_mul(tmp, q, mod);
    }

    /* G->x[i] = 1 / r**(i*i) */
    G->x[0] = 1;
    tmp = inv_r;
    for (i = 1; i < len; i++)
    {
        G->x[i] = nmod_mul(G->x[i - 1], tmp, mod);
        tmp = nmod_mul(tmp, inv_q, mod);
    }

    /* write Q_i for prod_{1 <= k <= i} (q**k - 1) */
    /* coeff(G->g1, i) = (-1)**i * q**(i*(i-1)/2) / Q_i */
    G->g1->length = len;
    G->g1->coeffs[0] = 1;

    /* coeff(G->g2, i) = q**(i*(i-1)/2) / Q_i */
    G->g2->length = len;
    G->g2->coeffs[0] = 1;

    /* G->y[i] = (-1)**i * Q_i / q**(i*(i-1)/2) = inverse of coeff(G->g1, i) */
    G->y[0] = 1;

    /* G->z = (-1)**i / Q_i */
    G->z[0] = 1;

    /* G->w[i] = 1 / Q_i */
    G->w[0] = 1;

    inv_diff  = _nmod_vec_init(len);
    diff      = _nmod_vec_init(len);
    prod_diff = _nmod_vec_init(len);
    inv_diff[0] = 1;
    diff[0] = 1;
    prod_diff[0] = 1;

    qk = q;  // Montgomery inversion
    /* FIXME improve with Shoup */
    /* FIXME diff not reused, could use e.g. G->y as buffer */
    /* FIXME similarly it seems G->w could store inv_diff, G->y could store prod_diff */
    /* diff[i] = q**i - 1 */
    /* prod_diff[i] = (q-1) ... (q**i - 1) */
    /* inv_diff[i] = 1 / (q**i - 1) */
    for (i = 1; i < len; i++)
    {
        diff[i] = qk - 1;
        prod_diff[i] = nmod_mul(qk - 1, prod_diff[i - 1], mod);
        qk = nmod_mul(qk, q, mod);
    }

    tmp = nmod_inv(prod_diff[len-1], mod);
    for (i = len - 1; i > 0; i--)
    {
        inv_diff[i] = nmod_mul(prod_diff[i - 1], tmp, mod);
        tmp = nmod_mul(tmp, diff[i], mod);
    }
    inv_diff[0] = tmp;
    // end Montgomery inversion

    qk = 1;  /* FIXME will recompute the powers of q */
    inv_qk = 1;
    qq = 1;
    s = 1;

    for (i = 1; i < len; i++)
    {
        qq = nmod_mul(qq, qk, mod);    /* q**(i*(i-1)/2) */
        s = nmod_mul(s, inv_qk, mod);  /* inverse of q**(i*(i-1)/2) */
        G->w[i] = nmod_mul(G->w[i - 1], inv_diff[i], mod); /* prod_{1 <= k <= i} 1/(q**k-1) */
        tmp = nmod_mul(qq, G->w[i], mod); /* prod_{1 <= k <= i} q**(k-1) / (q**k - 1) */
        G->g2->coeffs[i] = tmp;

        if (i % 2)  /* i is odd */
        {
            G->g1->coeffs[i] = mod.n - tmp;
            G->y[i] = mod.n - prod_diff[i];
            G->z[i] = mod.n - G->w[i];
        }
        else  /* i is even */
        {
            G->g1->coeffs[i] = tmp;
            G->y[i] = prod_diff[i];
            G->z[i] = G->w[i];
        }
        G->y[i] = nmod_mul(G->y[i], s, mod);

        qk = nmod_mul(qk, q, mod);  /* q**i */
        inv_qk = nmod_mul(inv_qk, inv_q, mod);  /* 1 / q**i */
    }

    _nmod_vec_clear(prod_diff);
    _nmod_vec_clear(inv_diff);
    _nmod_vec_clear(diff);
}

void
nmod_geometric_progression_clear(nmod_geometric_progression_t G)
{
    nmod_poly_clear(G->f);
    nmod_poly_clear(G->g2);
    nmod_poly_clear(G->g1);
    _nmod_vec_clear(G->x);
    _nmod_vec_clear(G->z);
    _nmod_vec_clear(G->y);
    _nmod_vec_clear(G->w);
}

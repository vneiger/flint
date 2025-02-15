/*
    Copyright (C) 2013 Fredrik Johansson

    This file is part of FLINT.

    FLINT is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.  See <https://www.gnu.org/licenses/>.
*/

#include <math.h>
#include "arb_poly.h"

void
_arb_poly_sinh_cosh_series(arb_ptr s, arb_ptr c, arb_srcptr h, slong hlen, slong n, slong prec)
{
    hlen = FLINT_MIN(hlen, n);

    if (hlen == 1)
    {
        arb_sinh_cosh(s, c, h, prec);
        _arb_vec_zero(s + 1, n - 1);
        _arb_vec_zero(c + 1, n - 1);
    }
    else if (n == 2)
    {
        arb_t t;
        arb_init(t);
        arb_set(t, h + 1);
        arb_sinh_cosh(s, c, h, prec);
        arb_mul(s + 1, c, t, prec);
        arb_mul(c + 1, s, t, prec);
        arb_clear(t);
    }
    else
    {
        slong cutoff;

        if (prec <= 128)
            cutoff = 400;
        else
            cutoff = 30000 / pow(log(prec), 3);

        if (hlen < cutoff)
            _arb_poly_sinh_cosh_series_basecase(s, c, h, hlen, n, prec);
        else
            _arb_poly_sinh_cosh_series_exponential(s, c, h, hlen, n, prec);
    }
}

void
arb_poly_sinh_cosh_series(arb_poly_t s, arb_poly_t c,
                                    const arb_poly_t h, slong n, slong prec)
{
    slong hlen = h->length;

    if (n == 0)
    {
        arb_poly_zero(s);
        arb_poly_zero(c);
        return;
    }

    if (hlen == 0)
    {
        arb_poly_zero(s);
        arb_poly_one(c);
        return;
    }

    if (hlen == 1)
        n = 1;

    arb_poly_fit_length(s, n);
    arb_poly_fit_length(c, n);
    _arb_poly_sinh_cosh_series(s->coeffs, c->coeffs, h->coeffs, hlen, n, prec);
    _arb_poly_set_length(s, n);
    _arb_poly_normalise(s);
    _arb_poly_set_length(c, n);
    _arb_poly_normalise(c);
}

/*=============================================================================

    This file is part of ARB.

    ARB is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    ARB is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ARB; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA

=============================================================================*/
/******************************************************************************

    Copyright (C) 2012 Fredrik Johansson

******************************************************************************/

#include "fmpr.h"

int
fmpr_cmp(const fmpr_t x, const fmpr_t y)
{
    int res, xsign, ysign;
    fmpr_t t;

    if (fmpr_equal(x, y))
        return 0;

    if (fmpr_is_special(x) || fmpr_is_special(y))
    {
        if (fmpr_is_nan(x) || fmpr_is_nan(y))
            return 0;
        if (fmpr_is_zero(y)) return fmpr_sgn(x);
        if (fmpr_is_zero(x)) return -fmpr_sgn(y);
        if (fmpr_is_pos_inf(x)) return 1;
        if (fmpr_is_neg_inf(y)) return 1;
        return -1;
    }

    xsign = fmpr_sgn(x);
    ysign = fmpr_sgn(y);

    if (xsign != ysign)
        return (xsign < 0) ? -1 : 1;

    /* Reduces to integer comparison of bottom exponents are the same */
    if (fmpz_equal(fmpr_expref(x), fmpr_expref(y)))
        return fmpz_cmp(fmpr_manref(x), fmpr_manref(y)) < 0 ? -1 : 1;

    /* TODO: compare position of top exponents to avoid subtraction */

    fmpr_init(t);
    fmpr_sub(t, x, y, 2, FMPR_RND_DOWN);
    res = fmpr_sgn(t);
    fmpr_clear(t);

    return res;
}

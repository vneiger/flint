/*============================================================================

    This file is part of FLINT.

    FLINT is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    FLINT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with FLINT; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

===============================================================================*/
/******************************************************************************

 Copyright (C) 2010 William Hart
 
******************************************************************************/

#ifndef FMPZ_VEC_H
#define FMPZ_VEC_H

#include <mpir.h>
#include "fmpz.h"

void _fmpz_vec_zero(fmpz * vec1, ulong len1);

void _fmpz_vec_copy(fmpz * vec1, fmpz * vec2, ulong len2);

void _fmpz_vec_neg(fmpz * vec1, fmpz * vec2, ulong len2);

void _fmpz_vec_scalar_mul_si(fmpz * vec1, fmpz * vec2, ulong len2, long c);

#endif







/*
    Copyright (C) 2023 Jean Kieffer

    This file is part of Arb.

    Arb is free software: you can redistribute it and/or modify it under
    the terms of the GNU Lesser General Public License (LGPL) as published
    by the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.  See <http://www.gnu.org/licenses/>.
*/

#include "acb_theta.h"

int main(void)
{
    slong iter;
    flint_rand_t state;

    flint_printf("all_sqr....");
    fflush(stdout);

    flint_randinit(state);

    /* Test: agrees with naive_all */
    for (iter = 0; iter < 20 * flint_test_multiplier(); iter++)
    {
        slong g = 1 + n_randint(state, 3);
        slong n2 = 1 << (2 * g);
        slong prec = 100;
        slong bits = n_randint(state, 5);
        acb_mat_t tau;
        acb_ptr z;
        acb_ptr th, test;
        slong k;

        acb_mat_init(tau, g, g);
        z = _acb_vec_init(g);
        th = _acb_vec_init(n2);
        test = _acb_vec_init(n2);

        /* Sample tau not too far from reduced domain */
        acb_siegel_randtest_reduced(tau, state, prec, bits);
        acb_mat_scalar_mul_2exp_si(tau, tau, -2);
        for (k = 0; k < g; k++)
        {
            acb_randtest_precise(z, state, prec, bits);
        }

        acb_theta_all_sqr(th, z, tau, prec);
        acb_theta_naive_all(test, z, 1, tau, prec);
        for (k = 0; k < n2; k++)
        {
            acb_sqr(&test[k], &test[k], prec);
        }

        if (!_acb_vec_overlaps(th, test, n2))
        {
            flint_printf("FAIL\n");
            flint_abort();
        }

        acb_mat_clear(tau);
        _acb_vec_clear(z, g);
        _acb_vec_clear(th, n2);
        _acb_vec_clear(test, n2);
    }

    flint_randclear(state);
    flint_cleanup();
    flint_printf("PASS\n");
    return 0;
}

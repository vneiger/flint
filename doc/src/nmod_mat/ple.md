---
title: Computing a PLE decomposition revealing the rank profile matrix
functions: nmod_mat_ple nmod_mat_ple_nullspace nmod_mat_ple_nullspace_compact NMOD_MAT_PLE_RANK_CHECK
---

`nmod_mat_ple` computes an LU decomposition with row pivoting, in the
format of `nmod_mat_lu`, that reveals the *rank profile matrix* of $A$:
from its row permutation and pivot columns one reads the first
linearly independent rows and columns of $A$, and those of every leading
submatrix of $A$. Its output is unique, so it does not depend on the
algorithm or on the machine. It has the same pivot columns as
`nmod_mat_lu`, but the rows come in a different order, and therefore the
factors $L$ and $E$ may differ too.

::: summary algorithms/linear-algebra/gaussian-elimination :::

## Rank profile matrix {#rank-profile}

Let $A$ be an $m \times n$ matrix of rank $r$ over $\Z/p\Z$. Its *row
rank profile* is the lexicographically smallest set of $r$ linearly
independent rows, that is, the rows that are not linear combinations of
the rows above them; the *column rank profile* is defined in the same
way. The *rank profile matrix* $\mathcal{R}_A$ is the unique $m \times n$
matrix with $r$ entries equal to 1, no two in the same row or column, and
zeros elsewhere, such that for all $i \le m$ and $j \le n$ the submatrices
formed by the first $i$ rows and $j$ columns of $A$ and of $\mathcal{R}_A$
have the same rank [@DPS2017]. Its rows and columns with a nonzero entry
are the row and column rank profiles of $A$, and the same holds for the
leading submatrices of $A$ with the entries of $\mathcal{R}_A$ that lie in
them.

## Output {#output}

Let $r$ be the returned rank. On return, as for `nmod_mat_lu`
([](nmod_mat/lu#output)):

- row $i$ of $LE$ is row `P[i]` of the original $A$, where $L$ is the
  $m \times r$ unit lower triangular matrix stored below the diagonal and
  $E$ the $r \times n$ matrix in row echelon form stored on and above it,
  whose pivot in row $k$ is in column $c_k$, with
  $c_0 < c_1 < \cdots < c_{r-1}$;
- the rows $i \ge r$ of the output contain $L$ in their first $r$ columns
  and zeros elsewhere;
- if `pivots_nonpivots` is not `NULL`, it receives
  $c_0, \ldots, c_{r-1}$ followed by the indices of the other columns in
  increasing order.

In addition, the rows without pivot are in increasing order,
`P[r]` $<$ `P[r+1]` $< \cdots <$ `P[m-1]`, and the nonzero entries of
$\mathcal{R}_A$ are at the positions (`P[k]`, $c_k$) for $k < r$. In
particular $c_0, \ldots, c_{r-1}$ is the column rank profile of $A$ and
`P[0]`, ..., `P[r-1]` sorted is its row rank profile.

These conditions determine `P` and the pivot columns, and then $L$ and
$E$, since an LU decomposition with unit $L$ is unique once the order of
the rows and the pivot columns are fixed. All the algorithms therefore
compute the same output.

`nmod_mat_lu` chooses the same pivot columns, but it brings each pivot row
into place by a swap with the row in that position, which changes the
order of the rows passed over: its pivot rows need not be the row rank
profile, as in the example below. `nmod_mat_ple` uses an upward rotation
instead: the pivot row moves up, and the rows passed over move down by
one position each, keeping their order, so that the rows without pivot
remain in increasing order of their indices in $A$. The algorithms use
this to avoid moving rows of the matrix too often. The classical
algorithm keeps the rows in place in memory and records their order in
`P` only, so that an upward rotation past $k$ rows costs $O(k)$
operations on indices, and permutes the rows once at the end. The
delayed and Crout algorithms swap the pivot row with the row in its
position, as `nmod_mat_lu` does, so that the eliminations run through
consecutive rows; the pivot of a column is then the row with a nonzero
entry of smallest index in $A$ among the rows without pivot, and these
rows are put back in increasing order at the end. The recursive algorithm puts the rows in
order once for each half at each level of the recursion.

## Kernels {#kernels}

From $A = \Pi L E$, the right nullspace of $A$ is that of $E$ and the
left nullspace is that of $L$ with the columns permuted by $P$. Writing
$E_1$ and $E_2$ for the pivot and the other columns of $E$, and $L_1$,
$L_2$ for the first $r$ and the other rows of $L$, the matrices
$-E_1^{-1} E_2$ and $-L_2 L_1^{-1}$ give the bases with an identity matrix
in the columns without pivot and in the rows `P[r]`, ..., `P[m-1]`; the
latter are the rows of $A$ that are not in its row rank profile, since
the decomposition reveals it. `nmod_mat_ple_nullspace` and
`nmod_mat_ple_left_nullspace` compute these bases from the output of
`nmod_mat_ple` with one triangular solve, and their `_compact` versions
only the blocks $-E_1^{-1} E_2$ and $-L_2 L_1^{-1}$ (with the columns of
the latter sorted by row index). These are the bases computed by
`nmod_mat_nullspace` and `nmod_mat_left_nullspace`, which use these
functions.

## Checking the rank {#rank-check}

With `NMOD_MAT_PLE_RANK_CHECK` in `flags`, the function returns the rank
if it is $\min(m, n)$, with the same output as without the flag, and
returns 0 otherwise, for all shapes of $A$; this differs from the
`rank_check` argument of `nmod_mat_lu`, which is only reliable for
$m \ge n$. The decomposition stops as soon as more than
$n - \min(m, n)$ columns without pivot have been met, and then leaves
`A`, `P` and `pivots_nonpivots` undefined.

## Moduli {#moduli}

The modulus should be prime. For a composite modulus the functions raise
a `FLINT_IMPINV` exception when they meet a nonzero pivot that is not
invertible, as `nmod_mat_lu` does.

## Algorithms {#algorithms}

`nmod_mat_ple_classical` is Gaussian elimination column by column, with
one reduction for each update of an entry; rows whose multiplier is zero
are not updated. The pivot of a column is the first row, in the order
recorded in `P`, among the rows that are not pivot rows yet, with a
nonzero entry in that column. Why this pivoting reveals the rank profile
matrix is explained in
[](algorithms/linear-algebra/gaussian-elimination#rank-profile).

`nmod_mat_ple_classical_delayed` performs the same operations, but
accumulates the updates of each entry unreduced, in the matrix itself if
$\min(m, n) (p - 1)^2 < 2^{\bits}$ and otherwise in a buffer of two or
three words per entry, and reduces an entry only when it is about to be
used, as `nmod_mat_lu_classical_delayed`
([](algorithms/linear-algebra/gaussian-elimination#delayed)).

`nmod_mat_ple_crout` performs the same operations in the order of the
Crout schedule ([](algorithms/linear-algebra/gaussian-elimination#crout)):
the entries of a column in the rows without pivot are computed when the
column is processed, by dot products of the multipliers of these rows
with the entries of $E$ above them, and the new pivot row by a
vector-matrix product of its multipliers with the rows of $E$. These
products delay their reductions, with SIMD code for all moduli, and no
buffer is needed. They skip the zero multipliers before the first and
after the last nonzero one of each row, but not the others.

`nmod_mat_ple_recursive` is the recursion of `nmod_mat_lu_recursive`: it
decomposes the left half of the columns, puts the rows of the right half
in the same order, updates them with a triangular solve and a matrix
multiplication, and decomposes the remaining block, calling
`nmod_mat_ple` for the two halves. Its cost is dominated by matrix
multiplication.

`nmod_mat_ple` uses the classical algorithm when $\min(m, n) \le 3$, or
$\le 7$ for $p \ge 2^{28}$, and the recursive algorithm when
$\min(m, n)$ is at least the following crossover $c$, or when
$m \ge 4n$, $n \ge c/2$ and $p \ge 4$:

| Modulus | $c$ |
|---|---|
| $p < 4$ | 64 |
| $4 \le p < 2^7$ | 244 to 332 |
| $2^7 \le p < 2^{28}$ | 320 |
| $2^{28} \le p < 2^{29}$ | 160 |
| $2^{29} \le p < 2^{32}$ | 64 |
| $2^{32} \le p < 2^{50}$ | 160 |
| $2^{50} \le p < 2^{52}$ | 256 |
| $p \ge 2^{52}$ | 768 |

Otherwise, it uses the Crout algorithm when the delayed algorithm would
need two or three words per entry and either $p \ge 2^{50}$ and
$mn \ge 2^{15}$, or $A$ has at least 32 rows and at least twice as many
columns as rows (four times for $p < 2^{32}$), where the vector-matrix
products are most of the work; otherwise the delayed algorithm. These
choices were measured on a Xeon Cascade Lake (AVX-512 without IFMA)
with uniformly random matrices, and may not be the best on other
machines. On matrices with many zero multipliers, the classical and
delayed algorithms, which skip them, can be faster than the others
beyond these crossovers.

## Example {#example}

The decomposition of a $4 \times 5$ matrix of rank 3 modulo 17 whose
first two rows are proportional, compared with `nmod_mat_lu`:

```c example name=nmod_mat_ple_rank_profile
#include <flint/flint.h>
#include <flint/nmod_mat.h>

int main(void)
{
    /* rows 0 and 1 are proportional; row 2 has the only nonzero entry
       of column 0 */
    ulong a[4][5] = { { 0, 3, 1, 0, 2 },
                      { 0, 6, 2, 0, 4 },
                      { 5, 1, 0, 2, 3 },
                      { 0, 0, 0, 4, 1 } };
    nmod_mat_t A, LU;
    slong P[4], Q[5], i, j, r;

    nmod_mat_init(A, 4, 5, 17);
    for (i = 0; i < 4; i++)
        for (j = 0; j < 5; j++)
            nmod_mat_entry(A, i, j) = a[i][j];

    nmod_mat_init_set(LU, A);
    r = nmod_mat_ple(P, Q, LU, 0);

    flint_printf("rank %wd, P = [%wd %wd %wd %wd]\n", r, P[0], P[1], P[2], P[3]);
    flint_printf("pivots_nonpivots = [%wd %wd %wd %wd %wd]\n",
                 Q[0], Q[1], Q[2], Q[3], Q[4]);
    nmod_mat_print_pretty(LU);

    flint_printf("rank profile matrix:");
    for (i = 0; i < r; i++)
        flint_printf(" (%wd, %wd)", P[i], Q[i]);
    flint_printf("\n");

    /* nmod_mat_lu swaps rows instead and selects row 1, not row 0 */
    nmod_mat_set(LU, A);
    r = nmod_mat_lu(P, LU, 0);
    flint_printf("nmod_mat_lu: P = [%wd %wd %wd %wd]\n", P[0], P[1], P[2], P[3]);

    nmod_mat_clear(A);
    nmod_mat_clear(LU);
    return 0;
}
```
```output
rank 3, P = [2 0 3 1]
pivots_nonpivots = [0 1 3 2 4]
<4 x 5 integer matrix mod 17>
[ 5  1  0  2  3]
[ 0  3  1  0  2]
[ 0  0  0  4  1]
[ 0  2  0  0  0]
rank profile matrix: (2, 0) (0, 1) (3, 3)
nmod_mat_lu: P = [2 1 3 0]
```

Column 0 has its only nonzero entry in row 2, which becomes the first
pivot row by an upward rotation: rows 0 and 1 move down one position
each and keep their order, so that row 0 is the pivot row of column 1. Row 1, twice row 0, then
becomes zero; its row of $L$, in the last row of the output, holds the
multiplier 2. Columns 2 and 4 have no pivot. The pivot rows 2, 0, 3 form
the row rank profile $\{0, 2, 3\}$, and the leading $2 \times 2$
submatrix of $A$, of rank 1, contains one entry of the rank profile
matrix, at $(0, 1)$. `nmod_mat_lu` swaps rows 0 and 2 instead, which
puts row 1 before row 0, and its pivot rows 2, 1, 3 are not the row rank
profile.

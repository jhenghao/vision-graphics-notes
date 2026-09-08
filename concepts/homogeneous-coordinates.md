---
title: Homogeneous Coordinates
aliases: [projective coordinates]
tags: [homogeneous-coordinates]
status: growing
---

## Motivating Question

Graphics and vision pipelines want to express every transform — rotate,
scale, translate, project — as "multiply by a matrix," so that a whole
chain of operations collapses into a single matrix product. Standard 3×3
matrices handle rotation and scaling for free. But translation needs
*addition*, and perspective projection needs *division* — neither fits the
"matrix times vector" mold. How do you fold those in too?

## Reasoning / Derivation

In ordinary 3D coordinates $(x,y,z)$:

- ✅ **Rotation** — a 3×3 matrix multiply.
- ✅ **Scaling** — a 3×3 matrix multiply.
- ❌ **Translation** — requires addition, not multiplication.
- ❌ **Perspective projection** — requires division by $z$ (see
  [[pinhole-camera-model]]).

The fix is to embed 3D points into 4D space by adding a 4th coordinate:

$$(x,y,z) \rightarrow (x,y,z,w)$$

For ordinary points, set $w=1$. This one addition is enough to make
translation a matrix multiply too. A translation by $(t_x,t_y,t_z)$
becomes:

$$
\begin{bmatrix} 1&0&0&t_x \\ 0&1&0&t_y \\ 0&0&1&t_z \\ 0&0&0&1 \end{bmatrix}
\begin{bmatrix} x\\y\\z\\1 \end{bmatrix} =
\begin{bmatrix} x+t_x\\y+t_y\\z+t_z\\1 \end{bmatrix}
$$

The extra $1$ in the last row/column is what lets the matrix "leak" the
translation constants into the output through multiplication rather than a
separate addition step. That's the whole trick: a 4×4 matrix acting on a
4-vector can now represent rotation, scaling, *and* translation uniformly,
and — as [[pinhole-camera-model]] shows — perspective projection too, once
you allow a nontrivial bottom row and a division by the resulting $w$.

## Formal Statement

$$(x,y,z) \rightarrow (x,y,z,w),\quad \text{ordinary points have } w=1$$

A 4×4 matrix multiply now expresses transforms that a 3×3 matrix alone
cannot (translation, perspective), because the extra coordinate gives the
matrix a channel to inject constants and to produce a nonzero divisor for
projection.

## Worked Example

Translate the point $(2,3,1)$ by $(t_x,t_y,t_z) = (5,-1,0)$:

$$
\begin{bmatrix} 1&0&0&5 \\ 0&1&0&-1 \\ 0&0&1&0 \\ 0&0&0&1 \end{bmatrix}
\begin{bmatrix} 2\\3\\1\\1 \end{bmatrix} =
\begin{bmatrix} 2+5\\3-1\\1+0\\1 \end{bmatrix} =
\begin{bmatrix} 7\\2\\1\\1 \end{bmatrix}
$$

giving $(7,2,1)$ — a plain vector addition, but reached through matrix
multiplication, which is what makes it composable with rotation/scaling
matrices in a single chained product.

## Connections

- [[pinhole-camera-model]] — this is the mechanism that lets that note's
  projection (a division by $Z$) be written as "matrix multiply, then
  dehomogenize," rather than as a separate non-linear step.
- [[points-vs-directions]] — the $w$ coordinate isn't just a bookkeeping
  trick for translation; setting $w=0$ instead of $1$ gives you a second,
  useful category of object (directions) with genuinely different
  transformation behavior.
- [[scaling-equivalence]] — once you have a 4th coordinate, a natural
  follow-up question is what $(2x,2y,2z,2w)$ *means* relative to
  $(x,y,z,w)$ — that note answers it.
- [[perspective-projection-matrix]] — the graphics pipeline's full
  projection matrix is a 4×4 homogeneous-coordinate matrix built exactly
  on this foundation.

## Open Questions

- None currently — the core trick (add a coordinate, multiply, divide) is
  solid. Revisit if a source explains *why* 4D specifically (vs. some other
  encoding) is the minimal way to unify these operations.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

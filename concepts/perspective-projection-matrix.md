---
title: The Perspective Projection Matrix
aliases: [projection matrix]
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

[[pinhole-camera-model]] shows the core idea — divide by depth — using a
minimal 3×4 matrix. A real graphics pipeline needs more: it has to respect
field of view, aspect ratio, near/far clipping planes, and produce a $w$
that the GPU can safely clip against *before* dividing (see
[[clip-space]]). What does a matrix have to do, structurally, to satisfy
all of that at once — and is there really only one way to build it?

## Reasoning / Derivation

The projection matrix $P$ transforms camera-space coordinates (see
[[camera-space]]) into clip space. Rather than guessing at a matrix and
checking it works, derive its shape from the properties it's required to
have:

**Five required properties:**

1. **Perspective**: $x_{ndc} = x/(-z)$, $y_{ndc} = y/(-z)$ — the same
   division-by-depth as [[pinhole-camera-model]], just written for the
   $-Z$-forward convention.
2. **Correct field of view (FOV)**.
3. **Correct aspect ratio**.
4. **Near and far planes map to fixed NDC $z$ values** — so depth
   comparisons downstream are well-defined.
5. **Points in front of the camera have positive $w$** — so the GPU's
   clip test (see [[clip-space]]) can distinguish visible geometry from
   geometry behind the camera.

Every entry in the matrix below exists to enforce one of these five. The
OpenGL perspective matrix is:

$$
P = \begin{bmatrix}
\frac{1}{a\tan(\theta/2)} & 0 & 0 & 0 \\
0 & \frac{1}{\tan(\theta/2)} & 0 & 0 \\
0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

where $\theta$ is the vertical field of view, $a$ is the aspect ratio
(width/height), $n$ is the near-plane distance, and $f$ is the far-plane
distance.

| Row | Purpose | Formula |
|-----|---------|---------|
| Row 1 | Horizontal FOV + aspect ratio | $1/(a\tan(\theta/2))$ |
| Row 2 | Vertical FOV | $1/\tan(\theta/2)$ |
| Row 3 | Nonlinear depth mapping | $A=-(f{+}n)/(f{-}n)$, $B=-2fn/(f{-}n)$ |
| Row 4 | Perspective divide ($w=-z$) | $[0,0,-1,0]$ |

**Why row 4 creates perspective.** That last row picks out
$w_{clip} = 0\cdot x + 0\cdot y + (-1)\cdot z + 0\cdot 1 = -z_{camera}$.
After the perspective divide (see [[ndc]]):

$$x_{ndc} = \frac{x_{clip}}{w_{clip}} = \frac{x}{-z}$$

— exactly the pinhole camera equation from [[pinhole-camera-model]],
recovered as a side effect of the matrix structure rather than bolted on
separately.

**Why $w = -z$ specifically.** The camera looks down $-Z$ (see
[[camera-coordinate-system]]), so points genuinely in front of the camera
have $z<0$. Requirement 5 wants $w>0$ for visible points. Setting
$w=-z$ satisfies both at once: visible points ($z<0$) get $w>0$; points
behind the camera ($z>0$) get $w<0$, which the clip test in
[[clip-space]] can then reject.

## Formal Statement

$$
P = \begin{bmatrix}
\frac{1}{a\tan(\theta/2)} & 0 & 0 & 0 \\
0 & \frac{1}{\tan(\theta/2)} & 0 & 0 \\
0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

> **Key insight**: the perspective projection matrix is the unique matrix
> that satisfies the pinhole camera model, perspective division, and
> near/far depth constraints simultaneously.

## Worked Example

Trace just the $w$ and $z$ rows for a point at camera-space $z=-5$ (5 units
in front of the camera):

$$w_{clip} = -1 \cdot (-5) = 5 > 0 \quad\checkmark\ \text{(visible, per requirement 5)}$$

Compare with a point behind the camera, $z=+2$:

$$w_{clip} = -1 \cdot 2 = -2 < 0$$

— which the GPU's clip test (see [[clip-space]]) rejects, before any
division ever happens. This is the whole reason clipping is done against
$w$ rather than after dividing by it: dividing by a negative or zero $w$
first would produce garbage or a crash, not a clean rejection.

## Connections

- [[pinhole-camera-model]] — this matrix is the full graphics-pipeline
  generalization of that note's minimal 2-row projection, with FOV,
  aspect ratio, and depth-range constraints folded in.
- [[camera-space]] — this matrix's input.
- [[clip-space]] — this matrix's output, and the reason requirement 5
  (positive $w$ for visible points) matters at all.
- [[ndc]] — reached by dividing this matrix's output by its own $w$
  component (row 4's whole purpose).
- [[depth-nonlinearity]] — row 3's $A, B$ coefficients are exactly what
  produces the nonlinear $1/z$ depth mapping discussed there.

## Open Questions

- The claim "this is the *unique* matrix satisfying these five properties"
  is stated as a takeaway in the source material but not proven here —
  worth deriving directly (e.g. by writing out a general 4×4 matrix and
  solving for the entries under each constraint) if this note gets
  revisited.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

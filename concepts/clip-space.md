---
title: Clip Space
aliases: []
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

Once [[perspective-projection-matrix]] produces homogeneous coordinates
$(x,y,z,w)$, why not just divide by $w$ immediately and get on with
rendering? What does the GPU gain by keeping an intermediate,
not-yet-divided space around at all?

## Reasoning / Derivation

Clip space is simply the direct result of applying the projection matrix
to a camera-space coordinate:

$$\text{clip\_coord} = P \times \text{camera\_coord}$$

producing a homogeneous 4-vector $(x_{clip}, y_{clip}, z_{clip}, w_{clip})$
that has *not yet* been divided by $w$ (see [[ndc]] for that next step).

The reason to pause here, rather than dividing immediately, is
**clipping**. The GPU needs to discard geometry that's outside the
viewable frustum — off to the side, beyond the far plane, or behind the
camera — and it needs to do that safely, before the division happens.
A point is kept only if:

$$-w_{clip} \le x_{clip} \le w_{clip}$$
$$-w_{clip} \le y_{clip} \le w_{clip}$$
$$-w_{clip} \le z_{clip} \le w_{clip}$$

Anything outside these bounds is discarded. Why bother clipping *before*
dividing, instead of dividing first and checking $x,y,z\in[-1,1]$
afterward? Because dividing first requires dividing by $w$ regardless of
its value — and $w$ can be zero (a point at infinity, see
[[projective-geometry]]) or negative (a point behind the camera, per
[[perspective-projection-matrix]]'s requirement 5). Clipping against
$\pm w$ first avoids:

- **Division by zero**.
- **Objects behind the camera** ($w<0$) leaking through as nonsensical
  divided coordinates.
- Wasted division work on **geometry outside the view frustum** entirely.

## Formal Statement

$$\text{clip\_coord} = P \times \text{camera\_coord} = (x_{clip}, y_{clip}, z_{clip}, w_{clip})$$

A point is kept if $-w_{clip} \le x_{clip}, y_{clip}, z_{clip} \le w_{clip}$; otherwise it's discarded.

> **Key insight**: clip space is GPU plumbing. NDC is geometry
> normalization. NeRF only needs the second (see [[why-nerf-skips-clip-space]]).

## Worked Example

Using [[perspective-projection-matrix]]'s worked example: a point at
camera-space $z=-5$ produced $w_{clip}=5$. Suppose its projected
$x_{clip}=3$. The clip test checks:

$$-5 \le 3 \le 5 \quad\checkmark$$

— kept. If instead $x_{clip}=8$ (the point is far off to the side of the
frustum), the test $-5 \le 8 \le 5$ fails, and the point is clipped away
*before* any division by $w$ is attempted.

## Connections

- [[perspective-projection-matrix]] — produces the clip-space coordinates
  this note operates on, and is the reason requirement 5 (positive $w$ for
  visible points) matters here.
- [[ndc]] — the very next pipeline stage: dividing clip-space coordinates
  by $w_{clip}$ to get Normalized Device Coordinates.
- [[projective-geometry]] — explains why $w=0$ specifically is meaningful
  (a point at infinity) rather than just a numerical edge case to avoid.
- [[why-nerf-skips-clip-space]] — NeRF has no triangles to clip and no GPU
  clipping hardware to feed, so it skips this stage entirely and goes
  straight to NDC.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

---
title: Projective Geometry Connection
aliases: [vanishing points, points at infinity]
tags: [homogeneous-coordinates]
status: growing
---

## Motivating Question

[[scaling-equivalence]] shows that dehomogenizing $(x,y,z,w)$ means
dividing by $w$. What happens when $w=0$? Division by zero doesn't produce
a finite 3D point — so is $(x,y,z,0)$ meaningless, or does it correspond to
something real? (Photos of railroad tracks converging to a point on the
horizon suggest it should mean *something*.)

## Reasoning / Derivation

In ordinary Euclidean geometry, parallel lines never meet — that's
practically the definition of "parallel." But **projective geometry**
extends the plane (or space) with extra points "at infinity," and in that
extended space, parallel lines *do* meet, at one of these points at
infinity. This isn't just a mathematical curiosity: it's exactly what a
photograph shows you. Railroad tracks are parallel in 3D, yet in the image
they visibly converge toward a single vanishing point.

Homogeneous coordinates give this idea an algebraic home. If $w=0$, you
literally cannot dehomogenize — $(x,y,z,0) \rightarrow$ division by zero.
Rather than treating that as an error, projective geometry treats it as
the point at infinity in the direction $(x,y,z)$:

$$(x,y,z,0) \rightarrow \text{division by zero} \rightarrow \text{point at infinity}$$

This lines up exactly with [[points-vs-directions]]'s observation that
$w=0$ marks a *direction* rather than a location — a direction vector
really is a point infinitely far away in that direction:

```
(1, 0, 0, 0) → direction along +x (point at infinity along the x-axis)
(0, 1, 0, 0) → direction along +y (point at infinity along the y-axis)
```

**Connecting it back to vanishing points.** Rails that are parallel in 3D
space share a common direction vector $(dx,dy,dz,0)$ — a single point at
infinity. When that shared point gets projected into the image (by the
same matrix machinery used for ordinary points — see
[[perspective-projection-matrix]]), it lands at one specific, finite pixel
location: the vanishing point you see in the photo. The rails don't
actually meet in 3D; their shared *point at infinity* does, and perspective
projection is what makes that visible.

## Formal Statement

A homogeneous coordinate with $w=0$ does not correspond to any finite 3D
point; it represents a **point at infinity** in the direction $(x,y,z)$.
Parallel lines sharing a direction vector share a point at infinity, whose
projection into an image is a **vanishing point**.

> **Vanishing points are points at infinity made visible by perspective
> projection.**

## Worked Example

Two parallel rails running in the $+z$ direction share the direction
vector $(0,0,1,0)$ — a single point at infinity. Feed that direction
through the same projection matrix that maps ordinary 3D points to image
coordinates (see [[perspective-projection-matrix]]). Unlike an ordinary
point, it produces a coordinate that — after the usual perspective divide
— lands at one fixed, finite pixel: that's the vanishing point where the
two rails visually converge in the photo, even though in the actual 3D
scene they never intersect.

## Connections

- [[scaling-equivalence]] — this note is the $w=0$ edge case that note's
  $w\neq0$ equivalence-class story deliberately sets aside.
- [[points-vs-directions]] — the same $w=0$ condition, reinterpreted:
  that note frames it as "immune to translation," this note frames it
  geometrically as "infinitely far away." Both descriptions are true of
  the same object.
- [[perspective-projection-matrix]] — the mechanism that actually turns a
  point at infinity into a finite, visible vanishing point in an image.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

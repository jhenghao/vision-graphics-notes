---
title: Camera to World Transformation
aliases: [c2w matrix]
tags: [vision]
status: growing
---

## Motivating Question

[[pixel-to-ray-conversion]] and [[why-y-is-negated]] produce a ray
direction expressed in *camera* space — relative to wherever the camera
happens to be pointing. To actually reconstruct or reason about a 3D
scene, that ray needs to be expressed in **world** space, a single fixed
frame shared across every camera view. What transform does that, and why
does it treat the ray's origin differently from its direction?

## Reasoning / Derivation

The **camera-to-world** (c2w) matrix does exactly what its name says: it
converts a point or direction expressed in camera space into world space.
Structurally it's a standard rigid transform — rotation plus translation —
expressed in homogeneous form (see [[homogeneous-coordinates]]):

$$
c2w = \begin{bmatrix} R_{3\times3} & t_{3\times1} \\ 0_{1\times3} & 1 \end{bmatrix}
$$

It has two logically separate components, and — crucially — they get
applied to two different kinds of ray data:

1. **Rotation** (`c2w[:3,:3]`) — rotates ray *directions* from the camera's
   local frame into the world's frame. Since a direction has $w=0$ (per
   [[points-vs-directions]]), the translation part of $c2w$ has no effect
   on it — only the rotation block matters.
2. **Translation** (`c2w[:3,-1]`) — the camera's own position in world
   coordinates. Since a ray's *origin* is a point with $w=1$, this becomes
   its world-space location directly: every ray starts exactly where the
   camera is.

This split — rotate directions, translate origins — is a direct
consequence of the $w=1$ vs. $w=0$ distinction from
[[points-vs-directions]]; it isn't a special rule for cameras, it's what
the homogeneous transform already does automatically once you tag each
quantity with the right $w$.

## Formal Statement

$$
c2w = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}
$$

```python
# Rotate directions: camera → world
rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)

# Ray origins = camera position in world
rays_o = c2w[:3, -1].expand(rays_d.shape)
```

Every ray **starts** at the camera center (`c2w[:3,-1]`) and **points** in
the rotated direction (`c2w[:3,:3] × dir`).

## Worked Example

Take the camera-space ray direction from [[why-y-is-negated]]'s worked
example, $(x_n,y_n,z)=(0.333,\,-1)$ — say the full direction vector is
$\text{dir}=(0.1,\,0.333,\,-1)$ — and a camera positioned at world
coordinates $t=(0,0,5)$ with no rotation ($R=I$, identity):

- **Rotated direction**: $R\cdot\text{dir} = (0.1,0.333,-1)$ (unchanged,
  since $R=I$).
- **Ray origin**: `c2w[:3,-1]` $= (0,0,5)$ — the ray starts exactly at the
  camera's world position.

If the camera were instead rotated $180°$ about the world $y$-axis (facing
the opposite way), the same camera-space direction would rotate to point
in roughly $(-0.1, 0.333, 1)$ in world space — the origin stays at
$(0,0,5)$ regardless, since rotation never touches the origin
(translation-only effect on directions is exactly zero, per
[[points-vs-directions]]).

## Connections

- [[points-vs-directions]] — the $w=1$/$w=0$ split this note's "rotate
  directions, translate origins" behavior falls directly out of.
- [[why-y-is-negated]] / [[pixel-to-ray-conversion]] — supply the
  camera-space ray direction this transform converts to world space.
- [[nerf-ray-generation]] — the concrete `get_rays` function applies
  exactly this c2w transform as its final step, after computing camera-space
  directions.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

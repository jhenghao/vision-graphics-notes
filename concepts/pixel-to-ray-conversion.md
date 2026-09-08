---
title: Pixel to Ray Conversion
aliases: [normalized camera coordinates]
tags: [vision]
status: growing
---

## Motivating Question

[[camera-intrinsics]] shows how to go from a 3D camera-space point to a 2D
pixel. Vision needs the reverse: given only a pixel location in an image,
what can you say about the 3D ray that produced it?

## Reasoning / Derivation

Graphics projects 3D points to 2D pixels; vision does the reverse. Start
from $K$'s forward mapping (see [[camera-intrinsics]]) and invert it. Given
a pixel location $(i,j)$, undo the two operations $K$ performed — scale by
focal length, then shift by principal point — in reverse order: subtract
the principal point first, then divide by focal length:

$$x_n = \frac{i-c_x}{f_x}, \qquad y_n = \frac{j-c_y}{f_y}$$

Concretely, this does two things:

1. **Subtract $(c_x,c_y)$**: recenters the pixel from the top-left-origin
   image coordinate system to the optical center.
2. **Divide by $(f_x,f_y)$**: removes the focal-length scaling, converting
   a pixel offset into an angle-like quantity (a slope, really).

The result, $(x_n, y_n)$, is only **2D** — it tells you a direction on the
image plane, but says nothing about depth. It only becomes a full 3D
camera-space direction once you assign it a depth. By convention (matching
the virtual image plane's position under the OpenGL/NeRF convention — see
[[camera-coordinate-system]]), that depth is $z=-1$:

$$(x_n, y_n, -1)$$

is a direction vector in camera space, corresponding to the ray this pixel
lies on. Note this is a *direction*, not a full 3D point — it needs
$w=0$ in homogeneous terms (see [[points-vs-directions]]), since scaling
it further just moves along the same ray.

| Concept | Dimensions | Has depth? | Meaning |
|---------|------------|------------|---------|
| Normalized camera coords | 2D | ❌ No | Direction on image plane |
| Camera space | 3D | ✅ Yes | Full 3D coordinates |
| $(x_n,y_n,-1)$ | 3D | ⚠️ Fixed | Direction vector in camera space |

## Formal Statement

$$x_n = \frac{i-c_x}{f_x}, \qquad y_n = \frac{j-c_y}{f_y}$$

Normalized camera coordinates become a camera-space ray direction only
after fixing a depth (conventionally $z=-1$): $(x_n, y_n, -1)$.

## Worked Example

Using [[camera-intrinsics]]'s worked-example $K$ ($f_x=f_y=900$,
$c_x=500$, $c_y=400$), invert the pixel found there, $(i,j)=(590,445)$:

$$x_n = \frac{590-500}{900} = 0.1 \qquad y_n = \frac{445-400}{900} = 0.05$$

giving the camera-space ray direction $(0.1, 0.05, -1)$. Compare with the
original 3D point used to produce that pixel, $(X,Y,Z)=(1,0.5,-10)$:
normalizing that point by its own (negative) depth, $(X/{-Z}, Y/{-Z}) =
(0.1, 0.05)$ — matching exactly, since inverting $K$ recovers the ray's
direction, not the original point's specific depth (which is lost in the
forward projection).

## Connections

- [[camera-intrinsics]] — this note is the algebraic inverse of that
  note's forward $K$-matrix projection.
- [[why-y-is-negated]] — the formula here uses $j-c_y$ without a sign
  flip; that note explains why NeRF's actual ray-generation code negates
  this term, due to the image-vs-camera $y$-axis mismatch this note
  doesn't yet address.
- [[points-vs-directions]] — $(x_n,y_n,-1)$ is exactly the kind of
  "direction, not location" object that note describes with $w=0$.
- [[nerf-ray-generation]] — applies this exact formula, pixel-by-pixel
  across a whole image, as the first step of generating a batch of rays.

## Open Questions

- None currently — see [[why-y-is-negated]] for the refinement this
  formula needs before it matches real ray-generation code.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

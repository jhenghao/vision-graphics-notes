---
title: Camera Coordinate System
aliases: []
tags: [camera]
status: growing
---

## Motivating Question

[[pinhole-camera-model]] derives projection assuming the camera looks down
$+Z$, since that's the direction that falls naturally out of the "object in
front, image plane behind" geometry. But most graphics engines, and NeRF
implementations in particular, define the camera as looking down $-Z$
instead. Before any of the projection math can be trusted, you need to pin
down: relative to the camera itself, which way do $X$, $Y$, and $Z$ actually
point?

## Reasoning / Derivation

The camera coordinate system is a local 3D frame attached to the camera —
it answers "where is this point, relative to me" before anything gets
projected down to 2D. There's no single universally correct choice of
axes; different tools pick different conventions. The one used throughout
the rest of this vault (and standard in OpenGL and NeRF) is:

```
        y (up)
        ↑
        │
        │
        │
        ●───────→ x (right)
       /
      /
     ↓
    z (backward)
```

- **Origin**: the camera center (the pinhole).
- **+X**: right.
- **+Y**: up.
- **+Z**: backward — behind the camera.

The consequence worth internalizing is the one that trips people up: since
$+Z$ points *backward*, the camera looks down $-Z$. So:

- Points **in front of** the camera have $z < 0$.
- Points **behind** the camera have $z > 0$.

This is the opposite sign convention from the "standard computer vision"
derivation in [[pinhole-camera-model]], where the camera looks down $+Z$
and objects in view have $Z>0$. Neither is "wrong" — they're just different
choices for which way the axis points, and every projection formula that
follows has to be read in light of whichever one is in force.

## Formal Statement

**OpenGL / NeRF convention** (used throughout this vault unless noted
otherwise):

| Axis | Direction |
|------|-----------|
| +X | right |
| +Y | up |
| +Z | backward (behind camera) |

Camera looks down $-Z$. A point in camera space with $z<0$ is visible
(in front); $z>0$ is behind the camera.

## Worked Example

A point at camera-space coordinates $(x_c, y_c, z_c) = (0.5, 0.3, -5.0)$:

- $z_c = -5.0 < 0$ → the point is 5 units **in front of** the camera (the
  camera looks down $-Z$, so negative-$z$ points are exactly where the
  camera is looking).
- $x_c = 0.5 > 0$ → slightly to the **right** of camera-forward.
- $y_c = 0.3 > 0$ → slightly **above** camera-forward.

If instead this point had $z_c = +5.0$, it would sit 5 units *behind* the
camera — outside the field of view no matter what $x_c, y_c$ are, since the
camera never looks in the $+Z$ direction.

## Connections

- [[pinhole-camera-model]] — that note's derivation assumes $+Z$-forward
  (objects in view have $Z>0$); this note is the OpenGL/NeRF convention
  actually used downstream, where the sign is flipped. That note's
  "Important" callout works through the conversion between the two
  explicitly.
- [[focal-length-and-image-plane]] — because the camera looks down $-Z$
  here, the virtual image plane sits at $Z=-f$ (a negative coordinate),
  which is easy to misread as "the plane is behind the camera" — it isn't;
  it's exactly in the direction the camera looks.
- [[camera-space]] — camera space *is* this coordinate system; that note
  picks up from here to describe it as the starting point of the graphics
  pipeline.
- [[camera-to-world-transform]] — the c2w matrix's job is to convert points
  and directions expressed in this local frame into world-space
  coordinates.

## Open Questions

- None yet — this note is mostly a fixed convention rather than something
  with open derivational threads. Revisit if a source introduces a
  left-handed or Y-down camera convention (e.g. some robotics or
  photogrammetry tools) and a comparison becomes useful.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

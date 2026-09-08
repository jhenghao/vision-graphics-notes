---
title: NDC (Normalized Device Coordinates)
aliases: [normalized device coordinates]
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

[[clip-space]] leaves geometry in homogeneous coordinates, not-yet-divided
by $w$. At some point the GPU actually needs plain 3D coordinates it can
rasterize onto a screen. What does that final division actually do, and
why does it land everything in the range $[-1,1]$ specifically?

## Reasoning / Derivation

After clipping, the GPU performs the **perspective divide** — dividing
every component of the clip-space coordinate by $w_{clip}$:

$$x_{ndc} = \frac{x_{clip}}{w_{clip}}, \quad y_{ndc} = \frac{y_{clip}}{w_{clip}}, \quad z_{ndc} = \frac{z_{clip}}{w_{clip}}$$

This is exactly the **dehomogenization** step from [[scaling-equivalence]]:
"choose the representative of this equivalence class where $w=1$." Before
the divide, $(x_{clip}, y_{clip}, z_{clip}, w_{clip})$ and any scalar
multiple of it represent the same point (per [[scaling-equivalence]]);
dividing through by $w$ just picks the one canonical representative.
Before: $(2x, 2y, 2z, 2w)$. After: $(x/w, y/w, z/w)$ — same point,
different representation.

Because [[perspective-projection-matrix]]'s clip test already constrained
$-w \le x,y,z \le w$, dividing through by (positive) $w$ necessarily lands
every coordinate in $[-1, 1]$:

$$x_{ndc}, y_{ndc}, z_{ndc} \in [-1, 1]$$

This normalized cube is **NDC**.

## Formal Statement

$$x_{ndc} = \frac{x_{clip}}{w_{clip}}, \quad y_{ndc} = \frac{y_{clip}}{w_{clip}}, \quad z_{ndc} = \frac{z_{clip}}{w_{clip}}, \qquad x_{ndc},y_{ndc},z_{ndc}\in[-1,1]$$

| Axis | Value −1 | Value +1 |
|------|----------|----------|
| x | Left edge of screen | Right edge |
| y | Bottom of screen | Top of screen |
| z | Far plane | Near plane (OpenGL) |

## Worked Example

Continuing the running example from [[clip-space]]: a kept point with
$x_{clip}=3$, $w_{clip}=5$:

$$x_{ndc} = \frac{3}{5} = 0.6$$

This lands within $[-1,1]$ as guaranteed by the clip test having already
passed — $0.6$ places the point 60% of the way from center toward the
right edge of the screen.

## Connections

- [[clip-space]] — the stage immediately before this one; NDC is what you
  get by dividing clip-space coordinates through by $w$.
- [[scaling-equivalence]] — the perspective divide *is* that note's
  dehomogenization operation, applied specifically to clip-space
  coordinates.
- [[screen-space]] — the next stage: mapping this $[-1,1]$ cube onto
  actual pixel coordinates.
- [[depth-nonlinearity]] — describes what happens specifically to
  $z_{ndc}$ under this divide, and why it isn't a linear function of
  camera-space depth.
- [[why-nerf-uses-ndc]] — NeRF needs exactly this normalized $[-1,1]$
  representation for numerical stability, without needing the
  clip-space/clipping machinery that produces it in a rasterizer.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

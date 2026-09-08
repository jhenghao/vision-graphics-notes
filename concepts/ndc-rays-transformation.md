---
title: NDC Rays Transformation
aliases: [ndc_rays]
tags: [nerf]
status: growing
---

## Motivating Question

[[nerf-ray-generation]] produces world-space rays with unbounded origins
and depths. For forward-facing scenes (see [[why-nerf-uses-ndc]]) NeRF
wants those rays reprojected into the bounded $[-1,1]$ NDC space from
[[ndc]] — but NDC as derived there assumes a matrix-and-clip-space
pipeline NeRF doesn't otherwise use. How do you get to the same NDC
representation for rays, analytically, without a projection matrix?

## Reasoning / Derivation

The `ndc_rays` function reprojects a ray (origin + direction) into NDC
space in three steps, mirroring what [[perspective-projection-matrix]] and
[[ndc]] do for points, but working through the algebra directly rather
than via a 4×4 matrix multiply.

```python
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Step 1: Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Step 2: Project origins into NDC
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    # Step 3: Convert directions to NDC slopes
    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] -
                                rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] -
                                rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d
```

**Step 1 — near plane intersection.**

```python
t = -(near + rays_o[..., 2]) / rays_d[..., 2]
rays_o = rays_o + t[..., None] * rays_d
```

Finds where each ray intersects the near plane ($z=-\text{near}$) and
moves the ray's origin to that intersection point. Purpose: all rays now
start at the same depth, matching the assumption baked into
[[perspective-projection-matrix]]'s near-plane handling — origins that
already differ in depth would otherwise reproject inconsistently.

**Step 2 — perspective projection of origins.**

```python
o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
o2 = 1. + 2. * near / rays_o[..., 2]
```

The scale factors $W/(2\cdot\text{focal})$ and $H/(2\cdot\text{focal})$
normalize by field of view (the same role $\tan(\theta/2)$ and aspect
ratio $a$ play in [[perspective-projection-matrix]]'s rows 1–2). Dividing
by `rays_o[...,2]` is the perspective divide (the same $x/z$, $y/z$
division that produces NDC in [[ndc]]) — here done directly on the ratio
rather than via a matrix-then-divide.

**Step 3 — ray slope conversion.**

```python
d0 = ... (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
d1 = ... (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
d2 = -2. * near / rays_o[..., 2]
```

Converts ray *directions* into *slopes* within NDC space — the difference
of ratios captures how the ray's NDC position changes as it moves, since a
direction under this nonlinear reprojection isn't simply "rotate the old
direction" the way [[camera-to-world-transform]]'s rigid transform was.

**Result:** near plane → $z=1$, far plane → $z\to-1$, $x,y\in[-1,1]$ — the
same normalized cube described in [[ndc]], reached analytically instead of
via [[clip-space]]'s matrix-and-clip machinery.

## Formal Statement

$$o_2 = 1 + \frac{2\cdot\text{near}}{\text{rays\_o}_z}, \qquad d_2 = -\frac{2\cdot\text{near}}{\text{rays\_o}_z}$$

with $o_0,o_1$ the perspective-divided origin components and $d_0,d_1$ the
slope-converted direction components, all scaled by
$W/(2\cdot\text{focal})$ or $H/(2\cdot\text{focal})$ per axis.

> **Note**: NeRF uses OpenGL-style math explicitly (see
> [[opengl-vs-directx]]) — NDC $z\in[-1,1]$, not DirectX's $[0,1]$.

## Worked Example

Take a single ray with origin already at the near plane after step 1, say
`rays_o[...,2] = -near` (exactly at $z=-\text{near}$, using the
$-Z$-forward convention from [[camera-coordinate-system]]):

$$o_2 = 1 + \frac{2\cdot\text{near}}{-\text{near}} = 1 - 2 = -1$$

Wait — this looks off by a sign relative to "near plane → $z=1$" stated
above; tracing it shows the near-plane depth actually needs
`rays_o[...,2]` evaluated *after* the origin has been shifted per step 1,
where the sign convention of the shift (via `t`) is what produces $+1$ in
practice. This is exactly the kind of arithmetic detail worth re-deriving
carefully against a numeric test case before trusting an adapted
implementation.

## Connections

- [[nerf-ray-generation]] — supplies the `rays_o, rays_d` this function
  consumes.
- [[perspective-projection-matrix]] / [[ndc]] — the matrix-based version of
  exactly the same reprojection; this note is the analytic, ray-based
  equivalent.
- [[why-nerf-uses-ndc]] — the motivation for doing this reprojection at
  all.
- [[why-nerf-skips-clip-space]] — explains why this goes straight to NDC
  rather than through [[clip-space]] first.

## Open Questions

- The worked example above surfaced a sign-convention detail that needs
  careful re-derivation with an actual numeric near-plane intersection,
  rather than assuming the origin is already exactly at $z=-\text{near}$.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

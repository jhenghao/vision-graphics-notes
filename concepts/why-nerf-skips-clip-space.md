---
title: Why NeRF Skips Clip Space
aliases: []
tags: [nerf]
status: growing
---

## Motivating Question

[[clip-space]] exists specifically to let a GPU safely discard
out-of-frustum triangle geometry before dividing by $w$. [[ray-marching-vs-rasterization]]
already notes NeRF doesn't use clip space — but *why* not, precisely? Is
it just that NeRF happens to skip a step, or is clip space actually
solving a problem NeRF doesn't have?

## Reasoning / Derivation

Clip space's entire reason for existing, per [[clip-space]], is to let a
GPU clip *triangles* against the view frustum before the (potentially
unsafe) perspective divide. Each of the three specific problems clipping
avoids there — division by zero, geometry behind the camera, geometry
outside the frustum — traces back to the fact that a rasterizer has to
process arbitrary triangle geometry supplied by the application, which
might extend anywhere in space, including partially or fully outside the
visible frustum.

NeRF's situation is structurally different in a way that removes the need
for that safeguard entirely:

1. **No triangles to clip.** NeRF renders a continuous volumetric field
   evaluated by a neural network, not a mesh of triangles that could
   straddle the frustum boundary and need subdividing.
2. **Rays are analytically defined.** Each ray, from
   [[nerf-ray-generation]], already has a well-defined origin and
   direction computed directly from the pixel it came from — there's no
   arbitrary incoming geometry that might be positioned outside the
   frustum by construction. Sampling only ever happens between the near
   and far bounds chosen for that ray.
3. **Direct path to NDC.** Since there's no clipping step needed, NeRF (for
   forward-facing scenes) goes straight from camera space to NDC via
   [[ndc-rays-transformation]]'s analytic reprojection, rather than via a
   matrix multiply followed by a hardware clip test.

> **"NeRF performs the perspective divide steps analytically, instead of
> via matrices and clip space."**

## Formal Statement

Clip space exists to safely clip arbitrary triangle geometry against the
view frustum before dividing by $w$. NeRF has no triangle geometry, and
its rays are already scoped to valid sampling bounds by construction — so
the safeguard clip space provides is unnecessary, and NeRF reprojects
directly from camera space to NDC.

## Worked Example

In rasterization, a large triangle might have one vertex inside the view
frustum and two vertices far outside it — [[clip-space]]'s clip test (and
associated triangle-subdivision logic) is exactly what's needed to handle
that case correctly before shading. In NeRF, there's no equivalent
situation: a ray for a given pixel is generated with a bounded
near/far sampling range (see [[ndc-rays-transformation]]'s near-plane
shift), so every sample point along it is already, by construction, within
the region the renderer cares about — there's nothing analogous to a
triangle vertex falling outside the frustum that needs to be clipped away.

## Connections

- [[clip-space]] — the mechanism, and the specific triangle-geometry
  problem, that this note explains NeRF doesn't need.
- [[ray-marching-vs-rasterization]] — the broader comparison this note is
  one specific piece of (clip space is one of several rasterization-only
  requirements NeRF skips).
- [[ndc-rays-transformation]] — the analytic reprojection NeRF uses
  instead of the matrix-and-clip-space route.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

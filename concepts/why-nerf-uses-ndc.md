---
title: Why NeRF Uses NDC
aliases: []
tags: [nerf]
status: growing
---

## Motivating Question

[[ndc-rays-transformation]] goes to real effort to reproject rays into
$[-1,1]$ NDC space, analytically, without a rasterizer's clip-space
pipeline. Why bother? What specifically about NeRF's training makes this
worth the extra step?

## Reasoning / Derivation

The motivating scenario is **forward-facing scenes** — the kind captured
by, e.g., the LLFF dataset, where every camera faces roughly the same
direction (think a row of photos taken while walking past a static scene).
Two properties of this setup create a problem for training a neural
network directly on raw world-space rays:

- **Rays are nearly parallel** — since all cameras face the same way, there's
  little angular diversity in world space to exploit for normalization.
- **Depth values vary wildly** — close foreground objects and a distant
  background can differ in depth by orders of magnitude, all within the
  same scene.

Feeding raw, unbounded, wildly-varying-scale depth values directly into a
neural network (as NeRF's MLP does, sampling points along each ray) is
numerically unfriendly — large and small values distort gradients and
sampling density unevenly.

Reprojecting rays into NDC — the same bounded $[-1,1]$ cube described in
[[ndc]] — fixes this:

1. **Normalizes depth** — compresses the unbounded depth range into
   $[-1,1]$, the same range [[ndc]] already established as a
   resolution/scale-independent representation for the graphics pipeline.
2. **Stabilizes training** — sampling uniformly in NDC space corresponds
   to *adaptively* denser sampling in world space near the camera (where
   more detail usually matters for forward-facing captures), and sparser
   sampling far away.
3. **Numerical stability** — avoids feeding the network very large raw
   coordinate values.

So NeRF is borrowing a purely *graphics* representation ([[ndc]]) and
applying it for a *training-stability* reason that has nothing to do with
GPU rasterization — the original motivation for NDC's existence.

## Formal Statement

NDC reprojection benefits NeRF training on forward-facing scenes by:
bounding the depth range to $[-1,1]$, converting uniform NDC sampling into
depth-adaptive world-space sampling, and avoiding large unbounded
coordinate values during optimization.

## Worked Example

Consider a forward-facing scene where the nearest surface is 1 unit from
the camera and the farthest visible background is 1000 units away — a
1000× dynamic range in raw depth. In world space, sampling 64 points
uniformly between depth 1 and depth 1000 would place the overwhelming
majority of samples in empty space near the far end, wasting model
capacity there while under-sampling the near foreground where most of the
interesting geometry actually is. Reprojecting into NDC first and then
sampling uniformly in $[-1,1]$ effectively concentrates samples near the
camera (small world-space steps map to larger, denser NDC steps there —
the same $1/z$ relationship from [[depth-nonlinearity]]), which is exactly
the sampling density forward-facing scenes need.

## Connections

- [[ndc]] — the representation being repurposed here for a training
  (rather than rasterization) reason.
- [[ndc-rays-transformation]] — the concrete mechanism that performs this
  reprojection for rays.
- [[depth-nonlinearity]] — the same $1/z$ compression effect that causes
  z-fighting in rasterized graphics is, here, exactly the *desired*
  behavior for sampling density.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

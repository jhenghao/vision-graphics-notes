---
title: OpenGL vs DirectX Differences
aliases: [graphics API conventions]
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

Every convention used throughout [[camera-coordinate-system]],
[[perspective-projection-matrix]], and [[ndc]] — $-Z$-forward, NDC $z \in
[-1,1]$ — is presented as *the* convention. But it's really just OpenGL's
choice. What does DirectX (and Vulkan) do differently, and does it
actually matter, or is it just a historical naming difference?

## Reasoning / Derivation

The two major API families diverge on a handful of specific conventions,
mostly around the NDC depth range and handedness:

| Feature | OpenGL | DirectX / Vulkan |
|---------|--------|--------------------|
| z range | $[-1,1]$ | $[0,1]$ |
| Near plane z | $+1$ | $0$ |
| Far plane z | $-1$ | $1$ |
| Handedness | Right-handed | Left-handed (configurable) |

The DirectX choice of $z\in[0,1]$ instead of OpenGL's $[-1,1]$ isn't
arbitrary — it's motivated by hardware and precision concerns tied
directly to [[depth-nonlinearity]]: depth buffers are naturally unsigned,
so a $[0,1]$ range maps onto unsigned buffer storage without wasting a
sign bit, and it can be arranged to give better precision distribution
near the camera than OpenGL's symmetric $[-1,1]$ range does.

> **Note**: NeRF uses OpenGL-style math throughout (see [[ndc-rays-transformation]]),
> so the $[-1,1]$ convention, not DirectX's $[0,1]$, is what appears
> elsewhere in this vault's NeRF notes.

**Putting the whole forward pipeline together.** With the convention fixed
(OpenGL-style, used throughout this vault), the full path from a 3D scene
to a rendered pixel is:

```
Camera space (3D)
     ↓ Projection matrix (P ×)
Clip space (4D, homogeneous)
     ↓ Clipping against ±w
Still clip space
     ↓ Perspective divide (÷ w)
NDC (3D, normalized to [-1,1])
     ↓ Viewport transform
Screen space (pixels)
```

Each arrow in this chain is one of the notes in this vault:
[[camera-space]] → [[perspective-projection-matrix]] → [[clip-space]] →
[[ndc]] → [[screen-space]].

## Formal Statement

| Feature | OpenGL | DirectX / Vulkan |
|---------|--------|--------------------|
| z range | $[-1,1]$ | $[0,1]$ |
| Near plane z | $+1$ | $0$ |
| Far plane z | $-1$ | $1$ |
| Handedness | Right-handed | Left-handed (configurable) |

## Worked Example

A point exactly at the near plane produces $z_{ndc}=+1$ under OpenGL's
convention, but the *same physical point* would produce $z_{ndc}=0$ under
DirectX's. Code (or a depth-comparison shader) written assuming one
convention will silently misbehave — near/far comparisons will be
inverted, or off by a scale factor — if fed data produced under the other.
This is why porting a renderer between OpenGL and DirectX/Vulkan requires
explicitly remapping the projection matrix's depth row, not just swapping
API calls.

## Connections

- [[camera-coordinate-system]], [[perspective-projection-matrix]], [[ndc]]
  — every one of these notes describes the OpenGL-side convention; this
  note is the explicit acknowledgment that DirectX/Vulkan made different,
  equally valid choices at each of those same points.
- [[depth-nonlinearity]] — the precision motivation behind DirectX's
  $[0,1]$ choice only makes sense in light of that note's $1/z$ mapping.
- [[ndc-rays-transformation]] — confirms which convention NeRF actually
  follows in practice (OpenGL-style).

## Open Questions

- Worth numerically verifying the claim that $[0,1]$ genuinely improves
  depth precision distribution vs. $[-1,1]$, rather than taking it as
  received wisdom.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

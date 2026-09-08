---
title: Reference Tables
aliases: [summary, glossary, cheat sheet]
tags: [reference]
status: growing
---

A quick-reference summary across every note in this vault — not itself a
concept with its own reasoning, just tables and takeaways to glance at
once the individual notes have been read. Every row links back to the note
that actually explains it.

## Full Pipeline Summary

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

See [[camera-space]] → [[perspective-projection-matrix]] → [[clip-space]]
→ [[ndc]] → [[screen-space]] for the full derivation of each arrow.

## Coordinate Systems at a Glance

| Space | Dimensions | Range | Origin | Purpose | Note |
|-------|------------|-------|--------|---------|------|
| World | 3D | Unbounded | Scene origin | Global scene representation | [[camera-to-world-transform]] |
| Camera | 3D | Unbounded | Camera center | Relative to camera | [[camera-space]] |
| Clip | 4D | Unbounded | — | Pre-division, enables clipping | [[clip-space]] |
| NDC | 3D | $[-1,1]$ | Screen center | Normalized for rendering | [[ndc]] |
| Screen | 2D | $[0,W]\times[0,H]$ | Top-left | Final pixels | [[screen-space]] |

## Symbol Glossary

| Symbol | Meaning | Note |
|--------|---------|------|
| $i, j$ | Pixel coordinates (horizontal, vertical) | [[pixel-to-ray-conversion]] |
| $f_x, f_y$ | Focal lengths in pixels | [[camera-intrinsics]] |
| $c_x, c_y$ | Principal point (image center) | [[camera-intrinsics]] |
| $K$ | Camera intrinsic matrix (3×3) | [[camera-intrinsics]] |
| c2w | Camera-to-world transformation (4×4) | [[camera-to-world-transform]] |
| $w$ | Homogeneous coordinate (4th component) | [[homogeneous-coordinates]] |
| $w_{clip}$ | $w$ after projection ($=-z_{camera}$) | [[perspective-projection-matrix]] |
| dirs | Ray directions in camera space | [[nerf-ray-generation]] |
| rays_o | Ray origins in world space | [[nerf-ray-generation]] |
| rays_d | Ray directions in world space | [[nerf-ray-generation]] |
| NDC | Normalized Device Coordinates | [[ndc]] |
| $n, f$ | Near and far plane distances | [[perspective-projection-matrix]] |
| $\theta$ | Vertical field of view | [[perspective-projection-matrix]] |
| $a$ | Aspect ratio (width / height) | [[perspective-projection-matrix]] |

## Pipeline Comparison

| Aspect | Graphics (Rasterization) | Vision (Ray Casting) | NeRF |
|--------|---------------------------|------------------------|------|
| Direction | 3D → 2D | 2D → 3D | 2D → 3D → 2D |
| Primitives | Triangles | Rays | Rays + Volume |
| Uses Clip Space | ✅ Yes | ❌ No | ❌ No |
| Uses NDC | ✅ Yes | ❌ Usually no | ✅ For forward-facing |
| Homogeneous Coords | ✅ Yes | ❌ No | ❌ No |

See [[ray-marching-vs-rasterization]] for the full reasoning behind this
table.

## Key Takeaways

1. **Pinhole camera** ([[pinhole-camera-model]]): perspective projection
   divides by $z$ to make distant objects appear smaller.
2. **Camera intrinsics** ([[camera-intrinsics]]): the $K$ matrix maps
   between 3D geometry and 2D pixels via focal length and principal point.
3. **Homogeneous coordinates** ([[homogeneous-coordinates]]): add $w$ to
   enable translation and perspective with matrix multiplication.
4. **Points vs. directions** ([[points-vs-directions]]): $w=1$ for points
   (affected by translation), $w=0$ for directions (not affected).
5. **Scaling equivalence** ([[scaling-equivalence]]):
   $(x,y,z,w) \sim (kx,ky,kz,kw)$ — same point, different representation.
6. **Clip space** ([[clip-space]]): GPU engineering space for safe
   clipping using $\pm w$.
7. **NDC** ([[ndc]]): normalized cube $[-1,1]$ after perspective divide —
   this is where perspective "happens."
8. **Depth nonlinearity** ([[depth-nonlinearity]]): $1/z$ mapping causes
   near objects to get more precision than far objects.
9. **Y negation** ([[why-y-is-negated]]): fixes the mismatch between image
   coordinates ($y$ down) and camera coordinates ($y$ up).
10. **NeRF** ([[nerf-ray-generation]], [[why-nerf-uses-ndc]]): uses
    graphics concepts (NDC) for a vision task (reconstruction), skipping
    clip space ([[why-nerf-skips-clip-space]]).

> [!QUOTE] Master Quotes
> - "Homogeneous coordinates work because geometry cares about ratios, not
>   scale — and perspective, infinity, clipping, and ray directions all
>   fall out naturally from that single idea." — [[projective-geometry]]
> - "The perspective projection matrix is the unique matrix that satisfies
>   the pinhole camera model, perspective division, and near/far depth
>   constraints simultaneously." — [[perspective-projection-matrix]]
> - "Clip space is GPU plumbing. NDC is geometry normalization. NeRF only
>   needs the second." — [[clip-space]], [[why-nerf-skips-clip-space]]

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

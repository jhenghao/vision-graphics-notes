---
title: Ray Marching vs Rasterization
aliases: [volume rendering]
tags: [nerf]
status: growing
---

## Motivating Question

Every note in [[camera-space]] through [[screen-space]] describes the
classic graphics pipeline built around projecting triangles. NeRF renders
images too, but has no triangles at all — it's a continuous volumetric
function. What does NeRF's rendering approach actually look like instead,
and which parts of the classic pipeline does it need or skip?

## Reasoning / Derivation

There are two fundamentally different philosophies for turning a 3D scene
representation into a 2D image:

**Rasterization (traditional graphics)** — the pipeline covered in
[[camera-space]] through [[screen-space]]:

```
triangles → project to screen → determine pixel coverage → shade pixels
```

This needs: [[clip-space]], [[homogeneous-coordinates]], triangle
clipping, and interpolation across triangle surfaces.

**Ray marching (NeRF)** — the opposite direction of traversal:

```
pixels → cast rays → sample points along ray → evaluate volume → composite color
```

This needs *none* of rasterization's machinery: no [[clip-space]], no
[[homogeneous-coordinates]] for the rendering step itself (they were used
earlier, to generate rays — see [[nerf-ray-generation]]), and no triangle
clipping, because there's nothing triangle-shaped to clip.

**Pipeline comparison, side by side:**

```
Rasterization:
camera space → clip space → NDC → screen → pixels

NeRF:
camera space → rays → (optional) NDC → ray marching → pixels
```

The key structural difference: rasterization pushes *geometry* (triangles)
through a pipeline toward the screen. Ray marching pulls *samples*
backward from each pixel into the scene, evaluating a continuous volumetric
field (NeRF's neural network) at points along the ray, then composites
those samples into a final pixel color. It's the vision-style "pixel → ray
→ scene" direction from [[pixel-to-ray-conversion]], extended all the way
into rendering itself, rather than the graphics-style "scene → pixel"
direction.

## Formal Statement

| Aspect | Rasterization | Ray Marching (NeRF) |
|--------|----------------|-----------------------|
| Direction | Geometry → screen | Pixel → scene |
| Needs clip space | ✅ Yes | ❌ No |
| Needs triangle clipping | ✅ Yes | ❌ No |
| Needs homogeneous coords for rendering | ✅ Yes | ❌ No (only for ray generation) |

## Worked Example

To render one pixel via rasterization: a triangle's three vertices are
projected through [[perspective-projection-matrix]] into clip space,
clipped, divided into NDC, and mapped to screen space; if that triangle
covers this pixel, its color (interpolated from the vertices) is written.

To render the same pixel via NeRF's ray marching: [[nerf-ray-generation]]
produces one ray for this pixel; points are sampled at increasing depths
along that ray; the network evaluates a color and density at each sampled
point; and those samples are composited (weighted by accumulated
transmittance) into a single final color for the pixel. No triangle, no
clip space, no interpolation across a surface — just repeated network
evaluations along a line.

## Connections

- [[camera-space]] through [[screen-space]] — the rasterization pipeline
  this note contrasts NeRF against, stage by stage.
- [[nerf-ray-generation]] — supplies the rays that ray marching then walks
  along; homogeneous coordinates and c2w transforms *are* used here, just
  earlier in the process than rasterization uses them.
- [[why-nerf-skips-clip-space]] — expands specifically on why the
  clip-space stage is unnecessary for this rendering approach.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

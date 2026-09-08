---
title: Camera Space
aliases: [view space]
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

Before a 3D scene can be turned into a 2D image, every object's position
needs to be expressed relative to the *camera*, not relative to some
arbitrary world origin. What exactly is that frame, and what does the
data look like once everything is expressed in it?

## Reasoning / Derivation

This is really just [[camera-coordinate-system]] under a different, more
pipeline-oriented name: **camera space** (also called **view space**) is
the coordinate frame whose origin is the camera center and whose axes are
aligned with how the camera is oriented, following the OpenGL/NeRF
convention already fixed there — $+X$ right, $+Y$ up, $+Z$ backward, camera
looking down $-Z$.

What makes it worth a separate note is its role as a *pipeline stage*: it's
the starting point for the forward graphics path. Before rendering, world
objects are transformed (rotated + translated) into this frame — the
inverse of the [[camera-to-world-transform]]. Once there, rays and
geometry are defined relative to the camera, ready for
[[perspective-projection-matrix]] to turn them into clip space.

## Formal Statement

Camera space (view space):
- **Origin**: camera center
- **Axes**: aligned with camera orientation (see [[camera-coordinate-system]])
- **Coordinates are 3D**
- **Camera looks down $-Z$**

## Worked Example

A point in camera space might look like:

```
(x_c, y_c, z_c) = (0.5, 0.3, -5.0)
                              ↑
                    5 units in front of camera (negative z)
```

This is the exact same worked example used in
[[camera-coordinate-system]] — camera space *is* that coordinate system;
here it's just the name used once the pipeline starts consuming it as
input to projection.

## Connections

- [[camera-coordinate-system]] — camera space is that coordinate frame; this
  note is the same thing viewed as a pipeline stage rather than an abstract
  convention.
- [[perspective-projection-matrix]] — takes camera-space coordinates as
  input and produces clip-space coordinates as output; this is the very
  next stage in the forward pipeline.
- [[camera-to-world-transform]] — the inverse direction: converts
  camera-space rays/points into world space, used on the vision
  (reconstruction) side rather than the graphics (rendering) side.

## Open Questions

- None currently — this note is mostly a naming/role clarification on top
  of [[camera-coordinate-system]].

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

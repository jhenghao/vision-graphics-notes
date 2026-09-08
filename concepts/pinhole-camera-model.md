---
title: Pinhole Camera Model
aliases: []
tags: [camera]
status: growing
---

## Motivating Question

A pinhole camera forms an image by letting light from a 3D scene pass
through a single point and land on a flat surface behind it. What's the
actual mathematical relationship between a 3D point out in the world and
the 2D point it produces on the image — and why does the textbook
derivation hand you an upside-down image that nobody actually wants to work
with?

## Reasoning / Derivation

**The geometry.** The physical setup has four pieces:

- **Pinhole (center of projection)**: at the origin $(0,0,0)$. Every light
  ray passes through this single point.
- **Object**: at world coordinates $(X,Y,Z)$, with $Z>0$ (in front of the
  camera, in this standard formulation).
- **Image plane**: a physical distance $f$ (the focal length) **behind**
  the pinhole, at $Z=-f$. The image forms here.
- **Optical axis**: the line through the pinhole, perpendicular to the
  image plane (the Z-axis).

Because light travels in straight lines through the single pinhole, the
image that lands on that plane is inverted — flipped both upside down and
left-right relative to the object.

**Deriving the projection.** The relationship between the object and its
image falls out of similar triangles: the triangle formed by the object and
the optical axis is similar to the triangle formed by the image point and
the optical axis, just scaled by the ratio of distances $f/Z$. Working that
out for a point $(X,Y,Z)$, its projection $(x,y)$ onto the physical plane
at $Z=-f$ is:

$$x = -f \frac{X}{Z} \qquad y = -f \frac{Y}{Z}$$

The minus sign is exactly the inversion described above — it isn't an
error, it's the physical consequence of light crossing over at the pinhole.

**Removing the inversion.** In vision and graphics we'd rather not carry
that flip through every downstream calculation. So we make a modeling
choice: place a **virtual image plane** at $Z=+f$, in front of the pinhole,
instead of the real one behind it. This doesn't change any physics — no
photon actually lands there — but it's mathematically equivalent (same
rays, same angles) and it removes the sign flip:

$$x = f \frac{X}{Z} \qquad y = f \frac{Y}{Z}$$

This upright, unflipped form is what you'll see written in most textbooks,
usually without any comment on why the minus sign disappeared.

**Expressing it as a matrix.** Both of the equations above are a
projection followed by a division — exactly the shape that homogeneous
coordinates are built to express as a single matrix multiply (see
[[homogeneous-coordinates]]). Convert $(X,Y,Z)$ to $(X,Y,Z,1)$ and multiply
by a camera matrix:

$$
\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} =
\begin{bmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

then recover the actual 2D coordinates by dividing by the third component
$w$ (which comes out equal to $Z$ here):

$$
x = \frac{x'}{w} = f\frac{X}{Z} \qquad y = \frac{y'}{w} = f\frac{Y}{Z}
$$

## Formal Statement

For a 3D point $(X,Y,Z)$ viewed through a pinhole with focal length $f$,
using the virtual (upright) image plane convention:

$$x = f \frac{X}{Z} \qquad y = f \frac{Y}{Z}$$

or, in matrix form with homogeneous coordinates and dehomogenization by $w$:

$$
\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} =
\begin{bmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix},
\qquad (x,y) = (x'/w,\ y'/w)
$$

> [!IMPORTANT]
> **Coordinate system conflict alert.** This derivation uses the
> **standard computer vision convention**, where the camera looks down
> $+Z$. Most graphics APIs (OpenGL, Blender) and NeRF implementations
> instead use the **OpenGL convention**, where the camera looks down $-Z$
> — see [[camera-coordinate-system]] for the full picture. To convert: flip
> the axis ($Z_{gl} = -Z_{cv}$), and the projection becomes
> $x = fX/(-Z_{gl})$. The physics doesn't change — only the sign, because
> of which way the camera is defined to look.

## Worked Example

Take a point at $(X,Y,Z) = (2, 1, 4)$ with focal length $f = 2$.

Using the virtual-plane equations:

$$x = 2 \cdot \frac{2}{4} = 1 \qquad y = 2 \cdot \frac{1}{4} = 0.5$$

In matrix form, the point $(2,1,4,1)$ maps to:

$$
\begin{bmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix} 2 \\ 1 \\ 4 \\ 1 \end{bmatrix} =
\begin{bmatrix} 2\cdot2 \\ 2\cdot1 \\ 4 \end{bmatrix} =
\begin{bmatrix} 4 \\ 2 \\ 4 \end{bmatrix}
$$

Dividing by $w=4$ gives $(x,y) = (1, 0.5)$ — matching the direct
calculation. Doubling the distance ($Z=8$, same $X,Y$) halves both $x$ and
$y$: this is the "objects farther away look smaller" effect falling
straight out of the division by $Z$.

## Connections

- [[camera-coordinate-system]] — this note derives projection assuming the
  camera looks down $+Z$; that note fixes the actual 3D axis convention
  (OpenGL's $-Z$-forward) used everywhere else in this vault.
- [[camera-intrinsics]] — generalizes this single scalar $f$ into the full
  $K$ matrix with separate $f_x, f_y$ (pixels, not physical distance) and a
  principal point offset, to go from the virtual image plane to actual
  pixel coordinates.
- [[homogeneous-coordinates]] — the matrix form above is only possible
  because homogeneous coordinates let a division-by-depth be expressed as
  "multiply then dehomogenize."
- [[perspective-projection-matrix]] — the graphics pipeline's projection
  matrix is the full 3D→clip-space generalization of the same idea
  introduced here in its simplest 2-row form.

## Open Questions

- Is the choice of $-Z$-forward in OpenGL purely historical, or does it
  interact with other conventions (e.g. right-handed coordinate systems,
  triangle winding order) in a way that made it the "natural" choice at the
  time?

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

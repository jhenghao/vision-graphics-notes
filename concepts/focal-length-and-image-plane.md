---
title: Focal Length and Image Plane
aliases: []
tags: [camera]
status: growing
---

## Motivating Question

Focal length shows up with different signs and different units depending
on which textbook or codebase you're reading — sometimes it's negative,
sometimes it's in millimeters, sometimes in pixels, and sometimes it's
implied that the image plane is confusingly "behind" the camera. Is $f$
positive or negative, physical or pixel-based, and what does "behind"
actually mean once you've fixed a coordinate convention?

## Reasoning / Derivation

**Sign, in physical optics vs. computer vision.** In strict optics, the
*sign* of the focal length tells you the type of optical system:

- **Positive focal length ($+f$)**: a **converging** system — parallel
  rays bend toward each other to a focus. Camera lenses, the human eye, a
  magnifying glass.
- **Negative focal length ($-f$)**: a **diverging** system — rays spread
  apart, appearing to come from a virtual point. Nearsighted glasses,
  peepholes, passenger-side mirrors.

| Sign of $f$ | Type | Effect on light | Example |
|:---|:---|:---|:---|
| Positive (+) | Converging | Brings rays together | Camera lens |
| Negative (−) | Diverging | Spreads rays apart | Myopia glasses |

Computer vision and graphics models almost exclusively deal with
converging systems (cameras), so *physically* $f$ is always a positive
magnitude. Any minus sign you see in the equations of
[[pinhole-camera-model]] is not telling you the lens is diverging — it's
coming from somewhere else: the coordinate system.

**Where the sign in the equations actually comes from.** Compare two
cases:

- **Case A — standard computer vision ($+Z$ forward)**: the virtual image
  plane sits at $Z=+f$ (positive coordinate), and the projection is
  $x = f\cdot(X/Z)$. Everything reads as positive — the "textbook"
  version.
- **Case B — OpenGL & NeRF ($-Z$ forward, used in this vault)**: the
  virtual image plane sits at $Z=-f$ (negative coordinate), and the
  projection becomes $x = f\cdot(X/{-Z})$. Here $f$ is still a *positive
  magnitude* — the plane's position is negative purely because the camera
  looks down the negative axis (see [[camera-coordinate-system]]).

> [!NOTE]
> **Common misconception, clarified.** "The image plane is behind the
> camera in OpenGL" is misleading. The virtual image plane at $z=-f$ is
> **in front of** the camera — in the direction the camera looks. It
> avoids the physical image's upside-down flip and keeps the equations
> simple; it does not mean the camera sees backward.

So the takeaway: in this vault, $f$ is always treated as a positive
distance. A negative sign attached to it (like $z=-f$) is purely a
statement about which way the coordinate axis points, never a statement
about lens type.

**Physical units vs. pixel units.** Focal length is the same physical
property described in two different unit systems:

1. **Physics/optics ($f_{mm}$)** — millimeters, the number printed on a
   lens (e.g. "50mm").
2. **Computer vision ($f_{pix}$)** — pixels, what the algorithms actually
   consume.

Why pixels? Look at the projection equation $x = f\cdot(X/Z)$: $X$ and $Z$
are both in physical units (e.g. meters), so they cancel in the ratio, but
$x$ needs to come out in *pixels* to be useful for image processing.
That forces $f$ itself to be expressed in pixels to balance the equation.

The conversion needs the sensor's pixel density $m_x$ (pixels per
millimeter):

$$f_{pix} = f_{mm} \times m_x = \frac{f_{mm}}{\text{sensor pixel size (mm)}}$$

And this is exactly why the intrinsic matrix (see [[camera-intrinsics]])
tracks $f_x$ and $f_y$ separately rather than a single $f$: ideally pixels
are square and $f_x = f_y$, but on sensors with non-square pixels the
mm→pixel conversion differs by axis, so $f_x \neq f_y$.

## Formal Statement

$$f_{pix} = \frac{f_{mm}}{\text{sensor pixel size in mm}} = f_{mm} \times m_x$$

where $m_x$ is the sensor's pixel density (pixels per mm) along that axis.
$f$ itself is always a positive physical distance; any negative sign
attached to a *coordinate* like $z=-f$ comes from the coordinate system's
forward-direction convention (see [[camera-coordinate-system]]), not from
the lens.

## Worked Example

- Lens focal length: $f = 4\text{mm}$.
- Sensor pixel width: $0.002\text{mm}$ ($2\mu\text{m}$).

$$f_{pix} = \frac{4}{0.002} = 2000 \text{ pixels}$$

So a $K$ matrix (see [[camera-intrinsics]]) for this camera would use
$f_x = f_y = 2000$ (assuming square pixels) — not $4$. This is a common
source of confusion when someone reads a lens spec sheet and tries to plug
"4" directly into vision code expecting pixel-space math to work.

## Connections

- [[pinhole-camera-model]] — this note resolves the "is $f$ positive or
  negative" ambiguity left implicit in that note's derivation.
- [[camera-coordinate-system]] — the source of the sign flip discussed
  above; $f$ itself never changes sign, only the coordinate it's attached
  to.
- [[camera-intrinsics]] — consumes $f_{pix}$ (as $f_x, f_y$) directly; this
  note supplies the mm→pixel conversion that produces those values from a
  physical lens spec.

## Open Questions

- For a real camera with lens distortion, is the mm→pixel conversion still
  a clean linear scale by $m_x$, or does it only hold in the idealized
  pinhole model before distortion correction is applied?

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

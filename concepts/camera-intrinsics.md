---
title: Camera Intrinsics (K Matrix)
aliases: [intrinsic matrix, K matrix]
tags: [camera]
status: growing
---

## Motivating Question

[[pinhole-camera-model]] gives $x = fX/Z$ using a single scalar focal
length and coordinates centered on the optical axis. Real images don't
work that way: pixels are indexed from a corner, not the center, and a
sensor's pixels aren't always perfectly square. How do you package
"focal length" plus "where pixel (0,0) actually is" into something a
vision or graphics pipeline can just multiply by?

## Reasoning / Derivation

Start from the virtual-plane projection in [[pinhole-camera-model]]:
$x = fX/Z$, $y = fY/Z$. Two real-world details are missing from this:

1. **Non-square or asymmetric pixel scaling.** As discussed in
   [[focal-length-and-image-plane]], converting $f$ from physical units to
   pixels can differ per axis, giving $f_x$ and $f_y$ instead of one $f$.
2. **Where the origin is.** The projection equations assume the optical
   axis passes through image coordinate $(0,0)$. But pixel coordinates are
   conventionally indexed from a corner of the image, not its center. So
   after scaling by $f_x, f_y$, you need to add an offset — the
   **principal point** $(c_x, c_y)$ — to land on the actual pixel grid.

Packaging "scale by $f_x, f_y$" and "shift by $c_x, c_y$" together as a
single linear map (which also needs to leave the homogeneous $w$ coordinate
alone) is exactly what a 3×3 matrix multiplying homogeneous 2D coordinates
gives you — see [[homogeneous-coordinates]] for why encoding an affine
shift as a matrix multiply requires that extra coordinate at all.

## Formal Statement

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

| Parameter | Meaning | Typical value |
|-----------|---------|----------------|
| $f_x$ | Focal length in pixels (x-direction) | ~500–2000 for most cameras |
| $f_y$ | Focal length in pixels (y-direction) | usually ≈ $f_x$ |
| $c_x$ | Principal point x-coordinate | ≈ width / 2 |
| $c_y$ | Principal point y-coordinate | ≈ height / 2 |

- $f_x, f_y$ convert physical angles to pixel distances — a larger focal
  length means more "zoom" (narrower field of view).
- $c_x, c_y$ mark the principal point, where the optical axis intersects
  the image plane. For an ideal camera this is the image center; real
  cameras may have a slight offset.

Accessing $K$ in code:

```python
f_x = K[0, 0]  # Focal length (x)
f_y = K[1, 1]  # Focal length (y)
c_x = K[0, 2]  # Principal point (x)
c_y = K[1, 2]  # Principal point (y)
```

## Worked Example

Take a $1000\times800$ pixel image from a camera with $f_x = f_y = 900$
pixels, and an ideally centered principal point:

$$c_x = 1000/2 = 500 \qquad c_y = 800/2 = 400$$

$$
K = \begin{bmatrix} 900 & 0 & 500 \\ 0 & 900 & 400 \\ 0 & 0 & 1 \end{bmatrix}
$$

A 3D point at camera-space $(X,Y,Z) = (1, 0.5, -10)$ (OpenGL convention,
in front of the camera — see [[camera-coordinate-system]]) projects to:

$$x_{pix} = f_x \cdot \frac{X}{-Z} + c_x = 900 \cdot \frac{1}{10} + 500 = 590$$
$$y_{pix} = f_y \cdot \frac{Y}{-Z} + c_y = 900 \cdot \frac{0.5}{10} + 400 = 445$$

So this point lands at pixel $(590, 445)$ — right and below the image
center, matching that it's slightly right ($X>0$) and slightly above
camera-forward but plotted with $y$ increasing downward in image space
(see [[why-y-is-negated]] for why that vertical flip matters when going
the other direction, from pixels back to rays).

## Connections

- [[pinhole-camera-model]] — $K$ is this note's single-focal-length
  projection, generalized with separate $f_x,f_y$ and a principal-point
  offset to land on actual pixel coordinates.
- [[focal-length-and-image-plane]] — supplies the mm→pixel conversion that
  produces the $f_x, f_y$ values $K$ stores.
- [[pixel-to-ray-conversion]] — is exactly the *inverse* operation: given a
  pixel and $K$, recover the normalized camera-space direction that
  produced it.

## Open Questions

- This note doesn't cover lens distortion coefficients, which real
  calibration (e.g. OpenCV's `calibrateCamera`) estimates alongside $K$.
  Worth a follow-up note once distortion comes up in a source.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

---
title: Why Y is Negated
aliases: []
tags: [vision]
status: growing
---

## Motivating Question

[[pixel-to-ray-conversion]] derives $y_n = (j-c_y)/f_y$ with no sign flip.
But real ray-generation code (e.g. NeRF's `get_rays`) negates that term:
`-(j - cy) / fy`. Without it, images come out vertically flipped. Why does
the formula need that extra minus sign?

## Reasoning / Derivation

The mismatch is between two coordinate systems that both use "y" but mean
opposite things by it:

**Image space (pixels):**
- Origin: top-left
- $x$ increases → right
- $y$ increases → **down**

**Camera / 3D space** (per [[camera-coordinate-system]]):
- Origin: camera center
- $x$ increases → right
- $y$ increases → **up**

The $x$-axis agrees between the two systems, but the $y$-axis is flipped.
[[pixel-to-ray-conversion]]'s formula $y_n=(j-c_y)/f_y$ treats a pixel
below the image center (larger $j$, since image $y$ grows downward) as
producing a *positive* $y_n$ — but in camera space, positive $y$ means
*up*. Left uncorrected, a ray meant to point toward the bottom of the
scene would instead point up, and the rendered or reconstructed image
comes out upside down.

The fix is a single negation:

```python
dirs = torch.stack([
    (i - cx) / fx,      # x: no change needed
    -(j - cy) / fy,     # y: NEGATED
    -torch.ones_like(i) # z: forward direction
], -1)
```

**What the negation does, concretely:** for a pixel *above* image center
($j < c_y$, so $(j-c_y)<0$), negating flips this to $y_n>0$ — pointing
*upward* in camera space, which is correct, since a pixel near the top of
an image should correspond to a ray pointing up and away from
camera-forward.

```
Pixel above center:   j < cy  →  (j - cy) < 0
After negation:       y > 0   →  ray points upward ✓
```

This same top-left-origin / y-down vs. y-up mismatch is also why
[[screen-space]]'s forward mapping uses $(1-y_{ndc})$ rather than
$(y_{ndc}+1)$ — it's the identical fix applied in the opposite direction of
the pipeline.

**OpenCV vs. OpenGL, side by side:**

| Feature | OpenCV / NeRF | OpenGL |
|---------|----------------|--------|
| Forward direction | $-Z$ | $-Z$ |
| Image origin | Top-left | Bottom-left |
| Typical use | Vision, reconstruction | Rendering |
| Uses NDC? | Usually no | Yes |

Interesting to note: OpenCV/NeRF and OpenGL actually agree on forward
direction ($-Z$) — the y-negation issue here is purely about image-origin
convention (top-left vs. bottom-left), not about the camera-space axis
convention from [[camera-coordinate-system]].

## Formal Statement

$$y_n = -\frac{j - c_y}{f_y}$$

(compare [[pixel-to-ray-conversion]]'s un-negated $y_n=(j-c_y)/f_y$, which
is only correct if the image coordinate's $y$-axis already matches camera
space's $y$-axis — it usually doesn't.)

## Worked Example

Take a pixel at $j=100$ with $c_y=400$, $f_y=900$ (above the image center,
since $j<c_y$):

Without negation: $y_n = (100-400)/900 = -0.333$ — a *negative* $y_n$,
which in camera space means "pointing down." Wrong — this pixel is above
center and should produce a ray pointing up.

With negation: $y_n = -(100-400)/900 = +0.333$ — positive, correctly
pointing up. This is the concrete failure mode "no negation → image
vertically flipped" made numeric.

## Connections

- [[pixel-to-ray-conversion]] — supplies the un-negated formula; this note
  is the correction needed before it matches actual working code.
- [[camera-coordinate-system]] — defines the $y$-up convention that
  image-space $y$-down conflicts with.
- [[screen-space]] — applies the same fix in the opposite (forward,
  graphics) direction of the pipeline.
- [[nerf-ray-generation]] — the concrete code (`get_rays`) where this
  negation actually appears, alongside the rest of ray generation.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

---
title: Screen Space
aliases: [viewport transform]
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

[[ndc]] leaves everything normalized into $[-1,1]$, a resolution-independent
cube. A display, though, has actual pixel dimensions — width $W$ and
height $H$ — and pixel coordinates conventionally start at the top-left
with $y$ increasing *downward*, the opposite of NDC's $y$-up convention.
What's the final mapping that bridges the two?

## Reasoning / Derivation

The last step of the forward pipeline maps NDC coordinates to actual pixel
positions:

$$x_{screen} = \frac{(x_{ndc}+1)\cdot W}{2} \qquad y_{screen} = \frac{(1-y_{ndc})\cdot H}{2}$$

Two things are happening in these formulas simultaneously:

1. **Rescaling** $[-1,1] \rightarrow [0,W]$ (and $[0,H]$) — a standard
   linear remap: add 1 to shift the range to $[0,2]$, then scale by
   $W/2$ (or $H/2$).
2. **Flipping $y$** — note the numerator is $(1-y_{ndc})$, not
   $(y_{ndc}+1)$. NDC has $y=+1$ at the top of the screen (see
   [[ndc]]'s axis table), but image/pixel coordinates increase downward.
   Using $(1-y_{ndc})$ instead of $(y_{ndc}+1)$ inverts that relationship
   so that $y_{ndc}=+1$ (top, in NDC) maps to $y_{screen}=0$ (top, in pixel
   space) rather than to the bottom.

The output of this step is the final raster position — the actual pixel a
GPU rasterizer will shade.

## Formal Statement

$$x_{screen} = \frac{(x_{ndc}+1)\cdot W}{2} \qquad y_{screen} = \frac{(1-y_{ndc})\cdot H}{2}$$

## Worked Example

For a $1920\times1080$ image, take the NDC point $(x_{ndc}, y_{ndc}) = (0.6, 0.8)$ (near the top-right of the screen):

$$x_{screen} = \frac{(0.6+1)\cdot 1920}{2} = \frac{1.6 \cdot 1920}{2} = 1536$$
$$y_{screen} = \frac{(1-0.8)\cdot 1080}{2} = \frac{0.2 \cdot 1080}{2} = 108$$

Pixel $(1536, 108)$ — far to the right ($x$ close to $W=1920$) and near the
top ($y$ small, since $y_{ndc}=0.8$ was close to NDC's top value of $+1$,
and the flip maps that to a small pixel $y$).

## Connections

- [[ndc]] — this note's input; screen space is reached by rescaling and
  y-flipping the NDC cube.
- [[why-y-is-negated]] — the vision-side counterpart of this note's y-flip:
  going from pixels back to camera-space rays requires undoing the exact
  same top-left-origin, y-down convention handled here for the forward
  direction.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

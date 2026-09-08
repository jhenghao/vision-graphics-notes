---
title: Depth Nonlinearity (1/z)
aliases: [z-fighting]
tags: [graphics-pipeline]
status: growing
---

## Motivating Question

Depth buffers in real-time graphics are notorious for **z-fighting** —
flickering, indeterminate ordering between two surfaces that are far from
the camera, even though the same surfaces render fine up close. Why does
depth precision depend on distance from the camera at all?

## Reasoning / Derivation

Row 3 of [[perspective-projection-matrix]] maps camera-space depth $z_c$
to clip-space $z$ using coefficients $A = -(f+n)/(f-n)$ and
$B = -2fn/(f-n)$, giving (after the perspective divide described in
[[ndc]]):

$$z_{ndc} = A + \frac{B}{z_c}$$

The key feature here is the $1/z_c$ term. Because it's a reciprocal rather
than a linear function of $z_c$, equal steps in NDC depth correspond to
very unequal steps in camera-space depth depending on where you are:

- For **small $|z_c|$** (near the camera), $1/z_c$ changes rapidly — a
  tiny change in camera depth produces a large change in $z_{ndc}$. Depth
  precision is high here.
- For **large $|z_c|$** (far from the camera), $1/z_c$ changes slowly — a
  large range of camera depths gets compressed into a small range of
  $z_{ndc}$. Depth precision is low here.

$$z_{ndc} \propto \frac{1}{z_c}$$

Since depth buffers store $z_{ndc}$ with finite precision (a fixed number
of bits), and many different far-away camera depths map to nearly
identical $z_{ndc}$ values, the buffer can't reliably tell which of two
distant surfaces is actually closer. That ambiguity is exactly what
produces **z-fighting** — visibly flickering surfaces at a distance.

## Formal Statement

$$z_{ndc} = A + \frac{B}{z_c}, \qquad A = -\frac{f+n}{f-n},\ B = -\frac{2fn}{f-n}$$

| Camera depth | NDC depth | Precision |
|--------------|-----------|-----------|
| Near | Spread out | High |
| Far | Compressed | Low |

## Worked Example

Compare the NDC depth change for two equal-sized camera-space steps, one
near and one far. Near the camera, moving from $z_c=-1$ to $z_c=-2$ changes
$1/z_c$ from $-1$ to $-0.5$ — a swing of $0.5$. Far from the camera, moving
from $z_c=-100$ to $z_c=-101$ changes $1/z_c$ from $-0.01$ to
$\approx-0.0099$ — a swing of about $0.0001$, roughly 5000× smaller for the
same 1-unit step in camera space. That's the concrete mechanism behind "far
objects get compressed": the *same size step* in world depth produces a
wildly different-sized step in stored NDC/buffer depth depending on
distance.

## Connections

- [[perspective-projection-matrix]] — row 3 of that matrix is exactly
  where the $A, B$ coefficients producing this nonlinearity come from.
- [[ndc]] — this note describes the specific behavior of $z_{ndc}$ under
  the same perspective divide that note introduces for all three axes.
- [[opengl-vs-directx]] — DirectX's choice of $z\in[0,1]$ instead of
  OpenGL's $[-1,1]$ is partly motivated by getting better depth precision
  where it's needed, given this same nonlinearity.

## Open Questions

- The source material states DirectX's $[0,1]$ range has better precision
  characteristics near the camera than OpenGL's $[-1,1]$ — worth deriving
  numerically (not just asserting) if this becomes relevant to actual
  rendering work.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

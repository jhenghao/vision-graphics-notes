---
title: Scaling Equivalence
aliases: [dehomogenization]
tags: [homogeneous-coordinates]
status: growing
---

## Motivating Question

If $(x,y,z,1)$ represents the 3D point $(x,y,z)$, what does $(2x,2y,2z,2)$
represent? It has the same ratios, but is it the "same" homogeneous
coordinate, a different one, or something that needs a rule to interpret?

## Reasoning / Derivation

Since a point's homogeneous form is defined as "3D coordinates plus $w=1$,"
it's tempting to think $w$ always has to literally equal $1$. But nothing
in how a homogeneous coordinate gets *used* — as a 4-vector fed through
matrix multiplies — requires that. The only place $w$ ever gets resolved
back into an actual 3D location is at the final step, where you divide
through by it:

$$(x,y,z,w) \rightarrow \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)$$

This step is called **dehomogenization**. And because it's a division, any
uniform scaling of all four components cancels out identically:

$$(x,y,z,w) \quad\text{and}\quad (kx,ky,kz,kw) \quad\text{for any } k\neq0$$

both dehomogenize to the exact same 3D point. So $(2,4,6,1)$, $(4,8,12,2)$,
and $(1,2,3,0.5)$ aren't three different points that happen to be nearby —
they're three different *representations* of the identical point
$(2,4,6)$.

**Geometric intuition.** Picture homogeneous coordinates as rays from the
origin through 4D space. Every point along one such ray —
$(x,y,z,w), (2x,2y,2z,2w), (kx,ky,kz,kw), \ldots$ — represents the same 3D
point once dehomogenized. Only the *direction* of the 4D vector carries
information; its length is discardable. This is also precisely why
[[perspective-projection-matrix]] can afford to produce clip-space
coordinates with an arbitrary nonzero $w$ and defer the actual division —
the scale doesn't matter until the very last step.

## Formal Statement

$$(x,y,z,w) \sim (kx,ky,kz,kw) \quad \text{for any } k \neq 0$$

represent the same 3D point, recovered by dehomogenizing (dividing through
by $w$):

$$(x,y,z,w) \rightarrow \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)$$

## Worked Example

Take the 3D point $(2,4,6)$:

| Homogeneous | Dehomogenize | Result |
|-------------|--------------|--------|
| $(2,4,6,1)$ | $(2/1, 4/1, 6/1)$ | $(2,4,6)$ ✅ |
| $(4,8,12,2)$ | $(4/2, 8/2, 12/2)$ | $(2,4,6)$ ✅ |
| $(1,2,3,0.5)$ | $(1/0.5, 2/0.5, 3/0.5)$ | $(2,4,6)$ ✅ |

All three rows are the same 3D point, scaled by $k=1$, $k=2$, and
$k=0.5$ respectively.

## Connections

- [[homogeneous-coordinates]] — this note answers the natural follow-up
  question that note leaves open: what does it mean for $w \ne 1$?
- [[ndc]] — the perspective divide that produces Normalized Device
  Coordinates is literally this note's dehomogenization step, applied to
  clip-space coordinates: "pick the representative of this equivalence
  class where $w=1$."
- [[projective-geometry]] — pushes this idea to its logical extreme: what
  happens to the equivalence class when $w=0$, so dehomogenizing would mean
  dividing by zero?

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

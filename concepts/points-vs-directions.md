---
title: Points vs Directions (w=1 vs w=0)
aliases: []
tags: [homogeneous-coordinates]
status: growing
---

## Motivating Question

A ray has both an origin (a location) and a direction (which way it
points). If you apply the *same* transform matrix to both, you'd expect
different behavior: moving the camera should move where a ray starts, but
it shouldn't change which way the ray is pointing. Ordinary 3D vectors
$(x,y,z)$ can't express that distinction — so how do homogeneous
coordinates (see [[homogeneous-coordinates]]) tell "location" and "pure
direction" apart?

## Reasoning / Derivation

The 4th coordinate $w$, introduced in [[homogeneous-coordinates]] to make
translation a matrix multiply, turns out to also be exactly the right
place to encode this distinction. Recall the translation matrix acts as:

$$
\begin{bmatrix} 1&0&0&t_x \\ 0&1&0&t_y \\ 0&0&1&t_z \\ 0&0&0&1 \end{bmatrix}
\begin{bmatrix} x\\y\\z\\w \end{bmatrix}
$$

The translation constants $t_x,t_y,t_z$ get multiplied by $w$ before being
added in. If $w=1$, they pass through unchanged — the point gets
translated. If $w=0$, they vanish entirely — the vector is *unaffected* by
translation. That single algebraic fact gives you two categories for free:

| Type | Homogeneous form | Affected by translation? |
|------|-------------------|---------------------------|
| Point | $(x,y,z,1)$ | ✅ Yes |
| Direction | $(x,y,z,0)$ | ❌ No |

This is precisely the behavior a ray needs: the **origin** is a location
(should move when the camera moves) → $w=1$. The **direction** is "which
way," not "where" → $w=0$, so translating the camera rotates the direction
along with everything else but never shifts it.

## Formal Statement

$$(x,y,z,1) \;\text{— a point, translated by matrix multiplication}$$
$$(x,y,z,0) \;\text{— a direction, unaffected by translation}$$

## Worked Example

```python
# In NeRF conceptually:
ray_origin = (x, y, z, 1)        # A point in space
ray_direction = (dx, dy, dz, 0)  # Just a direction, no location
```

Apply the same translation matrix from [[homogeneous-coordinates]]'s
worked example, $(t_x,t_y,t_z)=(5,-1,0)$, to a direction $(1,0,0,0)$:

$$
\begin{bmatrix} 1&0&0&5 \\ 0&1&0&-1 \\ 0&0&1&0 \\ 0&0&0&1 \end{bmatrix}
\begin{bmatrix} 1\\0\\0\\0 \end{bmatrix} =
\begin{bmatrix} 1+5\cdot0\\0-1\cdot0\\0+0\cdot0\\0 \end{bmatrix} =
\begin{bmatrix} 1\\0\\0\\0 \end{bmatrix}
$$

Unchanged — exactly as it should be, since "pointing in $+x$" doesn't
depend on where you're standing.

## Connections

- [[homogeneous-coordinates]] — supplies the 4th coordinate this note
  repurposes; this note is the specific case of $w=0$ that
  [[homogeneous-coordinates]] doesn't itself dwell on.
- [[camera-to-world-transform]] — applies exactly this distinction in
  practice: rotating ray *directions* by the c2w rotation block while
  translating ray *origins* by the camera's world position.
- [[projective-geometry]] — reinterprets $w=0$ geometrically, as a "point
  at infinity" rather than just "an object immune to translation" — the
  same algebra, a different and richer interpretation.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

---
title: Ray Generation Code Explained
aliases: [get_rays]
tags: [nerf]
status: growing
---

## Motivating Question

[[pixel-to-ray-conversion]], [[why-y-is-negated]], and
[[camera-to-world-transform]] each describe one piece of turning a pixel
into a world-space ray. What does it look like when all three are actually
chained together, for every pixel of an entire image at once, in real
NeRF code?

## Reasoning / Derivation

`get_rays` is exactly the composition of the three preceding notes,
vectorized over a full $H\times W$ image using PyTorch:

```python
def get_rays(H, W, K, c2w):
    # Step 1: Create a pixel grid
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H)
    )
    i = i.t()  # Transpose for correct indexing
    j = j.t()

    # Step 2: Convert pixels → camera-space directions
    dirs = torch.stack([
        (i - K[0][2]) / K[0][0],    # (i - cx) / fx
        -(j - K[1][2]) / K[1][1],   # -(j - cy) / fy  [negated!]
        -torch.ones_like(i)          # z = -1 (forward)
    ], -1)

    # Step 3: Rotate directions into world space
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)

    # Step 4: Set ray origins (camera position in world)
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    return rays_o, rays_d
```

**Step-by-step:**

1. **Create pixel grid** — `meshgrid` generates an $(i,j)$ coordinate for
   every pixel in the image at once.
2. **Normalized coords** — subtract the principal point $(c_x,c_y)$ and
   divide by focal length, exactly as in [[pixel-to-ray-conversion]].
3. **Y negation** — fixes the image-vs-camera coordinate mismatch, exactly
   as derived in [[why-y-is-negated]].
4. **$z=-1$** — fixes each ray's camera-space direction to point forward
   (camera looks down $-Z$, per [[camera-coordinate-system]]).
5. **Rotate to world** — apply the c2w rotation block, per
   [[camera-to-world-transform]].
6. **Ray origins** — the camera's world position, the same for every
   pixel, per [[camera-to-world-transform]].

**Output:**

- `rays_o`: shape $(H,W,3)$ — ray origins in world space (identical across
  every pixel, since all rays share one camera center).
- `rays_d`: shape $(H,W,3)$ — ray directions in world space (differs per
  pixel).

**PyTorch vs. NumPy.** `get_rays_np` is functionally identical, except it
uses `np.meshgrid(..., indexing='xy')`, which sidesteps the transpose
PyTorch's default `meshgrid` indexing requires.

## Formal Statement

For every pixel $(i,j)$:

$$\text{dir}_{cam} = \left(\frac{i-c_x}{f_x},\ -\frac{j-c_y}{f_y},\ -1\right)$$
$$\text{rays\_d} = R \cdot \text{dir}_{cam}, \qquad \text{rays\_o} = t$$

where $R,t$ come from the c2w matrix, applied identically per
[[camera-to-world-transform]].

## Worked Example

Trace one pixel through the whole function: say $H=W=800$, $K$ has
$f_x=f_y=1111$, $c_x=c_y=400$ (image center), and pixel $(i,j)=(400,400)$
— dead center.

- Step 2: $\text{dirs} = ((400-400)/1111,\ -(400-400)/1111,\ -1) = (0,0,-1)$
  — pointing straight down the camera's forward axis, as expected for the
  exact center pixel.
- Step 3–4: if `c2w[:3,:3]` is the identity (camera facing world $-Z$
  directly, no rotation), `rays_d = (0,0,-1)` unchanged.
- If the camera sits at world position $(0,0,5)$, `rays_o = (0,0,5)` for
  every pixel, including this one.

So the center pixel of the image corresponds to a world-space ray starting
at $(0,0,5)$ and heading straight in the $-z$ direction — exactly what
you'd expect for a camera looking directly down the world $z$-axis.

## Connections

- [[pixel-to-ray-conversion]], [[why-y-is-negated]],
  [[camera-to-world-transform]] — the three ideas this function composes,
  each corresponding to one numbered step above.
- [[ndc-rays-transformation]] — the function that (optionally) takes
  `rays_o, rays_d` from `get_rays` and reprojects them into NDC space for
  forward-facing scenes.

## Open Questions

- None currently.

## Sources

- [[sources/ndc-discussion-raw-2026-01-11]]

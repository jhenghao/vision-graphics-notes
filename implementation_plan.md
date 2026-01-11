# Reorganize NDC Discussion: Unified Graphics & Vision Notes

## Background

The user has a 72-page PDF (`ndc_discussion.pdf`) containing a ChatGPT discussion about graphics and computer vision concepts. The goal is to reorganize this into a logical, easy-to-understand markdown document that **unifies graphics and vision perspectives** while preserving ALL knowledge and derivations.

## Key Insight: Graphics & Vision Are Two Sides of the Same Coin

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHARED FOUNDATION                            │
│  Pinhole Camera Model • Projection Geometry • Coordinates       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                                       ▼
┌───────────────────┐                 ┌───────────────────┐
│     GRAPHICS      │                 │      VISION       │
│   (Forward Path)  │                 │  (Inverse Path)   │
│                   │                 │                   │
│ 3D Scene → Image  │                 │ Image → 3D Scene  │
│ "Rendering"       │                 │ "Reconstruction"  │
└───────────────────┘                 └───────────────────┘
        │                                       │
        │              ┌───────┐                │
        └──────────────│ NeRF  │────────────────┘
                       │       │
                       └───────┘
                 Uses graphics concepts (NDC)
                 for vision task (reconstruction)
```

## Proposed Reorganized Topic Order

---

### Part 1: The Pinhole Camera (Shared Foundation)

> Both graphics and vision start from the same physical model: how light passes through a point and creates an image.

1. **Pinhole Camera Model** - The fundamental geometry
   - Similar triangles: why projection divides by z
   - x_proj = x / (-z), y_proj = y / (-z)
   - ASCII diagram showing similar triangles derivation
   - Why dividing by -z_c makes projected x,y positive in correct direction
2. **Camera Coordinate System** - The 3D reference frame
   - Origin at camera center
   - x→right, y→up, z→backward (OpenGL/NeRF convention)
3. **Focal Length and Image Plane** - Where the image forms
   - f > 0 always (physical distance)
   - Image plane at z = -f (in front of camera)
   - Common misconception: "image plane behind camera" clarified
4. **Camera Intrinsics (K Matrix)** - Mapping geometry to pixels
   - fx, fy: focal length in pixels
   - cx, cy: principal point (image center)
   - The full K matrix structure

---

### Part 2: Homogeneous Coordinates (The Mathematical Trick)

> The math that makes both graphics pipelines and vision algorithms elegant.

5. **Why We Need Homogeneous Coordinates** - The problem they solve
   - Can't represent translation or perspective with 3×3 matrices
   - Solution: add a 4th coordinate (w)
   - Translation matrix example in 4×4 form
6. **Points vs Directions** - w=1 vs w=0
   - Points (w=1): affected by translation
   - Directions (w=0): not affected by translation, represent infinity
   - Why ray origins use w=1, ray directions use w=0
   - Directions are unaffected by translation (key insight)
7. **Scaling Equivalence** - Same point, different representations
   - (x, y, z, w) ≡ (kx, ky, kz, kw) for any k≠0
   - Dehomogenization: divide by w to get 3D point
   - **Concrete numeric example**: (2,4,6,1) ~ (4,8,12,2) ~ (1,2,3,0.5) all = (2,4,6)
   - Geometric intuition: rays from origin in 4D, equivalence classes
8. **Projective Geometry Connection** - Why this works
   - Parallel lines meet at infinity (vanishing points)
   - Points at infinity have w=0
   - Vanishing points explained (railroad tracks example)
   - **Master takeaway**: "Geometry cares about ratios, not scale"

---

### Part 3: The Graphics Pipeline (Forward Path: 3D → 2D)

> Graphics asks: "Given a 3D scene, what 2D image do we see?"

9. **Camera Space** - Starting point for rendering
   - 3D coordinates relative to camera
   - Camera looks down −Z
10. **The Perspective Projection Matrix** - Full derivation
    - **5 Required Properties**:
      1. Perspective: x_ndc = x/(-z)
      2. Correct field of view (FOV)
      3. Correct aspect ratio
      4. Near/far planes map to fixed NDC z values
      5. Points in front of camera have positive w
    - Row 1: Horizontal FOV + aspect ratio (1/(a·tan(θ/2)))
    - Row 2: Vertical FOV (1/tan(θ/2))
    - Row 3: Depth mapping with A = -(f+n)/(f-n) and B = -2fn/(f-n)
    - Row 4: [0, 0, -1, 0] → creates w = -z
    - Why w = -z: ensures visible points have w > 0, behind camera w < 0
    - "This matrix is the unique solution satisfying pinhole model + depth constraints"
11. **Clip Space** - Homogeneous coordinates before division
    - Result of projection matrix × camera point
    - Why clip against ±w (safe clipping before division)
    - Prevents: division by zero, objects behind camera, geometry outside frustum
    - **Key insight**: "Clip space is GPU plumbing, NDC is geometry normalization"
12. **NDC (Normalized Device Coordinates)** - After perspective divide
    - x_ndc = x_clip / w_clip
    - Everything in [-1, 1] cube
    - This is where perspective "happens"
    - **Perspective divide = choosing the representative where w=1**
    - NDC axis interpretation table (x=-1 left, x=+1 right, etc.)
13. **Depth Nonlinearity (1/z)** - Why far objects are compressed
    - Near objects get more depth precision
    - Causes z-fighting at far distances
    - z_ndc ∝ 1/z_c mapping
14. **Screen Space** - Final pixel coordinates
    - Map [-1, 1] → [0, width] and [0, height]
    - x_screen = (x_ndc + 1) * width / 2
    - y_screen = (1 - y_ndc) * height / 2 (note y flip)
15. **OpenGL vs DirectX Differences** - Convention variations
    - OpenGL: z ∈ [-1, 1], near = +1, far = -1
    - DirectX: z ∈ [0, 1], near = 0, far = 1
    - DirectX has better depth buffer precision near camera
    - NeRF uses OpenGL-style math

---

### Part 4: Computer Vision (Inverse Path: 2D → 3D)

> Vision asks: "Given a 2D image, what 3D scene produced it?"

16. **Pixel to Ray Conversion** - The inverse of projection
    - From pixel (i, j) to normalized camera coordinates
    - x_n = (i - cx) / fx, y_n = (j - cy) / fy
    - Distinction: normalized camera coords (2D) vs camera space (3D)
17. **Why Y is Negated** - Coordinate system mismatch
    - Image: y increases downward (origin top-left)
    - Camera/3D: y increases upward
    - -(j - cy) / fy fixes this
    - Without negation: image would be vertically flipped
    - **OpenCV vs OpenGL convention comparison table**:
      | Feature | OpenCV/NeRF | OpenGL |
      |---|---|---|
      | Forward direction | −Z | −Z |
      | Image origin | Top-left | Bottom-left |
      | Typical use | Vision, reconstruction | Rendering |
      | Uses NDC? | Usually no | Yes |
18. **Camera to World Transformation** - c2w matrix
    - Rotate ray directions to world frame: c2w[:3,:3]
    - Translate ray origin to camera position: c2w[:3,-1]

---

### Part 5: NeRF — Where Graphics Meets Vision

> NeRF is a reconstruction (vision) algorithm that uses graphics concepts (NDC) for numerical stability.

19. **Ray Generation Code Explained** - `get_rays` and `get_rays_np`
    - Pixel grid creation via meshgrid
    - Pixel → normalized coords → camera-space direction → world-space ray
    - Full code walkthrough with annotations
    - PyTorch vs NumPy differences (transpose, indexing)
20. **NDC Rays Transformation** - The `ndc_rays` function
    - **Step 1**: Near plane intersection
      - `t = -(near + rays_o[...,2]) / rays_d[...,2]`
      - Shifts ray origins to near plane
    - **Step 2**: Perspective projection of origins
      - W/(2*focal) and H/(2*focal) scaling terms explained
      - o0, o1, o2 formulas with division by z
    - **Step 3**: Ray slope conversion for directions
      - d0, d1, d2: slopes instead of absolute directions
      - `rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2]`
    - Near plane → z=1, Far plane → z=-1
    - NeRF uses OpenGL-style math explicitly
21. **Ray Marching vs Rasterization** - Two rendering philosophies
    - Rasterization: triangles → pixels (needs clip space, homogeneous coords, clipping)
    - Ray marching: rays → samples → pixels (skips clip space)
    - Pipeline comparison diagram
22. **Why NeRF Skips Clip Space** - No GPU clipping needed
    - No triangles to clip
    - Rays are analytically defined
    - Goes directly: camera space → NDC
    - "NeRF performs these steps analytically instead of via matrices"

---

### Part 6: Summary & Reference Tables

23. **Coordinate Systems at a Glance** - Quick reference table
24. **Symbol Glossary** - i, j, fx, fy, cx, cy, w, w_clip, dirs, rays_o, rays_d, etc.
25. **Pipeline Comparison** - Graphics vs Vision vs NeRF side-by-side
26. **Key Takeaways** - One-sentence summaries including master quotes:
    - "Homogeneous coordinates work because geometry cares about ratios, not scale"
    - "Perspective projection matrix is the unique solution satisfying pinhole + depth constraints"
    - "Clip space is GPU plumbing, NDC is geometry normalization"

---

## Verification Plan

### Manual Verification
- Cross-reference reorganized notes against original PDF to ensure no knowledge is lost
- Verify all mathematical derivations are preserved
- Check that all code snippets are included
- Ensure all concepts have proper context and build logically

---

## User Review Required

> [!NOTE]
> The reorganized structure:
> 1. **Shared Foundation** (Pinhole Camera, Intrinsics, Homogeneous Coords)
> 2. **Graphics Pipeline** (Forward: 3D → 2D)
> 3. **Vision** (Inverse: 2D → 3D)
> 4. **NeRF** (Bridge between both)
>
> This shows how graphics and vision use the same math in opposite directions, with NeRF as the practical application that combines both.

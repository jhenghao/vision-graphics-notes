# Understanding Graphics and Vision: From Cameras to Neural Rendering

> A comprehensive guide to the mathematics connecting computer graphics and computer vision, culminating in NeRF (Neural Radiance Fields).

---

## Overview: Graphics & Vision Are Two Sides of the Same Coin

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

Both fields use the same mathematical foundation but in opposite directions:
- **Graphics**: Given a 3D scene, render a 2D image
- **Vision**: Given a 2D image, reconstruct the 3D scene

---

# Part 1: The Pinhole Camera (Shared Foundation)

> Both graphics and vision start from the same physical model: how light passes through a point and creates an image.

## 1.1 Pinhole Camera Model

The pinhole camera model is the simplest representation of a camera, describing the mathematical relationship between coordinates of a 3D point in the world and its projection onto a 2D image plane.

### 1.1.1 The Geometry

The physical setup of a pinhole camera involves:

- **Pinhole (Center of Projection)**: Located at the origin $(0,0,0)$. All light rays pass through this single point.
- **Object**: Located in the world at coordinates $(X,Y,Z)$. In this standard formulation, objects are **in front** of the camera ($Z > 0$).
- **Image Plane**: Located at a physical distance $f$ (focal length) **behind** the pinhole center (at $Z=-f$). This is where the image is formed.
- **Optical Axis**: The line passing through the center and perpendicular to the image plane (the Z-axis).

Notice that the image formed on the physical image plane is inverted (upside down and reversed left-to-right) relative to the object.

### 1.1.2 Mathematical Formulation

The relationship is derived using similar triangles formed by the object and optical axis, and the image and optical axis.

For a point $(X,Y,Z)$ in 3D space, its projected point $(x,y)$ on the physical image plane (at $Z=-f$) is given by:

$$x = -f \frac{X}{Z}$$
$$y = -f \frac{Y}{Z}$$

**Where**:
- $(x,y)$ are the coordinates on the image plane.
- $f$ is the focal length.
- The negative sign indicates the image is inverted.

### 1.1.3 The "Virtual" Image Plane

In computer vision and computer graphics, we often prefer not to deal with inverted coordinates. To solve this, we mathematically place a **virtual image plane** in front of the pinhole (at $Z=+f$) rather than behind it.

In this model, the projection equations become:

$$x = f \frac{X}{Z}$$
$$y = f \frac{Y}{Z}$$

This creates an **upright image** and simplifies the math, which is why you will often see the equation written without the negative sign in textbooks.

### 1.1.4 Matrix Form (Homogeneous Coordinates)

This projection can be elegantly expressed as a linear mapping using homogeneous coordinates. We convert the 3D point $(X,Y,Z)$ to $(X,Y,Z,1)$ and multiply it by a camera matrix $P$:

$$
\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = 
\begin{bmatrix} 
f & 0 & 0 & 0 \\ 
0 & f & 0 & 0 \\ 
0 & 0 & 1 & 0 
\end{bmatrix} 
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

To get the final 2D coordinates, we divide by the third component $w$ (which equals $Z$ here):

$$x = x'/w = fX/Z$$
$$y = y'/w = fY/Z$$

### 1.1.5 Important: Converting to OpenGL Convention (-Z)

> [!IMPORTANT]
> **Coordinate System Conflict Alert**
> The derivation above uses the **Standard Computer Vision Convention** (Camera looks down $+Z$). However, most graphics APIs (OpenGL, Blender) and NeRF implementations (colmap) use the **OpenGL Convention** (Camera looks down $-Z$).

In the rest of these notes, we use the **OpenGL Convention**:
- **Forward** is $-Z$.
- **Object** is at $Z_{gl} < 0$.
- **Virtual Image Plane** is at $Z_{gl} = -1$ (looking down $-Z$).

To map the standard math to OpenGL:
1.  Flip the Z axis: $Z_{gl} = -Z_{cv}$
2.  The projection equation $x = f X / Z_{cv}$ becomes $x = f X / (-Z_{gl})$.
3.  This matches the OpenGL projection formula used in Section 1.1: $x = X / (-Z)$.

**Key Takeaway**: The physics is the same, but the sign of $Z$ flips depending on which way the camera "looks".


## 1.2 Camera Coordinate System

The camera coordinate system defines how we measure positions relative to the camera.

### OpenGL / NeRF Convention

```
        y (up)
        ↑
        │
        │
        │
        ●───────→ x (right)
       /
      /
     ↓
    z (backward)
```

- **Origin**: Camera center (the pinhole)
- **+X**: Points to the right
- **+Y**: Points up
- **+Z**: Points backward (behind the camera)
- **Camera looks down −Z** (forward direction is −Z)

This means:
- Points **in front** of the camera have **z < 0**
- Points **behind** the camera have **z > 0**

---

## 1.3 Focal Length and Image Plane

### Focal Length is Always Positive

The focal length **f** is a physical distance:

$$f > 0 \quad \text{(always)}$$

It represents the distance from the camera center (pinhole) to the image plane. Since distances are non-negative by definition, f is always positive.

> If you ever see a "negative focal length," that's not physics — it's a coordinate convention issue.

### Image Plane Location

Since the camera looks down −Z, the image plane is placed at:

$$z = -f$$

The negative sign comes from the coordinate system choice, not from f being negative:
- **f is positive** (physical distance)
- **z = −f is negative** because "forward" is the negative Z direction

### Common Misconception Clarified

> "The image plane is behind the camera in OpenGL"

This statement is misleading. The **virtual image plane** used for mathematical convenience is at z = −f, which is **in front of the camera** (in the direction the camera looks). This avoids image flipping and keeps equations simple. It does not mean the camera sees backward.

---

## 1.4 Camera Intrinsics (K Matrix)

The intrinsic matrix **K** encodes the camera's internal parameters that map 3D camera coordinates to 2D pixel coordinates.

### The Intrinsic Matrix

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

### Parameters Explained

| Parameter | Meaning | Typical Value |
|-----------|---------|---------------|
| **f_x** | Focal length in pixels (x-direction) | ~500-2000 for most cameras |
| **f_y** | Focal length in pixels (y-direction) | Usually ≈ f_x |
| **c_x** | Principal point x-coordinate | ≈ Width / 2 |
| **c_y** | Principal point y-coordinate | ≈ Height / 2 |

### What Each Parameter Does

- **f_x, f_y**: Convert physical angles to pixel distances. A larger focal length means more "zoom" (narrower field of view).

- **c_x, c_y**: The principal point — where the optical axis intersects the image plane. In an ideal camera, this is the image center, but real cameras may have slight offsets.

### Accessing K in Code

```python
f_x = K[0, 0]  # Focal length (x)
f_y = K[1, 1]  # Focal length (y)
c_x = K[0, 2]  # Principal point (x)
c_y = K[1, 2]  # Principal point (y)
```

---

# Part 2: Homogeneous Coordinates (The Mathematical Trick)

> The math that makes both graphics pipelines and vision algorithms elegant.

## 2.1 Why We Need Homogeneous Coordinates

### The Problem

In standard 3D coordinates (x, y, z), we can represent:
- ✅ **Rotation** — using a 3×3 matrix
- ✅ **Scaling** — using a 3×3 matrix
- ❌ **Translation** — requires addition, not multiplication
- ❌ **Perspective projection** — requires division by z

Graphics and vision pipelines want everything to be unified as:

```
matrix × vector
```

Homogeneous coordinates make this possible.

### The Solution: Add a 4th Coordinate

We embed 3D points into 4D space:

$$(x, y, z) \rightarrow (x, y, z, w)$$

For ordinary points, we set **w = 1**.

### Translation Now Works as Matrix Multiplication

A 3D translation by (t_x, t_y, t_z) becomes:

$$\begin{bmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \\ z + t_z \\ 1 \end{bmatrix}$$

Translation is now just a matrix multiply — that's the key benefit.

---

## 2.2 Points vs Directions (w=1 vs w=0)

### The Distinction

| Type | Homogeneous Form | Affected by Translation? |
|------|------------------|--------------------------|
| **Point** | (x, y, z, 1) | ✅ Yes |
| **Direction** | (x, y, z, 0) | ❌ No |

### Why This Matters

When you multiply by a translation matrix:
- **Points (w=1)**: Get translated
- **Directions (w=0)**: Stay unchanged

This is exactly what we want:
- **Ray origins** are locations → use w = 1
- **Ray directions** are vectors → use w = 0

```python
# In NeRF conceptually:
ray_origin = (x, y, z, 1)      # A point in space
ray_direction = (dx, dy, dz, 0)  # Just a direction, no location
```

> **Key Insight**: Directions are unaffected by translation because they represent "which way" not "where".

---

## 2.3 Scaling Equivalence

### Same Point, Different Representations

The coordinates:
- (x, y, z, w)
- (2x, 2y, 2z, 2w)
- (0.5x, 0.5y, 0.5z, 0.5w)

all represent **the same point** (for any nonzero k).

### Dehomogenization: Getting Back to 3D

To convert from homogeneous to regular 3D coordinates, divide by w:

$$(x, y, z, w) \rightarrow \left(\frac{x}{w}, \frac{y}{w}, \frac{z}{w}\right)$$

This step is called **dehomogenization**.

### Concrete Numeric Example

Take the 3D point **(2, 4, 6)**:

| Homogeneous | Dehomogenize | Result |
|-------------|--------------|--------|
| (2, 4, 6, 1) | (2/1, 4/1, 6/1) | (2, 4, 6) ✅ |
| (4, 8, 12, 2) | (4/2, 8/2, 12/2) | (2, 4, 6) ✅ |
| (1, 2, 3, 0.5) | (1/0.5, 2/0.5, 3/0.5) | (2, 4, 6) ✅ |

All three homogeneous coordinates represent the **same** 3D point!

### Geometric Intuition

Think of homogeneous coordinates as **rays from the origin in 4D space**. All points along this ray:
- (x, y, z, w)
- (2x, 2y, 2z, 2w)
- (kx, ky, kz, kw)

represent one single 3D point. Only the **direction** of the 4D vector matters, not its length.

---

## 2.4 Projective Geometry Connection

### Parallel Lines Meet at Infinity

In Euclidean geometry, parallel lines never meet.

In **projective geometry**, parallel lines meet at a **point at infinity**.

This matches what we see in real images — railroad tracks converging to a **vanishing point**.

### Points at Infinity Have w = 0

If w = 0, you cannot divide by it to get a finite 3D point:

$$(x, y, z, 0) \rightarrow \text{division by zero} \rightarrow \text{point at infinity}$$

Direction vectors are points at infinity:

```
(1, 0, 0, 0) → direction along +x (point at infinity along x-axis)
(0, 1, 0, 0) → direction along +y (point at infinity along y-axis)
```

### Vanishing Points Explained

Example: Railroad tracks
- The rails are parallel in 3D space
- In an image, they converge to a **vanishing point**
- Mathematically, the vanishing point is the projection of a direction vector (dx, dy, dz, 0)

> **Vanishing points are points at infinity made visible by perspective projection.**

This only works because homogeneous coordinates allow w = 0.

---

> ## Master Takeaway
> 
> **"Homogeneous coordinates work because geometry cares about ratios, not scale — and perspective, infinity, clipping, and ray directions all fall out naturally from that single idea."**

---

# Part 3: The Graphics Pipeline (Forward Path: 3D → 2D)

> Graphics asks: "Given a 3D scene, what 2D image do we see?"

## 3.1 Camera Space

Camera space (also called view space) is where:
- **Origin**: Camera center
- **Axes**: Aligned with camera orientation
- **Coordinates are 3D**
- **Camera looks down −Z**

A point in camera space might look like:

```
(x_c, y_c, z_c) = (0.5, 0.3, -5.0)
                              ↑
                    5 units in front of camera (negative z)
```

Rays are defined here before any projection.

---

## 3.2 The Perspective Projection Matrix

The projection matrix P transforms camera-space coordinates into **clip space**.

### 5 Required Properties

The matrix must satisfy these geometric requirements:

1. **Perspective**: x_ndc = x / (−z), y_ndc = y / (−z)
2. **Correct field of view (FOV)**
3. **Correct aspect ratio**
4. **Near and far planes map to fixed NDC z values**
5. **Points in front of camera have positive w**

Everything in the matrix exists to enforce one of these properties.

### The OpenGL Perspective Matrix

$$P = \begin{bmatrix} \frac{1}{a \cdot \tan(\theta/2)} & 0 & 0 & 0 \\ 0 & \frac{1}{\tan(\theta/2)} & 0 & 0 \\ 0 & 0 & -\frac{f+n}{f-n} & -\frac{2fn}{f-n} \\ 0 & 0 & -1 & 0 \end{bmatrix}$$

Where:
- θ = vertical field of view
- a = aspect ratio (width / height)
- n = near plane distance
- f = far plane distance

### What Each Row Does

| Row | Purpose | Formula |
|-----|---------|---------|
| **Row 1** | Horizontal FOV + aspect ratio | 1 / (a · tan(θ/2)) |
| **Row 2** | Vertical FOV | 1 / tan(θ/2) |
| **Row 3** | Nonlinear depth mapping | A = −(f+n)/(f−n), B = −2fn/(f−n) |
| **Row 4** | Perspective divide (w = −z) | [0, 0, −1, 0] |

### Why Row 4 Creates Perspective

The last row [0, 0, −1, 0] ensures:

$$w_{clip} = 0 \cdot x + 0 \cdot y + (-1) \cdot z + 0 \cdot 1 = -z_{camera}$$

After the perspective divide:

$$x_{ndc} = \frac{x_{clip}}{w_{clip}} = \frac{x}{-z}$$

This is exactly the pinhole camera equation!

### Why w = −z?

- Camera looks down −Z
- Objects **in front** have z < 0
- We want w > 0 for visible points
- So: w = −z ensures visible points → w > 0, behind camera → w < 0 (clipped)

> **Key Insight**: "The perspective projection matrix is the unique matrix that satisfies the pinhole camera model, perspective division, and near/far depth constraints simultaneously."

---

## 3.3 Clip Space

Clip space is the result of applying the projection matrix to camera-space coordinates.

### What is Clip Space?

```
clip_coord = P × camera_coord
```

This produces homogeneous coordinates:

$$(x_{clip}, y_{clip}, z_{clip}, w_{clip})$$

### Why Does Clip Space Exist?

Clip space allows the GPU to:
- Perform **frustum clipping** before perspective divide
- Keep perspective information in **w**
- **Delay division** until after clipping

### Clipping Against ±w

Before dividing by w, the GPU clips geometry. A point is **kept** only if:

$$-w_{clip} \leq x_{clip} \leq w_{clip}$$
$$-w_{clip} \leq y_{clip} \leq w_{clip}$$
$$-w_{clip} \leq z_{clip} \leq w_{clip}$$

Anything outside is **discarded**.

### Why Clip Before Dividing?

Clipping before division prevents:
- **Division by zero** (when w = 0)
- **Objects behind the camera** (w < 0)
- **Geometry outside the view frustum**

> **Key Insight**: "Clip space is GPU plumbing. NDC is geometry normalization. NeRF only needs the second."

---

## 3.4 NDC (Normalized Device Coordinates)

### What is NDC?

After clipping, the GPU performs the **perspective divide**:

$$x_{ndc} = \frac{x_{clip}}{w_{clip}}, \quad y_{ndc} = \frac{y_{clip}}{w_{clip}}, \quad z_{ndc} = \frac{z_{clip}}{w_{clip}}$$

Now everything is in a normalized cube:

$$x_{ndc}, y_{ndc}, z_{ndc} \in [-1, 1]$$

This is NDC — **Normalized Device Coordinates**.

### Perspective Divide = Choosing a Representative

Remember that (x, y, z, w) and (kx, ky, kz, kw) are the same point?

The perspective divide is simply:
> "Choose the representative of this equivalence class where w = 1."

Before divide: (2x, 2y, 2z, 2w)
After divide: (x/w, y/w, z/w) — same point, different representation.

### NDC Axis Interpretation

| Axis | Value −1 | Value +1 |
|------|----------|----------|
| x | Left edge of screen | Right edge |
| y | Bottom of screen | Top of screen |
| z | Far plane | Near plane (OpenGL) |

Everything visible fits inside this cube.

---

## 3.5 Depth Nonlinearity (1/z)

### Why Far Objects are Compressed

The depth mapping from camera space to NDC is:

$$z_{ndc} = A + \frac{B}{z_c}$$

This means:
- **Near points** (small |z_c|) → NDC changes rapidly
- **Far points** (large |z_c|) → NDC changes slowly

$$z_{ndc} \propto \frac{1}{z_c}$$

### The 1/z Effect

| Camera Depth | NDC Depth | Precision |
|--------------|-----------|-----------|
| Near | Spread out | High |
| Far | Compressed | Low |

### Why This Causes Z-Fighting

Depth buffers in graphics have more precision near the camera. This is why games often have **z-fighting** (flickering surfaces) for distant objects — there's not enough depth resolution to distinguish between them.

---

## 3.6 Screen Space

### From NDC to Pixels

The final step maps NDC coordinates to actual pixel positions:

$$x_{screen} = \frac{(x_{ndc} + 1) \cdot width}{2}$$
$$y_{screen} = \frac{(1 - y_{ndc}) \cdot height}{2}$$

This:
- Maps [-1, 1] → [0, width] and [0, height]
- **Flips y** for image coordinates (y increases downward in images)
- Produces final raster positions

---

## 3.7 OpenGL vs DirectX Differences

### Convention Variations

| Feature | OpenGL | DirectX / Vulkan |
|---------|--------|------------------|
| z range | [-1, 1] | [0, 1] |
| Near plane z | +1 | 0 |
| Far plane z | −1 | 1 |
| Handedness | Right-handed | Left-handed (configurable) |

### Why the Difference?

Historical and hardware reasons:
- **DirectX** chose [0, 1] for depth because:
  - Depth buffers are unsigned
  - Precision is better near the camera with this mapping

> **Note**: NeRF uses OpenGL-style math.

---

## Full Pipeline Summary

```
Camera space (3D)
     ↓ Projection matrix (P ×)
Clip space (4D, homogeneous)
     ↓ Clipping against ±w
Still clip space
     ↓ Perspective divide (÷ w)
NDC (3D, normalized to [-1,1])
     ↓ Viewport transform
Screen space (pixels)
```

---

# Part 4: Computer Vision (Inverse Path: 2D → 3D)

> Vision asks: "Given a 2D image, what 3D scene produced it?"

## 4.1 Pixel to Ray Conversion

### The Inverse of Projection

While graphics projects 3D points to 2D pixels, vision does the reverse: convert 2D pixels into 3D rays.

### Normalized Camera Coordinates

From a pixel location (i, j), compute normalized camera coordinates:

$$x_n = \frac{i - c_x}{f_x}, \quad y_n = \frac{j - c_y}{f_y}$$

Where:
- **(i, j)**: Pixel coordinates
- **(c_x, c_y)**: Principal point (image center)
- **(f_x, f_y)**: Focal lengths in pixels

### What These Formulas Do

1. **Subtract (c_x, c_y)**: Recenter from top-left to optical center
2. **Divide by (f_x, f_y)**: Remove focal length effect, convert to angles

### Important Distinction

| Concept | Dimensions | Has Depth? | Meaning |
|---------|------------|------------|---------|
| **Normalized camera coords** | 2D | ❌ No | Direction on image plane |
| **Camera space** | 3D | ✅ Yes | Full 3D coordinates |
| **(x_n, y_n, −1)** | 3D | ⚠️ Fixed | Direction vector in camera space |

**Normalized camera coordinates become camera-space rays only after you assign a depth** (typically z = −1).

---

## 4.2 Why Y is Negated

### Coordinate System Mismatch

**Image space (pixels):**
- Origin: **top-left**
- x increases → right
- y increases → **down**

**Camera / 3D space:**
- Origin: camera center
- x increases → right
- y increases → **up**

The y-axis is **flipped**!

### The Fix: Negate Y

In NeRF code:

```python
dirs = torch.stack([
    (i - cx) / fx,      # x: no change needed
    -(j - cy) / fy,     # y: NEGATED
    -torch.ones_like(i) # z: forward direction
], -1)
```

### What the Negation Does

```
Pixel above center:   j < cy  →  (j - cy) < 0
After negation:       y > 0   →  ray points upward ✓
```

Without the minus sign, the image would be vertically flipped.

### OpenCV vs OpenGL Convention Comparison

| Feature | OpenCV / NeRF | OpenGL |
|---------|---------------|--------|
| Forward direction | −Z | −Z |
| Image origin | Top-left | Bottom-left |
| Typical use | Vision, reconstruction | Rendering |
| Uses NDC? | Usually no | Yes |

---

## 4.3 Camera to World Transformation

### The c2w Matrix

The **camera-to-world** (c2w) matrix transforms from camera space to world space.

$$c2w = \begin{bmatrix} R_{3\times3} & t_{3\times1} \\ 0_{1\times3} & 1 \end{bmatrix}$$

### Two Components

**1. Rotation (c2w[:3, :3])**
- Rotates ray directions from camera frame to world frame
- A 3×3 rotation matrix

**2. Translation (c2w[:3, -1])**
- The camera's position in world coordinates
- This becomes the ray origin

### In Code

```python
# Rotate directions: camera → world
rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)

# Ray origins = camera position in world
rays_o = c2w[:3, -1].expand(rays_d.shape)
```

Every ray:
- **Starts** at the camera center (c2w[:3, -1])
- **Points** in the rotated direction (c2w[:3, :3] × dir)

---

# Part 5: NeRF — Where Graphics Meets Vision

> NeRF is a reconstruction (vision) algorithm that uses graphics concepts (NDC) for numerical stability.

## 5.1 Ray Generation Code Explained

### The `get_rays` Function (PyTorch)

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

### Step-by-Step Explanation

1. **Create pixel grid**: `meshgrid` generates (i, j) for every pixel
2. **Normalized coords**: Subtract principal point, divide by focal length
3. **Y negation**: Fixes image-to-camera coordinate mismatch
4. **z = −1**: Points forward (camera looks down −Z)
5. **Rotate to world**: Apply c2w rotation matrix
6. **Ray origins**: Camera position in world (same for all rays)

### Output

- `rays_o`: (H, W, 3) — ray origins in world space
- `rays_d`: (H, W, 3) — ray directions in world space

### PyTorch vs NumPy Version

The `get_rays_np` function is identical except:
- Uses NumPy instead of PyTorch
- Uses `np.meshgrid(..., indexing='xy')` so no transpose is needed

---

## 5.2 NDC Rays Transformation

### The `ndc_rays` Function

```python
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Step 1: Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    # Step 2: Project origins into NDC
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]
    
    # Step 3: Convert directions to NDC slopes
    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - 
                                rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - 
                                rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]
    
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    
    return rays_o, rays_d
```

### Step 1: Near Plane Intersection

```python
t = -(near + rays_o[..., 2]) / rays_d[..., 2]
rays_o = rays_o + t[..., None] * rays_d
```

- Find where each ray intersects the near plane (z = −near)
- Move ray origins to this intersection point
- **Purpose**: All rays now start at the same depth

### Step 2: Perspective Projection of Origins

```python
o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
o2 = 1. + 2. * near / rays_o[..., 2]
```

The scaling factors:
- `W / (2 * focal)` and `H / (2 * focal)` normalize by field of view
- Division by `rays_o[..., 2]` is the perspective divide (x/z, y/z)
- This maps coordinates into NDC range [-1, 1]

### Step 3: Ray Slope Conversion

```python
d0 = ... (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
d1 = ... (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
d2 = -2. * near / rays_o[..., 2]
```

- Converts ray **directions** to **slopes** in NDC space
- Uses the difference of ratios to compute how the ray moves through NDC

### NDC Space Result

After transformation:
- **Near plane** → z = 1
- **Far plane** → z → −1
- x, y ∈ [-1, 1]

> **Note**: NeRF uses OpenGL-style math explicitly.

---

## 5.3 Why NeRF Uses NDC

### Forward-Facing Scenes

For forward-facing captures (like the LLFF dataset):
- All cameras face roughly the same direction
- Rays are nearly parallel
- Depth values vary wildly (close objects vs. far background)

### NDC Benefits

1. **Normalizes depth**: Compresses the unbounded depth range into [-1, 1]
2. **Stabilizes training**: Uniform sampling in NDC = adaptive sampling in world space
3. **Numerical stability**: Avoids very large coordinate values

---

## 5.4 Ray Marching vs Rasterization

### Two Rendering Philosophies

**Rasterization (Traditional Graphics)**
```
triangles → project to screen → determine pixel coverage → shade pixels
```

Needs:
- Clip space
- Homogeneous coordinates
- Triangle clipping
- Interpolation

**Ray Marching (NeRF)**
```
pixels → cast rays → sample points along ray → evaluate volume → composite color
```

No need for:
- Clip space
- Homogeneous coordinates
- Triangle clipping

### Pipeline Comparison

```
Rasterization:
camera space → clip space → NDC → screen → pixels

NeRF:
camera space → rays → (optional) NDC → ray marching → pixels
```

---

## 5.5 Why NeRF Skips Clip Space

NeRF doesn't need clip space because:

1. **No triangles to clip**: NeRF renders a continuous volume, not triangle meshes
2. **Rays are analytically defined**: No need for GPU clipping hardware
3. **Direct NDC**: Goes straight from camera space to NDC when needed

> **"NeRF performs the perspective divide steps analytically, instead of via matrices and clip space."**

---

# Part 6: Summary & Reference Tables

## 6.1 Coordinate Systems at a Glance

| Space | Dimensions | Range | Origin | Purpose |
|-------|------------|-------|--------|---------|
| **World** | 3D | Unbounded | Scene origin | Global scene representation |
| **Camera** | 3D | Unbounded | Camera center | Relative to camera |
| **Clip** | 4D | Unbounded | — | Pre-division, enables clipping |
| **NDC** | 3D | [-1, 1] | Screen center | Normalized for rendering |
| **Screen** | 2D | [0, W] × [0, H] | Top-left | Final pixels |

---

## 6.2 Symbol Glossary

| Symbol | Meaning |
|--------|---------|
| **i, j** | Pixel coordinates (horizontal, vertical) |
| **f_x, f_y** | Focal lengths in pixels |
| **c_x, c_y** | Principal point (image center) |
| **K** | Camera intrinsic matrix (3×3) |
| **c2w** | Camera-to-world transformation (4×4) |
| **w** | Homogeneous coordinate (4th component) |
| **w_clip** | w after projection (= −z_camera) |
| **dirs** | Ray directions in camera space |
| **rays_o** | Ray origins in world space |
| **rays_d** | Ray directions in world space |
| **NDC** | Normalized Device Coordinates |
| **n, f** | Near and far plane distances |
| **θ** | Vertical field of view |
| **a** | Aspect ratio (width / height) |

---

## 6.3 Pipeline Comparison

| Aspect | Graphics (Rasterization) | Vision (Ray Casting) | NeRF |
|--------|--------------------------|----------------------|------|
| **Direction** | 3D → 2D | 2D → 3D | 2D → 3D → 2D |
| **Primitives** | Triangles | Rays | Rays + Volume |
| **Uses Clip Space** | ✅ Yes | ❌ No | ❌ No |
| **Uses NDC** | ✅ Yes | ❌ Usually no | ✅ For forward-facing |
| **Homogeneous Coords** | ✅ Yes | ❌ No | ❌ No |

---

## 6.4 Key Takeaways

1. **Pinhole Camera**: Perspective projection divides by z to make distant objects appear smaller.

2. **Camera Intrinsics**: The K matrix maps between 3D geometry and 2D pixels via focal length and principal point.

3. **Homogeneous Coordinates**: Add w to enable translation and perspective with matrix multiplication.

4. **Points vs Directions**: w=1 for points (affected by translation), w=0 for directions (not affected).

5. **Scaling Equivalence**: (x,y,z,w) ~ (kx,ky,kz,kw) — same point, different representation.

6. **Clip Space**: GPU engineering space for safe clipping using ±w.

7. **NDC**: Normalized cube [-1,1] after perspective divide — this is where perspective "happens."

8. **Depth Nonlinearity**: 1/z mapping causes near objects to get more precision than far objects.

9. **Y Negation**: Fixes the mismatch between image coordinates (y down) and camera coordinates (y up).

10. **NeRF**: Uses graphics concepts (NDC) for a vision task (reconstruction), skipping clip space.

---

> ## Master Quotes
> 
> - **"Homogeneous coordinates work because geometry cares about ratios, not scale — and perspective, infinity, clipping, and ray directions all fall out naturally from that single idea."**
> 
> - **"The perspective projection matrix is the unique matrix that satisfies the pinhole camera model, perspective division, and near/far depth constraints simultaneously."**
> 
> - **"Clip space is GPU plumbing. NDC is geometry normalization. NeRF only needs the second."**

---

*This document was reorganized from a ChatGPT discussion to present graphics and vision concepts in a unified, pedagogically-ordered structure.*

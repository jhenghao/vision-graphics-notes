---
type: chatgpt-conversation
date: 2026-01-11
---

# ChatGPT: NDC, homogeneous coordinates, graphics pipeline, and NeRF

Raw transcript, extracted from the exported PDF ([[sources/ndc_discussion.pdf]]).
Pasted as-is — page breaks below are artifacts of the PDF extraction.

---

Total pages: 72
================================================================================
--- Page 1 ---
can you explain what these three functions do?
def get_rays(H, W, K, c2w):
i, j = torch.meshgrid(torch.linspace(0, W-1, W),
torch.linspace(0, H-1, H))  # pytorch's meshgrid has
indexing='ij'
i = i.t()
j = j.t()
dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1]
[1], -torch.ones_like(i)], -1)
# Rotate ray directions from camera frame to the world
frame
rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3],
-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
# Translate camera frame's origin to the world frame. It
is the origin of all rays.
rays_o = c2w[:3,-1].expand(rays_d.shape)
return rays_o, rays_d
def get_rays_np(H, W, K, c2w):
i, j = np.meshgrid(np.arange(W, dtype=np.float32),
np.arange(H, dtype=np.float32), indexing='xy')
dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1],
-np.ones_like(i)], -1)
# Rotate ray directions from camera frame to the world
frame
rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
# dot product, equals to: [c2w.dot(dir) for dir in dirs]
# Translate camera frame's origin to the world frame. It
is the origin of all rays.
rays_o = np.broadcast_to(c2w[:3,-1],
np.shape(rays_d))
return rays_o, rays_d
def ndc_rays(H, W, focal, near, rays_o, rays_d):1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 1/72

--- Page 2 ---
# Shift ray origins to near plane
t = -(near + rays_o[...,2]) / rays_d[...,2]
rays_o = rays_o + t[...,None] * rays_d
# Projection
o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
o2 = 1. + 2. * near / rays_o[...,2]
d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] -
rays_o[...,0]/rays_o[...,2])
d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] -
rays_o[...,1]/rays_o[...,2])
d2 = -2. * near / rays_o[...,2]
rays_o = torch.stack([o0,o1,o2], -1)
rays_d = torch.stack([d0,d1,d2], -1)
return rays_o, rays_d
These three functions are standard utilities from NeRF-style rendering pipelines.
Together, they turn camera parameters into rays in 3D space, and optionally convert
those rays into Normalized Device Coordinates (NDC).
Iʼll explain them step by step, from intuition to math.
1. get_rays(H, W, K, c2w)  — PyTorch version
What it does:
For every pixel in an image, it computes:
rays_o: the origin of the ray (camera position in world space)
rays_d: the direction of the ray (where that pixel looks in 3D)
These rays are used for volume rendering (e.g., NeRF).
Step-by-step1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 2/72

--- Page 3 ---
1. Create a pixel grid
i: x-coordinates of pixels
j: y-coordinates of pixels
Shape: (H, W)
Each (i, j) corresponds to a pixel.
2. Convert pixels → camera-space directions
This uses the camera intrinsics matrix K:
(i - cx) / fx , (j - cy) / fy : converts pixels to normalized camera
coordinates
-1 in z: camera looks down the negative z-axis (NeRF convention)
Result: a direction vector for each pixel in camera space
3. Rotate directions into world spacepython
i, j = torch.meshgrid(torch.linspace( 0, W-1, W),
                      torch.linspace( 0, H-1, H))
i = i.t()
j = j.t()
Copy code
python
dirs = torch.stack([
    (i - K[ 0][2]) / K[0][0],
    -(j - K[ 1][2]) / K[1][1],
    -torch.ones_like(i)
], -1)
Copy codeK=      f  x
0
00
f y
0c  x
c y
1
python1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 3/72

--- Page 4 ---
c2w[:3, :3]  is the rotation matrix
Converts directions from camera coordinates → world coordinates
4. Set ray origins
c2w[:3, -1]  is the camera position in world space
Every ray starts at the same camera origin
Output
rays_o: (H, W, 3)  ray origins
rays_d: (H, W, 3)  ray directions
2. get_rays_np(H, W, K, c2w)  — NumPy version
This function does the exact same thing as get_rays , but:
Uses NumPy instead of PyTorch
Uses np.meshgrid(..., indexing='xy')  so no transpose is needed
Everything else (math, output meaning) is identical.
3. ndc_rays(H, W, focal, near, rays_o, rays_d)  — Convert rays
to NDC
What it does:
Transforms rays from world space into Normalized Device Coordinates (NDC).
This is mainly used in:
Forward-facing NeRF scenesrays_d = torch. sum(dirs[..., None, :] * c2w[: 3, :3], -1)
Copy code
python
rays_o = c2w[: 3, -1].expand(rays_d.shape)
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 4/72

--- Page 5 ---
Stabilizing training
Matching OpenGL-style projection
Step-by-step
1. Move ray origins to the near plane
Intersects each ray with the near plane
Ensures all rays start at the same depth
Helps with numerical stability
2. Project origins into NDC space
This:
Applies a perspective projection
Maps coordinates into the range roughly [-1, 1]
Similar to what a GPU does before rasterization
3. Project directions into NDC spacepython
t = -(near + rays_o[..., 2]) / rays_d[..., 2]
rays_o = rays_o + t[..., None] * rays_d
Copy code
python
o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
o2 = 1. + 2. * near / rays_o[..., 2]
Copy code
python
d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/ray
d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/ray1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 5/72

--- Page 6 ---
Converts ray slopes instead of absolute directions
Ensures rays remain consistent after projection
Final output
Now:
Rays live in NDC space
Near plane is at z = 1
Far plane approaches z = -1
Big Picture Summary
Function Purpose
get_rays Generate world-space rays from camera pixels (PyTorch)
get_rays_np Same as above, but NumPy
ndc_rays Convert rays to normalized device coordinates
If you want, I can:
Draw a diagram of camera → rays → NDC
Explain why NeRF uses -z
Walk through a single pixel example numericallyd2 = -2. * near / rays_o[..., 2]
Copy code
python
rays_o = torch.stack([o0, o1, o2], - 1)
rays_d = torch.stack([d0, d1, d2], - 1)
Copy code
1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 6/72

--- Page 7 ---
* `(i - cx) / fx`, `(j - cy) / fy`: converts pixels to normalized camera coordinates * `-1` in z:
camera looks down the **negative z-axis** (NeRF convention) * Result: a direction vector for
each pixel **in camera space**
are camera space and normalized camera coordinates
same thing?
Short answer: no — theyʼre related, but not the same thing.
People often blur the terms, which is why this is confusing.
Letʼs clear it up cleanly.
1. Camera space (a.k.a. camera coordinates / view space)
Camera space is a 3D coordinate system:
Origin: camera center
Axes: defined by the camera orientation
(in NeRF: +x right, +y up, −z forward)
A point in camera space looks like:
Example:
This is a full 3D position, with depth.
2. Normalized camera coordinates (a.k.a. normalized image plane
coordinates)
Normalized camera coordinates live on a 2D plane, not full 3D space.(x  ,y  ,z  ) c cc
text
(0, 0, -3)   # 3 units in front of the camera
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 7/72

--- Page 8 ---
They are obtained by:
or equivalently from pixels:
Key properties:
Unitless
Independent of focal length
Represent a ray direction, not a point in space
They answer:
3. What your code is actually doing
This line:
produces normalized camera coordinates.
Then this:
lifts them into camera space by assigning a depth:x  =n ,y  =z  cx  c
n z  cy  c
x =n ,y =f  xi−c  x
n f  yj−c  y
“In what direction does this pixel look?”
python
(i - cx) / fx, (j - cy) / fy
Copy code
python
dirs = [..., ..., - 1]
Copy code
text
(x_n, y_n, -1)
Cd1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 8/72

--- Page 9 ---
So the vector:
is:
✅ a direction vector in camera space
❌ not just normalized camera coordinates
4. Why this distinction matters
Concept Dimensionality Has depth?Meaning
Camera space 3D ✅ Actual coordinates relative
to camera
Normalized camera
coords2D ❌ Direction on image plane
(x_n, y_n, -1) 3D ⚠ fixed Direction vector in camera
space
Normalized camera coordinates become camera-space rays only after you choose a
depth (here z = -1).
5. Why people mix the terms (especially in NeRF code)
In ray generation:
We donʼt care about the exact depth
Any point along the same ray is equivalent
So people casually say:
even though itʼs constructed from normalized image plane coordinates.Copy code
text
[(i-cx)/fx, (j-cy)/fy, -1]
Copy code
“camera-space direction”1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 9/72

--- Page 10 ---
Thatʼs correct in context, but imprecise in theory.
One-sentence takeaway
Normalized camera coordinates are 2D ray directions; camera space is 3D.
Your code converts normalized camera coordinates into camera-space ray
directions by setting z = -1.
If you want, I can also explain how this differs from NDC or OpenGL camera space,
which adds another layer of confusion.
xn \=fx i−cx  ,yn \=fy j−cy  
what are i, Cx and Cy?
Good question — these symbols come straight from the camera intrinsics model.
Iʼll explain each one clearly and relate them back to the code and equation you wrote.
The equation youʼre looking at
This converts pixel coordinates → normalized camera coordinates.
1. What is i (and j)?
i: horizontal pixel coordinate (x-axis)
j: vertical pixel coordinate (y-axis)
They come from the image grid.x  =n ,y  =f  xi−c  x
n f  yj−c  y1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 10/72

--- Page 11 ---
Example (for a 640×480 image):
In your code:
i and j store the pixel location of each ray.
2. What are Cx and Cy (c_x, c_y)?
c_x: x-coordinate of the principal point
c_y: y-coordinate of the principal point
The principal point is where the cameraʼs optical axis hits the image plane.
In an ideal camera:
But in real cameras, it might be slightly off-center.
They come from the intrinsic matrix:
So:text
top-left pixel:     i = 0,   j = 0
center pixel:       i ≈ 320, j ≈ 240
bottom-right pixel: i = 639, j = 479
Copy codepython
i, j = meshgrid(...)
Copy code
text
c_x = W / 2
c_y = H / 2
Copy code
K=      f  x
0
00
f y
0c  x
c y
1
python1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 11/72

--- Page 12 ---
3. Why subtract Cx and Cy?
Pixel coordinates are measured from the top-left corner of the image.
But camera geometry is defined relative to the optical center.
So:
This recenters the coordinate system.
4. Why divide by f_x and f_y?
This:
Removes the effect of focal length
Converts pixel distances into unitless direction slopes
After division:
x_n, y_n describe ray direction, not pixel distance
Same scene + different focal length → same normalized coordinates
5. Putting it all together (intuition)
For a single pixel:c_x = K[ 0, 2]
c_y = K[ 1, 2]
Copy code
text
i - c_x   → how far left/right from the center
j - c_y   → how far up/down from the center
Copy code
text
Pixel (i, j)
   ↓ subtract (c_x, c_y)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 12/72

--- Page 13 ---
Then your code does:
Which means:
One-line summary
i, j: pixel coordinates
c_x, c_y: image center (principal point)
(i - c_x) / f_x , (j - c_y) / f_y : normalized camera coordinates (ray
direction)
If you want, I can also explain why the y term is negated in your code or how this
relates to OpenCV vs OpenGL conventions.Offset from image center
   ↓ divide by (f_x, f_y)
Normalized direction
Copy codepython
(x_n, y_n, - 1)
Copy code
“This pixel shoots a ray forward from the camera through this direction.”
I can also explain how this differs from **NDC** or **OpenGL camera space**, which adds
another layer of confusion
please do.
Great — this is the part that trips most people up. Iʼll explain it in three layers and tie
them directly to your code:
1. Why the y-coordinate is negated
2. OpenCV vs OpenGL camera conventions
3. How NDC fits into all of this1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 13/72

--- Page 14 ---
1. Why is the y term negated?
In your code:
The key line is:
The root cause: image coordinates vs 3D coordinates
Image space (pixels):
Origin: top-left
x increases → right
y increases → down
Camera / 3D space:
Origin: camera center
x increases → right
y increases → up
So the y-axis is flipped.
What the negation does
Without the minus sign:python
dirs = torch.stack([
    (i - cx) / fx,
    -(j - cy) / fy,
    -1
], -1)
Copy codepython
-(j - cy) / fy
Copy code
text
Pixel above center:   j < cy  →  (j - cy) < 0
After negation:       y > 0   →  ray points upward1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 14/72

--- Page 15 ---
The image would be vertically flipped
Rays would go in the wrong direction
✅ The negation fixes the coordinate system mismatch
2. OpenCV vs OpenGL camera conventions
This is where terminology gets messy.
OpenCV convention (used by NeRF)
Camera looks down −Z
Right-handed system
Image y-axis points down
Thatʼs why your code uses:
This means:
OpenGL convention (classic graphics pipeline)
Camera looks down −Z (same)
BUT after projection, coordinates are transformed into NDC
OpenGL pipeline:Copy code
text
x → right
y → up
z → backward
Copy codepython
z = -1
Copy code
“Shoot the ray forward”
text1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 15/72

--- Page 16 ---
OpenCV usually stops at camera space.
Quick comparison
Concept OpenCV / NeRF OpenGL
Forward direction −Z −Z
Image origin Top-left Bottom-left
Typical use Vision, reconstruction Rendering
Uses NDC? ❌ (usually) ✅
3. How NDC fits in (your ndc_rays  function)
NDC = Normalized Device Coordinates
After projection:
Why NeRF uses NDC
For forward-facing scenes:
All rays originate from roughly the same direction
NDC:
Normalizes depth
Keeps rays well-behaved
Improves training stabilitycamera space → clip space → NDC → screen
Copy code
text
x, y, z ∈ [-1, 1]
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 16/72

--- Page 17 ---
What ndc_rays is doing conceptually
1. Intersect rays with the near plane
2. Apply a perspective projection
3. Rescale to [-1, 1]
After this:
Near plane → z = 1
Far plane → z → -1
This mirrors OpenGLʼs NDC (with NeRF-specific tweaks).
4. Putting it all together (mental model)
Ray generation pipeline in your codepython
t = -(near + rays_o[..., 2]) / rays_d[..., 2]
Copy code
python
x_ndc = x / z
y_ndc = y / z
Copy codepython
W / (2 * focal)
H / (2 * focal)
Copy code
text
pixels (i, j)
   ↓ subtract (cx, cy)
image-centered pixels
   ↓ flip y
camera-aligned coordinates
   ↓ divide by focal
normalized camera coords
   ↓ set z = -11/5/26, 9:00 AM ChatGPT
https://chatgpt.com 17/72

--- Page 18 ---
Final takeaway (one paragraph)
The y negation fixes the mismatch between image coordinates (downward y) and
3D camera coordinates (upward y).
NeRF follows an OpenCV-style camera model where the camera looks down −Z.
NDC is a later projection step that normalizes rays for rendering and training, similar
to OpenGLʼs pipeline but adapted for ray tracing.
If you want next, I can:
Walk through a single pixel numerically
Draw a coordinate diagram in ASCII
Explain why NeRF doesnʼt always use NDCcamera-space ray direction
   ↓ rotate + translate (c2w)
world-space rays
   ↓ (optional) ndc_rays
NDC rays
Copy code
camera space → clip space → NDC → screen
can you explain the clip space and NDC? also explain
how the conversion of each step does.
Below is a clean, end-to-end explanation of the graphics pipeline your developer note
mentions:
Iʼll explain:
1. What clip space is
2. What NDC is
3. What each conversion step actually does mathematically and conceptuallycamera space → clip space → NDC → screen1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 18/72

--- Page 19 ---
4. How this relates to your NeRF ndc_rays  function
1. Camera space (starting point)
You already mostly understand this, but it anchors everything else.
Camera (view) space
Origin: camera center
Coordinates are 3D
Camera looks down −Z (OpenGL & NeRF)
A point looks like:
Rays are defined here before any projection.
2. Clip space — before perspective division
What is clip space?
Clip space is the result of applying the projection matrix to camera-space
coordinates.
Mathematically:
This produces homogeneous coordinates:scss
(x_c, y_c, z_c, 1)
Copy code
ini
clip_coord  = ProjectionMatrix × camera_coord
Copy code
scss
(x_clip, y_clip, z_clip, w_clip)
Copycode1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 19/72

--- Page 20 ---
Why does clip space exist?
Clip space allows the GPU to:
Perform frustum clipping
Keep perspective information in w
Delay division until after clipping
This is why itʼs called clip space.
What the projection matrix does
For perspective projection, it:
Encodes focal length (field of view)
Encodes near/far planes
Moves depth info into w
Conceptually:
⚠ Nothing is normalized yet
3. Clipping (why “clip” space matters)
Before dividing by w, the GPU clips geometry:
A point is kept only if:Copy code
nginx
x_clip ≈ x_c
y_clip ≈ y_c
z_clip ≈ depth mapping
w_clip ≈ -z_c
Copy code
diff
-w ≤ x_clip ≤ w
-w ≤ y_clip ≤ w
-w ≤ z_clip ≤ w1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 20/72

--- Page 21 ---
Anything outside is discarded.
This prevents:
Division by zero
Objects behind the camera
Geometry outside the view frustum
4. NDC — Normalized Device Coordinates
What is NDC?
After clipping, the GPU performs perspective division:
Now:
This is NDC.
Why divide by w?
This is the step that creates perspective.
Effects:
Far objects appear smaller
Parallel lines converge
Depth becomes nonlinear
This is the mathematical core of perspective projection.Copy code
ini
x_ndc = x_clip / w_clip
y_ndc = y_clip / w_clip
z_ndc = z_clip / w_clip
Copy codecss
x_ndc, y_ndc, z_ndc ∈ [-1, 1]
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 21/72

--- Page 22 ---
Interpretation of NDC
Axis Meaning
x = −1 left edge of screen
x = +1 right edge
y = −1 bottom
y = +1 top
z = −1 far plane
z = +1 near plane (OpenGL)
Everything visible fits inside a cube.
5. Screen space (final step)
NDC → screen coordinates:
This:
Maps [-1, 1] → pixel coordinates
Flips y for image coordinates
Produces final raster positions
6. Full pipeline summary (compact)ini
x_screen  = (x_ndc + 1) * width  / 2
y_screen  = (1 - y_ndc) * height / 2
Copy code
scss1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 22/72

--- Page 23 ---
7. How this relates to NeRFʼs ndc_rays
NeRF does not rasterize, so it skips:
Clip space
Homogeneous coordinates
GPU clipping
Instead, it manually recreates the math.
What NeRF does directly
In ndc_rays :
That is exactly the perspective divide, but done analytically.
Then it rescales by:
Which corresponds to:
Field of view normalizationCamera space ( 3D)
   ↓ Projection matrix
Clip space ( 4D, homogeneous)
   ↓ Clipping
Still clip space
   ↓ Perspective  divide (÷ w)
NDC (3D, normalized)
   ↓ Viewport transform
Screen space (pixels)
Copy code
python
x_ndc = x / z
y_ndc = y / z
Copy code
python
W / (2 * focal)
H / (2 * focal)
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 23/72

--- Page 24 ---
Projection matrix scaling terms
So NeRF:
Skips clip space
Skips w
Goes camera space → NDC directly
Because rays never need clipping.
8. Key conceptual distinction (important)
Space Purpose
Camera space Geometry relative to camera
Clip space Enable clipping using w
NDC Unified, normalized viewing cube
Screen Actual pixels
Clip space is a GPU engineering space, not a geometric one.
Final takeaway (one sentence)
Clip space exists to make clipping and perspective mathematically safe using
homogeneous coordinates; NDC is what you get after dividing by w, where
everything visible lies in a normalized [-1,1] cube.
If you want next, I can:
Derive the projection matrix numerically
Show why w = -z
Compare OpenGL vs DirectX NDC differences
Tie this back to ray marching vs rasterization
1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 24/72

--- Page 25 ---
\-w ≤ x\_clip ≤ w \-w ≤ y\_clip ≤ w \-w ≤ z\_clip ≤ w
is w the w_clip?
Yes. That w is specifically w_clip.
When you see inequalities like:
the w there is the same w component of the clip-space coordinate:
Why this matters
After applying the projection matrix, every point is in clip space and represented in
homogeneous coordinates:
Before doing the perspective divide, the GPU performs clipping using w_clip.
Only points that satisfy:
are kept.−w ≤ x_clip ≤ w
−w ≤ y_clip ≤ w
−w ≤ z_clip ≤ w
Copy code(x  , y  , z  , w  ) clip clip clip clip
   x clip
y  clip
z  clip
w  clip
−w  ≤ clipx  ≤ clipw  clip
−w  ≤ clipy  ≤ clipw  clip
−w ≤ clipz ≤ clipw clip1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 25/72

--- Page 26 ---
Why clip against ±w_clip?
Because perspective hasnʼt happened yet.
Later, the GPU will divide:
If clipping were done after division, youʼd risk:
division by zero
geometry behind the camera
broken interpolation
Clipping against ±w_clip guarantees that after division:
What is w_clip conceptually?
In a standard OpenGL-style projection:
So:
Points in front of the camera → w_clip > 0
Points behind the camera → w_clip < 0  (discarded)
This is why the camera looks down −Z.
Pipeline recap (minimal)x  = ndc w  clipx  clip
x  ,y  ,z  ∈ ndcndcndc[−1,1]
w  = clip−z  camera
scss
camera space (x, y, z, 1)
   ↓ projection matrix
clip space (x_clip, y_clip, z_clip, w_clip)
   ↓ clip against ±w_clip1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 26/72

--- Page 27 ---
One-line takeaway
Yes — the w in those inequalities is w_clip, the homogeneous coordinate produced
by the projection matrix, and it is what makes safe clipping and perspective division
possible.
If you want, I can also show:
a concrete numeric example
why DirectX clips z differently
how this relates to NeRF skipping clip space   ↓ divide by w_clip
NDC (x_ndc, y_ndc, z_ndc)
Copy code
homogeneous coordinates
what does homogeneous coordinates mean?
Homogeneous coordinates are a mathematical trick that lets us represent translation,
projection, and perspective using matrix multiplication.
Theyʼre the reason the graphics pipeline (and camera geometry) works so cleanly.
Iʼll explain this in a structured way and tie it directly to clip space and w.
1. The core problem homogeneous coordinates solve
In normal 3D coordinates:
css
(x, y, z)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 27/72

--- Page 28 ---
you cannot represent translation with a matrix multiply alone.
Example:
Rotation ✔  (matrix)
Scaling ✔  (matrix)
Translation ❌ (needs addition)
Graphics pipelines want everything to be:
Homogeneous coordinates make that possible.
2. The idea: add one extra dimension
We embed 3D points into 4D space:
This is a homogeneous coordinate.
For ordinary points:
3. Why this works (translation example)
A 3D translation by (tx, ty, tz)  becomes:Copy code
cpp
matrix × vector
Copy code
css
(x, y, z)  →  ( x, y, z, w)
Copy code
ini
w = 1
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 28/72

--- Page 29 ---
Translation is now just a matrix multiply.
Thatʼs the key benefit.
4. Homogeneous coordinates are not unique
The coordinates:
and
represent the same point (for any nonzero k).
To get the actual 3D point, you divide by w:
This step is called dehomogenization.
5. Why perspective works (the magic part)
Perspective projection sets:
So after projection:        =1
0
0
00
1
0
00
0
1
0t  x
t y
t  z
1x
y
z
1  x+tx
y+t y
z+t  z
1
css
(x, y, z, w)
Copy code
scss
(kx, ky, kz, kw)
Copy code
(x,y,z)=′′′
 ,  ,  (wx
wy
wz)
ini
w = -z_camera
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 29/72

--- Page 30 ---
This:
Makes far objects smaller
Creates depth-dependent scaling
Produces realistic perspective
All from a single divide by w.
6. Points vs directions (important distinction)
Type Homogeneous form Why
Point (x, y, z, 1) Affected by translation
Direction / vector (x, y, z, 0) Not affected by translation
Example:
Ray origin → w = 1
Ray direction → w = 0
This is exactly what your NeRF code relies on conceptually.
7. Why clip space uses homogeneous coordinates
Clip space coordinates are:
Before dividing by w_clip, the GPU:
Clips against ±w_clip
Ensures perspective divide is safe
After division:x  = ndc =w  clipx  clip
 −zx
scss
(x_clip, y_clip, z_clip, w_clip)
Copy code
scss1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 30/72

--- Page 31 ---
So homogeneous coordinates:
Enable clipping
Enable perspective
Enable a unified matrix pipeline
8. Intuition (one-sentence mental model)
Homogeneous coordinates store “scale” in w, and dividing by w turns that scale
into perspective.
9. Minimal takeaway
Homogeneous coordinates add an extra dimension
They make translation and projection linear
w controls perspective and clipping
Dividing by w gives back real 3D coordinates
If you want next, I can:
Show a numeric camera → clip → NDC example
Connect this to projective geometry
Explain why points at infinity have w = 0(x_ndc, y_ndc, z_ndc)
Copy code
If you want next, I can: * Derive the **projection matrix numerically** * Show **why `w = -
z`** * Compare **OpenGL vs DirectX NDC differences** * Tie this back to **ray marching vs
rasterization**
can you do all four items?
Absolutely — letʼs do all four, cleanly and in order.
Iʼll keep the math concrete but not overwhelming, and Iʼll constantly connect it back to1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 31/72

--- Page 32 ---
what itʼs for.
1️⃣ Deriving the perspective projection matrix
(numerically)
We want a matrix that maps camera space → clip space.
Camera space assumptions (OpenGL / NeRF style)
Camera at origin
Looking down −Z
Near plane at n
Far plane at f
Vertical field of view fov
Aspect ratio a = width / height
Step 1: Perspective intuition
For a point in camera space:
Perspective says:
So we want:
division by z
but only after clipping
Thatʼs why we use homogeneous coordinates.scss
(x_c, y_c, z_c)
Copy code
nginx
x_proj ∝ x_c / -z_c
y_proj ∝ y_c / -z_c
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 32/72

--- Page 33 ---
Step 2: Projection matrix (OpenGL)
The standard OpenGL perspective projection matrix is:
Apply it to:
Result:
✅ Perspective divide happens later via w_clip
2️⃣ Why w = -z?
This is a huge conceptual point.
What w is doing
After projection:
Then:P=        atan(θ/2)1
0
0
00
 tan(θ/2)1
0
00
0
−  f−nf+n
−10
0
−  f−n2fn
0
(x ,y ,z,1) c cc
ini
x_clip = x_c / (a tan(fov/ 2))
y_clip = y_c / tan(fov/ 2)
z_clip = A z_c + B
w_clip = -z_c
Copy code
scss
(x_clip, y_clip, z_clip, w_clip)
Copy code
ini1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 33/72

--- Page 34 ---
If:
Then:
Thatʼs exactly perspective projection.
Why the minus sign?
Because:
Camera looks down −Z
Objects in front have z < 0
We want w > 0 for visible points
So:
ensures:
visible points → w > 0
behind camera → w < 0 → clipped
Deep intuitionx_ndc = x_clip / w_clip
Copy code
ini
w_clip = -z_camera
Copy code
ini
x_ndc = x / -z
Copy code
ini
w = -z
Copy code
w stores depth so that dividing by it shrinks distant objects.1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 34/72

--- Page 35 ---
No division → no perspective
Division by z → perspective
3️⃣ OpenGL vs DirectX NDC differences
This causes a lot of confusion.
OpenGL NDC (classic)
After perspective divide:
Near plane → z = +1
Far plane → z = -1
Right-handed
Camera looks down −Z
DirectX / Vulkan NDC
Near plane → z = 0
Far plane → z = 1
Left-handed by default (configurable)
Better depth buffer precision
Why the difference?
Historical + hardware reasons.css
x, y ∈ [-1, 1]
z ∈ [-1, 1]
Copy code
css
x, y ∈ [-1, 1]
z ∈ [0, 1]
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 35/72

--- Page 36 ---
DirectX chose [0, 1] for depth because:
Depth buffers are unsigned
Precision is better near the camera
Key takeaway
Feature OpenGL DirectX
z range [-1, 1] [0, 1]
Near depth +1 0
Far depth −1 1
Used by NeRF ✅ ❌
NeRFʼs ndc_rays  matches OpenGL-style math.
4️⃣ Ray marching vs rasterization (why NeRF skips clip
space)
This ties everything together.
Rasterization pipeline (graphics)
Needs:
Clip spacecss
triangles
 → camera space
 → clip space
 → NDC
 → screen
 → pixels1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 36/72

--- Page 37 ---
Homogeneous coordinates
Triangle clipping
Interpolation
Ray marching (NeRF)
Key differences:
No triangles
No clipping
No screen projection
So NeRF:
does not need clip space
does not need homogeneous coords
does not rasterize
Why NeRF still uses NDC
For forward-facing scenes:
Rays are nearly parallel
Depth values vary wildly
NDC:
Normalizes depth
Makes ray sampling stable
Keeps values numerically well-behaved
Thatʼs why NeRF does:Copy code
nginx
pixels
 → rays
 → sample points along ray
 → evaluate volume
Copy code
nginx1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 37/72

--- Page 38 ---
directly, skipping clip space entirely.
🔁 Final mental model (important)
Clip space is GPU plumbing.
NDC is geometry normalization.
NeRF only needs the second.
✅ Final one-paragraph takeaway
The projection matrix encodes perspective by putting depth into w
w = -z makes perspective division work and enforces visibility
OpenGL and DirectX differ mainly in z range
NeRF skips clip space entirely because rays donʼt need clipping, but still uses NDC
for numerical stability
If you want next, I can:
Walk one point numerically through every space
Show how depth nonlinearity comes from 1/z
Explain why points at infinity have w = 0camera space → NDC
Copy code
vbnet
Rasterization:
camera → clip → NDC → screen
NeRF:
camera → rays → ( optional ) NDC → ray marching
Copy code
Perspective says: nginx Copy code `x_proj ∝  x_c / -z_c y_proj ∝  y_c / -z_c`1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 38/72

--- Page 39 ---
why?
Ah! Youʼre asking why perspective projection divides by z_c (or -z_c). Letʼs break
this down carefully.
1️⃣ The intuitive reason: “things farther away look smaller”
Perspective projection is literally trying to model how the human eye or a camera sees
the world:
If an object is twice as far away, it should appear half as big.
That means the projected coordinate on the image plane should shrink
proportionally to distance.
Imagine a simple pinhole camera
Camera at origin
Image plane at z = -f (negative because camera looks down −Z)
Point in space: (x_c, y_c, z_c)
Draw a line from the point to the camera center.
Where it intersects the image plane?
By similar triangles:
If we set image plane at z = −1 (NeRF convention):
✅ Thatʼs exactly the formula you saw.
2️⃣ Step-by-step with triangles =−zcx  proj
 z  image planex  c
x  = proj ,y  =−z  cx c
proj −z  cy c
markdown1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 39/72

--- Page 40 ---
C = camera center
* = 3D point
Image plane in front of camera
By similar triangles:
Rearranging gives:
In NeRF/OpenGL, z_plane = -1  →
3️⃣ The minus sign
Camera looks down −Z
Points in front of camera: z_c < 0
Dividing by -z_c makes the projected x, y positive in the right direction
4️⃣ Intuition in one sentence      *
     /|
    / |
   /  | y_c
  /   |
 C----+
      x_c
Copy code =image plane dept hcoordina te on image plane
 dept h of pointworld coordina te
x  = imagex   cz  cz  plane
x  = projx  /−c z  c
Perspective projection divides by z to make distant objects appear smaller,
exactly like how real cameras and human eyes work.1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 40/72

--- Page 41 ---
If you want, I can draw a simple ASCII diagram showing the similar triangles so you
can “see” why itʼs / -z_c. It really makes this click.
Do you want me to do that?
how can I save the conversation of this chat?
You have a few ways to save this conversation, depending on what you want to do with
it:
1️⃣ Copy & Paste (quickest)
Simply select all the text in the chat window.
Copy and paste it into:
A Word document
Google Docs
Notepad or any text editor
You can also format it if you want headings and code blocks.
2️⃣ Use the “Export” or “Download” feature (if available)
Some chat platforms (including OpenAI web chat) let you export the conversation
as a .txt or .pdf.
Look for buttons like “Export”, “Download”, or “Save chat” in the chat window.
3️⃣ Screenshots
For short sections or diagrams, you can take screenshots.
On Windows: Win + Shift + S  → Snip tool
On Mac: Cmd + Shift + 4  → Select area
On phones: usual screenshot gesture1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 41/72

--- Page 42 ---
4️⃣ Save in a coding-friendly format
If you want to keep code blocks intact:
Paste the conversation into Markdown or .md file
Use triple backticks for code:
markdown
```python
# your code here
Copy codeyaml
---
💡 **Tip:**  If you plan to revisit later for learning,  saving as Markdown  
---
If you want, I can make a **single  nicely formatted  version of this entire 
Do you want me to do that?
Copy code
* Show how **depth nonlinearity** comes from `1/z` * Explain why **points at infinity have
`w = 0`**
can you do these two?
Absolutely! Letʼs tackle these two concepts carefully and step by step.
1️⃣ Depth nonlinearity from 1/z1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 42/72

--- Page 43 ---
Step 1: Recall perspective projection
For a point (x, y, z)  in camera space, the perspective divide gives:
Similarly, the depth z is transformed (in OpenGL-style NDC) like:
Where A and B depend on the near and far plane.
Step 2: Why depth is nonlinear
Look at the mapping from camera depthz_c to NDC depthz_ndc:
Near points (z_c ≈ near ) → NDC changes rapidly
Far points (z_c ≈ far ) → NDC changes slowly
✅ This is depth nonlinearity.
Far objects get compressed in NDC.
Near objects get stretched.
Step 3: Intuition
Depth buffers in graphics are more precise near the camera — thatʼs why games
often have “z-fighting” far away.
The 1/z mapping comes directly from the perspective divide: distant points divide
by a larger z → smaller effect on projected depth.
Step 4: Connection to NeRF
NeRFʼs ndc_rays  does something similar:x  = ndc ,y  =−zx
ndc −zy
z = ndcA+ zB
nginx
z_ndc ∝ 1 / z_c
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 43/72

--- Page 44 ---
You can see the / rays_o[...,2]  term — this is the 1/z depth nonlinearity.
It ensures forward-facing rays are well-distributed in the NDC cube.
2️⃣ Points at infinity have w = 0
Step 1: Homogeneous coordinates refresher
In homogeneous coordinates:
3D point → w = 1
Direction (vector) → w = 0
Step 2: Why w = 0 represents infinity
Dehomogenization:
If w = 1 → normal point
If w = 0 → division by zero → point at infinity
Intuition:
Rays that “go forever” have no finite endpoint, so w = 0 encodes that the point is
infinitely far along a direction.
Step 3: Connection to rays and directions
In NeRF:
Ray origin → (x, y, z, 1)  (finite)python
o2 = 1. + 2. * near / rays_o[..., 2]
d2 = -2. * near / rays_o[..., 2]
Copy code
css
(x, y, z, w)
Copy code
(x ,y ,z )= realrealreal w(x,y,z)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 44/72

--- Page 45 ---
Ray direction → (dx, dy, dz, 0)  (infinite along that direction)
Homogeneous coordinates allow one matrix multiply to move both origin and
direction:
Directions (w=0) are not affected by translation
Points (w=1) are affected by translation
✅ One-sentence takeaway
Depth nonlinearity comes from perspective projection: farther objects divide by a
larger z → compressed depth.
Points at infinity have w=0 in homogeneous coordinates because dividing by zero
encodes “no finite endpoint” — exactly what you need for ray directions.
If you want, I can also draw a small diagram showing 1/z depth compression and
points at infinity as vectors — it makes both of these ideas visually obvious.
Do you want me to do that?
I think we had a great discussion. Could you reorganize
all concepts and details from the beginning of this chat
session entirely so reader can understand all concepts
easily? Don't loss any concept or details.
Absolutely! I can reorganize the entire discussion into a single, structured, self-
contained document, keeping all the concepts, code references, and details, in a
logical flow so that any reader can follow from start to finish. Iʼll break it into sections
with headings, explanations, math, and examples, just like a well-organized tutorial.
Hereʼs the full, reorganized version:1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 45/72

--- Page 46 ---
Understanding Ray Generation, Camera Coordinates,
and Projection in NeRF
This document consolidates all the concepts we discussed in this chat session,
including ray generation, camera coordinates, normalized coordinates, homogeneous
coordinates, clip space, NDC, and their connections to NeRF.
1️⃣ Ray Generation from a Camera
In NeRF and similar 3D rendering pipelines, we first generate rays from the camera into
the scene.
PyTorch function:
NumPy version:
Explanation of the code:python
def get_rays (H, W, K, c2w ):
    i, j = torch.meshgrid(torch.linspace( 0, W-1, W), torch.linspace( 0, H-1
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[ 0][2])/K[0][0], -(j-K[ 1][2])/K[1][1], -torch.
    rays_d = torch. sum(dirs[..., np.newaxis, :] * c2w[: 3,:3], -1)
    rays_o = c2w[: 3,-1].expand(rays_d.shape)
    return rays_o, rays_d
Copy codepython
def get_rays_np (H, W, K, c2w ):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=
    dirs = np.stack([(i-K[ 0][2])/K[0][0], -(j-K[ 1][2])/K[1][1], -np.ones_l
    rays_d = np. sum(dirs[..., np.newaxis, :] * c2w[: 3,:3], -1)
    rays_o = np.broadcast_to(c2w[: 3,-1], np.shape(rays_d))
    return rays_o, rays_d1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 46/72

--- Page 47 ---
1. Pixel grids (i, j)
i: horizontal pixel coordinates (0 → W-1)
j: vertical pixel coordinates (0 → H-1)
Generated using meshgrid .
2. Camera intrinsics (K)
Focal lengths: f_x = K[0,0] , f_y = K[1,1]
Principal point: c_x = K[0,2] , c_y = K[1,2]
These define how pixels map to directions.
3. Normalized camera coordinates
Formula:
Negation of y: because image coordinates grow down, but camera space y
grows up.
Result: ray direction in camera coordinates.
4. World coordinates
Rotate using camera-to-world matrix c2w[:3,:3] .
Translate origin to camera center c2w[:3,-1] .
Key concepts:
Symbol Meaning
i, j Pixel coordinates (image grid)
c_x, c_y Principal point of the camera (image center)
f_x, f_y Focal lengths in pixels
dirs Ray directions in camera space
rays_o Ray origins in world space
rays_d Ray directions in world spaceCopy code
x  =n ,y  =f  xi−c  x
n−  ,z  =f  yj−c  y
n−11/5/26, 9:00 AM ChatGPT
https://chatgpt.com 47/72

--- Page 48 ---
2️⃣ Normalized Device Coordinates (NDC) and ndc_rays
What this does:
1. Shift ray origins to near plane:
Ensures all rays start at the near clipping plane.
2. Apply perspective projection:
Divide x, y by z (x/z, y/z) → creates perspective (1/z depth nonlinearity)
Rescale to NDC cube [-1, 1]
3. Compute new ray directions in NDC.
Why use NDC?
Stabilizes ray marching for forward-facing scenes.
Depth is nonlinear, compressed for far points, expanded near the camera.
Makes training NeRF networks more stable.
Depth nonlinearitypython
def ndc_rays (H, W, focal, near, rays_o, rays_d ):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d
    
    o0 = - 1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = - 1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]
    d0 = - 1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/
    d1 = - 1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/
    d2 = - 2. * near / rays_o[..., 2]
    
    rays_o = torch.stack([o0,o1,o2], - 1)
    rays_d = torch.stack([d0,d1,d2], - 1)
    
    return rays_o, rays_d
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 48/72

--- Page 49 ---
From perspective:
Far points (z_c large) → smaller projected change
Near points (z_c small) → larger projected change
This is the 1/z effect.
3️⃣ Homogeneous Coordinates
Definition:
Represent 3D points in 4D: (x, y, z, w)
Points: w = 1
Directions: w = 0
Dehomogenize (convert back to 3D):
Why homogeneous coordinates?
Allows translation, rotation, scaling, and perspective projection all with matrix
multiplication.
Directions (w=0) are not translated, only rotated.
Points (w=1) are affected by translation.
Points at infinity:
w = 0 → dehomogenization gives division by zero → point at infinity
Used to represent rays/directions in graphics and NeRF.
4️⃣ Camera → Clip Space → NDC → Screen Pipeline
Camera space:
3D coordinates relative to camerax  = proj ,y  =−z  cx  c
proj −z  cy  c
(x,y,z)=′′′(x/w,y/w,z/w)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 49/72

--- Page 50 ---
Camera looks down −Z
Rays are defined here
Clip space:
Apply projection matrix
\text{clip_coord} = P \cdot \text{camera_coord}
Result: homogeneous (x_clip, y_clip, z_clip, w_clip)
Clipping is done against ±w_clip before perspective divide
Example: w_clip = -z_c
NDC:
Divide by w_clip:
Now all coordinates are in cube [-1, 1]
Perspective is applied automatically via division by z
Screen space:
Map NDC [-1, 1] → pixel coordinates
OpenGL vs DirectX NDC:
Feature OpenGL DirectX
z range [-1, 1] [0, 1]
Near plane +1 0
Far plane -1 1
Right-handed Yes Left-handed by default
Why NeRF skips clip space:x  = ndcx  /w  ,y  = clip clip ndcy  /w  ,z  = clip clip ndcz  /w  clip clip
x  = screen W,y  =2(x  +1) ndc
screen H2(1−y  )ndc1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 50/72

--- Page 51 ---
Rays do not need triangle clipping
No rasterization → no GPU pipeline needed
Only NDC is used for numerical stability
5️⃣ Perspective Divide and Depth Nonlinearity
Division by z after projection:
Far objects shrink → depth compression
Near objects expand → better precision near camera
Depth nonlinearity comes from the / z term in the projection
6️⃣ Summary of Symbols and Concepts
Symbol Meaning
i, j Pixel coordinates
c_x, c_y Principal point (image center)
f_x, f_y Focal length in pixels
dirs Ray directions in camera space
rays_o Ray origins in world space
rays_d Ray directions in world space
w / w_clip Homogeneous coordinate, controls perspective
NDC Normalized Device Coordinates, after perspective divide
Clip space Homogeneous coordinates before division, used for clipping
Points at infinity w = 0x  = projx/−z,y  = projy/−z1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 51/72

--- Page 52 ---
7️⃣ Key Takeaways
1. Camera space defines rays relative to the camera.
2. Normalized camera coordinates are obtained by centering pixels and dividing by
focal length.
3. Y-axis negation corrects the difference between image (downward) and camera
(upward) coordinates.
4. Homogeneous coordinates enable perspective projection and translation using
matrices.
5. Clip space stores points in homogeneous coordinates before perspective divide;
w_clip controls perspective and clipping.
6. NDC is the normalized cube after perspective divide; 1/z depth nonlinearity arises
here.
7. Points at infinity have w = 0 → used for ray directions.
8. NeRF skips clip space, applying NDC directly to rays for numerical stability.
✅ This is now a complete, self-contained explanation connecting:
Ray generation
Camera intrinsics
Coordinate systems (camera, world, NDC)
Homogeneous coordinates
Clip space and perspective
Depth nonlinearity
NeRF-specific simplifications
If you want, I can also draw a single diagram showing camera → NDC → rays with 1/z
depth so it becomes fully visual and intuitive.
Do you want me to do that?
represent the **same point** (for any nonzero `k`).1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 52/72

--- Page 53 ---
what do you mean they represent the same point?
Great question — this is the key subtle idea behind homogeneous coordinates.
When I said:
I meant the same point in ordinary 3D space after you convert back.
Letʼs unpack that very carefully.
1️⃣ How a homogeneous point becomes a 3D point
A homogeneous coordinate:
represents the actual 3D point:
This step is called dehomogenization.
2️⃣ Why scaling doesnʼt change the point
Now take any nonzero scalar .
Start with:
Scale it:
Now dehomogenize both.
Original: and  represent the same point (for any nonzero ) (x,y,z,w) (kx,ky,kz,kw) k
(x,y,z,w)
 ,   ,   (wx
wy
wz)
k
(x,y,z,w)
(kx,ky,kz,kw)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 53/72

--- Page 54 ---
Scaled:
✅ Exactly the same 3D point
Thatʼs what “they represent the same point” means.
3️⃣ Concrete numeric example
Take the 3D point:
One homogeneous representation:
Another:
Another:
Dehomogenize all of them:
✅ Same 3D point every time
4️⃣ Geometric intuition (very important)
Think of homogeneous coordinates as rays from the origin in 4D space.
All points along this ray: ,   ,   (wx
wy
wz)
 ,  ,  = (kwkx
kwky
kwkz) ,  ,  (wx
wy
wz)
(2,4,6)
(2,4,6,1)
(4,8,12,2)
(1,2,3,0.5)
(2/1,4/1,6/1)=(2,4,6)
(4/2,8/2,12/2)=(2,4,6)
(1/0.5,2/0.5,3/0.5)=(2,4,6)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 54/72

--- Page 55 ---
represent one single 3D point.
So a 3D point is not a single 4D vector — itʼs an entire equivalence class of vectors.
5️⃣ Why this is useful (not just abstract math)
This property allows:
Perspective projection to work via division by w
Clipping against ±w
Points at infinity (w = 0)
A single matrix pipeline for all transformations
Without this “same point under scaling” rule, none of that would work.
6️⃣ Why directions are special (w = 0)
If:
You cannot divide by w → no finite 3D point.
That means:
The point is at infinity
It represents a direction, not a location
This is why:
Ray origins use w = 1
Ray directions use w = 0
7️⃣ One-sentence takeaway
Homogeneous coordinates represent 3D points up to scale: multiplying all
components by the same nonzero number does not change the actual point after
dividing by w.(x,y,z,w), (2x,2y,2z,2w), (0.5x,0.5y,0.5z,0.5w)
(x,y,z,0)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 55/72

--- Page 56 ---
If you want, I can also:
Connect this directly to projective geometry
Show how this explains vanishing points
Tie it back explicitly to clip space and NDC
Sure
Great — Iʼll finish the picture by connecting homogeneous coordinates → projective
geometry → vanishing points → clip space & NDC.
This will make the “same point under scaling” idea feel inevitable, not arbitrary.
1️⃣ Homogeneous coordinates = projective geometry
(the big idea)
In Euclidean geometry, a point is a single coordinate:
In projective geometry, a point is an equivalence class:
All scaled versions represent the same geometric point.
Why?
Because the actual point is defined only after dehomogenization:css
( x, y, z )
Copy code
css
(x, y, z, w)  ~  (kx, ky, kz, kw)   for k ≠ 0
Copy code
(x/w, y/w, z/w)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 56/72

--- Page 57 ---
So projective geometry:
treats scale as meaningless
keeps only ratios
naturally models perspective
This is exactly how cameras work.
2️⃣ Points at infinity (why w = 0 exists)
In Euclidean space:
Parallel lines never meet
In projective geometry:
Parallel lines meet at infinity
That “point at infinity” is encoded as:
Why?
Because you cannot divide by w = 0 → the point has no finite location.
Direction vectors = points at infinity
Example:
These are not locations — they are directions.
This is why in graphics:
Points use w = 1css
(x, y, z, 0)
Copy code
scss
(1, 0, 0, 0) → direction  along + x
(0, 1, 0, 0) → direction  along + y
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 57/72

--- Page 58 ---
Directions / rays use w = 0
And why ray directions are unaffected by translation.
3️⃣ Vanishing points (the visual payoff)
Now letʼs connect this to what you see.
Example: railroad tracks
The rails are parallel in 3D
In an image, they converge to a point
That point is a vanishing point.
What is it mathematically?
It is the projection of a direction vector:
After projection:
w stays 0
the direction maps to a point at infinity in projective space
after normalization, it appears as a finite image location
So:
This only works because homogeneous coordinates allow w = 0.
4️⃣ Why scaling represents the same point (deep
intuition)
Think of homogeneous coordinates as describing a ray from the origin in 4D:scss
(dx, dy, dz, 0)
Copy code
Vanishing points are points at infinity made visible by perspective projection.
css1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 58/72

--- Page 59 ---
All points along this ray:
represent one geometric point.
Only the direction of the 4D vector matters — not its length.
Thatʼs why:
scaling doesnʼt change the point
division by w collapses the ray into a 3D point
5️⃣ How this explains clip space
Now we return to graphics.
After projection:
This is still projective space.
The GPU does not immediately divide by w_clip.
Why?
Because:
(x, y, z, w)  and (2x, 2y, 2z, 2w)  are the same point
clipping must respect that equivalence(x, y, z, w)
Copy code
scss
(x, y, z, w)
(2x, 2y, 2z, 2w)
(0.5x, 0.5y, 0.5z, 0.5w)
Copy code
scss
(x_clip, y_clip, z_clip, w_clip)
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 59/72

--- Page 60 ---
So clipping is done using:
These inequalities are scale-invariant.
That only works because points are defined up to scale.
6️⃣ Perspective divide = choosing a representative
When the GPU performs:
it is doing this:
Thatʼs all the perspective divide is.
Before divide:
After divide:
Same point. Different representation.−w_clip ≤ x_clip ≤ w_clip
−w_clip ≤ y_clip ≤ w_clip
−w_clip ≤ z_clip ≤ w_clip
Copy code
ini
x_ndc = x_clip / w_clip
y_ndc = y_clip / w_clip
z_ndc = z_clip / w_clip
Copy code“Choose the representative of this equivalence class where w = 1.”
scss
(2x, 2y, 2z, 2w)
Copy code
bash
(x/w, y/w, z/w)
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 60/72

--- Page 61 ---
7️⃣ Why w = -z gives perspective
In the projection matrix:
So after divide:
This is exactly the pinhole camera equation.
Perspective emerges automatically because:
farther points have larger |z|
dividing by z shrinks them
No special “perspective logic” is needed — projective geometry does it for free.
8️⃣ How NeRF fits perfectly into this
NeRF:
does not rasterize
does not clip triangles
does not need full homogeneous machinery
But it still relies on the same math:
ray directions ≈ points at infinity (w = 0)
ray origins ≈ finite points (w = 1)
NDC ≈ post–perspective divide space
depth nonlinearity ≈ 1/z
NeRF simply performs these steps analytically, instead of via matrices and clip space.py
ini
w_clip = -z_camera
Copy code
ini
x_ndc = x / -z
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 61/72

--- Page 62 ---
9️⃣ Final mental model (lock this in)
Homogeneous coordinates describe points up to scale
Scaling does not change the point
w = 0 encodes infinity (directions)
Perspective is just division by w
Clip space exists to safely manage this representation
NDC is the normalized result after choosing w = 1
One-sentence master takeaway
Homogeneous coordinates work because geometry cares about ratios, not scale —
and perspective, infinity, clipping, and ray directions all fall out naturally from that
single idea.
If you want, next we can:
walk one 3D point numerically through every space
show vanishing points in an actual image
connect this to multi-view geometry & epipolar lines
Just tell me which one you want.
Image plane at `z = -f` (negative because camera looks down −Z)
is f positive? Is the image plane before camera or after
camera?
Great question — this goes straight to camera sign conventions, and itʼs something that
everyone trips over at first.
Iʼll answer it carefully and unambiguously.1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 62/72

--- Page 63 ---
Short answer (then weʼll justify it)
Yes, the focal length f is positive.
In the OpenGL / NeRF camera convention, the image plane is in front of the
camera, at
The negative sign comes from the choice of coordinate axes, not from f being
negative.
1️⃣ Why f is always positive
The focal length is a physical distance:
Distance from the camera center (pinhole) to the image plane
Distances are non-negative by definition
So:
Always.
If you ever see a “negative focal length,” thatʼs not physics — itʼs a coordinate
convention issue.
2️⃣ Camera coordinate system being used
In NeRF and OpenGL-style camera space:
That means:
The camera is at the origin (0, 0, 0)z=−f
nginx
f > 0
Copy code
css
x → right
y → up
z → backward1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 63/72

--- Page 64 ---
The camera looks along −Z
Points in front of the camera have z < 0
This is the key fact.
3️⃣ Where is the image plane?
The image plane must be placed:
In front of the camera
Along the viewing direction
At a distance f from the camera center
Since the viewing direction is −Z, the image plane is at:
So:
f is positive
z = -f is negative because “forward” is negative Z
4️⃣ Why not put the image plane at z = +f?
You could, but then:
The camera would be looking down +Z
Or the image would be mirrored
Or all your projection equations would flip signs
Many computer vision systems (e.g., some OpenCV formulations) do use +Z forward —
they just adjust the math accordingly.
NeRF follows the OpenGL convention, so:
5️⃣ Pinhole projection (this explains everything)Copy code
z=−f
forward = −Z
image plane = z = −f1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 64/72

--- Page 65 ---
Let a 3D point in camera space be:
Image plane at:
By similar triangles:
So:
Same for y.
This is where the familiar formula comes from:
No negative focal length anywhere — only a negative z.
6️⃣ How this appears in NeRF code
In your ray-generation code:
This corresponds to:
Image plane at z = -1
Focal length implicitly absorbed into (i - cx)/fx
Ray points forward because z = -1 < 0(x  ,y  ,z  ),z  < c cc c0
z=−f
 =−fx  image
z  cxc
x  = imagef  −z cx  c
text
x_proj = x / -z
y_proj = y / -z
Copy code
python
dirs = [(i - cx)/fx, -(j - cy)/fy, - 1]
Copy code1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 65/72

--- Page 66 ---
Again:
f > 0
Image plane is in front of the camera
Negative z is just “forward”
7️⃣ Common source of confusion (important)
People sometimes say:
What they mean is:
The virtual image plane used for math is at z = -f
Not the physical sensor layout
This avoids image flipping and keeps equations simple
It does not mean the camera sees backward.
✅ Final clear statement
The focal length f is positive
The image plane is in front of the camera
It is at z = -f because the camera looks down −Z
The negative sign comes from the coordinate system, not from the optics
If you want next, I can:
compare this directly to OpenCVʼs +Z-forward convention
show how the equations change if the image plane were at +f
reconcile this with physical camera sensors
Just tell me 👍“The image plane is behind the camera in OpenGL”
The standard OpenGL perspective projection matrix is: P\=\
[1atan (θ/2)00001tan (θ/2)0000−f+nf−n−2fnf−n00−10\]P = \\begin{bmatrix} \\frac{1}{a
\\tan(\\theta/2)} & 0 & 0 & 0 \\\\ 0 & \\frac{1}{\\tan(\\theta/2)} & 0 & 0 \\\\ 0 & 0 & -\\frac{f+n}{…1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 66/72

--- Page 67 ---
Why do we have perspective projection matrix like this? 
Excellent question. This matrix looks mysterious until you see that every term is there
to satisfy a very specific geometric requirement.
Iʼll derive it from requirements, not magic, and explain why each row has exactly that
form.
Weʼll use the OpenGL / NeRF convention throughout.
The goal of the perspective projection matrix
We want a matrix  that transforms:
such that after perspective divide:
the following are true:
Required properties
1. Perspective:
2. Correct field of view (FOV)
3. Correct aspect ratio
4. Near and far planes map to fixed NDC z values
5. Points in front of camera have positive w
Everything in the matrix exists to enforce one of these.P
css
camera space  →  clip space
Copy code
css
clip space / w  →  NDC
Copy code
x  = ndc ,y  =−zx
ndc −zy1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 67/72

--- Page 68 ---
Step 1: Force perspective with w = -z
This is the most important step.
We want the perspective divide to produce:
That means:
So the last row must be:
Because:
✅ This single row creates perspective.
Step 2: Scale x and y by field of view
We want points on the edge of the view frustum to map to:
Vertical FOV
Let:
vertical FOV = 
near plane at 
At the top of the frustum:x  = ndc =w  clipx  clip
 −zx
w  = clip−z
[00−10]
w  = clip0⋅x+0⋅y−1⋅z+0⋅1=−z
ini
x_ndc = ±1
y_ndc = ±1
Copy code
θ
z=−n
y=ntan(θ/2)1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 68/72

--- Page 69 ---
After projection:
At the near plane:
We want:
So the y scaling term must be:
Aspect ratio correction (x direction)
If:
Then horizontal FOV is wider by factor .
So x must be scaled by:
This gives the first two rows:
Step 3: Mapping depth (near & far planes)y  = ndc =w clipy  clip
 −zAy
z=−n⇒y  = ndc =nAntan(θ/2)Atan(θ/2)
y = ndc1⇒A= tan(θ/2)1
 tan(θ/2)1
a=  heightwidt h
a
 atan(θ/2)1
    [ atan(θ/2)1
00
 tan(θ/2)10
00
0]1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 69/72

--- Page 70 ---
This is the trickiest part.
We want:
Camera z NDC z (OpenGL)
Assume:
After divide:
Apply constraints
Near plane: 
Far plane: 
Solve the system
Set them equal:
Solving:z=−n +1
z=−f −1
z  = clipAz+Band w  = clip−z
z = ndc −zAz+B
z=−n
1=  ⇒n−An+BB=n(1+A)
z=−f
−1=  ⇒f−Af+BB=f(A−1)
n(1+A)=f(A−1)
A=− f−nf+n
f1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 70/72

--- Page 71 ---
This gives the third row:
Step 4: Assemble the full matrix
Putting everything together:
What each row does (intuitively)
Row Purpose
Row 1 Horizontal FOV + aspect ratio
Row 2 Vertical FOV
Row 3 Nonlinear depth mapping
Row 4 Perspective divide (w = -z)
Why this matrix has to look like this
Because:
Perspective requires division by z
Division by z requires homogeneous coordinates
Homogeneous coordinates require w = -zB=− f−n2fn
[00−  −f−nf+n
 ]f−n2fn
P=        atan(θ/2)1
0
0
00
 tan(θ/2)1
0
00
0
− f−nf+n
−10
0
− f−n2fn
01/5/26, 9:00 AM ChatGPT
https://chatgpt.com 71/72

--- Page 72 ---
FOV requires tan(θ/2)
Depth mapping requires two constraints → two constants
There is no freedom left after imposing the geometry.
Key insight (very important)
One-sentence takeaway
This matrix exists because perspective is “divide by z,” and homogeneous
coordinates are the only way to encode that division using linear algebra before the
divide actually happens.
If you want next, I can:
derive the DirectX version side-by-side
show how this matrix changes for infinite far plane
derive it starting from similar triangles only
Just tell me which one you want.The perspective projection matrix is not arbitrary — it is the unique matrix that
satisfies the pinhole camera model, perspective division, and near/far depth
constraints simultaneously.
1/5/26, 9:00 AM ChatGPT
https://chatgpt.com 72/72


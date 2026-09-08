# Vision & Graphics Notes

An Obsidian vault of personal notes on computer vision and computer graphics —
the shared math (pinhole cameras, projective geometry) and where the two fields
diverge and reconverge (rendering vs. reconstruction, meeting again in NeRF).

## Graphics & Vision Are Two Sides of the Same Coin

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

See [[reference-tables]] in `concepts/` for a quick-reference summary across
every note, and `CLAUDE.md` for how this vault is organized and maintained.

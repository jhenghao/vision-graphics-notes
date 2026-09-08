# Vision & Graphics Notes Vault

This is an Obsidian vault where I'm learning computer vision & computer
graphics concepts (pinhole camera model, homogeneous coordinates, the
graphics pipeline, NeRF, and more over time) with help from Claude Code.

## Structure

- `sources/` — raw material: pasted ChatGPT conversations, exported PDFs,
  process docs. One file per source. **Never edit these after saving** —
  they're the historical record I cite, not something to rewrite.
- `concepts/` — one Markdown note per concept. This is the actual knowledge
  base I study from. Files are named in kebab-case matching the concept
  (e.g. `clip-space.md`, `pinhole-camera-model.md`).
- `templates/concept-template.md` — template for new concept notes.

## Workflow

1. I drop a new source into `sources/` (a ChatGPT transcript, a clipped
   webpage, a PDF export, etc.).
2. I ask you to read it and draft or update the relevant concept note(s) in
   `concepts/`, using `templates/concept-template.md` as the shape.
3. You cross-link related concepts with `[[wikilink]]` syntax and cite the
   source file at the bottom of the note under `## Sources`.
4. I review and rewrite the note in my own words in Obsidian — that's where
   the actual learning happens, so draft explanations for me to edit, don't
   try to produce a finished, publishable page.

## Purpose, and why it shapes everything below

This project is for *learning*, not documentation. What I actually need
from you is:

- **Reasoning, not just results.** Don't state a formula and then explain
  what it means — derive it. Show the chain of thought that gets from
  "here's a problem I can't solve" to "here's why this concept solves it,"
  including false starts if they're instructive.
- **Connections, not just links.** When you relate two concepts, say *how*
  they relate — "X is the specialization of Y for the OpenGL convention,"
  "X is what Y looks like after the perspective divide." A bare
  `[[wikilink]]` with no explanation is a missed opportunity.
- **Examples worked through, not stated.** Show the reasoning at each step
  of a worked example (numeric or code), not just the arithmetic.

## Conventions

- Follow `templates/concept-template.md`'s section order: Motivating
  Question → Reasoning/Derivation → Formal Statement → Worked Example →
  Connections → Open Questions → Sources. The formula/definition comes
  *after* the reasoning that leads to it, never before.
- **One concept = one file.** If a new source touches a concept that
  already has a note, UPDATE that file (deepen the reasoning, add an
  example, add the source link) — never create a second note for the same
  concept.
- Always use `[[wikilink]]` syntax for cross-links.
- A note's `status` frontmatter field moves `seed` → `growing` → `mature`
  as it fills in. Update it when you substantially edit a note.
- Tag each note by its sub-area, matching the vault's five areas: `camera`
  (the shared pinhole/intrinsics foundation), `homogeneous-coordinates`,
  `graphics-pipeline` (the forward 3D→2D path), `vision` (the inverse
  2D→3D path), `nerf`. The one cross-reference note (`reference-tables.md`)
  is tagged `reference` instead of a sub-area.
- If a source only states a fact/formula without justifying it, don't just
  transcribe it — reason out *why* it's true (or ask me to work through it
  together) before writing the note.
- If it's not obvious which concept(s) a new source relates to, ask me
  rather than guessing.
- Keep this file itself simple. Don't add new folders, tag schemes, or
  automation here unless I ask for it.

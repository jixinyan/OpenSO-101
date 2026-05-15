# Media

Logos, banner images, demo videos, screenshots, and any other binary
assets that the GitHub README, the docs site, or external promotional
material need to reference.

## Conventions

- **Logos:** `logo.png` (square or near-square, ~420px wide). Used by the
  top-level README hero.
- **Banners:** `banner.png` (16:9, 1200×630 recommended — also serves as
  the GitHub social-preview card).
- **Screenshots:** `screenshot_<feature>.png` — e.g. `screenshot_teleop.png`.
- **Demo videos:** `demo_<feature>.mp4` or `.gif`. Keep `.gif` under 10 MB.
- **Architecture diagrams:** `arch_<topic>.png` (or `.svg`).

## Why a top-level `media/` and not `docs/`

GitHub renders files in `media/` via raw URLs without any docs-tooling
involvement, so README image tags resolve fast both in the GitHub UI and
in mirrors (gitea, sourcehut, etc.). Keep doc-specific images in
`docs/` only when they are tied to a particular doc page.

## Adding a new asset

1. Drop the file in here.
2. Reference it from README/docs with `media/<filename>` (relative) so it
   works on GitHub and on a locally-served docs build.
3. Keep binaries small (<2 MB) when possible; for larger files prefer a
   `git lfs` track or an external CDN.

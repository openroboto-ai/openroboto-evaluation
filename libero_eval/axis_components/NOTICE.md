# Randomization component sources

This internal package assembles source snapshots, texture assets and evaluator
adapters. It is not an AXIS-published package or an endorsement of this evaluation
protocol. The project maintainer has confirmed permission to redistribute the
AXIS source snapshots included in this release. This does not imply that all
upstream generation scripts are publicly available.
Relocating these files does not change their authorship or licensing.

`camera.py` and `reset.py` are unmodified numerical function extractions from
the sources pinned in `SOURCES.json`. `material.py` contains the original
material-coefficient jitter and image luminance guard functions. Application
entry points, database access and configuration secrets were not copied.
The adapter and validation around these routines are maintained by validator.

Camera source: AXIS-Render-Robosuite, `robosuite_render.py`, recovered snapshot
`code_fixed_authority_20260723`; source SHA-256 and extraction names are recorded.
Reset source: AxisAIOrg/axis-mvp, commit
`66c7ff0151a5e90b7619f2fd17e13712602e4389`,
`backend/app/replay/submit_nonce_verify.py`.

`profiles/franka_v6.json` freezes the camera rules and **one specific surface
palette** from the released task 1952, attempt 1925874. Other room recipes use
different palettes; this is not a universal texture configuration for AXIS.
The material base coefficients were recovered from recorded random seeds and
outputs, then verified against 158 original samples with zero numerical error.

`visual_scene.py` is the complete, unmodified room generator from the recorded snapshot.
`config/cameras/franka.json` and `config/scenes/kujiale_mujoco_v2.json` are
upstream files with host-specific provenance paths removed; numerical settings
are unchanged. `scene_xml.py` extracts original XML helper
definitions without the upstream service/database imports. Their origins,
commit, extraction names and SHA-256 pins are recorded in `SOURCES.json`.

The complete library has 52 surface/material entries across all room themes.
Their source metadata and base coefficients come directly from the recorded
Poly Haven and curated-tabletop manifests. All required JPEG files in
`assets/` match those manifests' prepared SHA-256 digests; the textures are
CC0. `manifests/scene_materials.json` is the relevant union, with
`base_material` aliasing the upstream `material` field for the native adapter.

`robosuite/` contains the original Robosuite 1.5.2 TableArena source, XML,
required textures, license and authors from the renderer's geoadapt install.
`table_arena_adapter.py` keeps the original class unchanged with isolated
imports. `table_arena_base.py` is our minimal XML-only base implementation;
it resolves asset paths and performs the same collision coloring without
importing the Robosuite simulation stack.

Only reviewed profiles listed by digest in `SOURCES.json` are loadable. The
native adapter targets the validator's MuJoCo 3.11.0 environment; it does not
claim pixel-equivalence to the upstream MuJoCo 3.1.1/Robosuite renderer.

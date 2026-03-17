# Codex instructions — next steps for EBUS_simulator

Use these instructions as the **task brief / system-style handoff** for the next Codex run.

---

## Objective

Advance the repo from its current state into the next bounded milestone **without reopening already completed phases**.

The repo already has:
- portable manifest roots
- explicit render engine split
- a `localizer` renderer
- a first `physics` renderer
- artifact controls and debug-map export
- metadata sidecars and basic eval summaries
- CI smoke coverage

The next work should focus on:
1. making `review-presets` physics-aware and reviewer-friendly
2. adding a lightweight human review rubric / comparison output
3. tightening documentation so the repo guidance matches the actual implementation
4. only then considering desktop app work

Do **not** redo scaffold, loader, validation, pose generation, engine split, or initial physics-renderer implementation unless needed for a narrowly scoped fix.

---

## Hard constraints

This project is:
- **CP-EBUS / linear / convex-probe only**
- preset-driven
- local desktop / local CLI only
- outside 3D Slicer at runtime

This project is **not** for:
- radial EBUS
- full bronchoscopy navigation
- Slicer runtime module work
- web deployment
- scoring / quiz mode in this pass
- a GAN-first rewrite of the rendering path

Do not broaden scope.

---

## Current state you must assume

Treat the current repo state as follows:
- geometry and preset pose generation are implemented and test-covered
- `localizer_renderer.py` and `physics_renderer.py` exist
- physics rendering supports tunable artifact controls and debug-map export
- physics metadata includes basic evaluation summaries
- CI exists
- desktop app does not exist yet

Important implication:
- the next bottleneck is **review workflow + calibration support**, not renderer scaffolding

Also assume the docs are stale in places and need synchronization.

---

## Primary task

Implement a bounded **physics-centric review pass**.

### Goal
Make `review-presets` generate reviewer-ready bundles that incorporate the newer physics outputs, debug maps, and evaluation summaries, while keeping the workflow deterministic and easy to compare across presets and approaches.

### Deliverables

#### 1) Extend `review-presets`
Update the review workflow so that for each preset / approach it can emit:
- localizer render
- physics render
- sidecar JSON metadata
- physics debug maps, when requested
- extracted `eval_summary`
- a per-preset review summary entry

#### 2) Add structured review bundle output
Create a stable output layout under the chosen review directory, for example:

- `review_index.json`
- `review_index.csv`
- `review_index.md`
- `preset_name/approach_name/localizer.png`
- `preset_name/approach_name/physics.png`
- `preset_name/approach_name/metadata.json`
- `preset_name/approach_name/eval_summary.json`
- `preset_name/approach_name/debug_maps/...`

The exact structure can vary, but it must be deterministic and easy for a human reviewer to inspect.

#### 3) Add a lightweight human review rubric output
Add either:
- a generated markdown review sheet, or
- a checked-in rubric template that can be copied into review bundles

The rubric should keep scoring simple and descriptive, for example:
- airway wall plausibility
- vessel wall/lumen plausibility
- node conspicuity
- overall resemblance to CP-EBUS
- comments

Do not build a full scoring engine.

#### 4) Synchronize docs
Update the repo docs so they reflect current reality.
At minimum review and correct:
- `README.md`
- `Agents.md`
- `Plans.md`
- any package description text that still describes the repo as only a scaffold

Make the docs clearly state that the repo already has:
- portable manifest roots
- engine split
- localizer renderer
- first physics renderer
- CI smoke workflow

Docs should also state that the next active milestone is the **physics-aware review/calibration layer**.

---

## Secondary task (only if it stays narrowly scoped)

If there is a clean seam, reduce review-related coupling by moving review-specific extraction/packaging code out of heavier shared rendering orchestration.

Do this only if it is a small refactor with clear value.
Do **not** perform a broad architectural rewrite.

---

## File targets

Expect to work mainly in:
- `src/ebus_simulator/review.py`
- `src/ebus_simulator/review_cli.py`
- `src/ebus_simulator/render_cli.py`
- `src/ebus_simulator/render_all_cli.py` (only if needed)
- `src/ebus_simulator/rendering.py` (only if needed)
- `README.md`
- `Agents.md`
- `Plans.md`
- tests for review bundle generation and rubric / summary output

Possible additions:
- `src/ebus_simulator/review_models.py`
- `src/ebus_simulator/review_rubric.py`
- snapshot / fixture-based review bundle tests

Prefer adding small focused helpers over growing a monolithic review module.

---

## Acceptance criteria

The task is complete only if all of the following are true:

1. `review-presets` can generate a bundle that includes both localizer and physics outputs.
2. Physics metadata or its extracted review artifacts include `eval_summary` in a reviewer-usable form.
3. Physics debug maps can be bundled into the review export when requested.
4. The review output is deterministic and organized predictably by preset and approach.
5. Station 7 `lms` and `rms` remain distinct throughout the review export.
6. README / guidance docs no longer claim that engine split, portable roots, CI, or the physics renderer are unimplemented.
7. Focused tests pass.
8. Existing render smoke behavior is not broken.

---

## Explicitly out of scope

Do not do any of the following in this pass unless the user explicitly asks:
- build the PySide6 desktop app
- add free navigation
- add radial EBUS
- replace the physics renderer with a learned model
- implement real-image ingestion or dataset labeling tools
- build a scoring backend
- redesign the entire rendering architecture
- create new meshes or re-segment anatomy

---

## Test and validation expectations

Run focused tests that directly cover the changed surface.
At minimum add or run tests for:
- review bundle generation
- deterministic review index output
- physics artifact bundling
- station 7 dual-approach review separation
- doc/example command correctness if practical

Also run the existing smoke commands if they are still cheap enough.

Preferred evidence to return:
- exact files changed
- exact commands run
- exact test results
- one example review output tree
- known limitations remaining after the pass

---

## Suggested implementation order

1. Inspect current `review.py` / `review_cli.py` and find the narrowest seam for bundling physics artifacts.
2. Define stable review bundle data structures.
3. Implement per-preset review artifact collection.
4. Add index generation (`json` + `csv` or `md`).
5. Add rubric output.
6. Update docs.
7. Run focused tests and smoke checks.

---

## User-owned inputs that Codex should not invent

Codex should not fabricate or guess these:
- real CP-EBUS reference images
- expert review scores
- new airway meshes
- new segmentation masks
- changes to curated preset contacts/targets without evidence

If any of those seem necessary, Codex should note them as follow-up needs rather than inventing them.

---

## Nice-to-have if trivial

Only if it is truly low-risk and small:
- include thumbnail links or markdown image tables in the review index
- emit a one-page summary of physics parameter settings used for the batch
- include warnings in the review summary for presets with validation warnings

Do not let nice-to-haves expand the scope.

---

## Expected response format from Codex

When done, return:
1. brief implementation summary
2. exact files changed
3. exact commands run
4. exact test results
5. sample review bundle layout
6. remaining limitations / next recommended step


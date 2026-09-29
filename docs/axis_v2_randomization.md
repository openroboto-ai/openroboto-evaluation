# AXIS randomization in native MuJoCo

AXIS V2.0 is the OpenRoboto evaluator's combined randomization protocol for AXIS
tasks. It enables each frozen task's supported components. The evaluator uses
native MuJoCo and OSMesa; it does not require Isaac Sim or the Robosuite simulation stack. The implementation and public
release share one source tree.

## Current releases

| Bundle | Scope | Instances |
| --- | --- | ---: |
| `axis_v2.0.tar.zst` | Original 30-task benchmark | 600 |
| `axis_v20260928.33.tar.zst` | Full 1,233-task library, same combined protocol | 24,660 |

The full library is a larger task pool, not a different randomization method.
Both bundles and their `*-release.json` receipts are under
[`configs/benchmarks`](../configs/benchmarks). Receipts pin archive, manifest and
validation hashes. Each bundle includes the benchmark, randomization plan,
capability bindings, complete task payloads and native validation evidence.

```bash
uv run python libero_eval/run_eval.py \
  --benchmark axis_v2.0 --axis-randomization-seed 20260928 \
  --model /path/to/checkpoint --commit-id local \
  --backbone pi0.5 --num-trials 20 --gpus 0
```

The named entry point verifies and atomically extracts the bundle into
`.cache/axis/releases/axis_v2.0`. A changed archive or cached payload fails
validation. It cannot override the published randomization manifest.
An independent comparison must use the same environment seed and policy seed.
The worker takes the environment seed from the queue and rejects a missing or
invalid seed before downloading a model. See [the worker guide](../benchmark_worker/README.md).

For the full task library, prepare its verified bundle and pass its manifests:

```bash
uv run python -c 'from libero_eval.axis_release import prepare_release; print(prepare_release("axis_v20260928.33"))'
uv run python libero_eval/run_eval.py --benchmark axis \
  --axis-manifest .cache/axis/releases/axis_v20260928.33/benchmark.json \
  --axis-randomization-manifest .cache/axis/releases/axis_v20260928.33/randomization.json \
  --axis-randomization-seed 20260928 \
  --model /path/to/checkpoint --commit-id local --num-trials 20 --gpus 0
```

## Components and boundaries

| Component | Implementation |
| --- | --- |
| Object reset | Task-bound position/rotation ranges, PRNG, mocap handling and configured swaps |
| Front camera | Table-sector position, look-at/FOV sampler and visibility checks |
| Wrist camera | Optical-frame translation, rotation and FOV sampler when its mount matches |
| Background and geometry | 8,640-recipe library; full arena only where original task geometry remains aligned |
| Surface materials | 52-entry library, original textures, coefficient jitter and luminance guard |
| Physics and lighting | No added mass, friction or lighting noise |

All components supported by a task are enabled together. There is no factorized
score, reduced sampling range or model-dependent fallback. A task lacking one
component still uses its other compatible components. The native adapter runs
physics in the original model; the visual model is only forwarded, never stepped.

For tasks retaining source geometry, the renderer places new
backgrounds relative to the source robot base. This avoids the added rug/room
occluding the original work surface. The correction does not change physical
states, camera poses, material samples, collision geometry or success checkers.

The current 30-task release supports front-camera, background and floor-material
variation for all 30 tasks. Tasks 501–506, 514 and 757 additionally have matched
physical resets and wrist cameras; tasks 22 and 757 support the complete arena.
Observations are 320×180 before the policy's existing aspect-preserving resize.
All 30 tasks therefore run twenty different instances. An unsupported wrist
camera does not make a task wholly fixed.

The full library retains visual variation for all 1,233 tasks and physical
variation for 995. Seven task-bound reset distributions that could satisfy their
checker immediately are disabled as whole components in the frozen bindings;
visual components remain enabled. No individual seed is replaced to obtain an
easier or valid episode. The builder rejects non-finite states, MuJoCo warnings,
already-successful resets and non-repeatable render/state evidence.

## Selection and scoring

The subnet's `sha256-cycle-permutation-v1` contract uses the queue seed and task
ID to visit every frozen instance before repeating. It does not use model identity
or observed outcomes. Each episode records its instance ID, payload hash, seed,
selection digest and actual reset/render metadata. Pinned numeric samplers and
ranges are preserved; these frozen benchmark seeds are not a claim that exactly
the same rendered episodes appeared in training data.

A randomized task runs twenty trials. A task with no supported randomization
runs once. The score is the equal mean of per-task success rates, not a weighting
by trial count. Incomplete execution is an infrastructure error, not a zero or a
partial model score. The original 30-task V1.0 bundle remains available as a fixed
control and as source definitions for import/replay tools; it is not the current
randomized leaderboard protocol.

## Provenance, rebuilding and validation

Numerical routines, room definitions, textures and their source notices live in
the evaluator's internal [`axis_components`](../libero_eval/axis_components/NOTICE.md) package.
[`SOURCES.json`](../libero_eval/axis_components/SOURCES.json) records source revisions,
file hashes and extracted function names. Host-specific provenance paths and
archive owner metadata are removed in the public export. Distribution receipts
pin the sanitized archive and validation bytes; task payloads, randomization
parameters and numeric validation results remain unchanged. Frozen source-profile
identifiers are mapped to verified sanitized profile bytes in SOURCES.json.
Runtime imports use the vendored files, never another user's checkout.
The adapter targets MuJoCo 3.11.0 and does not claim pixel equivalence
to the upstream MuJoCo 3.1.1 / Robosuite renderer.

Frozen manifests retain historical identifiers containing `official` for byte
and hash compatibility with existing evaluation records. These identifiers do
not designate an AXIS-endorsed protocol or an upstream public release.

The current 600-instance benchmark was freshly reset, rendered and repeated in
full. For the full library, 1,955 instances were freshly verified for the coordinate
correction; 22,705 unchanged instances reuse prior validation only after checks of
their unchanged inputs and execution paths. The full-library receipt and
`validation.json` explicitly record this reuse. Native validation establishes
executable, repeatable instances, not policy success or universal task solvability.

To build a new frozen release, first run
[`plan_axis_randomized_release.py`](../tools/plan_axis_randomized_release.py) against
hash-pinned source tasks and matching reset records. Then run
[`build_axis_randomized_release.py`](../tools/build_axis_randomized_release.py) with
those bindings. The builder validates every instance and publishes no directory
on failure. Capability planning alone is not release validation. A new task set
requires its own validated release identity; never overwrite an existing version.

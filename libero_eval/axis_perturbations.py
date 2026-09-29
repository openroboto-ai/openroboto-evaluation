"""Native MuJoCo adapter for pinned AXIS reset and V2 rendering routines.

Task/scene bindings are explicit and part of the frozen payload. This module
does not infer surfaces from names, invent perturbation ranges, or replace a
missing upstream scene with a different one. No Isaac Sim or Robosuite runtime
is required. Pinned numeric routines and assets live in the internal
axis_components package; its NOTICE.md records their source provenance.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

if __package__:
    from .axis_components import camera, material, reset
else:
    from axis_components import camera, material, reset

COMPONENT_ROOT = pathlib.Path(__file__).resolve().parent / "axis_components"

RESET_REVISION = "66c7ff0151a5e90b7619f2fd17e13712602e4389"
PROFILE_PATH = COMPONENT_ROOT / "profiles" / "franka_v6.json"

# Frozen payload keys/modes and texture paths retain their historical spelling
# for hash and replay compatibility; they do not assert an upstream release.


def load_visual_profile(expected_hash: str) -> dict[str, Any]:
    """Resolve a reviewed upstream palette by content, never by a mutable URL."""
    root = COMPONENT_ROOT
    sources = json.loads((root / "SOURCES.json").read_bytes())
    for filename, digest in sources["profiles"].items():
        # Frozen manifests retain the source identifier; verify distributed bytes.
        source_digest = sources.get("profile_source_hashes", {}).get(filename)
        if expected_hash == digest or (source_digest is not None and expected_hash == source_digest):
            raw = (root / filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                break
            return json.loads(raw)
    raise ValueError("AXIS visual profile hash mismatch or unavailable profile")


def install_randomization_assets(config: dict[str, Any], asset_root: pathlib.Path) -> None:
    """Stage only hash-named, locally vendored CC0 assets; never fetch at reset."""
    visual = config.get("visual")
    if visual is None:
        return
    from axis_runtime import _atomic_write

    if visual.get("mode") in {"official_franka_v6", "official_franka_components", "official_franka_components_v2"}:
        _verify_vendor()
        entries = json.loads((COMPONENT_ROOT / "manifests/scene_materials.json").read_bytes())
        paths = {key: f"official_textures/{asset['sha256']}.jpg" for key, asset in entries.items()}
    else:
        entries = load_visual_profile(visual.get("profile_sha256"))["materials"]
        paths = visual.get("assets")
    if not isinstance(paths, dict) or set(paths) != set(entries):
        raise ValueError("visual.assets must pin the complete texture palette")
    for key, asset in entries.items():
        relative = f"official_textures/{asset['sha256']}.jpg"
        if paths[key] != relative:
            raise ValueError(f"texture {key} must use {relative}")
        source = COMPONENT_ROOT / "assets" / f"{asset['sha256']}.jpg"
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError(f"vendored texture hash mismatch: {key}")
        target = asset_root / relative
        if not target.resolve().is_relative_to(asset_root.resolve()):
            raise ValueError("texture cache path escapes its root")
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == asset["sha256"]:
            continue
        _atomic_write(target, data)


def _exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} requires exactly {sorted(keys)}")


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{label} requires {length} finite numbers")
    return result


def _named(model: Any, kind: str, name: Any) -> int:
    if not isinstance(name, str) or not name:
        raise ValueError(f"AXIS {kind} binding requires a name")
    try:
        return int(getattr(model, kind)(name).id)
    except KeyError as exc:
        raise ValueError(f"AXIS scene binding has no {kind} {name!r}") from exc


def _validate_domain(model: Any, cfg: Any) -> None:
    """Reject malformed/unsupported input before upstream can silently skip it."""
    if cfg is None:
        return
    if not isinstance(cfg, dict) or set(cfg) - {"objects", "swap_positions"}:
        raise ValueError("AXIS backend reset supports null, {}, objects and swap_positions only")
    objects = cfg.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("domain_randomization.objects must be an object")
    for name, entry in objects.items():
        if not isinstance(entry, dict) or set(entry) - {"mjcf", "pos_delta", "rot_delta"}:
            raise ValueError(f"unsupported domain_randomization.objects.{name} fields")
        body = reset._resolve_body_id(model, name, entry)
        if body < 0:
            raise ValueError(f"randomization object {name!r} is missing from this scene")
        if reset._find_free_joint_for_body(model, body) is None and reset._mocap_id_for_body(model, body) is None:
            raise ValueError(f"randomization object {name!r} has neither a free joint nor a mocap body")
        for field in ("pos_delta", "rot_delta"):
            if field not in entry:
                continue
            ranges = reset._parse_vec3_ranges(entry[field])
            if ranges is None or not np.isfinite(ranges).all() or any(lo > hi for lo, hi in ranges):
                raise ValueError(f"{name}.{field} must contain three finite, ordered ranges")
    if "swap_positions" in cfg:
        swap = cfg["swap_positions"]
        if isinstance(swap, dict):
            if set(swap) - {"pairs", "probability"}:
                raise ValueError("unsupported swap_positions fields")
            pairs, probability = swap.get("pairs"), swap.get("probability", 0.5)
        else:
            pairs, probability = swap, 0.5
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("swap_positions.probability must be in [0, 1]")
        if not isinstance(pairs, list):
            raise ValueError("swap_positions.pairs must be a list")
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(n not in objects for n in pair):
                raise ValueError("swap_positions must reference pairs of configured objects")
            for name in pair:
                body = reset._resolve_body_id(model, name, objects[name])
                if reset._find_free_joint_for_body(model, body) is None:
                    raise ValueError(f"swap_positions object {name!r} must have a free joint")


def _verify_vendor() -> str:
    root = COMPONENT_ROOT
    raw = (root / "SOURCES.json").read_bytes()
    for filename, expected in json.loads(raw)["files"].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned AXIS upstream routine changed: {filename}")
    return hashlib.sha256(raw).hexdigest()


def has_domain_randomization(model: Any, cfg: Any) -> bool:
    """Inspect the distribution, not a small sample of lucky seeds."""
    import mujoco

    _validate_domain(model, cfg)
    if cfg is None:
        return any(
            model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and reset._should_apply_default_free_joint_randomization(model.body(int(model.jnt_bodyid[j])).name)
            for j in range(model.njnt)
        )
    for entry in cfg.get("objects", {}).values():
        for key in ("pos_delta", "rot_delta"):
            if any(lo != hi for lo, hi in (reset._parse_vec3_ranges(entry.get(key)) or [])):
                return True
    swap = cfg.get("swap_positions") or {}
    pairs = swap if isinstance(swap, list) else swap.get("pairs", [])
    probability = swap.get("probability", 0.5) if isinstance(swap, dict) else 0.5
    return bool(pairs) and 0 < probability < 1


def sample_cameras(
    profile: dict[str, Any], geometry: dict[str, Any], *, camera_seed: int, replica_id: int
) -> dict[str, Any]:
    """Use the original field seeds, rotation order and visibility sampler."""
    pair_seed = camera.stable_seed(camera_seed, "randomized_pair", replica_id)
    result = {}
    if geometry.get("front_enabled", True):
        result["frontview"] = camera._sample_front_base_table_sector_lookat(
            profile=profile["camera_profiles"]["frontview"],
            camera_seed=camera.stable_seed(pair_seed, "frontview"),
            base_world_position=_finite_vector(geometry["base_position"], 3, "base_position"),
            base_world_quat_wxyz=_finite_vector(geometry["base_quat_wxyz"], 4, "base_quat_wxyz"),
            table_center_xy_world=_finite_vector(geometry["table_center_xy"], 2, "table_center_xy"),
            table_top_z_world=float(geometry["table_top_z"]),
            table_full_size=_finite_vector(geometry["table_full_size"], 3, "table_full_size"),
        )
    if geometry.get("wrist_base") is not None:
        wrist = geometry["wrist_base"]
        result["wrist"] = camera._sample_wrist_optical_pose_box(
            profile=profile["camera_profiles"]["wrist"],
            camera_seed=camera.stable_seed(pair_seed, "wrist"),
            base_position=_finite_vector(wrist["position"], 3, "wrist position"),
            base_quat_wxyz=_finite_vector(wrist["quat_wxyz"], 4, "wrist quaternion"),
            base_fovy_deg=float(wrist["fovy_deg"]),
        )
    return result


def sample_material(profile: dict[str, Any], surface: str, seed: int) -> dict[str, Any]:
    """Image-family branch of upstream _sample_and_apply_surface_material_family."""
    rng = np.random.RandomState(seed)
    candidates = profile["surface_asset_ids"][surface]
    index = int(rng.randint(len(candidates)))
    asset = profile["materials"][surface + "/" + candidates[index]]
    coefficients = material._jitter_material(rng, tuple(asset["base_material"]))
    # Preserve the upstream image branch's remaining random draws.
    rng.uniform()
    rng.uniform(-0.08, 0.08)
    return {"asset": asset, "surface_candidate_index": index, "material": coefficients.tolist()}


class AxisRandomizer:
    """One immutable frozen instance, reapplied from the same baseline on reset."""

    def __init__(
        self,
        model: Any,
        config: dict[str, Any],
        *,
        asset_root: pathlib.Path,
        task_id: int,
        task_name: str,
        width: int,
        height: int,
        resolved_profile: dict[str, Any] | None = None,
    ) -> None:
        _exact_keys(
            config,
            {
                "schema_version",
                "upstream_reset_revision",
                "submit_nonce",
                "domain_randomization",
                "object_order",
                "visual",
            },
            "official_randomization",
        )
        if (
            type(config["schema_version"]) is not int
            or config["schema_version"] != 1
            or config["upstream_reset_revision"] != RESET_REVISION
        ):
            raise ValueError("unsupported AXIS reset contract")
        self.nonce = reset._normalize_submit_nonce_text(config["submit_nonce"])
        self.config = config
        self.model = model
        self.sources_sha256 = _verify_vendor()
        _validate_domain(model, config["domain_randomization"])
        cfg = config["domain_randomization"]
        objects = (cfg or {}).get("objects", {})
        order = config["object_order"]
        if (
            not isinstance(order, list)
            or any(not isinstance(n, str) for n in order)
            or len(order) != len(set(order))
            or set(order) != set(objects)
        ):
            raise ValueError("object_order must explicitly list each configured object exactly once")
        # JSON object order is not covered by canonical hashes. The upstream PRNG
        # consumes objects in insertion order, so freeze that order as an array.
        self.domain = cfg if not objects else {**cfg, "objects": {name: objects[name] for name in order}}
        self.visual = config["visual"]
        self.profile = resolved_profile
        self.textures: dict[str, np.ndarray] = {}
        self.targets: dict[str, list[tuple[int, int, int]]] = {}
        if self.visual is not None:
            self._prepare_visual(asset_root, task_id, task_name, width, height)

    def _prepare_visual(self, asset_root: pathlib.Path, task_id: int, task_name: str, width: int, height: int) -> None:
        v = self.visual
        _exact_keys(
            v,
            {
                "profile_sha256",
                "global_seed",
                "attempt_id",
                "variant_id",
                "replica_id",
                "front_camera",
                "wrist_camera",
                "reference_body",
                "table_center_xy",
                "table_top_z",
                "table_full_size",
                "surfaces",
                "assets",
            },
            "visual",
        )
        if self.profile is None:
            self.profile = load_visual_profile(v["profile_sha256"])
        if width * 9 != height * 16:
            raise ValueError("AXIS V2 camera profile requires a 16:9 render size")
        for key in ("global_seed", "attempt_id", "variant_id", "replica_id"):
            if type(v[key]) is not int or v[key] < 0:
                raise ValueError(f"visual.{key} must be a non-negative integer")
        self.variant_seed = camera.stable_seed(v["global_seed"], task_id, v["attempt_id"], task_name, v["variant_id"])
        self.front_id = None if v["front_camera"] is None else _named(self.model, "camera", v["front_camera"])
        if self.front_id is not None and (
            self.model.cam_bodyid[self.front_id] != 0 or int(self.model.cam_mode[self.front_id]) != 0
        ):
            raise ValueError("frontview binding must be a fixed world camera")
        self.reference_id = _named(self.model, "body", v["reference_body"])
        self.wrist_id = None if v["wrist_camera"] is None else _named(self.model, "camera", v["wrist_camera"])
        self.wrist_base = None
        if self.wrist_id is not None:
            if self.wrist_id == self.front_id or int(self.model.cam_mode[self.wrist_id]) != 0:
                raise ValueError("wrist binding must be a separate fixed camera on its mount body")
            self.wrist_base = {
                "position": self.model.cam_pos[self.wrist_id].copy(),
                "quat_wxyz": self.model.cam_quat[self.wrist_id].copy(),
                "fovy_deg": float(self.model.cam_fovy[self.wrist_id]),
            }
        for name, length in (("table_center_xy", 2), ("table_full_size", 3)):
            _finite_vector(v[name], length, name)
        if not np.isfinite(v["table_top_z"]):
            raise ValueError("table_top_z must be finite")
        if not isinstance(v["surfaces"], dict) or set(v["surfaces"]) - {"table", "floor", "wall"}:
            raise ValueError("visual.surfaces must explicitly bind a subset of table, floor and wall")
        owners: dict[tuple[str, int], str] = {}
        self.material_samples = {}
        if not isinstance(v["assets"], dict):
            raise ValueError("visual.assets must map material keys to frozen texture paths")
        for surface, names in v["surfaces"].items():
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(n, str) for n in names)
                or len(names) != len(set(names))
            ):
                raise ValueError(
                    f"visual {surface} requires distinct existing geom names; missing surfaces cannot be guessed"
                )
            sample = sample_material(
                self.profile, surface, camera.stable_seed(self.variant_seed, surface + "_material")
            )
            self.material_samples[surface] = sample
            asset = sample["asset"]
            key = surface + "/" + asset["source_asset_id"]
            path_string = v["assets"].get(key)
            if (
                not isinstance(path_string, str)
                or pathlib.PurePosixPath(path_string).is_absolute()
                or ".." in pathlib.PurePosixPath(path_string).parts
            ):
                raise ValueError(f"missing or unsafe frozen texture path for {key}")
            path = (asset_root / path_string).resolve()
            if not path.is_relative_to(asset_root.resolve()):
                raise ValueError(f"texture {key} escapes the asset cache")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"missing or hash-mismatched texture {key}: {path}")
            self.targets[surface] = []
            for name in names:
                geom_id = _named(self.model, "geom", name)
                mat_id = int(self.model.geom_matid[geom_id])
                if mat_id < 0:
                    raise ValueError(f"surface geom {name!r} needs its material and texture")
                # MuJoCo 3.11 uses texture roles; mjTEXROLE_RGB is column 1.
                tex_ids = self.model.mat_texid[mat_id]
                tex_id = int(tex_ids[1]) if np.ndim(tex_ids) else int(tex_ids)
                if tex_id < 0 or int(self.model.tex_type[tex_id]) not in (0, 1):
                    raise ValueError(f"surface geom {name!r} requires an RGB 2D or cube texture")
                for kind, index in (("geom", geom_id), ("material", mat_id), ("texture", tex_id)):
                    prior = owners.setdefault((kind, index), surface)
                    if prior != surface:
                        raise ValueError(f"{kind} {index} is shared by {prior} and {surface}")
                # A material/texture shared with a task object must not recolor that object.
                bound_ids = {_named(self.model, "geom", n) for n in names}
                for other in range(self.model.ngeom):
                    other_mat = int(self.model.geom_matid[other])
                    if other_mat < 0 or other in bound_ids:
                        continue
                    other_tex = self.model.mat_texid[other_mat]
                    other_tex = int(other_tex[1]) if np.ndim(other_tex) else int(other_tex)
                    if other_mat == mat_id or other_tex == tex_id:
                        raise ValueError(f"surface {surface} shares a material/texture with unbound geom {other}")
                from PIL import Image

                w, h = int(self.model.tex_width[tex_id]), int(self.model.tex_height[tex_id])
                channels = int(self.model.tex_nchannel[tex_id]) if hasattr(self.model, "tex_nchannel") else 3
                if channels not in (3, 4):
                    raise ValueError("image textures require RGB or RGBA storage")
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if image.size != (w, h):
                        image = image.resize((w, h), Image.Resampling.BILINEAR)
                    bitmap = np.asarray(image).copy()
                if channels == 4:
                    bitmap = np.concatenate([bitmap, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
                guard = self.profile["surface_constraints"][surface].get("image_luminance_guard")
                if guard is not None:
                    bitmap, _ = material._apply_image_luminance_guard(bitmap, guard)
                self.textures[str(tex_id)] = bitmap
                self.targets[surface].append((geom_id, mat_id, tex_id))

    def apply(self, data: Any, renderer: Any) -> dict[str, Any]:
        import mujoco

        reset._apply_domain_randomization(self.model, data, self.domain, self.nonce)
        result = {
            "upstream_reset_revision": RESET_REVISION,
            "submit_nonce": self.nonce,
            "sources_sha256": self.sources_sha256,
            "domain_randomization": self.config["domain_randomization"],
            "visual": None,
        }
        if self.visual is None:
            return result
        v = self.visual
        geometry = {
            "base_position": data.xpos[self.reference_id],
            "base_quat_wxyz": data.xquat[self.reference_id],
            "table_center_xy": v["table_center_xy"],
            "table_top_z": v["table_top_z"],
            "table_full_size": v["table_full_size"],
            "wrist_base": self.wrist_base,
            "front_enabled": self.front_id is not None,
        }
        samples = sample_cameras(
            self.profile,
            geometry,
            camera_seed=camera.stable_seed(self.variant_seed, "camera"),
            replica_id=v["replica_id"],
        )
        if self.front_id is not None:
            front = samples["frontview"]["resolved"]
            self.model.cam_pos[self.front_id] = front["world_position"]
            self.model.cam_quat[self.front_id] = front["world_quat_wxyz"]
            self.model.cam_fovy[self.front_id] = front["fovy_deg"]
        if self.wrist_id is not None:
            wrist = samples["wrist"]["resolved"]
            self.model.cam_pos[self.wrist_id] = wrist["position"]
            self.model.cam_quat[self.wrist_id] = wrist["quat_wxyz"]
            self.model.cam_fovy[self.wrist_id] = wrist["fovy_deg"]
        for surface, targets in self.targets.items():
            coefficients = self.material_samples[surface]["material"]
            caps = self.profile["surface_constraints"][surface].get("material_caps", {})
            for geom_id, mat_id, _ in targets:
                self.model.geom_rgba[geom_id] = 1
                self.model.mat_rgba[mat_id] = 1
                for index, field in enumerate(("reflectance", "shininess", "specular")):
                    getattr(self.model, "mat_" + field)[mat_id] = min(
                        coefficients[index], caps.get("max_" + field, 1.0)
                    )
        pixels = self.model.tex_data if hasattr(self.model, "tex_data") else self.model.tex_rgb
        for key, bitmap in self.textures.items():
            tex_id = int(key)
            address = int(self.model.tex_adr[tex_id])
            pixels[address : address + bitmap.size] = bitmap.reshape(-1)
            # A Renderer owns its GL context; make it current before updating textures.
            if renderer._gl_context is not None:
                renderer._gl_context.make_current()
            mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, tex_id)
        mujoco.mj_forward(self.model, data)
        result["visual"] = {
            "profile_sha256": v["profile_sha256"],
            "variant_seed": self.variant_seed,
            "camera_samples": samples,
            "material_samples": self.material_samples,
            "lighting_randomization": False,
        }
        return result

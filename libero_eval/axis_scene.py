"""V6 room rendering beside the evaluator's unchanged physics model.

Upstream consumes replay states and replaces the scene shell for rendering.
Do the same here: forward a separate model, never step its replacement table
or removed collision meshes. Camera/room/material rules remain upstream's.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from .axis_perturbations import AxisRandomizer, COMPONENT_ROOT, _exact_keys, _verify_vendor
    from .axis_components import scene_xml, visual_scene
else:
    from axis_perturbations import AxisRandomizer, COMPONENT_ROOT, _exact_keys, _verify_vendor
    from axis_components import scene_xml, visual_scene

ROOT = COMPONENT_ROOT
# Frozen payloads use this upstream identity. SOURCES.json separately verifies
# the distributed config bytes after host-specific provenance path redaction.
CAMERA_CONFIG_SHA256 = "01f7f87dcc134ae3a485a7a297810e49c427b64e86291a1d303cd6cde2d5a417"


def _scene_library_source_digest() -> str:
    camera = json.loads((ROOT / "config/cameras/franka.json").read_bytes())
    return camera["render_profile"]["scene"]["library_sha256"]


def _camera_config_for_distributed_library() -> dict:
    """franka.json declares the scene library's source digest, which frozen
    payloads and validation evidence record. The distributed library has host
    paths redacted from its provenance fields, so its bytes are checked against
    the SOURCES.json digest (already verified by _verify_vendor); callers put
    the source digest back into the resolved profile."""
    camera = json.loads((ROOT / "config/cameras/franka.json").read_bytes())
    scene = camera["render_profile"]["scene"]
    files = json.loads((ROOT / "SOURCES.json").read_bytes())["files"]
    scene["library_sha256"] = files[scene["library_file"]]
    return camera


def resolve_profile(visual: dict, task_id: int, task_name: str) -> tuple[dict, dict]:
    partial = visual.get("mode") in {"official_franka_components", "official_franka_components_v2"}
    _exact_keys(
        visual,
        {
            "mode",
            "camera_config_sha256",
            "global_seed",
            "attempt_id",
            "variant_id",
            "replica_id",
        }
        | ({"components"} if partial else set()),
        "official_franka_v6 visual",
    )
    if (
        visual["mode"] not in {"official_franka_v6", "official_franka_components", "official_franka_components_v2"}
        or visual["camera_config_sha256"] != CAMERA_CONFIG_SHA256
    ):
        raise ValueError("Unsupported or unpinned Franka camera config")
    if partial:
        components = visual["components"]
        _exact_keys(components, {"arena", "front_camera", "wrist_camera", "background", "surfaces"}, "components")
        if any(type(components[key]) is not bool for key in ("arena", "front_camera", "wrist_camera", "background")):
            raise ValueError("component switches must be booleans")
        if not isinstance(components["surfaces"], dict) or set(components["surfaces"]) - {"table", "floor", "wall"}:
            raise ValueError("components.surfaces requires explicit surface bindings")
    for key in ("global_seed", "attempt_id", "variant_id", "replica_id"):
        if type(visual[key]) is not int or visual[key] < 0:
            raise ValueError(f"visual.{key} must be a non-negative integer")
    if visual["replica_id"] >= 4:
        raise ValueError("V6 has four randomized camera pairs: replica_id must be 0..3")
    _verify_vendor()
    config = visual_scene.resolve_visual_scene_profile(
        _camera_config_for_distributed_library(),
        project_root=ROOT,
        global_seed=visual["global_seed"],
        task_id=task_id,
        attempt_id=visual["attempt_id"],
        task_name=task_name,
        variant_id=visual["variant_id"],
    )
    render = config["render_profile"]
    render["scene"]["library_sha256"] = _scene_library_source_digest()
    palette = render["randomization"]["texture"]["asset_ids_by_surface"]
    catalog = json.loads((ROOT / "manifests/scene_materials.json").read_bytes())
    profile = {
        "profile_id": render["profile_id"],
        "camera_profiles": scene_xml._camera_randomization_by_name(config),
        "surface_asset_ids": palette,
        "surface_constraints": {surface: render["surface_constraints"].get(surface, {}) for surface in palette},
        "materials": {
            surface + "/" + aid: catalog[surface + "/" + aid] for surface, ids in palette.items() for aid in ids
        },
    }
    return config, profile


def _read_expanded_xml(path: Path, stack: tuple[Path, ...] = ()) -> ET.Element:
    """Expand includes in memory, so upstream camera injection cannot edit files."""
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Cyclic MJCF include: {path}")
    root = ET.parse(path).getroot()
    compiler = root.find("compiler")
    if compiler is not None and any(compiler.get(k) for k in ("assetdir", "meshdir", "texturedir")):
        raise ValueError("scene preparation requires materialized MJCF asset paths")
    scene_xml._absolutize_xml_file_attrs(root, xml_dir=path.parent)
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "include":
                included = _read_expanded_xml(Path(child.attrib["file"]), (*stack, path))
                index = list(parent).index(child)
                parent.remove(child)
                for offset, element in enumerate(list(included)):
                    parent.insert(index + offset, element)
    return root


def prepare_scene(
    path: Path, config: dict, components: dict | None = None, *, source_frame: bool = False
) -> tuple[str, dict]:
    root = _read_expanded_xml(path)
    needs_table = components is None or any(components[key] for key in ("arena", "front_camera", "background"))
    placement = scene_xml._infer_scene_placement(root, config["render_profile"]) if needs_table else None
    if placement is not None and (
        placement.preserve_source_table or placement.policy != "arm_robot_and_objects_on_table"
    ):
        raise ValueError("Franka V6 profile requires its upstream arm/table placement")
    arena = components is None or components["arena"]
    source_frame_z_translation = 0.0
    if source_frame and not arena and placement is not None:
        # Upstream's arena puts a zero-height arm root onto its 0.48 m table.
        # A preserved source scene may already use any arm/table height. Express
        # that SAME camera/room distribution in the source arm's world frame.
        reference = placement.metadata["fixed_table_placement"]["reference_body_world_pos"]
        source_top_z = float(reference[2])
        source_frame_z_translation = source_top_z - placement.table_top_z
        placement = replace(
            placement,
            table_top_z=source_top_z,
            metadata={
                **placement.metadata,
                "table_top_z": source_top_z,
                "source_frame_z_translation": source_frame_z_translation,
            },
        )
    # These geometry fields are unused when only a body-mounted camera and/or
    # native materials are selected. Do not infer a table for those components.
    metadata = (
        dict(placement.metadata)
        if placement is not None
        else dict(table_center_xy=[0, 0], table_top_z=0, table_full_size=[0, 0, 0], table_surface_geom_names=[])
    )
    if arena:
        metadata.update(scene_xml._apply_arm_scene_z_offset(root, placement))
        metadata.update(scene_xml._strip_invisible_collision_mesh_geoms(root))
        metadata.update(scene_xml._merge_robosuite_table_arena(root, ("frontview", "wrist"), placement))
    else:
        metadata.update(free_joint_qpos_z_offset=0.0, arm_scene_z_offset=0.0, source_geometry_preserved=True)
    if components is None or components["background"]:
        metadata.update(
            visual_scene.apply_visual_scene(
                root,
                config,
                table_center_xy=placement.table_center_xy,
                table_top_z=placement.table_top_z,
                table_full_size=placement.table_full_size,
                preserve_source_table=not arena,
                arena_prefix=scene_xml.ARENA_PREFIX,
            )
        )
        if source_frame_z_translation:
            # The library's fixtures use the canonical floor frame, independent
            # of table_top_z. Apply the corresponding rigid translation to only
            # those new visual geoms; source objects and physics remain intact.
            for fixture in metadata["fixtures"]:
                name = fixture["mjcf_geom_name"]
                geom = next(g for g in root.iter("geom") if g.get("name") == name)
                position = np.asarray([float(v) for v in geom.get("pos").split()])
                position[2] += source_frame_z_translation
                geom.set("pos", " ".join(format(float(v), ".17g") for v in position))
                fixture["absolute_center_pos"] = position.tolist()
            metadata["upstream_materialized_fixture_sha256"] = metadata.pop("materialized_fixture_sha256")
    if arena:
        metadata.update(scene_xml._configure_reference_lighting_rig(root, config, placement))
    cameras = copy.deepcopy(config)
    if components is not None:
        cameras["cameras"] = [
            c for c in cameras["cameras"] if components["front_camera" if c["name"] == "frontview" else "wrist_camera"]
        ]
    metadata.update(scene_xml._ensure_configured_cameras(root, cameras, placement, entry_xml=path))
    return ET.tostring(root, encoding="unicode"), metadata


class AxisSceneRenderer:
    def __init__(self, scene_path: Path, payload: dict, physics_model, *, width: int, height: int):
        import mujoco

        self.mujoco = mujoco
        self.physics_model = physics_model
        config = payload["official_randomization"]
        visual = config["visual"]
        camera_config, profile = resolve_profile(visual, int(payload["id"]), payload["name"])
        components = visual.get("components")
        xml, self.metadata = prepare_scene(
            scene_path,
            camera_config,
            components,
            source_frame=visual["mode"] == "official_franka_components_v2",
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # The replay contract requires identical state ordering, even
        # when visual-only bodies are added and original table bodies disappear.
        for size in ("nq", "nv", "nu", "na", "njnt", "nmocap"):
            if getattr(self.model, size) != getattr(physics_model, size):
                raise ValueError(f"render model changed {size}")
        for field in ("jnt_type", "jnt_qposadr", "jnt_dofadr"):
            if not np.array_equal(getattr(self.model, field), getattr(physics_model, field)):
                raise ValueError(f"render model changed {field}")
        for j in range(self.model.njnt):
            if self.model.joint(j).name != physics_model.joint(j).name:
                raise ValueError("render model changed joint ordering")
        self.qpos_offset = np.zeros(self.model.nq)
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.qpos_offset[self.model.jnt_qposadr[j] + 2] = self.metadata["free_joint_qpos_z_offset"]
        # Native MuJoCo uses the same physical near/far planes as upstream.
        self.model.vis.map.znear = 0.005 / self.model.stat.extent
        self.model.vis.map.zfar = 100.0 / self.model.stat.extent
        self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
        names = [self.model.geom(i).name for i in range(self.model.ngeom)]
        surfaces = {
            "table": self.metadata["table_surface_geom_names"],
            "floor": [scene_xml.ARENA_PREFIX + "floor"],
            "wall": sorted(
                n
                for n in names
                if n.startswith(scene_xml.ARENA_PREFIX)
                and not n.startswith(scene_xml.ARENA_PREFIX + "visual_scene_")
                and "wall" in n.lower()
                and "visual" in n.lower()
            ),
        }
        if components is not None:
            surfaces = (
                {key: surfaces[key] for key in components["surfaces"]}
                if components["arena"]
                else components["surfaces"]
            )
        derived = copy.deepcopy(config)
        derived.update(domain_randomization={}, object_order=[])
        derived["visual"] = {
            "profile_sha256": CAMERA_CONFIG_SHA256,
            **{key: visual[key] for key in ("global_seed", "attempt_id", "variant_id", "replica_id")},
            "front_camera": "frontview" if components is None or components["front_camera"] else None,
            "wrist_camera": "wrist" if components is None or components["wrist_camera"] else None,
            "reference_body": "franka/",
            **{key: self.metadata[key] for key in ("table_center_xy", "table_top_z", "table_full_size")},
            "surfaces": surfaces,
            "assets": {key: f"official_textures/{value['sha256']}.jpg" for key, value in profile["materials"].items()},
        }
        self.randomizer = AxisRandomizer(
            self.model,
            derived,
            asset_root=scene_path.parent.parent,
            task_id=int(payload["id"]),
            task_name=payload["name"],
            width=width,
            height=height,
            resolved_profile=profile,
        )
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = 0 if components is None or components["arena"] else 1
        self.option.geomgroup[1] = 1
        self.scene = camera_config["render_profile"]["scene"]

    def sync(self, data):
        self.data.qpos[:] = data.qpos + self.qpos_offset
        self.data.qvel[:] = data.qvel
        self.data.act[:] = data.act
        self.data.ctrl[:] = data.ctrl
        self.data.time = data.time
        # Match named mocap bodies rather than assuming equal body IDs.
        for bid in range(self.physics_model.nbody):
            mid = self.physics_model.body_mocapid[bid]
            if mid >= 0:
                target = self.model.body(self.physics_model.body(bid).name).id
                vid = self.model.body_mocapid[target]
                self.data.mocap_pos[vid] = data.mocap_pos[mid]
                self.data.mocap_quat[vid] = data.mocap_quat[mid]
        self.mujoco.mj_forward(self.model, self.data)

    def reset(self, data):
        self.mujoco.mj_resetData(self.model, self.data)
        self.sync(data)
        result = self.randomizer.apply(self.data, self.renderer)["visual"]
        result.update(scene=self.scene, xml_metadata=self.metadata, physics_model_unchanged=True)
        return result

    def render(self, data, camera: str):
        self.sync(data)
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        return self.renderer.render().copy()

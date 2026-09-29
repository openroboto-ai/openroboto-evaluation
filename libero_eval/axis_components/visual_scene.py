"""Deterministic procedural visual scenes for the native Robosuite renderer.

The module deliberately limits itself to MJCF assets and visual-only primitive
geoms.  It never adds bodies, joints, actuators, or collision geometry, so the
source replay layout remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any


SCENE_MODE = "kujiale_procedural_library"
SCENE_LIBRARY_SCHEMA_VERSION = "axis_kujiale_combinatorial_scene_v2"
SCENE_SELECTION = "stable_task_attempt_variant_mixed_radix_v2"
MIN_SCENE_RECIPE_COUNT = 1000
MIN_SCENE_GEOMETRY_RECIPE_COUNT = 1000
MAX_SCENE_FIXTURE_COUNT = 48
TABLE_STYLE_MODE = "four_leg_apron"
_MATERIAL_SCALAR_FIELDS = ("reflectance", "shininess", "specular", "emission")
_PRIMITIVE_SIZE_LENGTHS = {"box": 3, "sphere": 1, "cylinder": 2}
_FLOAT_TOLERANCE = 1e-8


def _as_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def _as_nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _as_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number.")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field_name} must be {operator} {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}.")
    return result


def _as_vector(
    value: Any,
    length: int,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} numbers.")
    return tuple(
        _as_number(
            item,
            f"{field_name}[{index}]",
            minimum=minimum,
            maximum=maximum,
            minimum_inclusive=minimum_inclusive,
        )
        for index, item in enumerate(value)
    )


def _material_specs(variant: dict[str, Any], field_name: str) -> dict[str, dict[str, Any]]:
    materials = _as_object(variant.get("materials"), f"{field_name}.materials")
    if not materials:
        raise ValueError(f"{field_name}.materials must not be empty.")
    result: dict[str, dict[str, Any]] = {}
    for raw_key, raw_spec in materials.items():
        key = _as_nonempty_string(raw_key, f"{field_name}.materials key")
        spec = _as_object(raw_spec, f"{field_name}.materials.{key}")
        _as_vector(spec.get("rgba"), 4, f"{field_name}.materials.{key}.rgba", minimum=0.0, maximum=1.0)
        for scalar_name in _MATERIAL_SCALAR_FIELDS:
            if scalar_name in spec:
                _as_number(
                    spec[scalar_name],
                    f"{field_name}.materials.{key}.{scalar_name}",
                    minimum=0.0,
                    maximum=1.0,
                )
        result[key] = spec
    return result


def _table_style(
    variant: dict[str, Any],
    materials: dict[str, dict[str, Any]],
    field_name: str,
) -> dict[str, Any]:
    style = _as_object(variant.get("table_style"), f"{field_name}.table_style")
    mode = _as_nonempty_string(style.get("mode"), f"{field_name}.table_style.mode")
    if mode != TABLE_STYLE_MODE:
        raise ValueError(
            f"{field_name}.table_style.mode must be {TABLE_STYLE_MODE!r}, got {mode!r}."
        )
    for material_field in ("frame_material", "edge_material", "foot_material"):
        material_key = _as_nonempty_string(
            style.get(material_field), f"{field_name}.table_style.{material_field}"
        )
        if material_key not in materials:
            raise ValueError(
                f"{field_name}.table_style.{material_field} references unknown material {material_key!r}."
            )
    for scalar_field in (
        "top_skin_thickness",
        "edge_thickness",
        "apron_height",
        "apron_thickness",
        "foot_height",
    ):
        _as_number(
            style.get(scalar_field),
            f"{field_name}.table_style.{scalar_field}",
            minimum=0.0,
            minimum_inclusive=False,
        )
    _as_vector(
        style.get("leg_half_width"),
        2,
        f"{field_name}.table_style.leg_half_width",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _as_vector(style.get("leg_inset"), 2, f"{field_name}.table_style.leg_inset", minimum=0.0)
    return style


def _fixture_identifier(fixture: dict[str, Any], field_name: str) -> str:
    # A human-readable name wins, while id remains supported for compact banks.
    raw_identifier = fixture.get("name") if fixture.get("name") is not None else fixture.get("id")
    return _as_nonempty_string(raw_identifier, f"{field_name}.name_or_id")


def _fixture_specs(
    variant: dict[str, Any],
    materials: dict[str, dict[str, Any]],
    field_name: str,
) -> list[dict[str, Any]]:
    fixtures = variant.get("fixtures")
    if not isinstance(fixtures, list):
        raise ValueError(f"{field_name}.fixtures must be a JSON array.")
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, raw_fixture in enumerate(fixtures):
        fixture_field = f"{field_name}.fixtures[{index}]"
        fixture = _as_object(raw_fixture, fixture_field)
        identifier = _fixture_identifier(fixture, fixture_field)
        if identifier in seen:
            raise ValueError(f"Duplicate fixture identifier {identifier!r} in {field_name}.fixtures.")
        seen.add(identifier)
        primitive_type = _as_nonempty_string(fixture.get("type"), f"{fixture_field}.type")
        if primitive_type not in _PRIMITIVE_SIZE_LENGTHS:
            raise ValueError(
                f"{fixture_field}.type must be one of {sorted(_PRIMITIVE_SIZE_LENGTHS)}, "
                f"got {primitive_type!r}."
            )
        frame = _as_nonempty_string(fixture.get("frame"), f"{fixture_field}.frame")
        if frame != "table_xy_floor":
            raise ValueError(f"{fixture_field}.frame must be 'table_xy_floor', got {frame!r}.")
        _as_vector(fixture.get("pos"), 3, f"{fixture_field}.pos")
        _as_vector(
            fixture.get("size"),
            _PRIMITIVE_SIZE_LENGTHS[primitive_type],
            f"{fixture_field}.size",
            minimum=0.0,
            minimum_inclusive=False,
        )
        material_key = _as_nonempty_string(fixture.get("material"), f"{fixture_field}.material")
        if material_key not in materials:
            raise ValueError(f"{fixture_field}.material references unknown material {material_key!r}.")
        if "quat" in fixture:
            quat = _as_vector(fixture["quat"], 4, f"{fixture_field}.quat")
            norm = math.sqrt(sum(component * component for component in quat))
            if norm <= _FLOAT_TOLERANCE:
                raise ValueError(f"{fixture_field}.quat must have non-zero norm.")
        result.append(fixture)
    return result


def _surface_asset_ids(variant: dict[str, Any], field_name: str) -> dict[str, list[str]]:
    raw_mapping = _as_object(variant.get("surface_asset_ids"), f"{field_name}.surface_asset_ids")
    if not raw_mapping:
        raise ValueError(f"{field_name}.surface_asset_ids must not be empty.")
    result: dict[str, list[str]] = {}
    for raw_surface, raw_asset_ids in raw_mapping.items():
        surface = _as_nonempty_string(raw_surface, f"{field_name}.surface_asset_ids key")
        if not isinstance(raw_asset_ids, list) or not raw_asset_ids:
            raise ValueError(f"{field_name}.surface_asset_ids.{surface} must be a non-empty array.")
        asset_ids = [
            _as_nonempty_string(asset_id, f"{field_name}.surface_asset_ids.{surface}[{index}]")
            for index, asset_id in enumerate(raw_asset_ids)
        ]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError(f"{field_name}.surface_asset_ids.{surface} contains duplicates.")
        result[surface] = asset_ids
    return result


def _validate_variant(variant: Any, field_name: str) -> dict[str, Any]:
    result = _as_object(variant, field_name)
    _as_nonempty_string(result.get("id"), f"{field_name}.id")
    materials = _material_specs(result, field_name)
    _table_style(result, materials, field_name)
    _fixture_specs(result, materials, field_name)
    _surface_asset_ids(result, field_name)
    return result


def _library_identifier(value: Any, field_name: str) -> str:
    identifier = _as_nonempty_string(value, field_name)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
        raise ValueError(
            f"{field_name} must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores."
        )
    return identifier


def _module_slot_bounds(value: Any, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field_name} must contain x/y/z [minimum, maximum] bounds.")
    bounds: list[tuple[float, float]] = []
    for axis_index, raw_bounds in enumerate(value):
        axis_bounds = _as_vector(raw_bounds, 2, f"{field_name}[{axis_index}]")
        if axis_bounds[0] >= axis_bounds[1]:
            raise ValueError(f"{field_name}[{axis_index}] minimum must be smaller than maximum.")
        bounds.append((axis_bounds[0], axis_bounds[1]))
    return tuple(bounds)


def _fixture_half_extents(fixture: dict[str, Any], field_name: str) -> tuple[float, float, float]:
    primitive_type = str(fixture["type"])
    size = _as_vector(
        fixture["size"],
        _PRIMITIVE_SIZE_LENGTHS[primitive_type],
        f"{field_name}.size",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if primitive_type == "box":
        return size[0], size[1], size[2]
    if primitive_type == "sphere":
        return size[0], size[0], size[0]
    if primitive_type == "cylinder":
        return size[0], size[0], size[1]
    raise AssertionError(f"Unhandled primitive type {primitive_type!r}.")


def _semantic_fixture_geometry_sha256(fixtures: list[dict[str, Any]]) -> str:
    """Hash physical primitive geometry without names or material appearance."""

    normalized = [
        {
            key: deepcopy(fixture[key])
            for key in ("type", "frame", "pos", "size", "quat")
            if key in fixture
        }
        for fixture in fixtures
    ]
    normalized.sort(key=lambda fixture: json.dumps(fixture, sort_keys=True, separators=(",", ":")))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounds_have_interior_overlap(
    first: tuple[tuple[float, float], ...],
    second: tuple[tuple[float, float], ...],
) -> bool:
    return all(
        first[axis][1] > second[axis][0] + _FLOAT_TOLERANCE
        and second[axis][1] > first[axis][0] + _FLOAT_TOLERANCE
        for axis in range(3)
    )


def _validate_theme(
    raw_theme: Any,
    *,
    field_name: str,
    material_slots: tuple[str, ...],
    material_roles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    theme = deepcopy(_as_object(raw_theme, field_name))
    _library_identifier(theme.get("id"), f"{field_name}.id")
    raw_palette = _as_object(theme.get("palette"), f"{field_name}.palette")
    missing_slots = sorted(set(material_slots) - set(raw_palette))
    extra_slots = sorted(set(raw_palette) - set(material_slots))
    if missing_slots or extra_slots:
        raise ValueError(
            f"{field_name}.palette must exactly match scene library.material_roles; "
            f"missing={missing_slots}, extra={extra_slots}."
        )
    theme["materials"] = {
        slot: {
            **deepcopy(material_roles[slot]),
            "rgba": list(
                _as_vector(
                    raw_palette[slot],
                    4,
                    f"{field_name}.palette.{slot}",
                    minimum=0.0,
                    maximum=1.0,
                )
            ),
        }
        for slot in material_slots
    }
    materials = _material_specs(theme, field_name)
    _table_style(theme, materials, field_name)
    surface_ids = _surface_asset_ids(theme, field_name)
    if set(surface_ids) != {"table", "floor", "wall"}:
        raise ValueError(f"{field_name}.surface_asset_ids must define table, floor, and wall.")
    return theme


def _validate_fixture_module(
    raw_module: Any,
    *,
    field_name: str,
    axis_name: str,
    material_slots: tuple[str, ...],
    slot_bounds: tuple[tuple[float, float], ...],
    workspace_keepout: tuple[tuple[float, float], ...],
    allow_workspace_underlap: bool,
) -> dict[str, Any]:
    module = _as_object(raw_module, field_name)
    module_id = _library_identifier(module.get("id"), f"{field_name}.id")
    dummy_materials = {slot: {"rgba": [0.0, 0.0, 0.0, 1.0]} for slot in material_slots}
    fixtures = _fixture_specs(module, dummy_materials, field_name)
    if not fixtures:
        raise ValueError(f"{field_name}.fixtures must not be empty.")
    for fixture_index, fixture in enumerate(fixtures):
        fixture_field = f"{field_name}.fixtures[{fixture_index}]"
        if "quat" in fixture:
            raise ValueError(
                f"{fixture_field}.quat is not supported in compositional slots; "
                "use axis-aligned primitives so bounds remain exact."
            )
        center = _as_vector(fixture["pos"], 3, f"{fixture_field}.pos")
        half_extents = _fixture_half_extents(fixture, fixture_field)
        fixture_minimum = tuple(
            coordinate - half_extent for coordinate, half_extent in zip(center, half_extents)
        )
        fixture_maximum = tuple(
            coordinate + half_extent for coordinate, half_extent in zip(center, half_extents)
        )
        for axis_index, ((minimum, maximum), coordinate, half_extent) in enumerate(
            zip(slot_bounds, center, half_extents)
        ):
            if (
                coordinate - half_extent < minimum - _FLOAT_TOLERANCE
                or coordinate + half_extent > maximum + _FLOAT_TOLERANCE
            ):
                raise ValueError(
                    f"{fixture_field} escapes {axis_name!r} slot bounds on axis {axis_index}: "
                    f"center={coordinate}, half_extent={half_extent}, bounds={(minimum, maximum)}."
                )
        overlaps_keepout = all(
            fixture_maximum[axis] > workspace_keepout[axis][0] + _FLOAT_TOLERANCE
            and workspace_keepout[axis][1] > fixture_minimum[axis] + _FLOAT_TOLERANCE
            for axis in range(3)
        )
        if overlaps_keepout and not allow_workspace_underlap:
            raise ValueError(
                f"{fixture_field} enters the protected robot/table workspace: "
                f"fixture_min={fixture_minimum}, fixture_max={fixture_maximum}, "
                f"keepout={workspace_keepout}."
            )
    return module


def _validate_library(payload: Any) -> dict[str, Any]:
    library = _as_object(payload, "scene library")
    schema_version = _as_nonempty_string(
        library.get("schema_version"), "scene library.schema_version"
    )
    if schema_version != SCENE_LIBRARY_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported scene library.schema_version "
            f"{schema_version!r}; expected {SCENE_LIBRARY_SCHEMA_VERSION!r}."
        )
    library_id = _library_identifier(library.get("library_id"), "scene library.library_id")
    raw_material_roles = _as_object(
        library.get("material_roles"), "scene library.material_roles"
    )
    if not raw_material_roles:
        raise ValueError("scene library.material_roles must not be empty.")
    material_roles: dict[str, dict[str, Any]] = {}
    for raw_slot, raw_properties in raw_material_roles.items():
        slot = _library_identifier(raw_slot, "scene library.material_roles key")
        properties = _as_object(
            raw_properties, f"scene library.material_roles.{slot}"
        )
        unknown_fields = sorted(set(properties) - set(_MATERIAL_SCALAR_FIELDS))
        if unknown_fields:
            raise ValueError(
                f"scene library.material_roles.{slot} has unknown fields {unknown_fields}."
            )
        for scalar_name, raw_value in properties.items():
            _as_number(
                raw_value,
                f"scene library.material_roles.{slot}.{scalar_name}",
                minimum=0.0,
                maximum=1.0,
            )
        material_roles[slot] = deepcopy(properties)
    material_slots = tuple(sorted(material_roles))
    workspace_keepout = _module_slot_bounds(
        library.get("workspace_keepout"), "scene library.workspace_keepout"
    )

    theme_axis = _library_identifier(library.get("theme_axis"), "scene library.theme_axis")
    raw_axis_order = library.get("axis_order")
    if not isinstance(raw_axis_order, list) or len(raw_axis_order) < 2:
        raise ValueError("scene library.axis_order must contain the theme and at least one module axis.")
    axis_order = tuple(
        _library_identifier(axis, f"scene library.axis_order[{index}]")
        for index, axis in enumerate(raw_axis_order)
    )
    if len(set(axis_order)) != len(axis_order) or axis_order.count(theme_axis) != 1:
        raise ValueError("scene library.axis_order must contain unique axes and exactly one theme axis.")
    if axis_order[-1] != theme_axis:
        raise ValueError(
            "scene library.theme_axis must be the final mixed-radix axis so consecutive "
            "variant ids enumerate geometry before material themes."
        )

    raw_themes = library.get("themes")
    if not isinstance(raw_themes, list) or not raw_themes:
        raise ValueError("scene library.themes must be a non-empty JSON array.")
    themes: list[dict[str, Any]] = []
    theme_ids: set[str] = set()
    for index, raw_theme in enumerate(raw_themes):
        theme = _validate_theme(
            raw_theme,
            field_name=f"scene library.themes[{index}]",
            material_slots=material_slots,
            material_roles=material_roles,
        )
        theme_id = str(theme["id"])
        if theme_id in theme_ids:
            raise ValueError(f"Duplicate scene theme id {theme_id!r}.")
        theme_ids.add(theme_id)
        themes.append(theme)
    themes.sort(key=lambda theme: str(theme["id"]))

    raw_modules = _as_object(library.get("modules"), "scene library.modules")
    module_axis_names = set(axis_order) - {theme_axis}
    if set(raw_modules) != module_axis_names:
        raise ValueError(
            "scene library.modules keys must exactly match non-theme axis_order entries; "
            f"expected={sorted(module_axis_names)}, actual={sorted(raw_modules)}."
        )
    modules: dict[str, dict[str, Any]] = {}
    for axis_name in axis_order:
        if axis_name == theme_axis:
            continue
        axis_field = f"scene library.modules.{axis_name}"
        raw_axis = _as_object(raw_modules[axis_name], axis_field)
        slot_bounds = _module_slot_bounds(raw_axis.get("slot_bounds"), f"{axis_field}.slot_bounds")
        allow_workspace_underlap = raw_axis.get("allow_workspace_underlap", False)
        if not isinstance(allow_workspace_underlap, bool):
            raise ValueError(f"{axis_field}.allow_workspace_underlap must be a boolean.")
        if allow_workspace_underlap and axis_name != "floor":
            raise ValueError(
                f"{axis_field}.allow_workspace_underlap may only be enabled for the floor axis."
            )
        if (
            allow_workspace_underlap
            and slot_bounds[2][1] > workspace_keepout[2][0] + _FLOAT_TOLERANCE
        ):
            raise ValueError(
                f"{axis_field}.slot_bounds must stay at or below the protected workspace "
                "when allow_workspace_underlap is enabled."
            )
        raw_options = raw_axis.get("options")
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"{axis_field}.options must be a non-empty JSON array.")
        options: list[dict[str, Any]] = []
        option_ids: set[str] = set()
        option_geometry_signatures: dict[str, str] = {}
        for option_index, raw_option in enumerate(raw_options):
            option = _validate_fixture_module(
                raw_option,
                field_name=f"{axis_field}.options[{option_index}]",
                axis_name=axis_name,
                material_slots=material_slots,
                slot_bounds=slot_bounds,
                workspace_keepout=workspace_keepout,
                allow_workspace_underlap=allow_workspace_underlap,
            )
            option_id = str(option["id"])
            if option_id in option_ids:
                raise ValueError(f"Duplicate module id {option_id!r} in axis {axis_name!r}.")
            geometry_signature = _semantic_fixture_geometry_sha256(option["fixtures"])
            duplicate_geometry_id = option_geometry_signatures.get(geometry_signature)
            if duplicate_geometry_id is not None:
                raise ValueError(
                    f"Module {option_id!r} in axis {axis_name!r} duplicates the physical geometry "
                    f"of module {duplicate_geometry_id!r}; names and materials do not create a new "
                    "geometry recipe."
                )
            option_ids.add(option_id)
            option_geometry_signatures[geometry_signature] = option_id
            options.append(option)
        options.sort(key=lambda option: str(option["id"]))
        modules[axis_name] = {
            "slot_bounds": slot_bounds,
            "allow_workspace_underlap": allow_workspace_underlap,
            "options": options,
        }

    module_axes = [axis_name for axis_name in axis_order if axis_name != theme_axis]
    for first_index, first_axis in enumerate(module_axes):
        first_bounds = modules[first_axis]["slot_bounds"]
        for second_axis in module_axes[first_index + 1 :]:
            second_bounds = modules[second_axis]["slot_bounds"]
            if _bounds_have_interior_overlap(first_bounds, second_bounds):
                raise ValueError(
                    "module slot bounds must be interior-disjoint for safe composition; "
                    f"axes {first_axis!r} and {second_axis!r} overlap."
                )

    maximum_fixture_count = sum(
        max(len(option["fixtures"]) for option in modules[axis_name]["options"])
        for axis_name in module_axes
    )
    if maximum_fixture_count > MAX_SCENE_FIXTURE_COUNT:
        raise ValueError(
            "scene library can compose up to "
            f"{maximum_fixture_count} fixtures in one recipe; maximum is "
            f"{MAX_SCENE_FIXTURE_COUNT}."
        )

    axis_sizes = {
        axis_name: len(themes) if axis_name == theme_axis else len(modules[axis_name]["options"])
        for axis_name in axis_order
    }
    recipe_count = math.prod(axis_sizes.values())
    if recipe_count < MIN_SCENE_RECIPE_COUNT:
        raise ValueError(
            f"scene library exposes only {recipe_count} recipes; "
            f"at least {MIN_SCENE_RECIPE_COUNT} are required."
        )
    geometry_recipe_count = math.prod(
        size for axis_name, size in axis_sizes.items() if axis_name != theme_axis
    )
    if geometry_recipe_count < MIN_SCENE_GEOMETRY_RECIPE_COUNT:
        raise ValueError(
            f"scene library exposes only {geometry_recipe_count} geometry recipes; "
            f"at least {MIN_SCENE_GEOMETRY_RECIPE_COUNT} are required."
        )
    return {
        "library_id": library_id,
        "material_slots": material_slots,
        "material_roles": material_roles,
        "workspace_keepout": workspace_keepout,
        "theme_axis": theme_axis,
        "axis_order": axis_order,
        "themes": themes,
        "modules": modules,
        "axis_sizes": axis_sizes,
        "recipe_count": recipe_count,
        "geometry_recipe_count": geometry_recipe_count,
        "maximum_fixture_count": maximum_fixture_count,
    }


def _stable_scene_base_seed(
    global_seed: int,
    task_id: Any,
    attempt_id: Any,
    library_id: str,
) -> int:
    if isinstance(global_seed, bool):
        raise ValueError("global_seed must be an integer.")
    try:
        seed = int(global_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("global_seed must be an integer.") from exc
    digest = hashlib.sha256()
    digest.update(str(seed).encode("utf-8"))
    # Scene identity deliberately excludes mutable task_name. Stable database
    # ids, the user seed, library id, and algorithm contract are sufficient.
    for part in (task_id, attempt_id, library_id, SCENE_MODE, SCENE_SELECTION):
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False) % (2**32)


def _scene_variant_offset(variant_id: Any) -> int:
    if isinstance(variant_id, bool) or not isinstance(variant_id, int):
        raise ValueError("variant_id must be an integer.")
    return int(variant_id)


def _compose_scene_recipe(
    library: dict[str, Any],
    recipe_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe_count = int(library["recipe_count"])
    if recipe_index < 0 or recipe_index >= recipe_count:
        raise ValueError(f"recipe_index must be in [0, {recipe_count}), got {recipe_index}.")
    remaining = int(recipe_index)
    component_indices: dict[str, int] = {}
    component_selection: dict[str, str] = {}
    selected_components: dict[str, dict[str, Any]] = {}
    for axis_name in library["axis_order"]:
        options = (
            library["themes"]
            if axis_name == library["theme_axis"]
            else library["modules"][axis_name]["options"]
        )
        component_index = remaining % len(options)
        remaining //= len(options)
        component = options[component_index]
        component_indices[axis_name] = int(component_index)
        component_selection[axis_name] = str(component["id"])
        selected_components[axis_name] = component
    if remaining != 0:
        raise AssertionError("Internal error: mixed-radix scene recipe did not fully decode.")

    theme = selected_components[library["theme_axis"]]
    fixtures: list[dict[str, Any]] = []
    for axis_name in library["axis_order"]:
        if axis_name == library["theme_axis"]:
            continue
        module = selected_components[axis_name]
        for fixture_index, raw_fixture in enumerate(module["fixtures"]):
            fixture = deepcopy(raw_fixture)
            original_name = _fixture_identifier(
                fixture, f"scene recipe.{axis_name}.fixtures[{fixture_index}]"
            )
            fixture["name"] = f"{axis_name}_{module['id']}_{original_name}"
            fixture.pop("id", None)
            fixtures.append(fixture)

    selection_json = json.dumps(component_selection, sort_keys=True, separators=(",", ":"))
    selection_hash = hashlib.sha256(selection_json.encode("utf-8")).hexdigest()[:10]
    geometry_index = int(recipe_index % int(library["geometry_recipe_count"]))
    selected_variant = {
        "id": f"recipe_{recipe_index:05d}_{selection_hash}",
        "description": "Composed Kujiale-inspired room: "
        + ", ".join(f"{axis}={value}" for axis, value in component_selection.items()),
        "source_scene_ids": list(theme.get("source_scene_ids") or []),
        "surface_asset_ids": deepcopy(theme["surface_asset_ids"]),
        "table_style": deepcopy(theme["table_style"]),
        "materials": deepcopy(theme["materials"]),
        "fixtures": fixtures,
        "recipe_index": int(recipe_index),
        "recipe_count": recipe_count,
        "geometry_index": geometry_index,
        "geometry_recipe_count": int(library["geometry_recipe_count"]),
        "component_indices": component_indices,
        "component_selection": component_selection,
    }
    if len(fixtures) > MAX_SCENE_FIXTURE_COUNT:
        raise ValueError(
            f"Composed scene recipe has {len(fixtures)} fixtures; "
            f"maximum is {MAX_SCENE_FIXTURE_COUNT}."
        )
    fixture_json = json.dumps(fixtures, sort_keys=True, separators=(",", ":"))
    selected_variant["materialized_fixture_sha256"] = hashlib.sha256(
        fixture_json.encode("utf-8")
    ).hexdigest()
    recipe_json = json.dumps(selected_variant, sort_keys=True, separators=(",", ":"))
    selected_variant["recipe_sha256"] = hashlib.sha256(recipe_json.encode("utf-8")).hexdigest()
    _validate_variant(selected_variant, "composed scene recipe")
    return selected_variant, {
        "recipe_index": int(recipe_index),
        "recipe_count": recipe_count,
        "recipe_sha256": selected_variant["recipe_sha256"],
        "materialized_fixture_sha256": selected_variant["materialized_fixture_sha256"],
        "geometry_index": geometry_index,
        "geometry_recipe_count": int(library["geometry_recipe_count"]),
        "component_indices": component_indices,
        "component_selection": component_selection,
        "geometry_validation_scope": "module_slots_and_declared_workspace_keepout",
        "workspace_keepout": [list(bounds) for bounds in library["workspace_keepout"]],
    }


def _path_inside_project(project_root: Path, raw_path: str) -> Path:
    project_root = project_root.expanduser().resolve()
    if not project_root.is_dir():
        raise FileNotFoundError(f"Project root does not exist or is not a directory: {project_root}")
    configured_path = Path(raw_path).expanduser()
    library_path = configured_path.resolve() if configured_path.is_absolute() else (project_root / configured_path).resolve()
    try:
        library_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Scene library must stay inside the project root: {library_path}") from exc
    if not library_path.is_file():
        raise FileNotFoundError(f"Configured scene library is missing: {library_path}")
    return library_path


def _mutable_child_object(parent: dict[str, Any], key: str, field_name: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        value = {}
        parent[key] = value
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def resolve_visual_scene_profile(
    camera_config: dict[str, Any],
    *,
    project_root: str | Path,
    global_seed: int,
    task_id: Any,
    attempt_id: Any,
    task_name: Any,
    variant_id: Any,
) -> dict[str, Any]:
    """Resolve one deterministic scene variant into a copied camera config.

    A scene-less profile is returned as an exact deep copy.  Once a scene mode
    is configured, every path, checksum, and schema error is fatal; there is no
    implicit fallback variant.
    """

    if not isinstance(camera_config, dict):
        raise ValueError("camera_config must be a JSON object.")
    resolved = deepcopy(camera_config)
    if "render_profile" not in resolved or resolved.get("render_profile") is None:
        return resolved
    render_profile = _as_object(resolved["render_profile"], "render_profile")
    if "scene" not in render_profile or render_profile.get("scene") is None:
        return resolved
    scene = _as_object(render_profile["scene"], "render_profile.scene")
    mode = str(scene.get("mode") or "").strip()
    if not mode:
        # Placement-only settings such as preserve_source_table are not a
        # procedural scene configuration and must remain backward compatible.
        return resolved
    if mode != SCENE_MODE:
        raise ValueError(f"Unsupported render_profile.scene.mode {mode!r}.")
    selection = _as_nonempty_string(
        scene.get("selection"), "render_profile.scene.selection"
    )
    if selection != SCENE_SELECTION:
        raise ValueError(
            f"Unsupported render_profile.scene.selection {selection!r}; "
            f"expected {SCENE_SELECTION!r}."
        )

    library_file = _as_nonempty_string(scene.get("library_file"), "render_profile.scene.library_file")
    declared_sha256 = _as_nonempty_string(
        scene.get("library_sha256"), "render_profile.scene.library_sha256"
    )
    if not re.fullmatch(r"[0-9a-f]{64}", declared_sha256):
        raise ValueError("render_profile.scene.library_sha256 must be a lowercase SHA-256 digest.")
    library_path = _path_inside_project(Path(project_root), library_file)
    library_bytes = library_path.read_bytes()
    actual_sha256 = hashlib.sha256(library_bytes).hexdigest()
    if actual_sha256 != declared_sha256:
        raise ValueError(
            "Configured scene library checksum mismatch: "
            f"expected={declared_sha256}, actual={actual_sha256}, path={library_path}"
        )
    try:
        library_payload = json.loads(library_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Configured scene library is not valid UTF-8 JSON: {library_path}") from exc
    library = _validate_library(library_payload)
    scene_base_seed = _stable_scene_base_seed(
        global_seed,
        task_id,
        attempt_id,
        str(library["library_id"]),
    )
    scene_base_index = scene_base_seed % int(library["recipe_count"])
    variant_offset = _scene_variant_offset(variant_id)
    if variant_offset < 0 or variant_offset >= int(library["recipe_count"]):
        raise ValueError(
            f"variant_id must be in [0, {library['recipe_count']}) for scene library "
            f"{library['library_id']!r}, got {variant_offset}."
        )
    scene_index = (scene_base_index + variant_offset) % int(library["recipe_count"])
    selected_variant, recipe_metadata = _compose_scene_recipe(library, scene_index)

    scene["library_sha256"] = actual_sha256
    scene["library_id"] = library["library_id"]
    scene["scene_seed"] = int(scene_base_seed)
    scene["scene_base_seed"] = int(scene_base_seed)
    scene["scene_base_index"] = int(scene_base_index)
    scene["scene_variant_offset"] = int(variant_offset)
    scene["scene_index"] = int(scene_index)
    scene["recipe_count"] = int(recipe_metadata["recipe_count"])
    scene["recipe_sha256"] = recipe_metadata["recipe_sha256"]
    scene["materialized_fixture_sha256"] = recipe_metadata["materialized_fixture_sha256"]
    scene["geometry_index"] = int(recipe_metadata["geometry_index"])
    scene["geometry_recipe_count"] = int(recipe_metadata["geometry_recipe_count"])
    scene["component_indices"] = deepcopy(recipe_metadata["component_indices"])
    scene["component_selection"] = deepcopy(recipe_metadata["component_selection"])
    scene["geometry_validation_scope"] = recipe_metadata["geometry_validation_scope"]
    scene["workspace_keepout"] = deepcopy(recipe_metadata["workspace_keepout"])
    scene["selected_variant"] = selected_variant

    randomization = _mutable_child_object(render_profile, "randomization", "render_profile.randomization")
    texture = _mutable_child_object(randomization, "texture", "render_profile.randomization.texture")
    asset_ids_by_surface = _mutable_child_object(
        texture,
        "asset_ids_by_surface",
        "render_profile.randomization.texture.asset_ids_by_surface",
    )
    for surface, selected_asset_ids in _surface_asset_ids(
        selected_variant, "render_profile.scene.selected_variant"
    ).items():
        existing = asset_ids_by_surface.get(surface)
        if existing is None:
            existing = []
        if not isinstance(existing, list):
            raise ValueError(
                f"render_profile.randomization.texture.asset_ids_by_surface.{surface} must be a JSON array."
            )
        normalized_existing = [
            _as_nonempty_string(
                asset_id,
                f"render_profile.randomization.texture.asset_ids_by_surface.{surface}[{index}]",
            )
            for index, asset_id in enumerate(existing)
        ]
        asset_ids_by_surface[surface] = list(dict.fromkeys([*normalized_existing, *selected_asset_ids]))
    return resolved


def _format_vector(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _xml_vector(
    element: ET.Element,
    attribute: str,
    length: int,
    field_name: str,
    *,
    default: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    raw_value = element.get(attribute)
    if raw_value is None:
        if default is None:
            raise ValueError(f"{field_name} is required.")
        return default
    try:
        values = tuple(float(part) for part in raw_value.split())
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain {length} finite numbers.") from exc
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} must contain {length} finite numbers.")
    return values


def _sanitize_name(value: Any) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_\-]", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        raise ValueError(f"Cannot derive a valid MJCF name from {value!r}.")
    return sanitized


def _ensure_top_level(root: ET.Element, tag: str) -> ET.Element:
    children = root.findall(tag)
    if children:
        # MuJoCo accepts repeated top-level compiler sections, and materialized
        # AXIS scenes commonly contain more than one <asset> / <worldbody>.
        # Adding generated elements to the first section matches the renderer's
        # existing arena merge policy without rewriting source sections.
        return children[0]
    element = ET.Element(tag)
    if tag == "asset":
        worldbody = root.find("worldbody")
        if worldbody is not None:
            root.insert(list(root).index(worldbody), element)
            return element
    root.append(element)
    return element


def _visual_geom(
    parent: ET.Element,
    *,
    name: str,
    primitive_type: str,
    pos: tuple[float, float, float],
    size: tuple[float, ...],
    material: str,
    quat: tuple[float, float, float, float] | None = None,
) -> ET.Element:
    attributes = {
        "name": name,
        "type": primitive_type,
        "pos": _format_vector(pos),
        "size": _format_vector(size),
        "material": material,
        "contype": "0",
        "conaffinity": "0",
        "group": "1",
    }
    if quat is not None:
        attributes["quat"] = _format_vector(quat)
    return ET.SubElement(parent, "geom", attributes)


def _table_elements(root: ET.Element, arena_prefix: str) -> tuple[ET.Element, ET.Element, ET.Element, list[ET.Element]]:
    table_name = f"{arena_prefix}table"
    bodies = [body for body in root.iter("body") if body.get("name") == table_name]
    if len(bodies) != 1:
        raise ValueError(f"Expected exactly one Robosuite table body {table_name!r}, found {len(bodies)}.")
    table_body = bodies[0]

    def direct_geom(name: str) -> ET.Element:
        matches = [geom for geom in table_body.findall("geom") if geom.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one direct table geom {name!r}, found {len(matches)}.")
        return matches[0]

    collision = direct_geom(f"{arena_prefix}table_collision")
    visual = direct_geom(f"{arena_prefix}table_visual")
    legs = [direct_geom(f"{arena_prefix}table_leg{index}_visual") for index in range(1, 5)]
    return table_body, collision, visual, legs


def _direct_worldbody_parent(root: ET.Element, body: ET.Element) -> ET.Element:
    for worldbody in root.findall("worldbody"):
        if body in list(worldbody):
            return worldbody
    raise ValueError("The Robosuite table body must be a direct child of <worldbody>.")


def _table_geometry_preflight(
    root: ET.Element,
    *,
    table_center_xy: tuple[float, float],
    table_top_z: float,
    table_full_size: tuple[float, float, float],
    table_style: dict[str, Any],
    arena_prefix: str,
) -> dict[str, Any]:
    table_body, collision, visual, legs = _table_elements(root, arena_prefix)
    _direct_worldbody_parent(root, table_body)
    if (collision.get("type") or "sphere") != "box" or (visual.get("type") or "sphere") != "box":
        raise ValueError("Robosuite table collision and visual geoms must both be boxes.")
    body_pos = _xml_vector(table_body, "pos", 3, f"{table_body.get('name')}.pos", default=(0.0, 0.0, 0.0))
    collision_pos = _xml_vector(
        collision, "pos", 3, f"{collision.get('name')}.pos", default=(0.0, 0.0, 0.0)
    )
    collision_size = _xml_vector(collision, "size", 3, f"{collision.get('name')}.size")
    visual_pos = _xml_vector(visual, "pos", 3, f"{visual.get('name')}.pos", default=(0.0, 0.0, 0.0))
    visual_size = _xml_vector(visual, "size", 3, f"{visual.get('name')}.size")
    if not all(value > 0.0 for value in collision_size) or not all(value > 0.0 for value in visual_size):
        raise ValueError("Robosuite table geom sizes must be positive.")
    collision_center = tuple(body_pos[index] + collision_pos[index] for index in range(3))
    visual_center = tuple(body_pos[index] + visual_pos[index] for index in range(3))
    expected_half_size = tuple(value / 2.0 for value in table_full_size)
    if any(abs(collision_size[index] - expected_half_size[index]) > _FLOAT_TOLERANCE for index in range(3)):
        raise ValueError(
            "table_full_size does not match the untouched Robosuite table collision geom: "
            f"full_size={table_full_size}, collision_half_size={collision_size}."
        )
    if any(abs(collision_center[index] - table_center_xy[index]) > _FLOAT_TOLERANCE for index in range(2)):
        raise ValueError(
            "table_center_xy does not match the Robosuite table collision center: "
            f"expected={table_center_xy}, actual={collision_center[:2]}."
        )
    collision_top = collision_center[2] + collision_size[2]
    visual_top = visual_center[2] + visual_size[2]
    if abs(collision_top - table_top_z) > _FLOAT_TOLERANCE:
        raise ValueError(
            f"table_top_z={table_top_z} does not match collision top z={collision_top}."
        )
    if abs(visual_top - table_top_z) > _FLOAT_TOLERANCE:
        raise ValueError(f"Existing table visual top z={visual_top} does not match table_top_z={table_top_z}.")

    full_x, full_y, full_z = table_full_size
    half_x, half_y = full_x / 2.0, full_y / 2.0
    top_skin = _as_number(
        table_style["top_skin_thickness"], "table_style.top_skin_thickness", minimum=0.0, minimum_inclusive=False
    )
    edge_thickness = _as_number(
        table_style["edge_thickness"], "table_style.edge_thickness", minimum=0.0, minimum_inclusive=False
    )
    leg_half_x, leg_half_y = _as_vector(
        table_style["leg_half_width"], 2, "table_style.leg_half_width", minimum=0.0, minimum_inclusive=False
    )
    inset_x, inset_y = _as_vector(table_style["leg_inset"], 2, "table_style.leg_inset", minimum=0.0)
    apron_height = _as_number(
        table_style["apron_height"], "table_style.apron_height", minimum=0.0, minimum_inclusive=False
    )
    apron_thickness = _as_number(
        table_style["apron_thickness"], "table_style.apron_thickness", minimum=0.0, minimum_inclusive=False
    )
    foot_height = _as_number(
        table_style["foot_height"], "table_style.foot_height", minimum=0.0, minimum_inclusive=False
    )
    underside_z = table_top_z - full_z
    if underside_z <= 0.0:
        raise ValueError(f"Table underside must be above floor z=0, got {underside_z}.")
    if top_skin > full_z:
        raise ValueError("table_style.top_skin_thickness cannot exceed the collision slab height.")
    if edge_thickness >= min(half_x, half_y):
        raise ValueError(
            "table_style.edge_thickness must be smaller than both tabletop half-extents "
            "so the inset top skin remains positive."
        )
    if apron_thickness >= min(full_x, full_y):
        raise ValueError("table_style.apron_thickness must be smaller than both tabletop dimensions.")
    if apron_height > underside_z:
        raise ValueError("table_style.apron_height cannot extend below floor z=0.")
    if foot_height > underside_z:
        raise ValueError("table_style.foot_height cannot exceed the available leg height.")
    leg_center_x = half_x - inset_x - leg_half_x
    leg_center_y = half_y - inset_y - leg_half_y
    if leg_center_x <= 0.0 or leg_center_y <= 0.0:
        raise ValueError("table_style leg widths and insets do not fit inside the tabletop footprint.")
    return {
        "table_body": table_body,
        "collision": collision,
        "visual": visual,
        "legs": legs,
        "body_pos": body_pos,
        "collision_pos": collision_pos,
        "collision_size": collision_size,
        "collision_center": collision_center,
        "collision_top": collision_top,
        "visual_top_before": visual_top,
        "full_x": full_x,
        "full_y": full_y,
        "full_z": full_z,
        "half_x": half_x,
        "half_y": half_y,
        "top_skin_half_x": half_x - edge_thickness,
        "top_skin_half_y": half_y - edge_thickness,
        "top_skin": top_skin,
        "edge_thickness": edge_thickness,
        "leg_half_x": leg_half_x,
        "leg_half_y": leg_half_y,
        "inset_x": inset_x,
        "inset_y": inset_y,
        "leg_center_x": leg_center_x,
        "leg_center_y": leg_center_y,
        "apron_height": apron_height,
        "apron_thickness": apron_thickness,
        "foot_height": foot_height,
        "underside_z": underside_z,
        "collision_xml_before": ET.tostring(collision, encoding="unicode"),
        "collision_attributes_before": dict(collision.attrib),
    }


def _apply_table_upgrade(
    plan: dict[str, Any],
    *,
    table_center_xy: tuple[float, float],
    table_top_z: float,
    table_style: dict[str, Any],
    material_names: dict[str, str],
    arena_prefix: str,
) -> tuple[dict[str, Any], list[str]]:
    table_body: ET.Element = plan["table_body"]
    collision: ET.Element = plan["collision"]
    visual: ET.Element = plan["visual"]
    legs: list[ET.Element] = plan["legs"]
    body_x, body_y, body_z = plan["body_pos"]
    collision_pos_x, collision_pos_y, _ = plan["collision_pos"]
    top_skin_half = plan["top_skin"] / 2.0
    visual.set("type", "box")
    visual.set("pos", _format_vector((collision_pos_x, collision_pos_y, table_top_z - top_skin_half - body_z)))
    # The four edge boxes occupy the tabletop perimeter up to table_top_z. Keep
    # the textured skin inside that border so differently-materialed top faces
    # never overlap or become coplanar over a positive area (which otherwise
    # produces view-dependent depth-buffer flicker from the wrist camera).
    visual.set(
        "size",
        _format_vector(
            (plan["top_skin_half_x"], plan["top_skin_half_y"], top_skin_half)
        ),
    )
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    visual.set("group", "1")

    frame_material = material_names[str(table_style["frame_material"]).strip()]
    edge_material = material_names[str(table_style["edge_material"]).strip()]
    foot_material = material_names[str(table_style["foot_material"]).strip()]
    leg_half_height = plan["underside_z"] / 2.0
    leg_center_z = leg_half_height
    leg_centers: list[tuple[float, float, float]] = []
    for index, (sign_x, sign_y) in enumerate(((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))):
        absolute_center = (
            table_center_xy[0] + sign_x * plan["leg_center_x"],
            table_center_xy[1] + sign_y * plan["leg_center_y"],
            leg_center_z,
        )
        local_center = (
            absolute_center[0] - body_x,
            absolute_center[1] - body_y,
            absolute_center[2] - body_z,
        )
        leg = legs[index]
        leg.set("type", "box")
        leg.set("pos", _format_vector(local_center))
        leg.set("size", _format_vector((plan["leg_half_x"], plan["leg_half_y"], leg_half_height)))
        leg.set("material", frame_material)
        leg.set("contype", "0")
        leg.set("conaffinity", "0")
        leg.set("group", "1")
        leg_centers.append(absolute_center)

    added_names: list[str] = []

    def add_table_box(
        suffix: str,
        absolute_pos: tuple[float, float, float],
        size: tuple[float, float, float],
        material: str,
    ) -> str:
        name = f"{arena_prefix}{suffix}"
        local_pos = (
            absolute_pos[0] - body_x,
            absolute_pos[1] - body_y,
            absolute_pos[2] - body_z,
        )
        _visual_geom(
            table_body,
            name=name,
            primitive_type="box",
            pos=local_pos,
            size=size,
            material=material,
        )
        added_names.append(name)
        return name

    edge_half = plan["edge_thickness"] / 2.0
    slab_half_z = plan["full_z"] / 2.0
    slab_center_z = table_top_z - slab_half_z
    edge_names = [
        add_table_box(
            "table_edge_front_visual",
            (table_center_xy[0], table_center_xy[1] + plan["half_y"] - edge_half, slab_center_z),
            (plan["half_x"], edge_half, slab_half_z),
            edge_material,
        ),
        add_table_box(
            "table_edge_back_visual",
            (table_center_xy[0], table_center_xy[1] - plan["half_y"] + edge_half, slab_center_z),
            (plan["half_x"], edge_half, slab_half_z),
            edge_material,
        ),
        add_table_box(
            "table_edge_left_visual",
            (table_center_xy[0] - plan["half_x"] + edge_half, table_center_xy[1], slab_center_z),
            (edge_half, max(plan["half_y"] - plan["edge_thickness"], edge_half), slab_half_z),
            edge_material,
        ),
        add_table_box(
            "table_edge_right_visual",
            (table_center_xy[0] + plan["half_x"] - edge_half, table_center_xy[1], slab_center_z),
            (edge_half, max(plan["half_y"] - plan["edge_thickness"], edge_half), slab_half_z),
            edge_material,
        ),
    ]

    apron_half_z = plan["apron_height"] / 2.0
    apron_half_thickness = plan["apron_thickness"] / 2.0
    apron_center_z = plan["underside_z"] - apron_half_z
    apron_names = [
        add_table_box(
            "table_apron_front_visual",
            (table_center_xy[0], table_center_xy[1] + plan["leg_center_y"], apron_center_z),
            (plan["half_x"] - plan["inset_x"], apron_half_thickness, apron_half_z),
            frame_material,
        ),
        add_table_box(
            "table_apron_back_visual",
            (table_center_xy[0], table_center_xy[1] - plan["leg_center_y"], apron_center_z),
            (plan["half_x"] - plan["inset_x"], apron_half_thickness, apron_half_z),
            frame_material,
        ),
        add_table_box(
            "table_apron_left_visual",
            (table_center_xy[0] - plan["leg_center_x"], table_center_xy[1], apron_center_z),
            (apron_half_thickness, plan["half_y"] - plan["inset_y"], apron_half_z),
            frame_material,
        ),
        add_table_box(
            "table_apron_right_visual",
            (table_center_xy[0] + plan["leg_center_x"], table_center_xy[1], apron_center_z),
            (apron_half_thickness, plan["half_y"] - plan["inset_y"], apron_half_z),
            frame_material,
        ),
    ]

    foot_half_z = plan["foot_height"] / 2.0
    foot_half_x = max(plan["leg_half_x"] * 1.25, plan["leg_half_x"] + 0.005)
    foot_half_y = max(plan["leg_half_y"] * 1.25, plan["leg_half_y"] + 0.005)
    foot_names: list[str] = []
    for index, center in enumerate(leg_centers, start=1):
        foot_names.append(
            add_table_box(
                f"table_foot_baseplate{index}_visual",
                (center[0], center[1], foot_half_z),
                (foot_half_x, foot_half_y, foot_half_z),
                foot_material,
            )
        )

    visual_pos_after = _xml_vector(visual, "pos", 3, f"{visual.get('name')}.pos")
    visual_size_after = _xml_vector(visual, "size", 3, f"{visual.get('name')}.size")
    visual_top_after = body_z + visual_pos_after[2] + visual_size_after[2]
    collision_xml_after = ET.tostring(collision, encoding="unicode")
    collision_unchanged = plan["collision_xml_before"] == collision_xml_after
    if not collision_unchanged:
        raise AssertionError("Internal error: table collision geom changed during visual table upgrade.")
    if abs(visual_top_after - table_top_z) > _FLOAT_TOLERANCE:
        raise AssertionError(
            f"Internal error: upgraded table visual top z={visual_top_after} differs from {table_top_z}."
        )
    leg_metadata = []
    for leg, absolute_center in zip(legs, leg_centers):
        leg_size = _xml_vector(leg, "size", 3, f"{leg.get('name')}.size")
        bottom_z = absolute_center[2] - leg_size[2]
        if abs(bottom_z) > _FLOAT_TOLERANCE:
            raise AssertionError(f"Internal error: table leg {leg.get('name')!r} does not reach floor z=0.")
        leg_metadata.append(
            {
                "name": leg.get("name"),
                "type": leg.get("type"),
                "absolute_center": list(absolute_center),
                "half_size": list(leg_size),
                "bottom_z": float(bottom_z),
                "top_z": float(absolute_center[2] + leg_size[2]),
            }
        )
    return (
        {
            "preserve_source_table": False,
            "table_upgrade_applied": True,
            "table_style_mode": TABLE_STYLE_MODE,
            "table_body_name": table_body.get("name"),
            "table_visual_geom_name": visual.get("name"),
            "table_collision_geom_name": collision.get("name"),
            "table_collision_attributes_before": plan["collision_attributes_before"],
            "table_collision_attributes_after": dict(collision.attrib),
            "table_collision_xml_unchanged": collision_unchanged,
            "table_center_xy": list(table_center_xy),
            "table_top_z_requested": float(table_top_z),
            "table_collision_top_z": float(plan["collision_top"]),
            "table_visual_top_z_before": float(plan["visual_top_before"]),
            "table_visual_top_z_after": float(visual_top_after),
            "table_top_z_preserved": abs(visual_top_after - table_top_z) <= _FLOAT_TOLERANCE,
            "table_full_size": [plan["full_x"], plan["full_y"], plan["full_z"]],
            "top_skin_thickness": float(plan["top_skin"]),
            "top_skin_half_size": [
                float(plan["top_skin_half_x"]),
                float(plan["top_skin_half_y"]),
                float(top_skin_half),
            ],
            "top_skin_xy_inset": [
                float(plan["edge_thickness"]),
                float(plan["edge_thickness"]),
            ],
            "top_skin_edge_overlap_policy": "boundary_contact_only",
            "edge_thickness": float(plan["edge_thickness"]),
            "leg_geometries": leg_metadata,
            "edge_geom_names": edge_names,
            "apron_geom_names": apron_names,
            "foot_baseplate_geom_names": foot_names,
        },
        added_names,
    )


def _replace_element(target: ET.Element, source: ET.Element) -> None:
    target.clear()
    target.tag = source.tag
    target.attrib.update(source.attrib)
    target.text = source.text
    target.tail = source.tail
    for child in list(source):
        target.append(child)


def _apply_visual_scene_mutating(
    root: ET.Element,
    camera_config: dict[str, Any],
    *,
    table_center_xy: tuple[float, float],
    table_top_z: float,
    table_full_size: tuple[float, float, float],
    preserve_source_table: bool,
    arena_prefix: str,
) -> dict[str, Any]:
    render_profile = _as_object(camera_config.get("render_profile"), "render_profile")
    scene = _as_object(render_profile.get("scene"), "render_profile.scene")
    mode = _as_nonempty_string(scene.get("mode"), "render_profile.scene.mode")
    if mode != SCENE_MODE:
        raise ValueError(f"Unsupported render_profile.scene.mode {mode!r}.")
    selected_variant = _validate_variant(
        scene.get("selected_variant"), "render_profile.scene.selected_variant"
    )
    variant_id = str(selected_variant["id"]).strip()
    materials = _material_specs(selected_variant, "render_profile.scene.selected_variant")
    table_style = _table_style(
        selected_variant, materials, "render_profile.scene.selected_variant"
    )
    fixtures = _fixture_specs(
        selected_variant, materials, "render_profile.scene.selected_variant"
    )

    namespace = f"{arena_prefix}visual_scene_{_sanitize_name(variant_id)}_"
    material_names = {
        key: f"{namespace}material_{_sanitize_name(key)}" for key in materials
    }
    if len(set(material_names.values())) != len(material_names):
        raise ValueError("Scene material names collide after MJCF name sanitization.")
    fixture_names = {
        _fixture_identifier(fixture, f"selected_variant.fixtures[{index}]"): (
            f"{namespace}fixture_{_sanitize_name(_fixture_identifier(fixture, f'selected_variant.fixtures[{index}]'))}"
        )
        for index, fixture in enumerate(fixtures)
    }
    if len(set(fixture_names.values())) != len(fixture_names):
        raise ValueError("Scene fixture names collide after MJCF name sanitization.")

    table_plan = None
    if not preserve_source_table:
        table_plan = _table_geometry_preflight(
            root,
            table_center_xy=table_center_xy,
            table_top_z=table_top_z,
            table_full_size=table_full_size,
            table_style=table_style,
            arena_prefix=arena_prefix,
        )

    generated_table_names = []
    if not preserve_source_table:
        generated_table_names = [
            f"{arena_prefix}table_edge_front_visual",
            f"{arena_prefix}table_edge_back_visual",
            f"{arena_prefix}table_edge_left_visual",
            f"{arena_prefix}table_edge_right_visual",
            f"{arena_prefix}table_apron_front_visual",
            f"{arena_prefix}table_apron_back_visual",
            f"{arena_prefix}table_apron_left_visual",
            f"{arena_prefix}table_apron_right_visual",
            *(f"{arena_prefix}table_foot_baseplate{index}_visual" for index in range(1, 5)),
        ]
    existing_names = {element.get("name") for element in root.iter() if element.get("name")}
    generated_names = [*material_names.values(), *fixture_names.values(), *generated_table_names]
    duplicate_generated = sorted(name for name in set(generated_names) if generated_names.count(name) > 1)
    if duplicate_generated:
        raise ValueError(f"Generated visual-scene MJCF names are not unique: {duplicate_generated}.")
    collisions = sorted(set(generated_names) & existing_names)
    if collisions:
        raise ValueError(f"Visual-scene MJCF names already exist: {collisions}.")

    joint_count_before = sum(1 for _ in root.iter("joint"))
    freejoint_count_before = sum(1 for _ in root.iter("freejoint"))
    body_count_before = sum(1 for _ in root.iter("body"))
    asset = _ensure_top_level(root, "asset")
    worldbody = _ensure_top_level(root, "worldbody")
    material_metadata: list[dict[str, Any]] = []
    for key, spec in materials.items():
        attributes = {
            "name": material_names[key],
            "rgba": _format_vector(_as_vector(spec["rgba"], 4, f"materials.{key}.rgba")),
        }
        for scalar_name in _MATERIAL_SCALAR_FIELDS:
            if scalar_name in spec:
                attributes[scalar_name] = f"{float(spec[scalar_name]):.9g}"
        ET.SubElement(asset, "material", attributes)
        material_metadata.append(
            {"library_key": key, "mjcf_name": material_names[key], "attributes": deepcopy(attributes)}
        )

    table_added_names: list[str] = []
    if table_plan is None:
        table_metadata = {
            "preserve_source_table": True,
            "table_upgrade_applied": False,
            "table_upgrade_skip_reason": "preserve_source_table",
            "table_center_xy": list(table_center_xy),
            "table_top_z_requested": float(table_top_z),
            "table_full_size": list(table_full_size),
        }
    else:
        table_metadata, table_added_names = _apply_table_upgrade(
            table_plan,
            table_center_xy=table_center_xy,
            table_top_z=table_top_z,
            table_style=table_style,
            material_names=material_names,
            arena_prefix=arena_prefix,
        )

    fixture_metadata: list[dict[str, Any]] = []
    fixture_added_names: list[str] = []
    for index, fixture in enumerate(fixtures):
        fixture_field = f"selected_variant.fixtures[{index}]"
        identifier = _fixture_identifier(fixture, fixture_field)
        primitive_type = str(fixture["type"])
        raw_pos = _as_vector(fixture["pos"], 3, f"{fixture_field}.pos")
        absolute_pos = (
            table_center_xy[0] + raw_pos[0],
            table_center_xy[1] + raw_pos[1],
            raw_pos[2],
        )
        size = _as_vector(
            fixture["size"],
            _PRIMITIVE_SIZE_LENGTHS[primitive_type],
            f"{fixture_field}.size",
            minimum=0.0,
            minimum_inclusive=False,
        )
        quat = _as_vector(fixture["quat"], 4, f"{fixture_field}.quat") if "quat" in fixture else None
        geom_name = fixture_names[identifier]
        _visual_geom(
            worldbody,
            name=geom_name,
            primitive_type=primitive_type,
            pos=absolute_pos,
            size=size,
            material=material_names[str(fixture["material"]).strip()],
            quat=quat,
        )
        fixture_added_names.append(geom_name)
        fixture_metadata.append(
            {
                "identifier": identifier,
                "mjcf_geom_name": geom_name,
                "type": primitive_type,
                "frame": "table_xy_floor",
                "configured_pos": list(raw_pos),
                "absolute_center_pos": list(absolute_pos),
                "size_mjcf_half_extent": list(size),
                "material_key": str(fixture["material"]).strip(),
                "material_name": material_names[str(fixture["material"]).strip()],
                "visual_only": True,
            }
        )

    joint_count_after = sum(1 for _ in root.iter("joint"))
    freejoint_count_after = sum(1 for _ in root.iter("freejoint"))
    body_count_after = sum(1 for _ in root.iter("body"))
    if (joint_count_before, freejoint_count_before, body_count_before) != (
        joint_count_after,
        freejoint_count_after,
        body_count_after,
    ):
        raise AssertionError("Internal error: visual-scene injection changed the MJCF kinematic layout.")
    added_geom_names = [*table_added_names, *fixture_added_names]
    for geom_name in added_geom_names:
        matches = [geom for geom in root.iter("geom") if geom.get("name") == geom_name]
        if len(matches) != 1:
            raise AssertionError(f"Internal error: expected one added visual geom {geom_name!r}.")
        geom = matches[0]
        if geom.get("contype") != "0" or geom.get("conaffinity") != "0" or geom.get("group") != "1":
            raise AssertionError(f"Internal error: added geom {geom_name!r} is not visual-only.")
    return {
        "visual_scene_applied": True,
        "mode": SCENE_MODE,
        "library_sha256": scene.get("library_sha256"),
        "scene_seed": scene.get("scene_seed"),
        "scene_base_seed": scene.get("scene_base_seed"),
        "scene_base_index": scene.get("scene_base_index"),
        "scene_variant_offset": scene.get("scene_variant_offset"),
        "scene_index": scene.get("scene_index"),
        "recipe_count": scene.get("recipe_count"),
        "recipe_sha256": scene.get("recipe_sha256"),
        "materialized_fixture_sha256": scene.get("materialized_fixture_sha256"),
        "geometry_index": scene.get("geometry_index"),
        "geometry_recipe_count": scene.get("geometry_recipe_count"),
        "component_indices": deepcopy(scene.get("component_indices")),
        "component_selection": deepcopy(scene.get("component_selection")),
        "geometry_validation_scope": scene.get("geometry_validation_scope"),
        "workspace_keepout": deepcopy(scene.get("workspace_keepout")),
        "selected_variant_id": variant_id,
        "arena_prefix": arena_prefix,
        "material_count": len(material_metadata),
        "materials": material_metadata,
        "fixture_count": len(fixture_metadata),
        "fixtures": fixture_metadata,
        "table": table_metadata,
        "added_geom_count": len(added_geom_names),
        "added_geom_names": added_geom_names,
        "visual_only_geom_names": added_geom_names,
        "added_body_count": body_count_after - body_count_before,
        "added_joint_count": joint_count_after - joint_count_before,
        "added_freejoint_count": freejoint_count_after - freejoint_count_before,
        "joint_count_before": joint_count_before,
        "joint_count_after": joint_count_after,
        "freejoint_count_before": freejoint_count_before,
        "freejoint_count_after": freejoint_count_after,
        "body_count_before": body_count_before,
        "body_count_after": body_count_after,
        "replay_dof_layout_preserved": True,
        "collision_geoms_added": 0,
    }


def apply_visual_scene(
    root: ET.Element,
    camera_config: dict[str, Any],
    *,
    table_center_xy: tuple[float, float] | list[float],
    table_top_z: float,
    table_full_size: tuple[float, float, float] | list[float],
    preserve_source_table: bool,
    arena_prefix: str,
) -> dict[str, Any]:
    """Inject a resolved scene and, for Franka-style arenas, a real table.

    The mutation is transactional: validation and construction happen on a
    deep copy, and the supplied root is replaced only after all invariants pass.
    """

    if not isinstance(root, ET.Element):
        raise ValueError("root must be an xml.etree.ElementTree.Element.")
    if not isinstance(camera_config, dict):
        raise ValueError("camera_config must be a JSON object.")
    center = _as_vector(table_center_xy, 2, "table_center_xy")
    top_z = _as_number(table_top_z, "table_top_z")
    full_size = _as_vector(
        table_full_size,
        3,
        "table_full_size",
        minimum=0.0,
        minimum_inclusive=False,
    )
    if not isinstance(preserve_source_table, bool):
        raise ValueError("preserve_source_table must be a boolean.")
    arena_prefix = _as_nonempty_string(arena_prefix, "arena_prefix")

    render_profile = camera_config.get("render_profile")
    if render_profile is None:
        return {
            "visual_scene_applied": False,
            "skip_reason": "render_profile_not_configured",
            "replay_dof_layout_preserved": True,
        }
    render_profile = _as_object(render_profile, "render_profile")
    scene = render_profile.get("scene")
    if scene is None:
        return {
            "visual_scene_applied": False,
            "skip_reason": "scene_not_configured",
            "replay_dof_layout_preserved": True,
        }
    scene = _as_object(scene, "render_profile.scene")
    mode = str(scene.get("mode") or "").strip()
    if not mode:
        return {
            "visual_scene_applied": False,
            "skip_reason": "procedural_scene_mode_not_configured",
            "replay_dof_layout_preserved": True,
        }
    if mode != SCENE_MODE:
        raise ValueError(f"Unsupported render_profile.scene.mode {mode!r}.")

    working_root = deepcopy(root)
    metadata = _apply_visual_scene_mutating(
        working_root,
        deepcopy(camera_config),
        table_center_xy=(center[0], center[1]),
        table_top_z=top_z,
        table_full_size=(full_size[0], full_size[1], full_size[2]),
        preserve_source_table=preserve_source_table,
        arena_prefix=arena_prefix,
    )
    _replace_element(root, working_root)
    return metadata


__all__ = ["apply_visual_scene", "resolve_visual_scene_profile"]

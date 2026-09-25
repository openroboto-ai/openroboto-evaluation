#!/usr/bin/env python3
"""Audit whether the official Axis API exposes archivable randomized tasks."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.parse
from typing import Any

import httpx


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_runtime import DEFAULT_MANIFEST, load_manifest, task_specs  # noqa: E402


NOT_READY_EXIT = 78
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TASK_FIELDS = (
    "mjcf_xml",
    "checker_config",
    "initial_state",
    "scene_variants",
    "scene_variants_count",
    "scene_variants_manifest_url",
    "scene_variants_sha256",
    "domain_randomization",
    "runtime_selection",
)


class UpstreamAuditError(RuntimeError):
    """An error that prevents a trustworthy upstream audit."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise UpstreamAuditError("Axis API returned invalid JSON") from exc


def canonical_sha256(value: Any) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise UpstreamAuditError("Axis API response cannot be canonicalized") from exc
    return hashlib.sha256(rendered).hexdigest()


def validate_api_base_url(value: Any) -> tuple[str, str]:
    if not isinstance(value, str):
        raise UpstreamAuditError("Axis API base URL must be a string")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise UpstreamAuditError("Axis API base URL must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise UpstreamAuditError("Axis API base URL must not contain credentials, query, or fragment")
    base_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    openapi_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/openapi.json", "", ""))
    return base_url, openapi_url


def expected_task_names(manifest: dict[str, Any]) -> dict[int, str]:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise UpstreamAuditError("Axis manifest tasks must be a non-empty list")
    try:
        specs = task_specs(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise UpstreamAuditError("Axis manifest task entries are invalid") from exc
    if len(specs) != len(tasks):
        raise UpstreamAuditError("Axis manifest contains duplicate task ids")
    names: dict[int, str] = {}
    for task_id, spec in specs.items():
        name = spec.get("instruction") if isinstance(spec, dict) else None
        if not isinstance(name, str) or not name:
            raise UpstreamAuditError(f"Axis manifest task {task_id} has no instruction")
        names[task_id] = name
    return names


def _url_pointer_present(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    parsed = urllib.parse.urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and parsed.username is None and parsed.password is None


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def analyze_openapi(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise UpstreamAuditError("Axis OpenAPI response must be an object")
    components = payload.get("components")
    components = components if isinstance(components, dict) else {}
    schemas = components.get("schemas")
    schemas = schemas if isinstance(schemas, dict) else {}
    task_read = schemas.get("TaskRead", {})
    task_properties = task_read.get("properties", {}) if isinstance(task_read, dict) else {}
    task_properties = task_properties if isinstance(task_properties, dict) else {}
    runtime = schemas.get("TaskRuntimeSelection", {})
    runtime_properties = runtime.get("properties", {}) if isinstance(runtime, dict) else {}
    runtime_properties = runtime_properties if isinstance(runtime_properties, dict) else {}
    missing_task_fields = sorted(set(TASK_FIELDS) - set(task_properties))
    version_schema = runtime_properties.get("version", {})
    version_schema = version_schema if isinstance(version_schema, dict) else {}
    runtime_v2_declared = version_schema.get("const") == 2
    paths = payload.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    task_path = paths.get("/api/tasks/{task_id}")
    task_path = task_path if isinstance(task_path, dict) else {}
    operation = task_path.get("get")
    operation = operation if isinstance(operation, dict) else {}
    parameters = operation.get("parameters", []) if isinstance(operation, dict) else []
    parameters = parameters if isinstance(parameters, list) else []
    session_cookie_declared = any(
        isinstance(item, dict) and item.get("in") == "cookie" and item.get("name") == "axis_session"
        for item in parameters
    )
    selection_contract_v2_declared = any(
        isinstance(item, dict)
        and item.get("in") == "query"
        and item.get("name") == "selection_contract"
        and isinstance(item.get("schema"), dict)
        and item["schema"].get("maximum") == 2
        for item in parameters
    )
    return {
        "version": payload.get("info", {}).get("version") if isinstance(payload.get("info"), dict) else None,
        "task_endpoint_declared": bool(operation),
        "selection_contract_v2_declared": selection_contract_v2_declared,
        "axis_session_cookie_declared": session_cookie_declared,
        "randomization_fields_declared": not missing_task_fields,
        "missing_task_fields": missing_task_fields,
        "runtime_selection_v2_declared": runtime_v2_declared,
        "schema_ready": bool(operation)
        and selection_contract_v2_declared
        and not missing_task_fields
        and runtime_v2_declared,
        "document_sha256": canonical_sha256(payload),
    }


def _embedded_variant_identifier(variant: Any) -> str | int | None:
    if not isinstance(variant, dict):
        return None
    identifier = next((variant.get(key) for key in ("variant_id", "id", "name") if variant.get(key) is not None), None)
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or identifier == "":
        return None
    return identifier


def _embedded_variant_archivable(variant: Any) -> bool:
    if not isinstance(variant, dict) or _embedded_variant_identifier(variant) is None:
        return False
    runtime = variant.get("payload", variant)
    embedded_runtime = (
        isinstance(runtime, dict)
        and isinstance(runtime.get("mjcf_xml"), str)
        and bool(runtime["mjcf_xml"])
        and isinstance(runtime.get("checker_config"), dict)
        and bool(runtime["checker_config"])
        and isinstance(runtime.get("initial_state"), dict)
        and bool(runtime["initial_state"])
    )
    pointer_url = next(
        (variant.get(key) for key in ("payload_url", "scene_variant_url", "url") if variant.get(key) is not None),
        None,
    )
    pointer_sha = next(
        (
            variant.get(key)
            for key in ("payload_sha256", "scene_variant_sha256", "sha256")
            if variant.get(key) is not None
        ),
        None,
    )
    return embedded_runtime or (_url_pointer_present(pointer_url) and _valid_sha256(pointer_sha))


def analyze_task(
    payload: Any,
    *,
    task_id: int,
    expected_name: str,
    manifest_digest_verified: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise UpstreamAuditError(f"Axis task {task_id} response must be an object")
    identity_matches = payload.get("id") == task_id and payload.get("name") == expected_name
    runtime_fields = {
        "mjcf_xml": isinstance(payload.get("mjcf_xml"), str) and bool(payload["mjcf_xml"]),
        "checker_config": isinstance(payload.get("checker_config"), dict) and bool(payload["checker_config"]),
        "initial_state": isinstance(payload.get("initial_state"), dict) and bool(payload["initial_state"]),
    }
    variants = payload.get("scene_variants")
    embedded_count = len(variants) if isinstance(variants, list) else 0
    embedded_archivable_count = (
        sum(_embedded_variant_archivable(variant) for variant in variants) if isinstance(variants, list) else 0
    )
    embedded_ids = [_embedded_variant_identifier(variant) for variant in variants] if isinstance(variants, list) else []
    embedded_ids_unique = all(identifier is not None for identifier in embedded_ids) and len(set(embedded_ids)) == len(
        embedded_ids
    )
    declared_count = payload.get("scene_variants_count")
    count_valid = type(declared_count) is int and declared_count >= 2
    manifest_pointer = _url_pointer_present(payload.get("scene_variants_manifest_url")) and bool(
        _valid_sha256(payload.get("scene_variants_sha256"))
    )
    embedded_contract_complete = (
        count_valid
        and embedded_count == declared_count
        and embedded_archivable_count == embedded_count
        and embedded_ids_unique
    )
    variants_discoverable = count_valid and (
        embedded_contract_complete or (manifest_pointer and manifest_digest_verified)
    )
    domain_randomization_present = isinstance(payload.get("domain_randomization"), dict) and bool(
        payload["domain_randomization"]
    )
    runtime_selection = payload.get("runtime_selection")
    runtime_selection_v2_present = (
        isinstance(runtime_selection, dict)
        and runtime_selection.get("version") == 2
        and _url_pointer_present(runtime_selection.get("scene_variant_url"))
        and _valid_sha256(runtime_selection.get("scene_variant_sha256"))
    )
    projection = {
        "task_id": task_id,
        "identity_matches": identity_matches,
        "runtime_fields": runtime_fields,
        "declared_variant_count": declared_count if type(declared_count) is int else None,
        "embedded_variant_count": embedded_count,
        "embedded_archivable_count": embedded_archivable_count,
        "embedded_variant_ids_unique": embedded_ids_unique,
        "embedded_contract_complete": embedded_contract_complete,
        "manifest_pointer_present": manifest_pointer,
        "manifest_digest_verified": manifest_digest_verified,
        "declared_variants_sha256": payload.get("scene_variants_sha256")
        if _valid_sha256(payload.get("scene_variants_sha256"))
        else None,
        "domain_randomization_present": domain_randomization_present,
        "runtime_selection_v2_present": runtime_selection_v2_present,
    }
    ready = (
        identity_matches
        and all(runtime_fields.values())
        and variants_discoverable
        and domain_randomization_present
        and runtime_selection_v2_present
    )
    return {
        **projection,
        "ready_for_archive": ready,
        "contract_projection_sha256": canonical_sha256(projection),
    }


def _fetch_bytes(client: httpx.Client, url: str, *, params: dict[str, int] | None = None) -> bytes:
    try:
        with client.stream("GET", url, params=params) as response:
            status_code = response.status_code
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise UpstreamAuditError("Axis API response exceeds 8 MiB")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise UpstreamAuditError("Axis API request failed") from exc
    if status_code != 200:
        raise UpstreamAuditError(f"Axis API returned HTTP {status_code}")
    return b"".join(chunks)


def _fetch_json(client: httpx.Client, url: str, *, params: dict[str, int] | None = None) -> Any:
    return strict_json(_fetch_bytes(client, url, params=params))


def verify_manifest_pointer(client: httpx.Client, payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    url = payload.get("scene_variants_manifest_url")
    declared_sha256 = payload.get("scene_variants_sha256")
    if not _url_pointer_present(url) or not isinstance(declared_sha256, str) or not SHA256.fullmatch(declared_sha256):
        return False
    return hashlib.sha256(_fetch_bytes(client, url)).hexdigest() == declared_sha256


def audit(manifest_path: pathlib.Path, *, session: str | None = None) -> tuple[dict[str, Any], int]:
    try:
        manifest = load_manifest(manifest_path.resolve())
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise UpstreamAuditError("AXIS manifest could not be loaded or validated") from exc
    expected_names = expected_task_names(manifest)
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("selection_contract") != 2:
        raise UpstreamAuditError("Axis manifest must select runtime contract v2")
    base_url, openapi_url = validate_api_base_url(runtime.get("task_api_base_url"))
    if session is not None and (
        not session or len(session) > 8192 or any(not 33 <= ord(character) <= 126 for character in session)
    ):
        raise UpstreamAuditError("Axis session cookie must be 1-8192 visible ASCII characters")
    cookies = httpx.Cookies()
    if session is not None:
        api_hostname = urllib.parse.urlsplit(base_url).hostname
        if api_hostname is None:  # Defensive: validate_api_base_url already requires a host.
            raise UpstreamAuditError("Axis API base URL has no hostname")
        cookies.set("axis_session", session, domain=api_hostname, path="/")
    headers = {"User-Agent": "validator-axis-upstream-audit/1"}
    with httpx.Client(
        follow_redirects=False,
        timeout=30.0,
        cookies=cookies,
        headers=headers,
    ) as api_client, httpx.Client(
        follow_redirects=False,
        timeout=30.0,
        headers=headers,
    ) as artifact_client:
        openapi = analyze_openapi(_fetch_json(api_client, openapi_url))
        tasks: list[dict[str, Any]] = []
        for task_id, expected_name in sorted(expected_names.items()):
            payload = _fetch_json(
                api_client,
                f"{base_url}/tasks/{task_id}",
                params={"selection_contract": runtime["selection_contract"]},
            )
            tasks.append(
                analyze_task(
                    payload,
                    task_id=task_id,
                    expected_name=expected_name,
                    manifest_digest_verified=verify_manifest_pointer(artifact_client, payload),
                )
            )
    ready_tasks = sum(item["ready_for_archive"] for item in tasks)
    ready = openapi["schema_ready"] and ready_tasks == len(tasks)
    report = {
        "schema_version": 1,
        "captured_at": datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds"),
        "api_origin": urllib.parse.urlsplit(base_url).netloc,
        "session_cookie_present": session is not None,
        "secret_value_or_digest_emitted": False,
        "benchmark": manifest["name"],
        "selection_contract": runtime["selection_contract"],
        "openapi": openapi,
        "task_count": len(tasks),
        "ready_task_count": ready_tasks,
        "ready_for_archive": ready,
        "tasks": tasks,
    }
    return report, 0 if ready else NOT_READY_EXIT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--session-env",
        default="AXIS_SESSION",
        help="Environment variable containing an optional Axis session cookie; the value is never emitted",
    )
    args = parser.parse_args()
    try:
        report, exit_code = audit(args.manifest, session=os.environ.get(args.session_env))
    except (OSError, ValueError, UpstreamAuditError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "audit_valid": False,
                    "error": str(exc),
                    "secret_value_or_digest_emitted": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(report, indent=2, sort_keys=True))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

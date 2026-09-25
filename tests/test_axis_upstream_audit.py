import hashlib
import json
import pathlib
import sys
import unittest

import httpx


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_axis_upstream import (  # noqa: E402
    UpstreamAuditError,
    analyze_openapi,
    analyze_task,
    expected_task_names,
    strict_json,
    validate_api_base_url,
    verify_manifest_pointer,
)


def _task(**overrides):
    payload = {
        "id": 501,
        "name": "Put the Brush in the Basket",
        "mjcf_xml": "<mujoco/>",
        "checker_config": {"checker_type": "CompositeChecker"},
        "initial_state": {"qpos": [0.0]},
        "scene_variants": [
            {
                "name": "a",
                "payload": {
                    "mjcf_xml": "<mujoco/>",
                    "checker_config": {"checker_type": "CompositeChecker"},
                    "initial_state": {"qpos": [0.0]},
                },
            },
            {
                "name": "b",
                "scene_variant_url": "https://assets.example.test/501/b.json",
                "scene_variant_sha256": "b" * 64,
            },
        ],
        "scene_variants_count": 2,
        "scene_variants_manifest_url": None,
        "scene_variants_sha256": None,
        "domain_randomization": {"object_pose": True},
        "runtime_selection": {
            "version": 2,
            "scene_variant_url": "https://assets.example.test/501/a.json",
            "scene_variant_sha256": "a" * 64,
        },
    }
    payload.update(overrides)
    return payload


def _openapi():
    task_properties = {
        name: {}
        for name in (
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
    }
    return {
        "info": {"version": "0.1.0"},
        "paths": {
            "/api/tasks/{task_id}": {
                "get": {
                    "parameters": [
                        {
                            "in": "query",
                            "name": "selection_contract",
                            "schema": {"maximum": 2},
                        },
                        {"in": "cookie", "name": "axis_session", "schema": {}},
                    ]
                }
            }
        },
        "components": {
            "schemas": {
                "TaskRead": {"properties": task_properties},
                "TaskRuntimeSelection": {"properties": {"version": {"const": 2}}},
            }
        },
    }


class TestAxisUpstreamAudit(unittest.TestCase):
    def test_release_manifest_supplies_all_expected_task_names(self):
        manifest = json.loads((ROOT / "configs" / "benchmarks" / "axis_v1.0.json").read_text(encoding="utf-8"))
        names = expected_task_names(manifest)
        self.assertEqual(len(names), 30)
        self.assertEqual(names[501], "Put the Brush in the Basket")

    def test_manifest_task_contract_rejects_missing_names_and_duplicate_ids(self):
        base = {"tasks": [{"task_id": 501, "instruction": "task 501"}]}
        self.assertEqual(expected_task_names(base), {501: "task 501"})
        for tasks in (
            [],
            [{"task_id": 501}],
            [
                {"task_id": 501, "instruction": "first"},
                {"task_id": 501, "instruction": "duplicate"},
            ],
        ):
            with self.subTest(tasks=tasks), self.assertRaises(UpstreamAuditError):
                expected_task_names({"tasks": tasks})

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self):
        for raw in (b'{"value": 1, "value": 2}', b'{"value": NaN}', b"\xff"):
            with self.subTest(raw=raw), self.assertRaises(UpstreamAuditError):
                strict_json(raw)

    def test_api_url_requires_https_and_strips_no_credentials(self):
        self.assertEqual(
            validate_api_base_url("https://api.axis.example/api"),
            ("https://api.axis.example/api", "https://api.axis.example/openapi.json"),
        )
        for url in ("http://api.example/api", "https://user:secret@api.example/api", "file:///tmp/api"):
            with self.subTest(url=url), self.assertRaises(UpstreamAuditError):
                validate_api_base_url(url)

    def test_openapi_requires_complete_randomization_contract(self):
        complete = analyze_openapi(_openapi())
        self.assertTrue(complete["schema_ready"])
        self.assertTrue(complete["axis_session_cookie_declared"])
        incomplete = _openapi()
        del incomplete["components"]["schemas"]["TaskRead"]["properties"]["scene_variants"]
        result = analyze_openapi(incomplete)
        self.assertFalse(result["schema_ready"])
        self.assertEqual(result["missing_task_fields"], ["scene_variants"])

    def test_malformed_openapi_fields_fail_closed_without_crashing(self):
        for payload in (
            {"components": None, "paths": None},
            {
                "components": {
                    "schemas": {
                        "TaskRead": {"properties": None},
                        "TaskRuntimeSelection": {"properties": {"version": 2}},
                    }
                },
                "paths": {"/api/tasks/{task_id}": {"get": {"parameters": None}}},
            },
        ):
            with self.subTest(payload=payload):
                self.assertFalse(analyze_openapi(payload)["schema_ready"])

    def test_embedded_variants_are_ready_without_manifest_pointer(self):
        result = analyze_task(_task(), task_id=501, expected_name="Put the Brush in the Basket")
        self.assertTrue(result["ready_for_archive"])
        self.assertEqual(result["embedded_variant_count"], 2)
        self.assertFalse(result["manifest_pointer_present"])

    def test_manifest_pointer_is_ready_without_embedded_variants(self):
        result = analyze_task(
            _task(
                scene_variants=None,
                scene_variants_manifest_url="https://assets.example.test/501/manifest.json",
                scene_variants_sha256="b" * 64,
            ),
            task_id=501,
            expected_name="Put the Brush in the Basket",
            manifest_digest_verified=True,
        )
        self.assertTrue(result["ready_for_archive"])
        self.assertEqual(result["embedded_variant_count"], 0)
        self.assertTrue(result["manifest_pointer_present"])

    def test_manifest_pointer_fetch_verifies_exact_bytes_without_redirects(self):
        manifest_bytes = b'{"variants":[]}'

        def handler(request):
            self.assertNotIn("cookie", request.headers)
            if request.url.path == "/redirect":
                return httpx.Response(302, headers={"location": "https://other.example.test/manifest"})
            return httpx.Response(200, content=manifest_bytes)

        payload = {
            "scene_variants_manifest_url": "https://assets.example.test/manifest",
            "scene_variants_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False) as client:
            self.assertTrue(verify_manifest_pointer(client, payload))
            self.assertFalse(verify_manifest_pointer(client, {**payload, "scene_variants_sha256": "0" * 64}))
            with self.assertRaises(UpstreamAuditError):
                verify_manifest_pointer(
                    client,
                    {**payload, "scene_variants_manifest_url": "https://assets.example.test/redirect"},
                )

    def test_null_or_partial_variant_contract_fails_closed(self):
        for overrides in (
            {
                "mjcf_xml": None,
                "checker_config": None,
                "initial_state": None,
                "scene_variants": None,
                "scene_variants_count": None,
                "domain_randomization": None,
                "runtime_selection": None,
            },
            {"scene_variants": [{"name": "label-only"}, {}], "scene_variants_count": 2},
            {
                "scene_variants": [
                    _task()["scene_variants"][0],
                    {**_task()["scene_variants"][1], "name": "a"},
                ]
            },
            {"scene_variants_count": 3},
            {"scene_variants": None, "scene_variants_sha256": "bad"},
            {"scene_variants": None, "scene_variants_sha256": 123},
            {"runtime_selection": {"version": 2}},
            {
                "runtime_selection": {
                    "version": 2,
                    "scene_variant_url": "https://assets.example.test/501/a.json",
                    "scene_variant_sha256": 123,
                }
            },
            {"id": 999},
        ):
            with self.subTest(overrides=overrides):
                result = analyze_task(
                    _task(**overrides),
                    task_id=501,
                    expected_name="Put the Brush in the Basket",
                )
                self.assertFalse(result["ready_for_archive"])
                serialized = json.dumps(result)
                self.assertNotIn("scene_variant_url", serialized)


if __name__ == "__main__":
    unittest.main()

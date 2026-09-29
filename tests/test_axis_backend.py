import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
import contextlib
import io
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libero_eval"))

from axis_backend import (  # noqa: E402
    AXIS_BENCHMARKS,
    _checkpoint_provenance,
    _manifest_path,
    _run_task,
    run as run_axis,
)
from run_eval import _reject_model, start_servers  # noqa: E402


class TestAxisBackend(unittest.TestCase):
    def test_early_model_rejection_uses_worker_contract_exit_code(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            _reject_model("empty checkpoint")
        self.assertEqual(raised.exception.code, 3)
        self.assertIn("model REJECTED: empty checkpoint", output.getvalue())

    def test_checkpoint_provenance_is_allowlisted(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = pathlib.Path(temporary)
            (checkpoint / "axis_vla_metadata.json").write_text(
                '{"artifact_sha256":"abc","eligible_for_scoring":false,"training_scope":"action-expert","secret":"drop-me"}'
            )
            provenance = _checkpoint_provenance(checkpoint)
        self.assertEqual(
            provenance, {"artifact_sha256": "abc", "eligible_for_scoring": False, "training_scope": "action-expert"}
        )
        self.assertEqual(AXIS_BENCHMARKS, ("axis_v2.0", "axis_v1.0", "axis"))

    def test_custom_manifest_is_explicit_and_cannot_replace_named_releases(self):
        with self.assertRaisesRegex(ValueError, "requires --axis-manifest"):
            _manifest_path("axis")
        with self.assertRaisesRegex(ValueError, "named releases cannot be overridden"):
            _manifest_path("axis_v1.0", "/tmp/different.json")
        self.assertEqual(_manifest_path("axis", "/tmp/new-release.json"), pathlib.Path("/tmp/new-release.json"))

    def test_axis_v1_dry_run_uses_frozen_base_scenes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            axis_python = root / "python"
            axis_python.touch()
            args = types.SimpleNamespace(
                benchmark="axis_v1.0",
                task_ids="501",
                num_trials=None,
                axis_randomization_manifest=None,
                axis_randomization_seed=None,
                axis_cache_root=str(root / "cache"),
                axis_asset_fetch_workers=2,
                axis_policy_host="127.0.0.1",
                axis_policy_port=None,
                base_port=9000,
                axis_replan_steps=10,
                axis_task_api_base_url=None,
                axis_asset_base_url=None,
                axis_max_control_steps=None,
                task_timeout=60,
                suites="",
                tasks="",
                init_seed=None,
                init_states_root=None,
                output_dir=str(root / "output"),
                dry_run=True,
                model=".",
                model_family="openpi",
                backbone="pi0.5",
                commit_id="local",
                evaluator_source_git_commit=None,
                seed=7,
            )
            task_result = {
                "status": "ok",
                "benchmark": "axis_v1.0",
                "randomization": False,
                "num_trials": 0,
                "num_successes": 0,
            }
            with mock.patch("axis_backend._run_task", return_value=task_result) as run_task:
                self.assertEqual(run_axis(args, None, [0], axis_python), 0)
            command = run_task.call_args.args[0]
            self.assertNotIn("--randomization-manifest", command)
            self.assertNotIn("--randomization-seed", command)
            summary = __import__("json").loads((root / "output" / "summary.json").read_text())
            self.assertEqual(summary["benchmark"], "axis_v1.0")
            self.assertEqual(summary["protocol_revision"], "axis_v1.0_30tasks_native_joint_osmesa_v1")
            self.assertIsNone(summary["randomization_seed"])
            self.assertEqual(summary["num_trials_per_task"], 0)

    def test_custom_manifest_keeps_version_identity_and_uses_its_own_tasks(self):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = json.loads((ROOT / "configs/benchmarks/axis_v1.0.json").read_text())
            manifest.update(name="axis_v99.1", protocol_revision="axis_demo_extension_test_v1")
            manifest["tasks"] = [{**manifest["tasks"][0], "task_id": 507}]
            manifest_path = root / "axis_v99.1.json"
            manifest_path.write_text(json.dumps(manifest))
            args = types.SimpleNamespace(
                benchmark="axis",
                axis_manifest=str(manifest_path),
                task_ids="507",
                num_trials=20,
                axis_cache_root=str(root / "cache"),
                axis_asset_fetch_workers=2,
                axis_policy_host="127.0.0.1",
                axis_policy_port=None,
                base_port=9000,
                axis_replan_steps=10,
                axis_task_api_base_url=None,
                axis_asset_base_url=None,
                axis_max_control_steps=None,
                task_timeout=60,
                suites="",
                tasks="",
                init_seed=None,
                init_states_root=None,
                output_dir=str(root / "output"),
                dry_run=True,
                model=".",
                model_family="openpi",
                backbone="pi0.5",
                commit_id="local",
                evaluator_source_git_commit=None,
                seed=0,
            )
            axis_python = root / "python"
            axis_python.touch()
            result = {"status": "ok", "benchmark": "axis_v99.1", "task_id": 507}
            with mock.patch("axis_backend._run_task", return_value=result) as run_task:
                self.assertEqual(run_axis(args, None, [0], axis_python), 0)
            command = run_task.call_args.args[0]
            self.assertEqual(command[command.index("--manifest") + 1], str(manifest_path))
            self.assertEqual(command[command.index("--task-id") + 1], "507")
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertEqual(summary["benchmark"], "axis_v99.1")
            self.assertEqual(summary["protocol_revision"], "axis_demo_extension_test_v1")
            self.assertEqual(list(summary["tasks"]), ["507"])
            args.dry_run = False
            args.config = "pi05_axis_joint"
            args.mem_fraction = 0.7
            args.server_impl = "upstream"
            args.max_batch = 1
            args.workers_per_gpu = 1
            args.server_timeout = 60
            servers = mock.Mock(return_value=[types.SimpleNamespace(port=9000)])
            wait, stop = mock.Mock(), mock.Mock()
            with mock.patch("axis_backend._run_task", return_value=result):
                self.assertEqual(
                    run_axis(
                        args,
                        root / "checkpoint",
                        [0],
                        axis_python,
                        start_servers=servers,
                        wait_for_servers=wait,
                        stop_servers=stop,
                    ),
                    0,
                )
            self.assertEqual(servers.call_args.kwargs["benchmark"], "axis")
            stop.assert_called_once()
            self.assertEqual(servers.call_args.kwargs["axis_gripper_mode"], "continuous")
            self.assertEqual(servers.call_args.kwargs["axis_policy_samples"], 1)
            self.assertEqual(servers.call_args.kwargs["axis_sample_reduction"], "mean")
            summary = json.loads((root / "output/summary.json").read_text())
            self.assertIsNone(summary["checkpoint_provenance"])
            self.assertTrue(summary["discrete_state_input"])
            args.axis_sample_reduction = "medoid"
            with self.assertRaisesRegex(ValueError, "sample-reduction differs"):
                run_axis(args, root / "checkpoint", [0], axis_python)
            args.axis_sample_reduction = "mean"
            args.axis_policy_samples = 5
            with self.assertRaisesRegex(ValueError, "policy-samples differs"):
                run_axis(args, root / "checkpoint", [0], axis_python)
            args.axis_policy_samples = 1
            args.axis_gripper_mode = "symmetric-binary"
            with self.assertRaisesRegex(ValueError, "gripper-mode differs"):
                run_axis(args, root / "checkpoint", [0], axis_python)
            args.axis_gripper_mode = "continuous"
            manifest["protocol"]["replan_steps"] = 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "replan-steps differs"):
                run_axis(args, root / "checkpoint", [0], axis_python)

    def test_task_process_bypasses_proxy_for_local_policy_server(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "os.environ",
            {"HTTP_PROXY": "http://proxy.invalid:8080", "NO_PROXY": "internal.example"},
            clear=True,
        ), mock.patch("axis_backend.subprocess.run") as run:
            result_path = pathlib.Path(temporary) / "result.json"

            def complete(command, **kwargs):
                self.assertEqual(kwargs["env"]["MUJOCO_GL"], "osmesa")
                self.assertNotIn("MUJOCO_EGL_DEVICE_ID", kwargs["env"])
                self.assertEqual(
                    kwargs["env"]["NO_PROXY"],
                    "internal.example,127.0.0.1,localhost,::1",
                )
                self.assertEqual(kwargs["env"]["no_proxy"], kwargs["env"]["NO_PROXY"])
                result_path.write_text('{"status":"ok"}')
                return subprocess.CompletedProcess(command, 0)

            run.side_effect = complete
            result = _run_task(
                ["axis-task"],
                pathlib.Path(temporary) / "task.log",
                result_path,
                gpu=0,
                timeout_s=1,
            )
        self.assertEqual(result["status"], "ok")

    def test_native_axis_checkpoint_uses_validator_owned_openpi_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            process = mock.Mock(pid=1234)
            with mock.patch("run_eval._find_free_ports", return_value=[9100]), mock.patch(
                "run_eval.subprocess.Popen", return_value=process
            ) as popen, mock.patch(
                "run_eval._base_env",
                return_value={
                    "JAX_COMPILATION_CACHE_DIR": "/old-cache",
                    "JAX_ENABLE_COMPILATION_CACHE": "true",
                    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
                    "XLA_FLAGS": "--xla_gpu_per_fusion_autotune_cache_dir=/old-tuning --xla_gpu_autotune_level=4",
                },
            ):
                start_servers(
                    [0],
                    9100,
                    "pi05_axis_joint",
                    root / "checkpoint",
                    logs,
                    0.7,
                    model_family="openpi",
                    benchmark="axis_v1.0",
                    axis_gripper_mode="symmetric-binary",
                )
        command = popen.call_args.args[0]
        self.assertTrue(command[1].endswith("serve_axis_openpi.py"))
        self.assertEqual(command[command.index("--config") + 1], "pi05_axis_joint")
        self.assertEqual(command[command.index("--openpi-root") + 1], str(ROOT / "third_party" / "openpi"))
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--gripper-mode") + 1], "symmetric-binary")
        self.assertEqual(command[command.index("--policy-samples") + 1], "1")
        self.assertEqual(command[command.index("--sample-reduction") + 1], "mean")
        self.assertEqual(popen.call_args.kwargs["cwd"], str(ROOT))
        self.assertEqual(
            popen.call_args.kwargs["env"]["XLA_FLAGS"],
            "--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0",
        )
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["JAX_ENABLE_COMPILATION_CACHE"], "false")
        self.assertEqual(environment["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "none")
        self.assertNotIn("JAX_COMPILATION_CACHE_DIR", environment)


if __name__ == "__main__":
    unittest.main()

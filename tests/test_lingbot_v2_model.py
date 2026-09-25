import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "libero_eval"))

from check_model import (  # noqa: E402
    _LINGBOT_REQUIRED_WEIGHT_KEYS,
    check_lingbot_data_contract,
    check_lingbot_vla_v2_model,
    detect_model_family,
)
from lingbot_runtime import (  # noqa: E402
    DEFAULT_LINGBOT_DATA_CONTRACT,
    DEFAULT_LINGBOT_NORM_STATS,
    LINGBOT_ACTION_REPLAY_ATOL,
    LingbotRequestSeedSelfTestFailure,
    LingbotRequestSeededSampler,
    copy_readonly_observation_arrays,
    disable_lingbot_batch1_vision_cache,
    lingbot_action_drift_summary,
    lingbot_actions_are_equivalent,
    load_data_contract,
    runtime_contract_metadata,
    validate_norm_stats,
    verify_lingbot_request_seed_determinism,
    verify_lingbot_request_seed_determinism_nonblocking,
    warm_up_lingbot_libero_policy,
)
from run_eval import (  # noqa: E402
    LINGBOT_LIBERO_PRO_STATIC_SCHEDULING_PROFILE,
    LINGBOT_LIBERO_PRO_STATIC_SUITE_COSTS,
    TaskSpec,
    _client_env,
    _find_checkpoint_root,
    _mujoco_egl_device_id,
    _policy_server_is_healthy,
    interleaved_static_lane_index,
    partition_static_lanes,
    static_scheduling_cost,
    static_scheduling_profile,
    start_servers,
)


def _write_training_config(path: pathlib.Path):
    moe_layers = ", ".join(str(index) for index in range(36))
    path.write_text(
        f"""\
model:
  config_key: LingbotVLAV2Config
  tokenizer_path: Qwen/Qwen3-VL-4B-Instruct
data:
  joints:
    - arm.position: 14
    - end.position: 14
    - effector.position: 2
  cameras: [camera_top, camera_wrist_left, camera_wrist_right]
train:
  action_dim: 55
  max_action_dim: 55
  max_state_dim: 55
  expert_hidden_size: 768
  token_num_experts: 32
  token_top_k: 4
  token_moe_layers: [{moe_layers}]
"""
    )


def _make_checkpoint(root: pathlib.Path) -> pathlib.Path:
    checkpoint = root / "checkpoints" / "global_step_50000" / "hf_ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps({
            "model_type": "lingbotvla",
            "vlm_family": "qwen3_vl",
            "architectures": ["LingbotVlaV2Policy"],
            "tokenizer_path": "Qwen/Qwen3-VL-4B-Instruct",
            "action_dim": 55,
            "max_action_dim": 55,
            "max_state_dim": 55,
            "chunk_size": 50,
            "expert_hidden_size": 768,
            "token_num_experts": 32,
            "token_top_k": 4,
            "token_moe_layers": list(range(36)),
        })
    )
    shard = checkpoint / "model-00001-of-00001.safetensors"
    # Sparse file: validates byte accounting without consuming 25 GB in tests.
    with shard.open("wb") as file:
        file.truncate(25_503_630_044)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 25_503_630_044},
            "weight_map": {key: shard.name for key in _LINGBOT_REQUIRED_WEIGHT_KEYS},
        })
    )
    _write_training_config(root / "lingbotvla_cli.yaml")
    return checkpoint


class TestLingbotV2Checkpoint(unittest.TestCase):
    def test_accepts_official_nested_package_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = _make_checkpoint(root)
            result = check_lingbot_vla_v2_model(checkpoint)
            self.assertTrue(result.ok, result.errors)
            self.assertEqual(detect_model_family(checkpoint), "lingbot_vla_v2")
            self.assertEqual(_find_checkpoint_root(root), checkpoint)

    def test_missing_architecture_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _make_checkpoint(pathlib.Path(tmp))
            index_path = checkpoint / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"].pop("model.action_out_proj.weight")
            index_path.write_text(json.dumps(index))
            result = check_lingbot_vla_v2_model(checkpoint)
            self.assertFalse(result.ok)
            self.assertTrue(any("architecture tensors" in error for error in result.errors))

    def test_architecture_defining_training_field_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _make_checkpoint(pathlib.Path(tmp))
            config_path = checkpoint / "config.json"
            config = json.loads(config_path.read_text())
            config["action_dim"] = 54
            config_path.write_text(json.dumps(config))
            result = check_lingbot_vla_v2_model(checkpoint)
            self.assertFalse(result.ok)
            self.assertIn("config.json action_dim must be 55", "\n".join(result.errors))

    def test_standard_hf_checkpoint_does_not_require_training_directory_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            checkpoint = _make_checkpoint(root)
            (root / "lingbotvla_cli.yaml").unlink()
            result = check_lingbot_vla_v2_model(checkpoint)
            self.assertTrue(result.ok, result.errors)
            self.assertEqual(
                check_lingbot_data_contract(
                    checkpoint,
                    ("camera_top", "camera_wrist"),
                    {"end.position": 14, "effector.position": 2},
                    metadata_required=False,
                ),
                [],
            )

    def test_benchmark_data_contract_distinguishes_robotwin_from_libero(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = _make_checkpoint(pathlib.Path(tmp))
            robotwin_errors = check_lingbot_data_contract(
                checkpoint,
                ("camera_top", "camera_wrist_left", "camera_wrist_right"),
                {"arm.position": 14, "end.position": 14, "effector.position": 2},
            )
            libero_errors = check_lingbot_data_contract(
                checkpoint,
                ("camera_top", "camera_wrist"),
                {"end.position": 14, "effector.position": 2},
            )
            self.assertEqual(robotwin_errors, [])
            self.assertTrue(any("data.cameras" in error for error in libero_errors))


class TestLingbotLiberoRuntimeContract(unittest.TestCase):
    def test_dynamic_batching_disables_nested_compiled_vision_cache(self):
        config_objects = [types.SimpleNamespace(precompute_grid_thw=True) for _ in range(4)]
        vision_model = types.SimpleNamespace(
            config=config_objects[3],
            pos_embeds="cached",
            position_embeddings="cached",
            cu_seqlens="cached",
            visual_split_sizes=[64, 64],
            visual_max_seqlen=64,
        )
        compiled_wrapper = types.SimpleNamespace(_orig_mod=vision_model)
        outer_model = types.SimpleNamespace(
            config=config_objects[2],
            qwenvl_with_expert=compiled_wrapper,
        )
        vla = types.SimpleNamespace(config=config_objects[1], model=outer_model)
        policy = types.SimpleNamespace(config=config_objects[0], vla=vla)

        self.assertTrue(disable_lingbot_batch1_vision_cache(policy))

        self.assertTrue(all(not config.precompute_grid_thw for config in config_objects))
        self.assertIsNone(vision_model.pos_embeds)
        self.assertIsNone(vision_model.position_embeddings)
        self.assertIsNone(vision_model.cu_seqlens)
        self.assertIsNone(vision_model.visual_split_sizes)
        self.assertIsNone(vision_model.visual_max_seqlen)
        self.assertFalse(disable_lingbot_batch1_vision_cache(policy))

    def test_readonly_observation_arrays_are_copied_before_tensor_conversion(self):
        class FakeArray:
            def __init__(self, writeable, marker):
                self.flags = types.SimpleNamespace(writeable=writeable)
                self.marker = marker

            def copy(self):
                return FakeArray(True, f"{self.marker}-copy")

        readonly = FakeArray(False, "image")
        writeable = FakeArray(True, "state")
        source = {"image": readonly, "state": writeable, "task": "pick object"}

        result = copy_readonly_observation_arrays(source)

        self.assertIs(source["image"], readonly)
        self.assertIsNot(result["image"], readonly)
        self.assertEqual(result["image"].marker, "image-copy")
        self.assertIs(result["state"], writeable)
        self.assertEqual(result["task"], "pick object")

    def test_policy_server_readiness_uses_http_health_endpoint(self):
        connection = mock.Mock()
        response = mock.Mock(status=200)
        connection.getresponse.return_value = response
        with mock.patch("run_eval.http.client.HTTPConnection", return_value=connection) as http_connection:
            self.assertTrue(_policy_server_is_healthy(9006, timeout_s=0.5))

        http_connection.assert_called_once_with("127.0.0.1", 9006, timeout=0.5)
        connection.request.assert_called_once_with("GET", "/healthz")
        response.read.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_action_replay_equivalence_accepts_only_bounded_finite_drift(self):
        tolerance = LINGBOT_ACTION_REPLAY_ATOL
        self.assertTrue(lingbot_actions_are_equivalent([0.0, 1.0], [tolerance / 2, 1.0 - tolerance / 2]))
        self.assertFalse(lingbot_actions_are_equivalent([0.0], [tolerance + 1e-6]))
        self.assertFalse(lingbot_actions_are_equivalent([0.0], [0.0, 0.0]))
        self.assertFalse(lingbot_actions_are_equivalent([float("nan")], [float("nan")]))

    def test_request_seeded_sampler_passes_reproducible_explicit_noise(self):
        class FakeGenerator:
            def __init__(self, *, device):
                self.device = device
                self.seed = None

            def manual_seed(self, seed):
                self.seed = seed

        calls = []

        def randn(shape, *, device, dtype, generator=None):
            return (shape, device, dtype, None if generator is None else generator.seed)

        def sample_actions(*args, **kwargs):
            calls.append((args, kwargs))
            return kwargs["noise"]

        sampler = LingbotRequestSeededSampler(
            sample_actions,
            generator_factory=FakeGenerator,
            randn=randn,
            action_steps=50,
            action_dimension=32,
        )
        state = types.SimpleNamespace(shape=(2, 8), device="cuda:0", dtype="bfloat16")
        model_args = ("images", "image_masks", "tokens", "token_masks", state)

        unseeded = sampler(*model_args, image_grid_thw="grid")
        with sampler.request_seed(123):
            first = sampler(*model_args, image_grid_thw="grid")
        with sampler.request_seed(456):
            interleaved = sampler(*model_args, image_grid_thw="grid")
        with sampler.request_seed(123):
            repeated = sampler(*model_args, image_grid_thw="grid")

        self.assertEqual(unseeded, ((2, 50, 32), "cuda:0", "bfloat16", None))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, interleaved)
        self.assertTrue(all(call[1]["image_grid_thw"] == "grid" for call in calls))

    def test_request_seeded_sampler_rejects_nested_seed_scopes(self):
        sampler = LingbotRequestSeededSampler(
            lambda *_args, **_kwargs: None,
            generator_factory=lambda **_kwargs: None,
            randn=lambda *_args, **_kwargs: None,
            action_steps=50,
            action_dimension=32,
        )
        with sampler.request_seed(1):
            with self.assertRaisesRegex(RuntimeError, "cannot be nested"):
                with sampler.request_seed(2):
                    pass

    def test_request_seeded_sampler_generates_each_batch_noise_independently(self):
        class FakeGenerator:
            def __init__(self, *, device):
                self.device = device
                self.seed = None

            def manual_seed(self, seed):
                self.seed = seed

        def randn(shape, *, device, dtype, generator=None):
            return {"shape": shape, "device": device, "dtype": dtype, "seed": generator.seed}

        def concatenate(chunks, *, dim):
            self.assertEqual(dim, 0)
            return chunks

        sampler = LingbotRequestSeededSampler(
            lambda *_args, **kwargs: kwargs["noise"],
            generator_factory=FakeGenerator,
            randn=randn,
            concatenate=concatenate,
            action_steps=50,
            action_dimension=32,
        )
        state = types.SimpleNamespace(shape=(3, 8), device="cuda:0", dtype="bfloat16")
        model_args = ("images", "image_masks", "tokens", "token_masks", state)

        with sampler.request_seeds([11, 22, 33]):
            noise = sampler(*model_args)

        self.assertEqual([item["seed"] for item in noise], [11, 22, 33])
        self.assertTrue(all(item["shape"] == (1, 50, 32) for item in noise))
        with sampler.request_seeds([11, 22]):
            with self.assertRaisesRegex(ValueError, "3 samples but 2 request seeds"):
                sampler(*model_args)

    def test_bundled_contract_and_norm_stats_are_valid(self):
        contract = load_data_contract(DEFAULT_LINGBOT_DATA_CONTRACT)
        stats = validate_norm_stats(DEFAULT_LINGBOT_NORM_STATS)
        metadata = runtime_contract_metadata(DEFAULT_LINGBOT_DATA_CONTRACT, DEFAULT_LINGBOT_NORM_STATS)

        self.assertEqual(contract.cameras, ["camera_top", "camera_wrist"])
        self.assertEqual(stats["count"], 273465)
        self.assertEqual(metadata["norm_stats_count"], 273465)
        self.assertEqual(len(metadata["norm_stats_sha256"]), 64)

    def test_norm_stats_dimension_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "stats.json"
            payload = json.loads(DEFAULT_LINGBOT_NORM_STATS.read_text())
            payload["norm_stats"]["action.end.position"]["q01"].pop()
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "must contain 6 finite numbers"):
                validate_norm_stats(path)

    def test_cold_start_warmup_uses_libero_contract_and_resets_policy(self):
        arrays = []
        synchronizations = []

        def zeros(shape, *, dtype):
            value = {"shape": shape, "dtype": dtype}
            arrays.append(value)
            return value

        class FakePolicy:
            def __init__(self):
                self.requests = []

            def infer(self, request):
                self.requests.append(request)
                if request.get("reset"):
                    return {"action": None}
                return {"action": [[0.0] * 7 for _ in range(5)]}

        policy = FakePolicy()
        duration = warm_up_lingbot_libero_policy(
            policy,
            zeros=zeros,
            synchronize=lambda: synchronizations.append(True),
        )

        self.assertGreaterEqual(duration, 0)
        self.assertEqual(policy.requests[0], {"reset": True, "robo_name": "libero"})
        self.assertEqual(policy.requests[-1], {"reset": True, "robo_name": "libero"})
        self.assertEqual(policy.requests[1]["observation.image"], arrays[0])
        self.assertEqual(policy.requests[1]["observation.wrist_image"], arrays[1])
        self.assertEqual(policy.requests[1]["observation.state"], arrays[2])
        self.assertEqual(arrays[0], {"shape": (256, 256, 3), "dtype": "uint8"})
        self.assertEqual(arrays[1], {"shape": (256, 256, 3), "dtype": "uint8"})
        self.assertEqual(arrays[2], {"shape": (8,), "dtype": "float32"})
        self.assertTrue(policy.requests[1]["task"])
        self.assertEqual(synchronizations, [True])

    def test_cold_start_warmup_rejects_short_action_and_still_resets(self):
        class FakePolicy:
            def __init__(self):
                self.requests = []

            def infer(self, request):
                self.requests.append(request)
                return {"action": None} if request.get("reset") else {"action": [[0.0] * 7] * 4}

        policy = FakePolicy()
        with self.assertRaisesRegex(RuntimeError, "returned 4 action steps"):
            warm_up_lingbot_libero_policy(
                policy,
                zeros=lambda shape, *, dtype: (shape, dtype),
                synchronize=lambda: None,
            )
        self.assertEqual(policy.requests[-1], {"reset": True, "robo_name": "libero"})

    def test_request_seed_self_test_interleaves_and_reproduces_actions(self):
        class FakePolicy:
            def __init__(self):
                self.requests = []

            def infer(self, request):
                self.requests.append(request)
                if request.get("reset"):
                    return {"action": None}
                seed = request["_evaluation_seed"]
                return {"action": [[float(seed % 997)] * 7 for _ in range(5)]}

        policy = FakePolicy()
        synchronizations = []
        duration = verify_lingbot_request_seed_determinism(
            policy,
            zeros=lambda shape, *, dtype: (shape, dtype),
            synchronize=lambda: synchronizations.append(True),
            actions_equal=lambda left, right: left == right,
            seed_field="_evaluation_seed",
        )
        sampled = [request["_evaluation_seed"] for request in policy.requests if "_evaluation_seed" in request]
        self.assertGreaterEqual(duration, 0)
        self.assertEqual(sampled[0], sampled[2])
        self.assertNotEqual(sampled[0], sampled[1])
        self.assertEqual(len(synchronizations), 3)
        self.assertEqual(policy.requests[-1], {"reset": True, "robo_name": "libero"})

    def test_request_seed_self_test_rejects_non_reproducible_policy(self):
        class FakePolicy:
            def __init__(self):
                self.counter = 0

            def infer(self, request):
                if request.get("reset"):
                    return {"action": None}
                self.counter += 1
                return {"action": [[float(self.counter)] * 7 for _ in range(5)]}

        with self.assertRaisesRegex(LingbotRequestSeedSelfTestFailure, "same-seed replay drift exceeded tolerance"):
            verify_lingbot_request_seed_determinism(
                FakePolicy(),
                zeros=lambda shape, *, dtype: (shape, dtype),
                synchronize=lambda: None,
                actions_equal=lambda left, right: left == right,
                seed_field="_evaluation_seed",
            )

    def test_action_drift_summary_reports_magnitude_and_old_tolerance(self):
        summary = lingbot_action_drift_summary([[0.0, 1.0]], [[0.001, 1.0]])
        self.assertIn("max_abs_diff=0.001", summary)
        self.assertIn("changed=1/2", summary)
        self.assertIn("within_atol_0.02=True", summary)

    def test_request_seed_self_test_checks_the_exact_static_batch_shape(self):
        class FakeBatchPolicy:
            def __init__(self):
                self.batch_sizes = []

            def infer(self, request):
                if request.get("reset"):
                    return {"action": None}
                raise AssertionError("static batch self-test must not call scalar infer")

            def infer_batch(self, requests):
                self.batch_sizes.append(len(requests))
                return [
                    {"action": [[float(request["_evaluation_seed"] % 997)] * 7 for _ in range(5)]}
                    for request in requests
                ]

        policy = FakeBatchPolicy()
        verify_lingbot_request_seed_determinism(
            policy,
            zeros=lambda shape, *, dtype: (shape, dtype),
            synchronize=lambda: None,
            actions_equal=lambda left, right: left == right,
            seed_field="_evaluation_seed",
            batch_size=4,
        )
        self.assertEqual(policy.batch_sizes, [4, 4, 4])

    def test_request_seed_self_test_rejects_policy_that_ignores_seed(self):
        class FakePolicy:
            def infer(self, request):
                if request.get("reset"):
                    return {"action": None}
                return {"action": [[1.0] * 7 for _ in range(5)]}

        with self.assertRaisesRegex(LingbotRequestSeedSelfTestFailure, "different seeds produced equivalent actions"):
            verify_lingbot_request_seed_determinism(
                FakePolicy(),
                zeros=lambda shape, *, dtype: (shape, dtype),
                synchronize=lambda: None,
                actions_equal=lambda left, right: left == right,
                seed_field="_evaluation_seed",
            )

    def test_request_seed_self_test_failure_can_be_reported_without_blocking_startup(self):
        class NonReproduciblePolicy:
            def __init__(self):
                self.counter = 0

            def infer(self, request):
                if request.get("reset"):
                    return {"action": None}
                self.counter += 1
                return {"action": [[float(self.counter)] * 7 for _ in range(5)]}

        failures = []
        duration = verify_lingbot_request_seed_determinism_nonblocking(
            NonReproduciblePolicy(),
            on_failure=failures.append,
            zeros=lambda shape, *, dtype: (shape, dtype),
            synchronize=lambda: None,
            actions_equal=lambda left, right: left == right,
            seed_field="_evaluation_seed",
        )

        self.assertIsNone(duration)
        self.assertEqual(len(failures), 1)
        self.assertIn("same-seed replay drift exceeded tolerance", failures[0])

    def test_nonblocking_request_seed_check_still_propagates_inference_failures(self):
        class BrokenPolicy:
            def infer(self, request):
                if request.get("reset"):
                    return {"action": None}
                raise RuntimeError("CUDA execution failed")

        with self.assertRaisesRegex(RuntimeError, "CUDA execution failed"):
            verify_lingbot_request_seed_determinism_nonblocking(
                BrokenPolicy(),
                on_failure=lambda _error: self.fail("infrastructure errors must not be softened"),
                zeros=lambda shape, *, dtype: (shape, dtype),
                synchronize=lambda: None,
                actions_equal=lambda left, right: left == right,
                seed_field="_evaluation_seed",
            )

    def test_policy_server_receives_evaluator_owned_contract_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            process = mock.Mock(pid=1234)
            with mock.patch("run_eval._find_free_ports", return_value=[9100]):
                popen_patch = mock.patch("run_eval.subprocess.Popen", return_value=process)
                with popen_patch as popen:
                    start_servers(
                        [0],
                        9100,
                        "lingbot-vla-v2",
                        root / "checkpoint",
                        logs,
                        0.7,
                        model_family="lingbot_vla_v2",
                        lingbot_norm_stats=DEFAULT_LINGBOT_NORM_STATS,
                        lingbot_robot_config_root=DEFAULT_LINGBOT_DATA_CONTRACT.parent / "robot_configs",
                        lingbot_data_contract=DEFAULT_LINGBOT_DATA_CONTRACT,
                        qwen3_vl_path=root / "qwen",
                    )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--data-contract") + 1], str(DEFAULT_LINGBOT_DATA_CONTRACT))
            self.assertEqual(command[command.index("--norm-stats") + 1], str(DEFAULT_LINGBOT_NORM_STATS))
            self.assertEqual(command[command.index("--seed") + 1], "7")
            self.assertEqual(command[command.index("--torch-threads") + 1], "4")
            self.assertEqual(command[command.index("--torch-interop-threads") + 1], "1")
            self.assertEqual(command[command.index("--dynamic-batching") + 1], "False")
            self.assertEqual(command[command.index("--max-batch") + 1], "4")
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["OMP_NUM_THREADS"], "4")
            self.assertEqual(environment["MKL_NUM_THREADS"], "4")
            self.assertEqual(environment["TORCHINDUCTOR_COMPILE_THREADS"], "4")

    def test_lingbot_batched_server_is_selected_by_server_impl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            process = mock.Mock(pid=1234)
            with mock.patch("run_eval._find_free_ports", return_value=[9100]):
                popen_patch = mock.patch("run_eval.subprocess.Popen", return_value=process)
                with popen_patch as popen:
                    start_servers(
                        [0],
                        9100,
                        "lingbot-vla-v2",
                        root / "checkpoint",
                        logs,
                        0.7,
                        server_impl="batched",
                        max_batch=4,
                        model_family="lingbot_vla_v2",
                        lingbot_norm_stats=DEFAULT_LINGBOT_NORM_STATS,
                        lingbot_robot_config_root=DEFAULT_LINGBOT_DATA_CONTRACT.parent / "robot_configs",
                        lingbot_data_contract=DEFAULT_LINGBOT_DATA_CONTRACT,
                        qwen3_vl_path=root / "qwen",
                    )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--dynamic-batching") + 1], "True")
            self.assertEqual(command[command.index("--max-batch") + 1], "4")

    def test_lingbot_static_server_enables_fixed_lanes_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            process = mock.Mock(pid=1234)
            with mock.patch("run_eval._find_free_ports", return_value=[9100]):
                popen_patch = mock.patch("run_eval.subprocess.Popen", return_value=process)
                with popen_patch as popen:
                    start_servers(
                        [0],
                        9100,
                        "lingbot-vla-v2",
                        root / "checkpoint",
                        logs,
                        0.7,
                        server_impl="static",
                        max_batch=4,
                        lane_count=8,
                        model_family="lingbot_vla_v2",
                        lingbot_norm_stats=DEFAULT_LINGBOT_NORM_STATS,
                        lingbot_robot_config_root=DEFAULT_LINGBOT_DATA_CONTRACT.parent / "robot_configs",
                        lingbot_data_contract=DEFAULT_LINGBOT_DATA_CONTRACT,
                        qwen3_vl_path=root / "qwen",
                    )
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--static-batching") + 1], "True")
            self.assertEqual(command[command.index("--lane-count") + 1], "8")
            self.assertEqual(command[command.index("--deterministic") + 1], "True")
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertEqual(environment["TORCHINDUCTOR_MAX_AUTOTUNE"], "0")

    def test_static_lane_partition_is_stable_and_longest_first_balanced(self):
        specs = [
            TaskSpec("suite", 0, 10),
            TaskSpec("suite", 1, 9),
            TaskSpec("suite", 2, 8),
            TaskSpec("suite", 3, 7),
            TaskSpec("suite", 4, 6),
        ]
        first = partition_static_lanes(specs, 2)
        second = partition_static_lanes(list(specs), 2)
        self.assertEqual([[spec.task_id for spec in lane] for lane in first], [[0, 3, 4], [1, 2]])
        self.assertEqual(first, second)

    def test_static_lane_partition_uses_explicit_runtime_cost(self):
        specs = [
            TaskSpec("suite", 0, 100, scheduling_cost=9),
            TaskSpec("suite", 1, 100, scheduling_cost=8),
            TaskSpec("suite", 2, 100, scheduling_cost=7),
            TaskSpec("suite", 3, 100, scheduling_cost=1),
        ]
        lanes = partition_static_lanes(specs, 2)
        self.assertEqual([[spec.task_id for spec in lane] for lane in lanes], [[0, 3], [1, 2]])

    def test_libero_pro_static_scheduling_profile_is_complete_and_versioned(self):
        expected_suites = {
            f"libero_{base}_{dimension}"
            for base in ("spatial", "object", "goal", "10")
            for dimension in ("object", "swap", "lan", "task")
        }
        self.assertEqual(set(LINGBOT_LIBERO_PRO_STATIC_SUITE_COSTS), expected_suites)
        self.assertEqual(
            static_scheduling_profile("libero_pro"),
            LINGBOT_LIBERO_PRO_STATIC_SCHEDULING_PROFILE,
        )
        self.assertEqual(static_scheduling_profile("libero"), "max_steps_v1")
        self.assertEqual(static_scheduling_cost("libero", "anything", 220), 220.0)
        with self.assertRaisesRegex(ValueError, "no fixed LingBot static scheduling cost"):
            static_scheduling_cost("libero_pro", "libero_new_suite", 220)

    def test_runtime_profile_does_not_queue_work_behind_known_longest_suites_at_112_lanes(self):
        specs = [
            TaskSpec(
                suite,
                task_id,
                max_steps=520 if suite.startswith("libero_10_") else 220,
                scheduling_cost=LINGBOT_LIBERO_PRO_STATIC_SUITE_COSTS[suite],
            )
            for suite in sorted(LINGBOT_LIBERO_PRO_STATIC_SUITE_COSTS)
            for task_id in range(10)
        ]
        specs.sort(key=lambda spec: spec.dispatch_cost, reverse=True)
        lanes = partition_static_lanes(specs, 112)
        longest = {spec.name for spec in specs if spec.suite in {"libero_10_swap", "libero_10_task"}}
        assignments = {spec.name: lane for lane in lanes for spec in lane}
        self.assertTrue(longest)
        self.assertTrue(all(len(assignments[name]) == 1 for name in longest))
        self.assertEqual(sorted(len(lane) for lane in lanes).count(2), 48)

    def test_static_lane_plan_interleaves_whole_cohorts_across_gpus(self):
        mapping = [[interleaved_static_lane_index(gpu, lane, 7, 4) for lane in range(8)] for gpu in range(7)]
        self.assertEqual(mapping[0], [0, 1, 2, 3, 28, 29, 30, 31])
        self.assertEqual(mapping[6], [24, 25, 26, 27, 52, 53, 54, 55])
        for gpu_lanes in mapping:
            self.assertEqual(gpu_lanes[:4], list(range(gpu_lanes[0], gpu_lanes[0] + 4)))
            self.assertEqual(gpu_lanes[4:], list(range(gpu_lanes[4], gpu_lanes[4] + 4)))
        self.assertEqual(sorted(index for gpu in mapping for index in gpu), list(range(56)))

    def test_mujoco_egl_device_can_be_remapped_by_container(self):
        with mock.patch.dict("os.environ", {"MUJOCO_EGL_DEVICE_ID": ""}):
            self.assertEqual(_mujoco_egl_device_id(6), "6")
        with mock.patch.dict("os.environ", {"MUJOCO_EGL_DEVICE_ID": "0"}):
            self.assertEqual(_mujoco_egl_device_id(6), "0")

    def test_simulator_client_thread_pools_are_bounded(self):
        bench = mock.Mock()
        bench.client_env.return_value = {
            "BENCHMARK_SETTING": "preserved",
            "OPENBLAS_NUM_THREADS": "64",
        }
        with mock.patch.dict("os.environ", {"MUJOCO_EGL_DEVICE_ID": ""}):
            environment = _client_env(3, bench)

        self.assertEqual(environment["BENCHMARK_SETTING"], "preserved")
        self.assertEqual(environment["MUJOCO_GL"], "egl")
        self.assertEqual(environment["MUJOCO_EGL_DEVICE_ID"], "3")
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(environment[variable], "1")


if __name__ == "__main__":
    unittest.main()

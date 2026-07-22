"""Evaluate a single LIBERO task (suite + task_id) against a running policy server.

Runs inside the LIBERO client environment (Python 3.8, MuJoCo/robosuite).
Based on openpi/examples/libero/main.py, restructured to evaluate exactly one
task and write a JSON result file, so that an external orchestrator can
dispatch the 40 (suite, task) pairs across many GPUs.

Preprocessing is selected by the policy family: openpi uses resize-with-pad and
5-step replanning, while OpenVLA-OFT receives the rotated 256px render and uses
its official server-side JPEG/Lanczos + center-crop path with 8-step chunks.
"""

import argparse
import collections
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
import torch
from init_mix import mix_counts  # sibling module; sys.path[0] is this script's dir
from libero.libero import benchmark
from prompt_clean import clean_task_prompt
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Longest training demo per base suite (from openpi examples/libero/main.py).
# Fallback when --max-steps is not given: derived suites (e.g. LIBERO-Pro's
# libero_goal_lan) reuse their base suite's limit.
BASE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _resolve_max_steps(suite):
    if suite in BASE_MAX_STEPS:
        return BASE_MAX_STEPS[suite]
    for base in sorted(BASE_MAX_STEPS, key=len, reverse=True):
        if suite.startswith(base + "_"):
            return BASE_MAX_STEPS[base]
    raise ValueError(f"Cannot derive max steps for suite {suite!r}; pass --max-steps explicitly.")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # Must be a str: LIBERO-plus's env_wrapper substring-parses the *filename*
    # for perturbation parameters (`"_view_" in bddl_file_name`, then .split());
    # camera/init-state/noise variants have no bddl on disk at this exact path.
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def eval_one_task(args):
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if not (0 <= args.task_id < task_suite.n_tasks):
        raise ValueError(
            f"task_id {args.task_id} out of range for suite {args.task_suite_name} ({task_suite.n_tasks} tasks)"
        )
    max_steps = args.max_steps or _resolve_max_steps(args.task_suite_name)

    task = task_suite.get_task(args.task_id)
    if args.init_states_root:
        # Anti-overfitting eval: use privately re-sampled init states (see
        # gen_init_states.py) instead of (--init-states-mix: for half of the
        # episodes alongside) the benchmark's official frozen files.
        init_path = pathlib.Path(args.init_states_root).expanduser() / task.problem_folder / task.init_states_file
        if not init_path.exists():
            raise FileNotFoundError(f"--init-states-root given but {init_path} does not exist")
        seeded_states = torch.load(str(init_path))
        if args.init_states_mix:
            official_states = task_suite.get_task_init_states(args.task_id)
            n_official, n_seeded = mix_counts(args.num_trials, len(official_states), len(seeded_states))
            initial_states = np.concatenate([official_states[:n_official], seeded_states[:n_seeded]])
            init_sources = ["official"] * n_official + ["seeded"] * n_seeded
        else:
            initial_states = seeded_states
            init_sources = ["seeded"] * len(seeded_states)
    else:
        initial_states = task_suite.get_task_init_states(args.task_id)
        init_sources = ["official"] * len(initial_states)
    if args.num_trials > len(initial_states):
        logging.warning(
            "Requested %d trials but only %d init states exist; clamping.",
            args.num_trials,
            len(initial_states),
        )
        args.num_trials = len(initial_states)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # task.language is derived from the bddl *filename*, so it keeps the
    # original wording even in LIBERO-Pro's lan/task perturbation suites; the
    # perturbed instruction only exists in the bddl (:language) block, exposed
    # as env.language_instruction. LIBERO-plus's published evaluation path uses
    # task.language unchanged. "task_clean" is retained only as a non-official
    # ablation and must not be used for leaderboard-comparable results.
    if args.prompt_source == "env":
        prompt = str(env.language_instruction)
    elif args.prompt_source == "task_clean":
        prompt = clean_task_prompt(task.name, task_description)
    else:
        prompt = str(task_description)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    video_dir = None
    if args.video_dir and args.save_videos > 0:
        video_dir = pathlib.Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    episodes = []
    num_successes = 0

    for episode_idx in range(args.num_trials):
        # Timing buckets: infer_s is the client-side wall clock of client.infer
        # (serialization + websocket round-trip + server), infer_server_s is the
        # server-reported handling time, env_s is MuJoCo stepping + EGL render.
        infer_s = infer_server_s = env_s = 0.0
        batch_size_sum = 0  # server-reported effective batch per call (1 on the upstream server)
        episode_start = time.time()

        t_mark = time.time()
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx])
        env_s += time.time() - t_mark

        t = 0
        done = False
        replay_images = []
        num_infer_calls = 0
        error = None

        while t < max_steps + args.num_steps_wait:
            try:
                # Wait for objects to stabilize before acting.
                if t < args.num_steps_wait:
                    t_mark = time.time()
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    env_s += time.time() - t_mark
                    t += 1
                    continue

                # Preprocess images: rotate 180 degrees to match train preprocessing.
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if args.policy_preprocess == "openpi":
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                if video_dir is not None and episode_idx < args.save_videos:
                    replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": prompt,
                        "task_suite": args.task_suite_name,
                    }
                    t_mark = time.time()
                    infer_result = client.infer(element)
                    infer_s += time.time() - t_mark
                    infer_server_s += infer_result.get("server_timing", {}).get("infer_ms", 0.0) / 1000.0
                    batch_size_sum += infer_result.get("server_timing", {}).get("batch_size", 1)
                    action_chunk = infer_result["actions"]
                    num_infer_calls += 1
                    assert len(action_chunk) >= args.replan_steps, (
                        f"Policy predicts {len(action_chunk)} steps, need >= {args.replan_steps}"
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                t_mark = time.time()
                obs, reward, done, info = env.step(action.tolist())
                env_s += time.time() - t_mark
                if done:
                    num_successes += 1
                    break
                t += 1

            except Exception as e:  # noqa: BLE001
                logging.exception("Caught exception during episode")
                error = str(e)
                break

        episodes.append({
            "trial": episode_idx,
            "init_source": init_sources[episode_idx],
            "success": bool(done),
            "steps": t,
            "num_infer_calls": num_infer_calls,
            "duration_s": round(time.time() - episode_start, 2),
            "timing": {
                "infer_s": round(infer_s, 2),
                "infer_server_s": round(infer_server_s, 2),
                "env_s": round(env_s, 2),
                "avg_batch": round(batch_size_sum / num_infer_calls, 2) if num_infer_calls else 0.0,
            },
            "error": error,
        })
        logging.info(
            "[%s/task%d] trial %d/%d -> %s (%d/%d successes so far)",
            args.task_suite_name,
            args.task_id,
            episode_idx + 1,
            args.num_trials,
            "SUCCESS" if done else "failure",
            num_successes,
            episode_idx + 1,
        )

        if video_dir is not None and episode_idx < args.save_videos and replay_images:
            suffix = "success" if done else "failure"
            video_path = (
                video_dir / f"{args.task_suite_name}_task{args.task_id:02d}_trial{episode_idx:02d}_{suffix}.mp4"
            )
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

    env.close()

    result = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_name": str(task.name),
        "task_description": str(task_description),
        "prompt": prompt,
        "prompt_source": args.prompt_source,
        "num_trials": args.num_trials,
        "num_successes": num_successes,
        "success_rate": num_successes / args.num_trials if args.num_trials else 0.0,
        "max_steps": max_steps,
        "seed": args.seed,
        "init_states_root": args.init_states_root,
        "init_states_mix": bool(args.init_states_root and args.init_states_mix),
        "duration_s": round(time.time() - start_time, 2),
        "timing_totals": {
            **{k: round(sum(e["timing"][k] for e in episodes), 2) for k in ("infer_s", "infer_server_s", "env_s")},
            "avg_batch": round(
                sum(e["timing"]["avg_batch"] * e["num_infer_calls"] for e in episodes)
                / max(1, sum(e["num_infer_calls"] for e in episodes)),
                2,
            ),
        },
        "episodes": episodes,
    }

    out_path = pathlib.Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(
        "[%s/task%d] DONE success_rate=%.2f -> %s",
        args.task_suite_name,
        args.task_id,
        result["success_rate"],
        out_path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--policy-preprocess", choices=("openpi", "openvla_oft"), default="openpi")
    parser.add_argument(
        "--task-suite-name", required=True, help="Any suite registered in the libero package on PYTHONPATH"
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="Step limit per episode (0 = derive from the suite's base name)"
    )
    parser.add_argument(
        "--prompt-source",
        choices=("task", "env", "task_clean"),
        default="task",
        help="Feed the model task.language ('task'), env.language_instruction "
        "('env'; required for LIBERO-Pro lan/task perturbations), or task.language "
        "with LIBERO-plus filename-suffix junk stripped ('task_clean')",
    )
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--init-states-root",
        default=None,
        help="Load init states from {root}/{suite}/{task}.pruned_init instead of the "
        "benchmark's official files (generated by gen_init_states.py)",
    )
    parser.add_argument(
        "--init-states-mix",
        action="store_true",
        help="With --init-states-root: draw half of the trials from the official init "
        "states (first episodes, same layouts as an unmixed run) and half from the root",
    )
    parser.add_argument("--out-json", required=True, help="Path to write the result JSON")
    parser.add_argument("--video-dir", default=None, help="Directory to save rollout videos")
    parser.add_argument("--save-videos", type=int, default=0, help="Save videos for the first N trials")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eval_one_task(args)


if __name__ == "__main__":
    main()

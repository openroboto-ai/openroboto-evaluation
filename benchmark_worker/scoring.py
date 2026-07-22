"""
把 validator run_eval.py 的 summary.json 映射为后端评分提交体。
Map a validator run_eval.py summary.json onto the backend score payload.

纯函数、无 I/O。目标 schema:POST /api/v1/benchmark/task/{id}/score
(prototype 仓库 docs/api_reference_zh.md 3.2 节)。
"""

import json

from benchmark_worker.profiles import ScoreTarget, get_profile

# 原版 LIBERO 自带的 suite;其余名字(如 libero_spatial_lan 等扰动 suite)
# 一律走 libero_pro benchmark。
_LIBERO_BASE_SUITES = {
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
}

# 后端 env_list 用的是"评测环境"词汇表,不保证每个名字都是可直接执行的
# suite:LIBERO-100 是 90+10 两个 suite 的合称,没有独立的 bddl/init 资产。
# 执行前按此表展开,上报时再按 env 名聚合回去(见 build_score_payload)。
_ENV_TO_SUITES = {
    "libero_100": ("libero_90", "libero_10"),
}

_CUSTOM_1_GROUPS = (
    ("libero_spatial", "libero_spatial"),
    ("libero_object", "libero_object"),
    ("libero_goal", "libero_goal"),
    # Custom 1 只取 LIBERO-10 的四个 Pro 随机化 suite；这里绝不运行或
    # 聚合 libero_90，上报名也使用后端当前约定的 libero_10。
    ("libero_10", "libero_10"),
)
_CUSTOM_1_EXTRA_SUITES = ("libero_object_swap", "libero_spatial_swap")
_CUSTOM_1_DIMENSIONS = ("object", "swap", "lan", "task")


def env_suites(env: str) -> tuple[str, ...]:
    """单个后端 env 名 → 实际执行的 suite(多数就是它自己)。"""
    return _ENV_TO_SUITES.get(env, (env,))


def expand_env_suites(env_list: list[str]) -> list[str]:
    """env_list → 去重保序的可执行 suite 列表(供 run_eval --suites)。"""
    suites: list[str] = []
    for env in env_list:
        for s in env_suites(env):
            if s not in suites:
                suites.append(s)
    return suites


def parse_env_list(raw) -> list[str]:
    """队列条目的 env_list 字段 → list[str]。

    /api/pending-tasks 给数组;旧端点 /api/pending-tasks 给 JSON 字符串
    (老 state 账本里存的任务也可能是这种形状),两种都接受。直接迭代字符串
    会拆成单字符,曾把任务错报成 "invalid env names",必须先解码。
    其余形状一律 ValueError,由调用方如实上报。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"env_list is not valid JSON: {raw!r}") from e
    if raw is None:
        raw = []
    if not isinstance(raw, list) or not all(isinstance(s, str) for s in raw):
        raise ValueError(f"env_list must be a list of strings, got: {raw!r}")
    return [s.strip() for s in raw if s.strip()]


def choose_benchmark(env_list: list[str]) -> str:
    if not env_list or all(s in _LIBERO_BASE_SUITES for s in env_list):
        return "libero"
    return "libero_pro"


def _aggregate_env(env: str, suites_agg: dict, suite_secs: dict) -> dict:
    """一个后端 env 名对应的若干 suite 结果 → 一条 env_scores 记录。

    多 suite 时 score 按 episode 数加权平均(各 task trial 数相同时等价于按
    task 数加权);samples/tasks/耗时直接求和。env 的 suite 全部缺席于 summary
    时如实给零分记录并注明,不让单个 env 的缺失掩盖整体结果。
    """
    parts = [suites_agg[s] for s in env_suites(env) if s in suites_agg]
    episodes = sum(a["episodes"] for a in parts)
    tasks = sum(a["tasks"] for a in parts)
    failed = sum(a.get("failed_tasks", 0) for a in parts)
    weighted = [
        (a["success_rate"], a["episodes"]) for a in parts if a["success_rate"] is not None and a["episodes"] > 0
    ]
    score = sum(sr * n for sr, n in weighted) / sum(n for _, n in weighted) if weighted else 0.0
    if not parts:
        error = "no results for this env"
    else:
        error = f"{failed}/{tasks} tasks failed" if failed else ""
    return {
        "env_name": env,
        "score": round(score, 6),
        "samples": episodes,
        "duration_sec": round(sum(suite_secs.get(s, 0.0) for s in env_suites(env)), 1),
        "error": error,
    }


def _aggregate_target(target: ScoreTarget, suites_agg: dict, suite_secs: dict) -> dict:
    aggregated = _aggregate_env(target.env_name, suites_agg, suite_secs)
    aggregated.update({
        "base_suite": target.base_suite,
        "perturbation": target.perturbation,
    })
    return aggregated


def _collapse_custom_1_scores(env_scores: list[dict]) -> list[dict]:
    """把 LIBERO-Pro 的 16 条 suite 分数折叠为 Custom 1 的六条上报记录。

    四个 base 条目各自等权平均其 object/swap/lan/task 分数；object_swap 和
    spatial_swap 同时保留为独立条目，因此这两个 score 会有意出现两次。
    输入若不是完整的 16-suite 结构则原样返回，避免误改其他/历史 payload。
    """
    by_name = {
        entry.get("env_name"): entry
        for entry in env_scores
        if isinstance(entry, dict) and isinstance(entry.get("env_name"), str)
    }
    expected = {
        f"{base}_{dimension}"
        for _, base in _CUSTOM_1_GROUPS
        for dimension in _CUSTOM_1_DIMENSIONS
    }
    if len(by_name) != len(env_scores) or set(by_name) != expected:
        return env_scores

    collapsed = []
    for output_name, base in _CUSTOM_1_GROUPS:
        parts = [by_name[f"{base}_{dimension}"] for dimension in _CUSTOM_1_DIMENSIONS]
        errors = [str(part.get("error") or "") for part in parts if part.get("error")]
        collapsed.append({
            "env_name": output_name,
            "score": round(sum(float(part.get("score") or 0.0) for part in parts) / len(parts), 6),
            "samples": sum(int(part.get("samples") or 0) for part in parts),
            "duration_sec": round(sum(float(part.get("duration_sec") or 0.0) for part in parts), 1),
            "error": "; ".join(errors),
        })

    for env_name in _CUSTOM_1_EXTRA_SUITES:
        part = by_name[env_name]
        collapsed.append({
            "env_name": env_name,
            "score": float(part.get("score") or 0.0),
            "samples": int(part.get("samples") or 0),
            "duration_sec": float(part.get("duration_sec") or 0.0),
            "error": str(part.get("error") or ""),
        })
    return collapsed


def _mean_scored_entries(env_scores: list[dict]) -> float:
    scores = [float(entry["score"]) for entry in env_scores if int(entry.get("samples") or 0) > 0]
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def build_score_payload(
    task: dict,
    summary: dict | None,
    duration_sec: float,
    error: str = "",
    init_seed: int | None = None,
    benchmark: str = "libero",
) -> dict:
    """构造评分提交体。

    task:         后端队列条目(miner_hotkey 等标识字段原样回传)。
    summary:      解析后的 summary.json;评测没产出结果时传 None。
    duration_sec: 处理该任务的端到端墙钟时间(下载 + 评测)。
    error:        顶层错误说明(一切正常时为空串)。
    init_seed:    init states 随机化种子(None = 未启用),即该队列条目自带的
                  seed(由 block_hash + round_num + drand 派生,见 prototype 仓库
                  docs/SEED_GENERATION.md)。随 payload 回传作确认:miner 可独立
                  验证 seed 本身,再用 gen_init_states.py --seed 逐 task 复现
                  评测所用的初始状态。

    `success` 的语义是「产出了可用分数」:个别 task 重试后仍失败的运行
    依然算成功,损失记录在顶层 error 和对应 env_scores[].error 里。
    """
    payload = {
        "success": False,
        "benchmark": benchmark,
        # 新 benchmark queue 返回 miner_hotkey;旧 /api/pending-tasks 直接
        # 暴露 submissions 行,字段名仍是 hotkey。两种部署都要可回传。
        "miner_hotkey": task.get("miner_hotkey") or task.get("hotkey", ""),
        "hf_repo_id": task.get("hf_repo_id", ""),
        "hf_commit": task.get("hf_commit", ""),
        "round_num": task.get("round_num", 0),
        "init_seed": init_seed,
        "env_scores": [],
        "total_score": 0.0,
        "duration_sec": round(duration_sec, 1),
        "error": error,
        "per_task_scores": [],
    }
    if summary is None:
        return payload

    # 每个 suite 的耗时 = 该 suite 各 task 用时之和(task 在多卡上并行,
    # suite 维度的墙钟时间没有意义,计算耗时才可比)。
    suite_secs: dict[str, float] = {}
    per_task = []
    for r in summary.get("tasks", {}).values():
        suite = r["task_suite_name"]
        suite_secs[suite] = suite_secs.get(suite, 0.0) + float(r.get("duration_s") or 0.0)
        if r.get("status") == "ok":
            per_task.append({
                "task_id": f"{suite}_{r['task_id']:02d}",
                "success_rate": r["success_rate"],
                "trials": r["num_trials"],
            })

    # 评测内容由 worker 的 benchmark profile 决定,不再相信队列中的 legacy
    # env_list。LIBERO-Pro 的平面 runtime 名在协议边界还原为
    # (base_suite, perturbation)；Custom 1 不改变这 16 个运行 suite，只改变
    # 最终的上报聚合格式。
    profile = get_profile(benchmark)
    suites_agg = summary.get("suites", {})
    env_scores = [_aggregate_target(target, suites_agg, suite_secs) for target in profile.targets]

    # 这是部署方约定的兼容协议，只适用于以 LIBERO-Pro 运行的 Custom 1。
    # 普通 libero_pro 仍提交完整 16 条；libero/libero_plus 也保持原格式。
    custom_1_six_env = benchmark == "libero_pro_custom_1" and profile.runtime_benchmark == "libero_pro"
    if custom_1_six_env:
        env_scores = _collapse_custom_1_scores(env_scores)

    # Custom 1 按六条兼容记录等权平均；其他 profile 使用各自运行 profile
    # 的权重生成本地预览值。
    if custom_1_six_env:
        weighted = [(entry["score"], 1.0) for entry in env_scores if entry["samples"] > 0]
    else:
        weighted = [
            (entry["score"], profile.weight_for(target))
            for target, entry in zip(profile.targets, env_scores, strict=True)
            if entry["samples"] > 0
        ]
    payload.update({
        "success": bool(weighted),
        "env_scores": env_scores,
        "total_score": (
            round(sum(score * weight for score, weight in weighted) / sum(weight for _, weight in weighted), 6)
            if weighted
            else 0.0
        ),
        "per_task_scores": per_task,
    })
    if not weighted and not error:
        payload["error"] = "evaluation produced no scored episodes"
    return payload


def prepare_submit_payload(payload: dict) -> dict:
    """构造实际提交体,省略只在本地结果中保留的 task 级明细。

    后端会逐条同步写入 ``per_task_scores``。完整 LIBERO-100 有 130 条,
    经 Cloudflare 访问时写库会超过代理超时并稳定返回 524。该字段在 API
    中是可选的;四环境汇总才是评分所需数据,完整 task 明细仍保存在
    ``score_payload.json`` 和本地 state 账本中。
    """
    custom_scores = payload.get("env_scores")
    collapsed_scores = (
        _collapse_custom_1_scores(custom_scores)
        if payload.get("benchmark") == "libero_pro_custom_1" and isinstance(custom_scores, list)
        else custom_scores
    )
    needs_custom_collapse = collapsed_scores is not custom_scores
    if not payload.get("per_task_scores") and not needs_custom_collapse:
        return payload
    slim = dict(payload)
    if payload.get("per_task_scores"):
        slim["per_task_scores"] = []
    if needs_custom_collapse:
        slim["env_scores"] = collapsed_scores
        slim["total_score"] = _mean_scored_entries(collapsed_scores)
    return slim

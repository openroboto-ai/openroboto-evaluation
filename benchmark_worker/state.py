"""
本地任务状态账本(单 JSON 文件,崩溃安全)。
Crash-safe local ledger of every backend task this worker has seen.

task_id -> entry,entry 的 status 状态机:

    pending              已从队列领取,等待评测 worker
    running              评测子进程进行中(重启后重置为 pending 重跑)
    done_pending_submit  评分 payload 已就绪(成功或失败),后端尚未确认,
                         会持续重试提交
    submitted            后端已确认收到评分(终态)
    abandoned            后端永久拒绝该提交(终态)

该文件是去重的唯一依据,粒度是"提交"而非 task_id:同一
(task_id, hf_repo_id, hf_commit) 绝不重复评测,跨轮询、跨重启均如此;
但 miner 换 commit 重新提交(task_id 不变)会重新评测,已 submitted 的
任务若仍出现在后端队列,则重发已存结果(见 worker.classify_queued_task)。
带 protocol_revision 的 benchmark 在协议升级后会使旧结果失效并重新评测,
避免把旧评测口径的缓存 payload 补交到新轮次。

(历史版本曾在顶层存 rounds:每轮本地生成的 init_seed;现 init seed 直接
取队列条目的 seed 字段,旧 state 文件里遗留的 rounds 键会被原样忽略。)
"""

import datetime
import json
import pathlib
import threading


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class StateStore:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self._lock = threading.Lock()
        self._data = {"tasks": {}}
        if path.exists():
            text = path.read_text()
            if text.strip():  # 空文件视为全新账本(手动清空 = 要求全部重评)
                try:
                    self._data = json.loads(text)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"state file {path} is not valid JSON ({e}); fix it or delete it to start fresh"
                    ) from None
                self._data.setdefault("tasks", {})

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            entry = self._data["tasks"].get(task_id)
            return dict(entry) if entry else None

    def all_tasks(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._data["tasks"].items()}

    def update(self, task_id: str, **fields) -> dict:
        with self._lock:
            entry = self._data["tasks"].setdefault(task_id, {})
            entry.update(fields)
            entry["updated_at"] = now_iso()
            self._save()
            return dict(entry)

    def _save(self) -> None:
        # 写临时文件再原子替换,进程被杀也不会留下半截 JSON。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        tmp.replace(self.path)

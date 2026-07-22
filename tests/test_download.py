"""libero_eval/download.py 的单元测试(全部无网络)。

运行:uv run python -m unittest discover tests
"""

import errno
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "libero_eval"))

import download  # noqa: E402
from download import (  # noqa: E402
    DEFAULT_STRATEGIES,
    OFFICIAL_ENDPOINT,
    DownloadError,
    _build_hfd_cmd,
    _dir_size,
    _is_permanent,
    _is_resumable,
    download_model,
    parse_strategies,
)


def _hub_error(cls, msg="x"):
    # huggingface_hub 1.x 的 HTTP 异常要求带 httpx.Response。
    return cls(msg, response=httpx.Response(404, request=httpx.Request("GET", "http://x")))


class TestParseStrategies(unittest.TestCase):
    def test_valid_order_preserved(self):
        self.assertEqual(parse_strategies("hub,hfd-mirror,hfd"), ["hub", "hfd-mirror", "hfd"])

    def test_default_is_valid(self):
        self.assertEqual(parse_strategies(DEFAULT_STRATEGIES), ["hfd-mirror", "hfd", "hub"])

    def test_whitespace_tolerated(self):
        self.assertEqual(parse_strategies(" hfd-mirror , hub "), ["hfd-mirror", "hub"])

    def test_unknown_name_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown.*hfdd"):
            parse_strategies("hfd-mirror,hfdd")

    def test_empty_fails(self):
        for spec in ("", " ", ","):
            with self.assertRaisesRegex(ValueError, "empty"):
                parse_strategies(spec)

    def test_duplicate_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicated.*hub"):
            parse_strategies("hub,hfd,hub")


class TestBuildHfdCmd(unittest.TestCase):
    def _build(self, **kw):
        defaults = dict(
            repo_id="user/repo",
            revision=None,
            local_dir=pathlib.Path("/abs/dir"),
            endpoint="https://hf-mirror.com",
            repo_type="model",
            send_token=False,
        )
        defaults.update(kw)
        return _build_hfd_cmd(**defaults)

    def test_argv_basics(self):
        cmd, _ = self._build()
        self.assertEqual(cmd[:3], ["bash", str(download.HFD_SCRIPT), "user/repo"])
        self.assertIn("--tool", cmd)
        self.assertEqual(cmd[cmd.index("--tool") + 1], "aria2c")
        self.assertEqual(cmd[cmd.index("--local-dir") + 1], "/abs/dir")
        self.assertEqual(cmd[cmd.index("-x") + 1], str(download._HFD_THREADS))
        self.assertEqual(cmd[cmd.index("-j") + 1], str(download._HFD_JOBS))
        self.assertNotIn("--revision", cmd)
        self.assertNotIn("--dataset", cmd)

    def test_revision_and_dataset(self):
        cmd, _ = self._build(revision="abc123", repo_type="dataset")
        self.assertEqual(cmd[cmd.index("--revision") + 1], "abc123")
        self.assertIn("--dataset", cmd)

    def test_token_never_in_argv(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret"}):
            cmd, _ = self._build(send_token=True)
        self.assertNotIn("hf_secret", " ".join(cmd))
        self.assertNotIn("--hf_token", cmd)

    def test_mirror_strips_token_env(self):
        secrets = {
            "HF_TOKEN": "hf_secret",
            "HUGGING_FACE_HUB_TOKEN": "hf_secret2",
            "HF_USERNAME": "someone",
        }
        with mock.patch.dict(os.environ, secrets):
            _, env = self._build(send_token=False)
        for k in secrets:
            self.assertNotIn(k, env)
        self.assertEqual(env["HF_ENDPOINT"], "https://hf-mirror.com")

    def test_official_keeps_token_env(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_secret"}):
            _, env = self._build(endpoint=OFFICIAL_ENDPOINT, send_token=True)
        self.assertEqual(env["HF_TOKEN"], "hf_secret")
        self.assertEqual(env["HF_ENDPOINT"], OFFICIAL_ENDPOINT)


class TestIsResumable(unittest.TestCase):
    """hfd.sh 三处 "Re-run to resume" 输出(带 ANSI 色码)都应判为可续传。"""

    def test_download_incomplete_true(self):
        out = "\x1b[0;31mDownload incomplete. Re-run to resume. Log: /d/.hfd/download.log\x1b[0m"
        self.assertTrue(_is_resumable(out))

    def test_listing_failure_true(self):
        self.assertTrue(_is_resumable("[Error] Failed to list repository files. Re-run to resume."))

    def test_interrupted_true(self):
        self.assertTrue(_is_resumable("Interrupted. Re-run to resume."))

    def test_other_failure_false(self):
        self.assertFalse(_is_resumable("aria2c: command not found"))
        self.assertFalse(_is_resumable(""))


class TestDirSize(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)

    def test_counts_files_recursively(self):
        (self.root / "a.bin").write_bytes(b"xxx")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "b.bin").write_bytes(b"yyyyy")
        self.assertEqual(_dir_size(self.root), 8)

    def test_excludes_hfd_metadata(self):
        (self.root / "a.bin").write_bytes(b"xxx")
        (self.root / ".hfd").mkdir()
        (self.root / ".hfd" / "console.log").write_bytes(b"z" * 100)
        self.assertEqual(_dir_size(self.root), 3)

    def test_missing_dir_is_zero(self):
        self.assertEqual(_dir_size(self.root / "nope"), 0)


class TestRunHfdResume(unittest.TestCase):
    """用 fake bash 脚本顶替 hfd.sh,验证 _run_hfd 的原地续传重跑循环。

    fake 脚本忽略 argv,靠计数文件区分第几次运行;"下载进度"靠往 local_dir
    追加 payload 字节模拟(.hfd/ 下的日志增长被 _dir_size 排除,不算进度)。
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        self.local_dir = self.root / "model"
        self.counter = self.root / "runs"
        self.logs: list[str] = []

    def _script(self, body: str) -> pathlib.Path:
        # 每次运行先自增计数文件,再执行 body(body 里可用 $count)。
        script = self.root / "fake_hfd.sh"
        script.write_text(
            "#!/bin/bash\n"
            f"count=$(cat {self.counter} 2>/dev/null || echo 0)\n"
            "count=$((count+1))\n"
            f"echo $count > {self.counter}\n" + body
        )
        return script

    def _runs(self) -> int:
        return int(self.counter.read_text().strip()) if self.counter.exists() else 0

    def _call(self, script: pathlib.Path, retries: str | None = None):
        with mock.patch.object(download, "HFD_SCRIPT", script):
            with mock.patch.dict(os.environ):
                os.environ.pop("MODEL_DOWNLOAD_RESUME_RETRIES", None)
                if retries is not None:
                    os.environ["MODEL_DOWNLOAD_RESUME_RETRIES"] = retries
                download._run_hfd("u/r", None, self.local_dir, OFFICIAL_ENDPOINT, "model", self.logs.append)

    def test_resumable_with_progress_retries_then_succeeds(self):
        script = self._script(
            "if [ $count -lt 3 ]; then\n"
            f"  head -c 100 /dev/zero >> {self.local_dir}/payload.bin\n"
            '  echo "Download incomplete. Re-run to resume."\n'
            "  exit 1\n"
            "fi\n"
            "exit 0\n"
        )
        self._call(script, retries="3")
        self.assertEqual(self._runs(), 3)
        self.assertTrue(any("resume attempt 1/3" in m for m in self.logs), self.logs)

    def test_resumable_without_progress_raises_immediately(self):
        script = self._script('echo "Failed to list repository files. Re-run to resume."\nexit 1\n')
        with self.assertRaisesRegex(RuntimeError, "hfd.sh exited 1"):
            self._call(script)
        self.assertEqual(self._runs(), 1)

    def test_non_resumable_failure_raises_immediately(self):
        script = self._script(
            f'head -c 100 /dev/zero >> {self.local_dir}/payload.bin\necho "aria2c: command not found"\nexit 1\n'
        )
        with self.assertRaisesRegex(RuntimeError, "hfd.sh exited 1"):
            self._call(script)
        self.assertEqual(self._runs(), 1)

    def test_retry_cap_exhausted(self):
        script = self._script(
            f"head -c 100 /dev/zero >> {self.local_dir}/payload.bin\n"
            'echo "Download incomplete. Re-run to resume."\n'
            "exit 1\n"
        )
        with self.assertRaisesRegex(RuntimeError, "hfd.sh exited 1"):
            self._call(script, retries="1")
        self.assertEqual(self._runs(), 2)

    def test_retries_zero_disables_retry(self):
        script = self._script(
            f"head -c 100 /dev/zero >> {self.local_dir}/payload.bin\n"
            'echo "Download incomplete. Re-run to resume."\n'
            "exit 1\n"
        )
        with self.assertRaisesRegex(RuntimeError, "hfd.sh exited 1"):
            self._call(script, retries="0")
        self.assertEqual(self._runs(), 1)

    def test_tail_is_from_last_attempt_only(self):
        # console.log 是 append 模式:预埋旧输出,异常里应只带本次输出片段。
        (self.local_dir / ".hfd").mkdir(parents=True)
        junk = "\n".join(f"OLDJUNK-{i}" for i in range(10)) + "\n"
        (self.local_dir / ".hfd" / "console.log").write_text(junk)
        script = self._script('echo "DISTINCTIVE-FAILURE"\nexit 1\n')
        with self.assertRaises(RuntimeError) as ctx:
            self._call(script)
        self.assertIn("DISTINCTIVE-FAILURE", str(ctx.exception))
        self.assertNotIn("OLDJUNK", str(ctx.exception))


class TestIsPermanent(unittest.TestCase):
    def _hub_errors(self):
        from huggingface_hub.errors import (
            GatedRepoError,
            RemoteEntryNotFoundError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

        return [
            _hub_error(RepositoryNotFoundError),
            _hub_error(RevisionNotFoundError),
            _hub_error(GatedRepoError),
            _hub_error(RemoteEntryNotFoundError),
        ]

    def test_typed_hub_errors_permanent_only_on_official(self):
        for exc in self._hub_errors():
            self.assertTrue(_is_permanent(exc, official=True), exc)
            self.assertFalse(_is_permanent(exc, official=False), exc)

    def test_enospc_always_permanent(self):
        exc = OSError(errno.ENOSPC, "no space left on device")
        self.assertTrue(_is_permanent(exc, official=True))
        self.assertTrue(_is_permanent(exc, official=False))

    def test_local_entry_not_found_not_permanent(self):
        # LocalEntryNotFoundError 表示离线且本地无缓存(临时故障),
        # 虽然继承 EntryNotFoundError,但不能判为永久。
        from huggingface_hub.errors import LocalEntryNotFoundError

        exc = LocalEntryNotFoundError("offline")
        self.assertFalse(_is_permanent(exc, official=True))
        self.assertFalse(_is_permanent(exc, official=False))

    def test_generic_errors_not_permanent(self):
        for exc in (
            RuntimeError("hfd.sh exited 1"),
            TimeoutError("hfd.sh timed out"),
            ConnectionError("boom"),
            OSError(errno.ECONNRESET, "reset"),
        ):
            self.assertFalse(_is_permanent(exc, official=True), exc)
            self.assertFalse(_is_permanent(exc, official=False), exc)


class TestDownloadModelChain(unittest.TestCase):
    """monkeypatch _run_hfd/_snapshot,验证策略链的回退/短路/汇总逻辑。"""

    def setUp(self):
        self.calls: list[tuple[str, str]] = []  # (step, endpoint)
        self.dir = pathlib.Path("/tmp/x")

    def _patch(self, hfd=None, snapshot=None):
        def fake_hfd(repo_id, revision, local_dir, endpoint, repo_type, log):
            self.calls.append(("hfd", endpoint))
            if hfd:
                hfd(endpoint)

        def fake_snapshot(repo_id, revision, local_dir, endpoint, use_mirror, repo_type):
            self.calls.append(("snapshot", endpoint))
            if snapshot:
                snapshot(endpoint)

        def fake_preflight(repo_id, revision, repo_type, log):
            pass  # 预检行为由 TestPreflightRepoCheck 单独覆盖

        return mock.patch.multiple(
            download, _run_hfd=fake_hfd, _snapshot=fake_snapshot, _preflight_repo_check=fake_preflight
        )

    def test_first_strategy_success_short_circuits(self):
        with self._patch():
            result = download_model("u/r", self.dir, strategies=["hfd-mirror", "hub"], log=lambda *_: None)
        self.assertEqual(result, self.dir.resolve())
        # hfd-mirror 成功 = hfd 预取(镜像)+ snapshot 收尾(固定官方端点,
        # 镜像的 resolve 重定向缺元数据头无法哈希校验),不再尝试 hub。
        self.assertEqual(
            self.calls,
            [("hfd", download._mirror_endpoint()), ("snapshot", OFFICIAL_ENDPOINT)],
        )

    def test_fallback_in_order(self):
        mirror = download._mirror_endpoint()

        def failing_hfd(endpoint):
            raise RuntimeError("hfd.sh exited 1")

        with self._patch(hfd=failing_hfd):
            download_model("u/r", self.dir, strategies=["hfd-mirror", "hfd", "hub"], log=lambda *_: None)
        # 两次 hfd 失败(镜像、官方),最后 hub 的 snapshot 成功。
        self.assertEqual(
            self.calls,
            [("hfd", mirror), ("hfd", OFFICIAL_ENDPOINT), ("snapshot", OFFICIAL_ENDPOINT)],
        )

    def test_permanent_error_on_official_aborts_chain(self):
        from huggingface_hub.errors import RepositoryNotFoundError

        def snapshot_404(endpoint):
            raise _hub_error(RepositoryNotFoundError, "404")

        with self._patch(snapshot=snapshot_404):
            with self.assertRaises(DownloadError) as ctx:
                download_model("u/r", self.dir, strategies=["hub", "hfd"], log=lambda *_: None)
        # hub(官方)404 是永久错误,不再尝试 hfd。
        self.assertEqual(self.calls, [("snapshot", OFFICIAL_ENDPOINT)])
        self.assertIn("hub", str(ctx.exception))
        self.assertNotIn("hfd:", str(ctx.exception))

    def test_permanent_error_on_mirror_does_not_abort(self):
        from huggingface_hub.errors import RepositoryNotFoundError

        def snapshot_404_on_mirror(endpoint):
            if endpoint != OFFICIAL_ENDPOINT:
                raise _hub_error(RepositoryNotFoundError, "404 (mirror stripped token)")

        with self._patch(snapshot=snapshot_404_on_mirror):
            download_model("u/r", self.dir, strategies=["hub-mirror", "hub"], log=lambda *_: None)
        self.assertEqual(self.calls[-1], ("snapshot", OFFICIAL_ENDPOINT))

    def test_all_failed_message_lists_every_strategy(self):
        def failing_hfd(endpoint):
            raise RuntimeError("hfd.sh exited 1")

        def failing_snapshot(endpoint):
            raise ConnectionError("network down")

        with self._patch(hfd=failing_hfd, snapshot=failing_snapshot):
            with self.assertRaises(DownloadError) as ctx:
                download_model(
                    "u/r",
                    self.dir,
                    revision="abc",
                    strategies=["hfd-mirror", "hfd", "hub"],
                    log=lambda *_: None,
                )
        msg = str(ctx.exception)
        self.assertIn("u/r@abc", msg)
        for name in ("hfd-mirror", "hfd", "hub"):
            self.assertIn(f"- {name}:", msg)

    def test_env_var_default_strategies(self):
        with self._patch():
            with mock.patch.dict(os.environ, {"MODEL_DOWNLOAD_STRATEGIES": "hub"}):
                download_model("u/r", self.dir, log=lambda *_: None)
        self.assertEqual(self.calls, [("snapshot", OFFICIAL_ENDPOINT)])

    def test_env_var_invalid_fails_fast(self):
        with mock.patch.dict(os.environ, {"MODEL_DOWNLOAD_STRATEGIES": "bogus"}):
            with self.assertRaises(ValueError):
                download_model("u/r", self.dir, log=lambda *_: None)


class TestPreflightRepoCheck(unittest.TestCase):
    """预检:官方端点 repo 不存在 → 一个策略都不试直接永久失败;临时故障 → 放行。"""

    def setUp(self):
        self.calls: list[str] = []
        self.dir = pathlib.Path("/tmp/x")

    def _patch_strategies(self):
        return mock.patch.multiple(
            download,
            _run_hfd=lambda *a, **k: self.calls.append("hfd"),
            _snapshot=lambda *a, **k: self.calls.append("snapshot"),
        )

    def _patch_hfapi(self, exc=None):
        api = mock.MagicMock()
        if exc is not None:
            api.repo_info.side_effect = exc
        return mock.patch("huggingface_hub.HfApi", return_value=api), api

    def test_repo_not_found_aborts_before_any_strategy(self):
        from huggingface_hub.errors import RepositoryNotFoundError

        patcher, _ = self._patch_hfapi(_hub_error(RepositoryNotFoundError, "404"))
        with patcher, self._patch_strategies():
            with self.assertRaisesRegex(DownloadError, "not found on"):
                download_model("u/r", self.dir, strategies=["hfd-mirror", "hub"], log=lambda *_: None)
        self.assertEqual(self.calls, [])

    def test_transient_preflight_failure_does_not_block_chain(self):
        patcher, _ = self._patch_hfapi(ConnectionError("boom"))
        with patcher, self._patch_strategies():
            result = download_model("u/r", self.dir, strategies=["hub"], log=lambda *_: None)
        self.assertEqual(self.calls, ["snapshot"])
        self.assertEqual(result, self.dir.resolve())

    def test_preflight_pins_official_endpoint_and_default_revision(self):
        patcher, api = self._patch_hfapi()
        with patcher as hfapi_cls:
            download._preflight_repo_check("u/r", None, "model", lambda *_: None)
        hfapi_cls.assert_called_once_with(endpoint=OFFICIAL_ENDPOINT)
        api.repo_info.assert_called_once_with(repo_id="u/r", repo_type="model", revision="main")


if __name__ == "__main__":
    unittest.main()

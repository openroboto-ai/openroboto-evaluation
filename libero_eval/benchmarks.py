"""Benchmark variants for the eval pipeline: LIBERO, LIBERO-Pro and LIBERO-plus.

A Benchmark bundles everything variant-specific: which `libero` package goes on
the client PYTHONPATH, where bddl/init files live (isolated per variant via
LIBERO_CONFIG_PATH), the default suite list, per-suite step limits, and which
field carries the language instruction. run_eval.py picks one by name
(--benchmark) and hands the resolved per-task values (max steps, prompt source)
to eval_task.py as plain CLI arguments, so eval_task.py stays variant-agnostic.

LIBERO-Pro (arXiv:2510.03827) forks the `libero` package: same envs/API, but
~80 registered suites. The perturbation suites `{base}_{object,swap,lan,task}`
(object swap / position swap / instruction paraphrase / task redefinition) ship
their bddl+init files in the HF dataset zhouxueyang/LIBERO-Pro, downloaded on
first use. The checkout is treated as read-only: downloaded assets and the
merged bddl/init roots live under ~/.cache/libero_pro.

LIBERO-plus (arXiv:2510.13626) also forks the `libero` package and keeps the
original suite names, but folds 10,030 perturbed task variants into
libero_spatial/object/goal/10 (7 dimensions: camera viewpoints, robot initial
states, language, light, background textures, sensor noise, objects layout).
Official protocol: 1 trial per task. bddl/init files ship in the checkout
(camera/init-state/noise variants have no bddl on disk — the fork's env_wrapper
parses their parameters out of the *filename*, upstream #60); the 6.4 GB asset
pack comes from the HF dataset Sylvest/LIBERO-plus on first use. The extracted
assets live under ~/.cache/libero_plus; the checkout only gains one gitignored
symlink `libero/libero/assets` pointing there (the fork resolves asset paths
relative to its package dir, not only via the config file).
"""

import dataclasses
import functools
import json
import os
import pathlib
import shutil
import typing
import zipfile

from paths import CLIENT_VENV_PY, LIBERO_PLUS_DIR, LIBERO_PRO_DIR, OPENPI_DIR

# Step limit per base suite = longest training demo (openpi examples/libero).
# LIBERO-Pro perturbation suites reuse the limit of their base suite;
# LIBERO-plus suites keep the base names and get the base limits directly.
BASE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

_PRO_CACHE = pathlib.Path(os.path.expanduser("~/.cache/libero_pro"))
_PRO_HF_DATASET = "zhouxueyang/LIBERO-Pro"
_PRO_BASES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
# Perturbation dimensions published in the HF dataset. The paper's fifth
# dimension (environment swap) has no released assets and is flagged unstable
# by the LIBERO-Pro README, so it is not offered here.
_PRO_DIMENSIONS = ("object", "swap", "lan", "task")

# Official LIBERO-Plus registry sizes.  These are part of the scoring protocol:
# the leaderboard Total is the micro-average over all 10,030 variants, not an
# equal average over suites or perturbation dimensions.
PLUS_SUITE_TASK_COUNTS = (
    ("libero_spatial", 2402),
    ("libero_object", 2518),
    ("libero_goal", 2591),
    ("libero_10", 2519),
)
PLUS_TOTAL_TASKS = sum(count for _, count in PLUS_SUITE_TASK_COUNTS)
PLUS_DIMENSIONS = {
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
}


@dataclasses.dataclass(frozen=True)
class Benchmark:
    name: str
    package_dir: pathlib.Path  # `libero` package root, goes on client PYTHONPATH
    default_suites: tuple
    # Where eval_task.py takes the model prompt from:
    #   "task"       -> task.language (filename-derived, matches openpi examples)
    #   "env"        -> env.language_instruction (parsed from the bddl
    #                   `(:language)` block; the only field that reflects
    #                   LIBERO-Pro's lan/task perturbations — task.language
    #                   keeps the original wording)
    #   "task_clean" -> non-official ablation that strips LIBERO-plus filename
    #                   suffixes (see prompt_clean.py); never use for leaderboard scores
    prompt_source: str = "task"
    # LIBERO_CONFIG_PATH (None = library default ~/.libero)
    config_dir: typing.Optional[pathlib.Path] = None
    # --num-trials default when the caller does not pass one.
    default_num_trials: int = 10
    # Whether --init-seed / --init-states-root make sense for this benchmark
    # (False where the benchmark's tasks carry their own perturbation-specific
    # init states that must not be resampled).
    supports_seeded_inits: bool = True
    # When populated, run_eval can distinguish an official result from an
    # explicitly requested development subset in summary.json.
    official_protocol_name: typing.Optional[str] = None
    official_suite_task_counts: tuple = ()
    official_num_trials: typing.Optional[int] = None
    official_prompt_source: typing.Optional[str] = None

    def client_env(self):
        """Extra environment for every client-venv process (eval_task, warm-up)."""
        env = {"PYTHONPATH": str(self.package_dir)}
        if self.config_dir is not None:
            env["LIBERO_CONFIG_PATH"] = str(self.config_dir)
        return env

    def resolve_max_steps(self, suite):
        if suite in BASE_MAX_STEPS:
            return BASE_MAX_STEPS[suite]
        for base in sorted(BASE_MAX_STEPS, key=len, reverse=True):
            if suite.startswith(base + "_"):
                return BASE_MAX_STEPS[base]
        raise ValueError(
            f"Cannot derive a step limit for suite '{suite}' (not derived from any of {sorted(BASE_MAX_STEPS)})."
        )

    def prepare(self, suites):
        """Make sure the benchmark's assets exist before servers are started."""

    def task_groups(self, suite) -> typing.Optional[dict]:
        """Optional {task_id: group_label} for per-group summary rollups.

        None (default) = the suite itself is the only grouping. LIBERO-plus
        returns the perturbation dimension per task.
        """
        return None

    def protocol_request(self, suites, suite_n_tasks, task_ids, num_trials):
        """Describe whether a requested run matches this benchmark's official protocol."""
        if self.official_protocol_name is None:
            return None

        expected = dict(self.official_suite_task_counts)
        deviations = []
        if len(suites) != len(expected) or set(suites) != set(expected):
            deviations.append(f"suites must be exactly {list(expected)}")
        for suite, expected_count in expected.items():
            actual_count = suite_n_tasks.get(suite)
            if actual_count != expected_count:
                deviations.append(f"{suite} registry has {actual_count} tasks, expected {expected_count}")
        if task_ids is not None:
            deviations.append("--task-ids selects a development subset")
        if num_trials != self.official_num_trials:
            deviations.append(f"--num-trials must be {self.official_num_trials}, got {num_trials}")
        if self.official_prompt_source is not None and self.prompt_source != self.official_prompt_source:
            deviations.append(
                f"prompt_source must be {self.official_prompt_source!r}, got {self.prompt_source!r}"
            )

        scheduled = (
            len(suites) * len(task_ids)
            if task_ids is not None
            else sum(suite_n_tasks.get(suite, 0) for suite in suites)
        )
        return {
            "name": self.official_protocol_name,
            "official_request": not deviations,
            "official_result": False,
            "expected_task_count": sum(expected.values()),
            "scheduled_task_count": scheduled,
            "expected_trials_per_task": self.official_num_trials,
            "deviations": deviations,
        }


class _LiberoProBenchmark(Benchmark):
    def prepare(self, suites):
        bddl_root = _PRO_CACHE / "bddl_files"
        init_root = _PRO_CACHE / "init_files"
        repo_libero = self.package_dir / "libero" / "libero"

        self._download_hf_assets(suites)
        for root, repo_src in ((bddl_root, repo_libero / "bddl_files"), (init_root, repo_libero / "init_files")):
            self._merge_root(root, repo_src, _PRO_CACHE / "hf" / repo_src.name)
        self._write_config(repo_libero, bddl_root, init_root)

        missing = [s for s in suites if not (bddl_root / s).is_dir() or not (init_root / s).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"No bddl/init files for suites {missing} under {_PRO_CACHE}. "
                f"Available perturbation suites: "
                f"{sorted(f'{b}_{d}' for b in _PRO_BASES for d in _PRO_DIMENSIONS)} "
                f"plus the base suites shipped with the LIBERO-PRO checkout."
            )

    def _download_hf_assets(self, suites):
        hf_dir = _PRO_CACHE / "hf"
        repo_libero = self.package_dir / "libero" / "libero"
        needed = [
            s
            for s in suites
            if not (repo_libero / "bddl_files" / s).is_dir() and not (hf_dir / "bddl_files" / s).is_dir()
        ]
        if not needed:
            return
        from huggingface_hub import snapshot_download  # lazy; openpi venv has it

        print(f"[run_eval] downloading LIBERO-Pro assets ({_PRO_HF_DATASET}) for suites {needed} ...")
        snapshot_download(repo_id=_PRO_HF_DATASET, repo_type="dataset", local_dir=str(hf_dir))

    @staticmethod
    def _merge_root(root, *sources):
        """(Re)build a merged suite root: one symlink per suite dir.

        Later sources win on name clashes (HF assets override repo copies).
        Real directories placed here by hand are left untouched.
        """
        root.mkdir(parents=True, exist_ok=True)
        for entry in root.iterdir():
            if entry.is_symlink():
                entry.unlink()
        for src in sources:
            if not src.is_dir():
                continue
            for suite_dir in src.iterdir():
                if not suite_dir.is_dir():
                    continue
                link = root / suite_dir.name
                if link.is_symlink():
                    link.unlink()
                if not link.exists():  # keep real dirs
                    link.symlink_to(suite_dir)

    def _write_config(self, repo_libero, bddl_root, init_root):
        # Written before any `import libero` so the interactive first-import
        # config prompt never triggers. Overwritten every run (self-healing).
        assert self.config_dir is not None
        datasets_dir = _PRO_CACHE / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"assets: {repo_libero / 'assets'}",
            f"bddl_files: {bddl_root}",
            f"benchmark_root: {repo_libero}",
            f"datasets: {datasets_dir}",
            f"init_states: {init_root}",
        ]
        (self.config_dir / "config.yaml").write_text("\n".join(lines) + "\n")


_PLUS_CACHE = pathlib.Path(os.path.expanduser("~/.cache/libero_plus"))
_PLUS_HF_DATASET = "Sylvest/LIBERO-plus"
# Pinned dataset commit (assets.zip): bump deliberately together with
# LIBERO_PLUS_REF in setup.sh, then re-run the smoke tests.
_PLUS_ASSETS_REVISION = "dd2bd61b7d9a6fef1abc52d606e983b41886a149"
# Top-level entries that prove an extraction completed (os.replace makes the
# extraction itself atomic; this guards against a wrong/partial zip).
_PLUS_ASSET_MARKERS = ("new_objects", "scenes", "textures", "wall.xml")


def _extract_assets(zip_path, assets_dir):
    """Extract the asset pack and move its `assets` root to assets_dir.

    The upstream zip nests everything under the author's absolute HPC path
    (`inspire/hdd/.../LIBERO-plus-0/assets/...`), so the real root is located
    via the marker entries instead of assuming a fixed layout.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        marker = _PLUS_ASSET_MARKERS[0] + "/"
        prefixes = {n[: n.index(marker)] for n in names if marker in n}
        if len(prefixes) != 1:
            raise FileNotFoundError(
                f"assets.zip does not contain exactly one '{marker}' tree (prefixes: {sorted(prefixes)[:3]})"
            )
        tmp = assets_dir.parent / "assets.extracting"
        shutil.rmtree(tmp, ignore_errors=True)
        zf.extractall(tmp)
    extracted = tmp / prefixes.pop() if prefixes != {""} else tmp
    missing = [m for m in _PLUS_ASSET_MARKERS if not (extracted / m).exists()]
    if missing:
        raise FileNotFoundError(f"assets.zip did not contain {missing} (extracted to {extracted})")
    shutil.rmtree(assets_dir, ignore_errors=True)
    os.replace(extracted, assets_dir)
    shutil.rmtree(tmp, ignore_errors=True)


@functools.lru_cache(maxsize=None)
def _plus_task_groups(package_dir: str, suite: str):
    """{task_id: perturbation dimension} from the fork's task_classification.json.

    The json lists tasks per suite in registry order with 1-based ids
    (verified against libero_suite_task_map for all four suites), so
    task_id = id - 1.
    """
    path = pathlib.Path(package_dir) / "libero" / "libero" / "benchmark" / "task_classification.json"
    data = json.loads(path.read_text())
    if suite not in data:
        return None
    return {entry["id"] - 1: entry["category"] for entry in data[suite]}


class _LiberoPlusBenchmark(Benchmark):
    def prepare(self, suites):
        self._check_client_deps()
        assets_dir = _PLUS_CACHE / "assets"
        if not all((assets_dir / m).exists() for m in _PLUS_ASSET_MARKERS):
            self._download_and_extract_assets(assets_dir)
        self._link_checkout_assets(assets_dir)
        self._write_config(assets_dir)

        repo_libero = self.package_dir / "libero" / "libero"
        missing = [s for s in suites if not (repo_libero / "bddl_files" / s).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"No bddl files for suites {missing} under {repo_libero / 'bddl_files'} — "
                f"is {self.package_dir} a LIBERO-plus checkout (setup.sh clones it)?"
            )

        expected_counts = dict(PLUS_SUITE_TASK_COUNTS)
        for suite in suites:
            expected_count = expected_counts.get(suite)
            groups = _plus_task_groups(str(self.package_dir), suite)
            if expected_count is None or groups is None:
                raise ValueError(f"{suite!r} is not part of the official LIBERO-Plus protocol")
            if len(groups) != expected_count or set(groups) != set(range(expected_count)):
                raise ValueError(
                    f"LIBERO-Plus classification for {suite} is incomplete: "
                    f"expected task ids 0..{expected_count - 1}, found {len(groups)} entries"
                )
            unknown_dimensions = set(groups.values()) - PLUS_DIMENSIONS
            if unknown_dimensions:
                raise ValueError(
                    f"LIBERO-Plus classification for {suite} has unknown dimensions: "
                    f"{sorted(unknown_dimensions)}"
                )

    @staticmethod
    def _check_client_deps():
        """Fail fast if the client venv can't run LIBERO-plus envs.

        The fork imports wand (needs the ImageMagick MagickWand system
        library) and scikit-image at env_wrapper import time, i.e. for every
        task regardless of its perturbation dimension.
        """
        import subprocess  # lazy: only benchmark that needs a probe

        probe = subprocess.run(
            [str(CLIENT_VENV_PY), "-c", "import wand, skimage"],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "LIBERO-plus needs `wand` and `scikit-image` in the LIBERO client venv "
                "plus the ImageMagick system library (apt install libmagickwand-dev). "
                f"Re-run setup.sh or install them manually. Probe error:\n{probe.stderr.strip()}"
            )

    def _download_and_extract_assets(self, assets_dir):
        from download import download_model  # sibling module; lazy like the HF imports

        hf_dir = _PLUS_CACHE / "hf"
        print(f"[run_eval] downloading LIBERO-plus asset pack ({_PLUS_HF_DATASET}, ~6.4 GB) ...")
        download_model(_PLUS_HF_DATASET, hf_dir, revision=_PLUS_ASSETS_REVISION, repo_type="dataset")
        print("[run_eval] extracting assets.zip ...")
        _extract_assets(hf_dir / "assets.zip", assets_dir)

    def _link_checkout_assets(self, assets_dir):
        # The fork resolves object XMLs relative to its package dir
        # (libero/libero/envs/objects/*.py: `absolute_path = <pkg>/libero/libero`),
        # so the assets must be reachable *inside* the checkout. Upstream's
        # .gitignore already ignores libero/libero/assets/ — a symlink keeps
        # the checkout effectively read-only and `git status` clean.
        link = self.package_dir / "libero" / "libero" / "assets"
        if link.is_symlink():
            link.unlink()
        if not link.exists():  # keep a real dir (user extracted assets by hand)
            link.symlink_to(assets_dir)

    def _write_config(self, assets_dir):
        # Written before any `import libero` so the interactive first-import
        # config prompt never triggers. Overwritten every run (self-healing).
        assert self.config_dir is not None
        repo_libero = self.package_dir / "libero" / "libero"
        datasets_dir = _PLUS_CACHE / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"assets: {assets_dir}",
            f"bddl_files: {repo_libero / 'bddl_files'}",
            f"benchmark_root: {repo_libero}",
            f"datasets: {datasets_dir}",
            f"init_states: {repo_libero / 'init_files'}",
        ]
        (self.config_dir / "config.yaml").write_text("\n".join(lines) + "\n")

    def task_groups(self, suite):
        return _plus_task_groups(str(self.package_dir), suite)


BENCHMARKS = {
    "libero": Benchmark(
        name="libero",
        package_dir=OPENPI_DIR / "third_party" / "libero",
        default_suites=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        prompt_source="task",
        config_dir=None,  # existing ~/.libero setup; behavior identical to before
    ),
    "libero_pro": _LiberoProBenchmark(
        name="libero_pro",
        package_dir=LIBERO_PRO_DIR,
        default_suites=tuple(f"{b}_{d}" for b in _PRO_BASES for d in _PRO_DIMENSIONS),
        prompt_source="env",
        config_dir=_PRO_CACHE / "config",
    ),
    "libero_plus": _LiberoPlusBenchmark(
        name="libero_plus",
        package_dir=LIBERO_PLUS_DIR,
        # Same names as the base benchmark but 2402/2518/2591/2519 perturbed
        # variants each (10,030 total). The isolated PYTHONPATH/config make the
        # name collision with "libero" harmless.
        default_suites=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        # Match the published evaluator/OpenPI example exactly: pass
        # task.language through unchanged.  task_clean is a useful ablation,
        # but scores produced with it are not leaderboard-comparable.
        prompt_source="task",
        config_dir=_PLUS_CACHE / "config",
        default_num_trials=1,  # official protocol: 1 rollout per task variant
        supports_seeded_inits=False,
        official_protocol_name="libero_plus_official_v1",
        official_suite_task_counts=PLUS_SUITE_TASK_COUNTS,
        official_num_trials=1,
        official_prompt_source="task",
    ),
}


def get_benchmark(name) -> Benchmark:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"Unknown benchmark '{name}'; choose from {sorted(BENCHMARKS)}")

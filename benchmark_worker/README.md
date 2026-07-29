# Benchmark Worker — 评测编排层

后端 Benchmark API 与本仓库评测流水线之间的隔离层:**worker 主进程只做 HTTP
通信和子进程管理,不运行任何评测逻辑**;评测(GPU、policy server、MuJoCo)
全部发生在 `libero_eval/run_eval.py` 子进程里。两层各自独立演进,接口就是
CLI 参数 + `summary.json`。

```
backend(评测队列 / 评分接口,HTTP + X-API-Key)
   ▲  ▲
   │  └── POST /api/v1/benchmark/task/{id}/score   ← admin key,评分回传
   └───── GET  /api/v1/benchmark/queue             ← public key,周期轮询
   │
benchmark_worker/worker.py(常驻主进程,本仓库 uv 环境)
   ├── 本地持久化队列 benchmark_worker_state.json(跨重启去重)
   ├── 模型下载:huggingface_hub,revision 锁定到任务的 hf_commit
   │   (hf_commit 缺失/非法的任务直接拒绝并上报,不评测未锚定的提交)
   └── 逐个派发 ──► libero_eval/run_eval.py 子进程(产出 summary.json)
```

接口定义见 prototype 仓库 `docs/api_reference_zh.md` 第三节。

## 运行

在仓库根目录:

```bash
BACKEND_PUBLIC_API_KEY=<control.json 的 public_key> \
BACKEND_ADMIN_API_KEY=<后端 backend.yaml 的 admin_key> \
    uv run benchmark_worker/worker.py --backend-url http://localhost:8001 \
    --benchmark libero_pro_custom_1 --num-trials 50
```

鉴权按最小权限拆分:`GET /api/v1/benchmark/queue` 和任务详情使用 `public_key`
(`control.json` → `public_key`),评分 POST 使用 `admin_key`
(`backend.yaml` → `admin_key`)。两个值都通过 `X-API-Key` 发送且启动时必填。
旧 `--api-key` / `BACKEND_API_KEY` 仍可临时把同一个 key 用于读写,但新部署应使用
两个独立参数或环境变量。

调试:`--once`(轮询一次、处理完即退出)配合 `--num-trials 2 --task-ids 0
--gpus 0` 可做单卡冒烟;`--allow-local-model` 允许 `hf_repo_id` 直接是本地
checkpoint 路径(仅测试)。

## 任务状态机(benchmark_worker_state.json)

```
pending → running → done_pending_submit → submitted(终态)
                                        ↘ abandoned(后端永久拒绝,终态)
```

- 队列里 `task_id` 已见过的任务绝不重复评测(跨轮询、跨重启);
- 进程被杀:评测中的任务下次启动自动重跑;已评完未确认的自动补交;
- 提交失败按 1/2/4/8/16/30 分钟持久化指数退避(跨轮询/重启);仅超时、断连和 5xx 这类结果不确定的错误会回查任务详情,严格核对任务身份、终态、总分和完整环境汇总,一致即视为提交成功;兼容旧 backend 的任务详情不含 `benchmark`,但远端明确返回该字段时必须一致;
- 同一 task 重新评测时会清零旧结果的提交次数、错误和退避时间,避免新结果继承历史失败状态;
- 明确的 400/404/413/422 不回查且转 `abandoned`;401/403/429 保留结果等待配置修复或限流恢复。
- 每次 POST 前会把实际发送的 JSON body 记到控制台和 `logs/benchmark_worker-YYYY-MM-DD.log`;API key 只存在于 HTTP header,不会写入该日志。

## 评分映射(summary.json → score payload)

| API 字段 | 来源 |
|----------|------|
| `benchmark` | worker 的 `--benchmark`;计分 profile 身份随每笔评分提交 |
| `env_scores[]` | 通常每个 summary suite 一条；Custom 1 按下述规则把 16 条折叠为 6 条兼容记录 |
| `total_score` | Custom 1 为六条兼容记录的等权平均;其他 profile 为本地计分预览值 |
| `per_task_scores[]` | summary `tasks` 中成功的条目(`libero_spatial_03` 格式),完整保存在本地产物/账本;API 提交时省略(可选字段,避免后端逐条同步写库触发 Cloudflare 524) |
| `duration_sec` | 该任务端到端墙钟时间(含模型下载) |
| `success` | 是否产出了可用分数;个别 task 重试后仍失败不算整体失败,损失记在 `error` / `env_scores[].error` |

benchmark 完全由启动参数 `--benchmark` 选择,队列中的 `env_list` 是数据库兼容字段,
worker 不读取。`libero_pro` 运行并上报完整的 16 个 Pro suite;
`libero_pro_custom_1` 也运行完全相同的 16 个 suite,但只在该 profile 下把结果折叠成
6 条兼容记录:四个 base 各取 object/swap/lan/task 的等权平均,其中 LIBERO-10
按后端当前约定上报为 `libero_10`(不涉及 libero_90),再单独上报
`libero_object_swap` 与 `libero_spatial_swap`。两个 swap 因此既参与 base 平均,
又各自作为独立记录出现。
`libero_plus` 强制 10,030 tasks × 1 trial、关闭 init-state 重采样并拒绝 task
子集;四个 suite 按官方 task 数 2402/2518/2591/2519 加权,等价于全 episode
micro-average。非官方或未完整跑完的 LIBERO-Plus summary 不会上报分数。
升级到 `libero_plus_official_v1` 后,本地账本里旧协议的缓存结果会自动失效并
等待重新评测,不会直接补交。
同一任务的本地账本 benchmark 不一致时不会重发旧结果。

## 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--backend-url` | `$BACKEND_URL` / `http://localhost:8001` | 后端地址 |
| `--public-api-key` | `$BACKEND_PUBLIC_API_KEY` | 只读 public key(`control.json` → `public_key`) |
| `--admin-api-key` | `$BACKEND_ADMIN_API_KEY` | 写入 admin key(`backend.yaml` → `admin_key`) |
| `--api-key` | `$BACKEND_API_KEY` | 旧版兼容:同一个 key 用于读写 |
| `--queue-path` | `/api/v1/benchmark/queue` | 上游公开任务队列路径 |
| `--benchmark` | 必填 | `libero` / `libero_pro` / `libero_pro_custom_1` / `libero_plus` |
| `--poll-interval` | 60 | 队列轮询间隔(秒) |
| `--num-trials` | 必填 | 每 task 试验数;LIBERO-Plus 必须为 1,其他 LIBERO 官方口径为 50 |
| `--gpus` | 0-7 | 透传 run_eval.py |
| `--init-workers-per-gpu` | 4 | 每张 GPU 并发的 init-state 生成进程数;按 task 切分,与正式评测的 `--workers-per-gpu` 独立 |
| `--no-init-randomization` | 关 | 关闭 init states 混合随机化(默认开:用队列条目自带的 seed(公开可验证,miner 可复现),每 task 一半 trial 用官方 init、一半用重采样 init,seed 随评分 payload 回传;见 docs/init_state_randomization.md) |
| `--eval-timeout` | 28800 | 单次评测超时(秒) |
| `--state-file` | `benchmark_worker_state.json` | 本地状态账本 |
| `--output-root` | `eval_runs/` | 评测产物(日志/视频/summary) |
| `--download-dir` | `hf_models/` | HF 模型下载缓存 |

# Validator — LIBERO / LIBERO-Pro / LIBERO-plus 并行评测

输入一个模型(本地 checkpoint 目录 / HF repo id / HF 链接),在选定的 benchmark 上评测,
8 张 GPU 并行加速(每卡一个 policy server,task 动态分发)。

Benchmark 通过 `--benchmark` 选择,评测逻辑与 benchmark 定义解耦
(`libero_eval/benchmarks.py` 是唯一的 benchmark 知识层,新增变体只需在那里注册):
- **`libero`**(默认):原版 LIBERO,4 个标准 suite × 10 task
- **`libero_pro`**:LIBERO-Pro(arXiv:2510.03827)扰动评测,16 个 suite × 10 task
  (见下方「LIBERO-Pro」)
- **`libero_plus`**:LIBERO-plus(arXiv:2510.13626)鲁棒性评测,4 个同名 suite
  共 10,030 个扰动任务变体 × 1 trial,按 7 个扰动维度出分(见下方「LIBERO-plus」)

支持 openpi 的两种 checkpoint 格式,按目录内容自动识别:
- **JAX**: 含`params/`(orbax OCDBT)
- **PyTorch**: 含 `model.safetensors`(见下方「JAX → PyTorch 转换」)

另支持 **OpenVLA-OFT** 的原生分片 checkpoint(`model.safetensors.index.json` +
`model-*.safetensors` + action head/proprio projector)。它使用独立 policy server,
不会把 OpenVLA 权重伪转换成不兼容的 openpi 架构。

## TODO List
- [x] Generating init states (when init seed is passed), slicing by suite -> by task.

## 架构

```
run_eval.py(调度器,本仓库 uv 环境)
 ├── 启动时独占锁定所选 GPU,拒绝另一轮 run_eval 重叠占用同一张卡
 ├── benchmarks.py:--benchmark 决定 libero 包路径、bddl/init 资产、suite 列表、
 │      步数上限、指令来源;eval_task.py 保持 benchmark 无关
 ├── 每张 GPU 启动一个 policy server(openpi 服务端 venv,websocket 端口 8000+i)
 │      --server-impl upstream = openpi 官方 serve_policy.py(单请求 batch=1)
 │      --server-impl batched  = serve_policy_batched.py(贪心动态凑批:队列里攒了
 │      几个就拼几个一次推理,稀疏时 batch=1 零等待;配 6-8 workers 用)
 ├── 任务队列:suites × 各自任务数 个 (suite, task_id)(libero 40 个,libero_pro 160 个,
 │      libero_plus 10,030 个),按步数上限降序派发(长任务先跑,避免收尾长尾)
 ├── 每张 GPU N 个 worker 线程(--workers-per-gpu,默认 3):从队列领任务 → 起 eval_task.py 子进程
 │      同卡多 worker 共享一个 policy server:一个 env 做 CPU 仿真时另一个 env 的推理占用 GPU,
 │      实测 4090 上 3 workers 单模型评测 ~1.8x 提速(GPU 利用率 35%→65%)
 │      eval_task.py(LIBERO client venv,Python 3.8)
 │      MuJoCo EGL 渲染(MUJOCO_EGL_DEVICE_ID=该卡)→ websocket 请求动作
 └── 汇总 results/*.json → summary.json + 控制台报表
```

沿用 openpi 官方 `examples/libero` 的评测协议(图像旋转 180°、resize-pad 224、
每 5 步重规划、每 episode 使用官方固定初始状态、seed=7),结果可与官方数字对齐。

## 目录与环境

本仓库自包含,外部依赖全部在 `third_party/`(gitignore,由 `setup.sh` 管理):

```
validator/
├── pyproject.toml       # 调度器环境(uv 管理,仅需 huggingface_hub)
├── setup.sh             # 一次性安装:克隆 third_party + 构建三套环境
├── libero_eval/         # run_eval.py / benchmarks.py / eval_task.py / paths.py
├── benchmark_worker/    # 对接后端评测队列的常驻编排层(见其 README)
├── tools/               # upload_hf.py
└── third_party/         # openpi、LIBERO-PRO、LIBERO-plus checkout(克隆或符号链接)
```

涉及三套 Python 环境(版本互不兼容,无法合并;`setup.sh` 全部装好):

| 环境 | 位置 | 用途 |
|------|------|------|
| validator(pyproject.toml)| `.venv/` | 调度器 run_eval.py、tools/ |
| openpi 服务端 | `third_party/openpi/.venv/` | policy server(JAX/torch CUDA)|
| LIBERO 客户端(Python 3.8)| `third_party/openpi/examples/libero/.venv/` | MuJoCo 仿真 |

已有 openpi / LIBERO-PRO / LIBERO-plus checkout 的机器可以不重复克隆:在
`third_party/` 放符号链接,或设环境变量 `OPENPI_DIR` / `LIBERO_PRO_DIR` /
`LIBERO_PLUS_DIR` 指向现有位置(setup.sh 与运行时都认)。

## 安装(一次性)

```bash
git clone https://github.com/openroboto-ai/openroboto-evaluation && cd openroboto-evaluation
bash setup.sh --with-checkpoint   # --with-checkpoint 会额外下载 12.4GB 演示模型
# 需要复现 OpenVLA-OFT+ 时,额外创建其独立推理环境
bash setup.sh --with-openvla-oft
```

## 用法

在仓库根目录用 `uv run` 执行(自动使用 pyproject.toml 对应的环境):

```bash
# 完整评测:4 suites × 10 tasks,8 卡并行,每 task 10 次试验(官方标准为 50)
# --commit-id 必填:本地 checkpoint 目录仅作记录,临时本地评测传 local
uv run libero_eval/run_eval.py \
    --model ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --commit-id local --num-trials 10

# 输入 HF 模型(要求 repo 内含 openpi 格式 checkpoint:params/ 或 model.safetensors)。
# 下载 pin 到 --commit-id 指定的提交(完整 40 位 hex,从 repo 的 commits 页复制),
# 缓存目录按 commit 区分:同一 repo 重新提交后,旧缓存不可能被误当成新提交。
uv run libero_eval/run_eval.py --model your-name/pi05-libero-finetuned \
    --commit-id 0123456789abcdef0123456789abcdef01234567 --num-trials 10

# LIBERO-Pro 扰动评测(16 个扰动 suite;bddl/init 资产首次使用时自动下载)
uv run libero_eval/run_eval.py \
    --model ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --commit-id local --benchmark libero_pro --num-trials 10

# Custom 1 与标准 Pro 执行完全相同的 16 个 suite;作为 worker profile 使用时,
# 后端会把 object_swap / spatial_swap 各额外计权一次,不增加 rollout。
uv run benchmark_worker/worker.py \
    --backend-url http://localhost:8001 \
    --benchmark libero_pro_custom_1 --num-trials 50

# LIBERO-plus 鲁棒性评测:10,030 个扰动任务 × 1 trial(官方协议,--num-trials 默认 1)。
# 首次使用自动下载 6.4GB 资产包;完整一轮 8 卡约 7-10 小时,建议 --save-videos 0
uv run libero_eval/run_eval.py \
    --model ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --commit-id local --benchmark libero_plus --save-videos 0

# 论文 Table 2 "Ours"(79.5%) checkpoint。必须 pin 官方提交;
# family 会从 checkpoint 自动识别,官方协议为 10,030 tasks x 1 trial。
uv run libero_eval/run_eval.py \
    --model Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata \
    --commit-id a85655ec941bae6644c9fbdf62db02b9726d7cf5 \
    --benchmark libero_plus --workers-per-gpu 1 --save-videos 0

# 快速冒烟测试:1 个 suite 的 2 个 task × 2 trial,单卡
uv run libero_eval/run_eval.py \
    --model ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --commit-id local --suites libero_spatial --task-ids 0,1 --num-trials 2 --gpus 0

# 抗过拟合混合评测:每 task 一半 trial 用官方 init、一半用该 seed 重采样的
# init(首次使用某 seed 时自动生成并缓存;见 docs/init_state_randomization.md)
uv run libero_eval/run_eval.py \
    --model ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --commit-id local --benchmark libero_pro --init-seed 12345
```

常用参数:

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | 必填 | 本地 checkpoint 目录 / HF repo id / HF URL |
| `--model-family` | `auto` | 从 checkpoint 自动识别 `openpi` / `openvla_oft`;可显式指定并校验一致性 |
| `--commit-id` | 必填 | 评测所 pin 的 HF commit(完整 40 位 hex);HF 模型按此下载并按 commit 缓存,本地目录仅记录进 summary.json(临时本地评测传 `local`) |
| `--benchmark` | `libero` | `libero`(原版)、`libero_pro`(扰动评测)、`libero_pro_custom_1`(同一 Pro runtime,Custom 1 计分身份)或 `libero_plus`(鲁棒性评测) |
| `--task-ids` | 该 suite 全部任务 | 逗号分隔的 task id 子集(调试用;LIBERO-plus 会标记为非官方结果) |
| `--num-trials` | benchmark 协议值 | 每 task 试验数:libero/libero_pro 默认 10(官方口径 50,上限受官方初始状态数限制);libero_plus 默认 1(官方协议) |
| `--gpus` | 0-7 | 使用哪些卡,server 与仿真渲染都按卡隔离 |
| `--workers-per-gpu` | 3 | 每卡并发评测客户端数,共享该卡的 policy server;默认 3 为 4090 实测最优(~1.8x),4 开始退化;batched server 配 6(~2.0x);传 1 可复现严格串行的旧行为。>1 时 JAX 采样噪声的 rng 顺序随请求到达序变化,分数存在与「任务换卡重跑」同量级的随机浮动 |
| `--init-workers-per-gpu` | 4 | 每卡并发的 init-state 生成进程数,与正式评测 worker 独立;按 task 切分,8 GPUs 时最多并行启动 32 个生成进程 |
| `--server-impl` | `upstream` | `batched` 启用动态凑批 server(见架构节)。仅对 JAX checkpoint 生效批处理;4090 上 batch 扩展性有限(每样本 80→62ms,-22%),整体比 upstream w=3 再快 ~13%,算力更强的卡(A100/H100)收益更大。PyTorch checkpoint 自动退化为 batch=1(其 torch.compile 按 shape 静态编译,动态 batch 会反复重编译) |
| `--max-batch` | 8 | batched server 单次推理的 batch 上限(pad 到 2 的幂) |
| `--init-seed` | 关闭 | 抗过拟合混合评测:每 task 一半 trial 用官方 init、一半用该 seed 重采样的 init;只生成 seeded 一半所需的状态并按 (seed, seeded 数量) 幂等缓存。benchmark_worker 默认传入队列条目自带的 seed(公开可验证,miner 可复现)。全替换版本用 `--init-states-root`(实验对比用,与本参数互斥)。详见 docs/init_state_randomization.md |
| `--save-videos` | 1 | 每 task 保存前 N 个 trial 的视频 |
| `--retries` | 1 | task 失败自动重试次数 |
| `--model-architectures` | `pi0.5` | 允许的模型架构,逗号分隔;调试可传 `pi0,pi0.5`(也接受 `π0,π0.5`),不使用特殊的 `both` 值 |
| `--config` | 自动 | openpi 训练配置名;默认按 checkpoint 架构选 `pi0_libero` / `pi05_libero`,也可显式覆盖 |
| `--output-dir` | `eval_runs/<时间戳>_<benchmark>_<模型名>` | 输出目录 |

## 输出

```
eval_runs/<run>/
├── summary.json          # 总分 + 每 suite 成功率 + 每 task 明细
├── results/<task>.json   # 单 task 结果(每 trial 成功与否、步数、耗时)
├── logs/                 # server 与各 task 的日志
└── videos/               # 抽样 rollout 视频(每 task 前 N 个 trial)
```

`summary.json` 关键字段:`total_success_rate`(总成功率)、`suites.<name>.success_rate`、
`tasks.<task>.episodes[]`。评测有 task 失败时进程退出码为 2,`tasks.<task>.status = "failed"`。
libero_plus 额外输出 `dimensions.<扰动维度>.success_rate`(7 个维度,控制台报表同步打印)。

## 对接后端评测队列(Benchmark Worker)

`benchmark_worker/worker.py` 是常驻编排层:轮询后端评测队列 → 按 `hf_commit`
锁定下载模型 → 调 `run_eval.py` 子进程 → 把 `summary.json` 映射成评分 JSON
回传后端,断点续跑、去重、失败重试都内置。详见 `benchmark_worker/README.md`:

```bash
uv run benchmark_worker/worker.py --backend-url http://localhost:8001 \
    --benchmark libero_pro_custom_1 --num-trials 50
# 需要鉴权时，通过 BACKEND_API_KEY 环境变量或 --api-key 参数提供 worker key
```

## 模型合法性检查(评测前置)

`run_eval.py` 在启动任何 policy server 之前,会按 openpi 的 checkpoint 格式校验模型
(`libero_eval/check_model.py`,纯标准库实现,不依赖 jax/torch)。不合法的模型直接被拒绝,
逐条打印原因,退出码为 3,不消耗 GPU 时间。校验内容与 openpi 实际加载路径一一对应:

- **checkpoint 结构**:根目录须有 `model.safetensors`(PyTorch)或 `params/`(JAX orbax);
  两者都有时以 PyTorch 为准(与 openpi 行为一致)
- **权重完整性**:safetensors 头部/偏移量自洽(防截断上传)、orbax `_METADATA` 可解析且
  树根为 `params`、git-lfs 指针文件检测
- **架构筛选与结构匹配**:`--model-architectures` 默认只接受 `pi0.5`;传
  `pi0` 或 `pi0,pi0.5` 时会识别 checkpoint 并自动选择对应推理配置。参数组须与配置的
  架构一致(如 pi05 要求 `time_mlp_*`,出现
  `state_proj`/`action_time_mlp_*` 则判定为 pi0 checkpoint)、投影层形状核对
  `action_dim`/expert 宽度、总参数量须在 PaliGemma+expert 的合理区间
- **归一化统计**:`assets/physical-intelligence/libero/norm_stats.json` 存在、`state`/`actions`
  条目齐全、数值有限;pi05 使用分位数归一化,`q01`/`q99` 必须存在

也可以独立运行(输入与 `--model` 相同,支持本地目录 / HF repo id / HF URL;
HF 引用同样必须用 `--commit-id` pin 到具体提交):

```bash
uv run libero_eval/check_model.py --model your-name/pi05-libero-finetuned \
    --commit-id <完整40位commit哈希>                                                # 人读输出
uv run libero_eval/check_model.py --model hf_models/xxx --json                     # 机器可读
```

合法退出码 0,不合法退出码 1 并列出全部问题。`run_eval.py --skip-model-check` 可跳过
(仅调试用)。

## JAX → PyTorch 转换

官方 checkpoint 是 JAX 格式;转成 PyTorch(`model.safetensors`)后评测流水线用法完全不变,
只需把 `--model` 指向转换后的目录:

```bash
cd third_party/openpi
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv/bin/python \
    examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --config_name pi05_libero --precision float32 \
    --output_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

# 官方脚本从 checkpoint_dir 的父目录找 assets(bug),norm_stats 不会被拷出,必须手动补:
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
      ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/
```

注意事项:
- 转换纯 CPU(`JAX_PLATFORMS=cpu` + 空 `CUDA_VISIBLE_DEVICES`),不占 GPU,约 2 分钟
- `--precision float32`:保存精度。推理时加载器统一转 bf16 混合精度(layernorm 等少数层
  保留 fp32),fp32 保存可让这些层无量化损失;bf16 保存则文件减半(14GB → 7GB)
- 转换脚本靠**路径里含 "pi05" 字符串**判断模型结构,checkpoint 目录名须含 `pi05`
- 依赖 transformers 补丁(`setup.sh` 已自动打;缺补丁时转换/serve 会报
  `transformers_replace is not installed correctly`)
- PyTorch 推理默认 `torch.compile(mode="max-autotune")`:**每个 server 首次推理有
  ~5 分钟编译**(inductor 缓存生效后跳过);编译期间 server 日志刷
  `No valid triton configs. OutOfResources` 是 autotune 淘汰不适配 4090 的候选 kernel,
  属正常噪声。编译出问题时可给 server 加环境变量 `TORCHDYNAMO_DISABLE=1` 禁用 compile
- PyTorch server 显存 ~8.6GB/卡(按需分配);JAX server 是预分配 mem-fraction(0.7 → 17.8GB)

## HF 模型闭环(miner 上传 → validator 下载评测)

**上传**(miner 侧,或用官方转换产物做测试):

```bash
HF_TOKEN=<write token> HF_ENDPOINT=https://huggingface.co \
uv run tools/upload_hf.py \
    --src <checkpoint目录> --repo <user>/<model-name>   # 默认 private,--public 可公开
```

repo 结构即 checkpoint 结构(`model.safetensors` + `assets/<asset_id>/norm_stats.json`),
`upload_large_folder` 按文件提交、断点续传,传输被杀直接重跑。

**下载评测**(validator 侧,run_eval 原生支持 HF 链接 / repo id):

```bash
HF_TOKEN=<token> uv run libero_eval/run_eval.py \
    --model https://huggingface.co/<user>/<model-name> \
    --commit-id <完整40位commit哈希> --num-trials 10
```

下载 pin 到 `--commit-id`,落在 `hf_models/<user>__<model>@<commit前12位>/`(按 commit
区分缓存,同 repo 重新提交必然重新拉取),中断重跑可续传。public repo 无需 token。

### 多策略下载(hfd.sh + 镜像 + 回退)

worker 与 run_eval 的模型下载统一走 `libero_eval/download.py` 的策略链,按序尝试、
失败自动回退,全部失败才报错(错误逐策略汇总)。策略:

| 策略名 | 下载方式 | 端点 |
|--------|----------|------|
| `hfd-mirror` | hfd.sh + aria2c 多连接 | 镜像(默认 hf-mirror.com) |
| `hfd` | hfd.sh + aria2c 多连接 | 官方 |
| `hub-mirror` | snapshot_download | 镜像 |
| `hub` | snapshot_download | 官方 |

默认顺序 `hfd-mirror,hfd,hub`。hfd 策略成功后会用 snapshot_download 做收尾校验
(对已落盘文件 sha256 比对,不重下,固定走官方端点——镜像的 resolve 重定向缺
元数据头,无法哈希校验),保证评测不会用到静默损坏的权重。

hfd 策略失败时若命中 hfd.sh 的 "Re-run to resume" 提示且本次确有净下载进度,
会先原地重跑同一策略续传(hfd.sh 原生断点续传:aria2c -c + manifest 差量),
重跑次数用尽或无进度才回退到下一策略。

配置(env + CLI,CLI 覆盖 env):

| 配置 | 形式 | 默认 |
|------|------|------|
| `MODEL_DOWNLOAD_STRATEGIES` | env | `hfd-mirror,hfd,hub` |
| `--download-strategies` | worker / run_eval CLI | 同上 |
| `HF_MIRROR_ENDPOINT` | env | `https://hf-mirror.com` |
| `MODEL_DOWNLOAD_TIMEOUT` | env,单个 hfd 策略墙钟秒数 | `7200` |
| `MODEL_DOWNLOAD_RESUME_RETRIES` | env,单条 hfd 策略可续传失败的原地重跑上限(0 = 关闭) | `3` |

单独验证下载(不跑评测):

```bash
uv run libero_eval/download.py <user>/<repo> [--revision r] [--dest DIR] [--strategies hfd-mirror,hub]
```

token 与镜像:**镜像会剥掉 Authorization 头**,因此镜像策略会自动从子进程环境
剥除 `HF_TOKEN`(既无用又防泄露),private repo 走镜像会快速失败并回退到官方
策略;官方策略(`hfd`/`hub`)显式钉死 `https://huggingface.co`,机器全局的
`HF_ENDPOINT` 不再影响下载行为。上传(`tools/upload_hf.py`)仍需如上显式
`HF_ENDPOINT=https://huggingface.co`。

## 实测成绩与耗时(8×4090,2026-07-09,40 tasks × 10 trials)

| suite | JAX | PyTorch | 官方参考(50 trials,JAX) |
|-------|-----|---------|--------------------------|
| libero_spatial | 99.0% | 100.0% | 98.8% |
| libero_object | 98.0% | 99.0% | 98.2% |
| libero_goal | 99.0% | 96.0% | 98.0% |
| libero_10 | 93.0% | 92.0% | 92.4% |
| **总计** | **97.2%** | **96.8%** | 96.85% |
| 墙钟 | 447s | 527s | — |

两后端成绩差 0.4%(2/400 个 episode),在 10-trial 随机波动内,视为对齐。
HF 闭环实测(同 PyTorch 权重经 HF private repo 上传→下载,md5 无损):96.5%,
评测段墙钟 511s,进一步印证 trial 级噪声幅度(96.5~96.8%)。
耗时特征:PyTorch 编译后同 task 耗时比 JAX 略短(约 10-15%),但启动开销更大
(每 server 读 14.5GB fp32 权重 + inductor 编译缓存;**冷缓存时每卡首次推理另有
~5 分钟 torch.compile**,已编译过的机器缓存命中则跳过)。
50 trials(官方口径)折算约 35-45 分钟一轮。

## 故障排查

**GPU 驱动死锁**(症状:`nvidia-smi` 卡住不返回、CUDA/EGL 进程进入 `D` 状态且 `kill -9` 无效):
大量 MuJoCo EGL 渲染进程挤在同一张卡、或与 JAX server 初始化并发冲突时可能触发。
恢复:`sudo reboot`(最可靠);重启后可用 `sudo dmesg | grep -i xid` 查看具体错误码。
预防:始终通过 `run_eval.py` 调度(每卡 1 个 server,同一轮评测的渲染 worker 受控并发),
不要手动同时启动多个 `run_eval.py` / `eval_task.py` 抢占相同 GPU。
`run_eval.py` 启动时会先执行 `nvidia-smi` 预检,驱动无响应会拒绝启动;随后独占锁定
所选 GPU,重叠评测将以明确错误快速失败。常驻 benchmark worker 如果发现 CUDA/EGL
基础设施错误会把任务重新排队且不提交部分分数,避免把 validator 故障算到模型头上。

## LIBERO-Pro

LIBERO-Pro(arXiv:2510.03827,`third_party/LIBERO-PRO`)是原版 LIBERO 的扰动增强版,
用于检验模型是否只是记住了训练场景。`--benchmark libero_pro` 的默认 suite 是
4 个 base(spatial/object/goal/10)× 4 个扰动维度 = 16 个 suite,每个 10 task:

| 后缀 | 扰动维度 | 示例 |
|------|----------|------|
| `_object` | 物体替换(外观/颜色/尺寸) | black bowl → akita black bowl |
| `_swap` | 位置扰动(物体互换/挪位,指令不变) | 碗和盘子交换位置 |
| `_lan` | 语义改写(指令同义转述) | "pick up ... place" → "lift ... set" |
| `_task` | 任务重定义(改目标状态) | 开中间抽屉 → 开底层抽屉 |

论文的第五个维度(环境替换 `_env`)官方未发布资产、且其 README 自述不稳定,故不提供。

`libero_pro_custom_1` 不是新的模拟器 benchmark。它仍调用 `libero_pro` runtime、
执行上述 16 个 suite;后端只在总分中把 `libero_object_swap` 和
`libero_spatial_swap` 的权重从 1 提高到 2。因此 50 trials 时仍是
16 × 10 × 50 = 8,000 episodes,canonical 分母为 18。

实现要点(都封装在 `libero_eval/benchmarks.py`,评测协议与原版完全一致):
- **资产自动下载**:扰动 suite 的 bddl/init 文件来自 HF dataset
  `zhouxueyang/LIBERO-Pro`(约几 MB),首次使用时下载到 `~/.cache/libero_pro/`,
  与 repo 自带的 base suite 合并成统一目录(符号链接)
- **配置隔离**:LIBERO-Pro 是 `libero` 包的 fork(envs 与原版逐字节一致,只扩充了
  benchmark 注册表)。两个包通过 `PYTHONPATH` + `LIBERO_CONFIG_PATH` 隔离,
  互不污染 `~/.libero/config.yaml`
- **指令来源**:`task.language` 由 bddl 文件名推导,不反映 `_lan`/`_task` 的扰动;
  扰动后的指令只存在于 bddl 的 `(:language)` 块中。因此 libero_pro 用
  `env.language_instruction` 作为模型 prompt(result JSON 里 `prompt` 是实际输入,
  `task_description` 是原始指令,`prompt_source` 标记来源)
- 步数上限、初始状态(每 task 50 个)、图像预处理、replan 等协议均与原版相同,
  扰动 suite 沿用其 base suite 的步数上限

实测(pi05_libero 官方 JAX checkpoint,16 suites × 10 tasks × 3 trials,7×4090,
墙钟 724s,2026-07-11)——各维度均值:

| 维度 | spatial | object | goal | 10 | 均值 |
|------|---------|--------|------|-----|------|
| `_lan` 语义改写 | 100% | 100% | 96.7% | 93.3% | 97.5% |
| `_object` 物体替换 | 96.7% | 86.7% | 100% | 70.0% | 88.3% |
| `_swap` 位置扰动 | 46.7% | 16.7% | 33.3% | 6.7% | 25.8% |
| `_task` 任务重定义 | 50.0% | 10.0% | 20.0% | 20.0% | 25.0% |

与论文方向一致(语义扰动几乎无损、位置/任务扰动大幅崩塌;论文报告 π0.5 位置扰动
0.1-0.4,与我们的 swap 结果吻合)。注意论文的任务扰动报 0.0,我们测得 10-50%
——prompt/场景/判定均已核验来自官方扰动资产,差异可能源于论文评测细节或
trial 数(3 trials/task 噪声较大),50-trial 口径下再对照。

## LIBERO-plus

LIBERO-plus(arXiv:2510.13626,`third_party/LIBERO-plus`)把原版 4 个 suite 的 40 个
任务沿 7 个扰动维度展开成 10,030 个任务变体,官方协议是**每个变体 1 次 rollout**
(`--num-trials` 默认即 1)。suite 名与原版相同(spatial/object/goal/10),但任务数
变为 2402/2518/2591/2519;`summary.json` 与控制台报表按扰动维度汇总
(映射来自 fork 的 `task_classification.json`,已核验与任务注册顺序逐条对齐):

| 维度 | 任务数 | 扰动内容 |
|------|--------|----------|
| Camera Viewpoints | 1599 | 相机位姿/视场(参数编码在 bddl 文件名 `_view_..._initstate_...` 中,运行时由 fork 的 env_wrapper 施加;这些变体在磁盘上没有 bddl 文件,属预期) |
| Robot Initial States | 1550 | 机械臂初始位姿 |
| Sensor Noise | 1601 | 图像噪声/退化(wand/scikit-image 运行时生成) |
| Language Instructions | 1537 | LLM 改写指令(存于 bddl `(:language)` 块) |
| Objects Layout | 1525 | 目标位置扰动/加入干扰物(`_add_`/`_level` 新物体任务带独立 init 文件) |
| Light Conditions | 1142 | 光照强度/方向/颜色/阴影 |
| Background Textures | 1076 | 桌面/背景纹理替换 |

七个维度共 10,030 条。官方 `Total` 是所有 task variant 的 **micro-average**，
不是七个维度或四个 suite 的等权平均。benchmark worker 在保持四-suite API
结构的同时，按 2402/2518/2591/2519 的 suite task 数加权，结果与 episode
micro-average 完全等价。

实现要点(封装在 `libero_eval/benchmarks.py`):
- **资产**:bddl/init 文件随 checkout 分发;6.4GB 资产包(新物体/场景/纹理)来自
  HF dataset `Sylvest/LIBERO-plus`(pin 到具体 commit),首次使用时经 download.py
  策略链下载并解压到 `~/.cache/libero_plus/assets`。fork 按包内相对路径找资产,
  因此在 checkout 里放一个 gitignored 符号链接 `libero/libero/assets` 指向缓存
- **官方 prompt**(`prompt_source="task"`):与发布的 LIBERO-Plus/OpenPI 评测路径
  一致,直接传入 `task.language`,包括非 language 任务名中由 fork 生成的参数文本。
  `task_clean` 只保留为研究消融工具;上游 issue #48 尚无 maintainer 结论,使用
  clean prompt 得到的分数不得与 leaderboard 或官方复现结果比较
- **客户端依赖**:fork 的 env_wrapper 对所有任务 import wand + scikit-image,
  wand 依赖系统库 ImageMagick(`apt install libmagickwand-dev`);setup.sh 已处理,
  评测启动前也会探测并给出可操作的报错
- **不支持 `--init-seed` / `--init-states-root`**:任务变体自带扰动特定的
  init states,重采样会破坏扰动定义(启动即拒绝)
- 步数上限沿用各 base suite(220/280/300/520),图像预处理、replan、seed 等协议不变
- `summary.json.evaluation_protocol.official_result` 只有在四个完整 suite、10,030
  tasks、每 task 1 rollout 且零失败时才为 `true`;使用 `--task-ids`、少 suite 或
  改写 `--num-trials` 会明确标成 `NON-OFFICIAL DEVELOPMENT RUN`

历史文档中的 π0.5 84.7% 使用了 clean prompt,不属于官方协议,已撤销。完整一轮
8×4090 约 7-10 小时;小样本仍可用于迭代筛选,但不能作为最终 benchmark 成绩。

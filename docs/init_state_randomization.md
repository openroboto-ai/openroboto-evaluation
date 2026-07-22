# Init States 种子随机化:机制与验证报告

## 动机

官方 LIBERO / LIBERO-Pro 评测把每个任务的初始状态冻结在 `.pruned_init` 文件里
(每任务 50 个,LIBERO-Pro 的来自 HF 数据集 `zhouxueyang/LIBERO-Pro`),评测集
完全确定。子网场景下,miner 可以针对这 50 个固定初始布局刷分(例如用评测初始
状态生成训练数据),使评测分数失去对泛化能力的度量意义。

## 可随机化维度调研结论

对 LIBERO-PRO 源码的调研确认:**物体初始位置 + 旋转是唯一"无限"的随机化维度**。

| 维度 | 性质 |
|---|---|
| 初始位置 + 旋转 | **连续/无限**:每次 env reset 由 `np.random.uniform` 在 BDDL 区域范围内重采(`base_region_sampler.py`),官方只是把 50 次采样冻结成文件 |
| 物体替换 (object) | 有限:每类 1–6 个候选(`ood_object.yaml`,受资产池约束) |
| 空间关系交换 (swap) | 有限候选列表(`ood_spatial_relation.yaml`) |
| 语言改写 (lan) | 有限:每任务 3 条(`ood_language.yaml`) |
| 任务改写 (task) | 有限:每任务约 2 条(`ood_task.yaml`) |
| 环境替换 (env) | 代码硬编码为 1 个目标且不稳定,validator 未启用 |

有限维度已被 validator 的 16 个 LIBERO-Pro 扰动 suite 用尽;抗过拟合的增量
手段就是用 miner 训练时不可预知的 seed 重采初始位置/旋转。重采样与官方 init
**同分布**(同一 BDDL 采样范围),差异幅度与官方 50 个状态内部的自然差异同量级。

## 机制与用法

```bash
# 1. 生成自定义 seed 的 init states(client venv,LIBERO-Pro 16 suites)
PYTHONPATH=third_party/LIBERO-PRO \
LIBERO_CONFIG_PATH=~/.cache/libero_pro/config \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
third_party/openpi/examples/libero/.venv/bin/python \
    libero_eval/gen_init_states.py --seed <SEED> \
    --output-root ~/.cache/libero_pro_custom/init_files_seed<SEED>

# base LIBERO 套件则换包与配置:
#   PYTHONPATH=third_party/openpi/third_party/libero(不设 LIBERO_CONFIG_PATH)
#   --suites libero_spatial,libero_object,libero_goal,libero_10

# 2. 评测时指向自定义 init 目录
python libero_eval/run_eval.py ... --init-states-root ~/.cache/libero_pro_custom/init_files_seed<SEED>
```

- 每任务 seed 由 base seed 稳定派生(`crc32(suite/task)`),与生成顺序无关,
  可按 suite 并行生成、可单独重生成子集,结果逐字节可复现(同 seed 两次生成
  bitwise 一致)。
- 每个 suite 目录带 `manifest.json`(base_seed、每任务派生 seed 与 shape),
  可审计;评测的 `summary.json` 与每任务结果 JSON 记录 `init_states_root`。
- 生成很快:16 suites × 10 任务 × 50 状态,4 进程并行约 10 分钟。

## 验证实验(2026-07-12)

模型(均为 LIBERO 数据上微调的 PyTorch checkpoint,评测 seed=7、10 trials/task):
- **pi05** = `Fisher-Wang/pi05-libero-pytorch@ab107fbe`
- **pi0** = `Fisher-Wang/pi0-libero-pytorch@23d60043`

Benchmark:LIBERO-Pro 16 扰动 suite(160 任务,1600 episodes/run);
对照:base LIBERO 4 suite(400 episodes/run)。自定义 init seed=10007。

### 结果 1:重采样不引入评测偏差

| run | official | seeded(10007) | delta | p 值 |
|---|---|---|---|---|
| pi05 × LIBERO-Pro | 58.9% (943/1600) | 58.1% (930/1600) | +0.8% | 0.64(噪声内) |
| pi0 × LIBERO-Pro | 46.8% (748/1600) | 46.7% (747/1600) | +0.1% | 0.97(噪声内) |
| pi05 × base LIBERO | 96.8% (387/400) | 98.5% (394/400) | −1.7% | 0.10(噪声内) |

两个模型都未针对固定 init 刷分,分数在重采样后保持不变——符合"同分布重采样
对诚实模型无偏"的预期;base 对照说明 95%+ 的高分也不依赖特定 init 布局。

### 结果 2:模型区分度完整保留

- 排序不变:pi05 > pi0,分差 official +12.2% / seeded +11.4%。
- task 级成功率相关(official vs seeded,160 任务):
  pi05 Pearson 0.967 / Spearman 0.894;pi0 Pearson 0.978 / Spearman 0.940。
- task 级模型分差相关:Pearson 0.833 / Spearman 0.700。
- （base 对照的 task 级相关性低是天花板效应:40 任务几乎全部 90–100%,
  方差过小,相关系数失去意义。）

### 结论

换 seed 重采 init states **不损失评测的一致性与区分度**,同时使"背初始
状态"型过拟合失效:对诚实模型分数不变,而记忆固定布局的模型在重采样的
那一半评测上将回落到真实泛化水平。方案可用于生产。

实验数据:`eval_runs/exp_m{1,2}_pro_{official,seeded}`、
`eval_runs/exp_m1_base_{official,seeded}`;
对比脚本:`tools/compare_init_randomization.py`。

## 生产集成(已实现,worker 默认启用)

每个任务一半 trial 用官方 init、一半用 seed 重采样的 init(episode 级 50/50
混合)。**seed 直接取该任务队列条目自带的 `seed` 字段**(uint32,每个 miner
每轮一个),miner 可端到端复现验证——这也是选它的原因:

1. `seed = int.from_bytes(sha256(f"{block_hash}:{round_num}:{drand_random}")[-4:], "big")`
   按公开协议派生(prototype 仓库 `protocol/seed.py` 与
   `docs/SEED_GENERATION.md`,drand 可从 https://api.drand.sh 独立验证)。
   block_hash 来自 miner 自己的链上提交、drand 在入队时才抓取,所以 **miner
   提交权重之前无法预知自己的 seed**——不存在"先拿 seed 微调"的过拟合窗口;
   而入队之后 seed 即公开可查、可验证。
2. `task_seed = (seed * 1000003 + crc32(f"{suite}/{task_name}")) % 2**32`
   (`libero_eval/init_mix.py` `derive_task_seed`,共享测试向量位于
   `tests/test_init_seed.py`)。
3. `gen_init_states.py` 以 `np.random.seed(task_seed)` 重采样,逐字节可复现。
4. 混合规则:前 `min((num_trials+1)//2, 官方可用数)` 个 trial 用官方 init,
   其余用 seeded(`init_mix.py` `mix_counts`)。

同一轮不同 miner 的 seeded 一半 init states 不同;上文验证实验表明 seed 变化
对分数无偏、区分度保留,且官方一半对所有 miner 完全相同,可比性有锚点。

- **`benchmark_worker/worker.py`**:默认启用。`select_init_seed` 读取队列条目
  的 `seed` 字段(缺失/非法时告警并退回纯官方评测,不本地造随机数——私自选
  的 seed 无法向 miner 证明来源);评测命令自动附加 `--init-seed`。调试/复现
  旧行为用 `--no-init-randomization` 关闭。
- **`run_eval.py --init-seed <int>`**:自动(幂等、按 (seed, num_inits) 缓存)
  为所请求的 suites 生成重采样 init states(policy server 启动前,各 task 轮转
  分配到 GPU worker slots 并行生成,默认每卡 4 个生成进程),只生成 50/50 混合
  中 seeded 一半实际需要的状态(50 trials 生成 25 个;生成是前缀稳定的:同 seed
  下多生成只是往后追加)。生产 seed 每 miner 每轮不同,缓存
  基本不复用,单个目录仅数 MB,磁盘紧张时手动清理缓存根即可。
  与 `--init-states-root`(全替换,实验用)互斥。
- **`eval_task.py --init-states-mix`**:前一半 trial 用官方 init(布局与未混合
  run 的前几个 episode 完全一致,保持可对比性;奇数 trial 官方多一个),后一半
  用 seeded init;每个 episode 的 `init_source` 记录在结果 JSON。
- **seed 回传**:评分 payload 带 `init_seed` 字段(= 队列条目 seed),后端存入
  result 供查询确认;`summary.json` 与 worker state 亦有记录。任何人可按上述
  派生链用 `gen_init_states.py --seed` 复现全部评测 init states 以审计。

单元测试:`tests/test_init_seed.py`(trial 分配、derive_task_seed 共享测试
向量、select_init_seed 选取与回退、payload 公布字段)。

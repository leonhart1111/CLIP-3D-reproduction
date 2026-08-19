# CLIP-3D 论文复现与扩展

本项目复现 CLIP-3D 的“架构模拟 → 功耗/缓存物理参数 → 3D 布局 → 热分析 →
持续频率 → 回标 gem5”闭环，并保留所有关键配置、哈希和中间证据。

> 当前状态（2026-08-20）：工作流和新的固定语义工作量测量协议已经实现；热代理已
> 修正为论文的 \(L_c=\) die 半宽，并增加可复现的共同 HotSpot 位置梯度诊断。论文级
> 全量实验和最终数值结论尚未完成。仓库中的 exploratory 配置不能当作论文正式结果。

## 最重要的测量口径

新的正式入口是 `configs/experiments/r1_semantic_cache_sweep.json`。五个 workload
都在完整 work-unit 边界调用 `m5_work_begin(1, 0)` 和 `m5_work_end(1, 0)`：先同步、
再 reset 统计，完成固定数量的完整工作单元后再次同步、dump 并停止。

fixed-bin 与 CLIP-3D 的主比较量是：

```text
work_units_per_cycle = measured_work_units / completion_cycles
work_units_per_ns    = work_units_per_cycle * sustainable_frequency_GHz
```

`completion_cycles` 来自两个 marker 的全局 tick 差，不再使用“CPU0 先完成”或“最慢
核心达到指令阈值”作为工作完成边界。IPC 只保留为诊断量；只有两次 R2 的四核动态
指令向量逐项完全相同，才允许报告 same-trace IPC/BIPS 对比。

历史 `cpu0`/`all-cores` 指令窗口结果仍可审计，但属于 legacy 数据，不能与
`semantic-work` 结果拼接、补点或共同计算提升率。协议细节及真实 marker smoke 证据见
[docs/semantic_roi_protocol_zh.md](docs/semantic_roi_protocol_zh.md)。

## 已实现

| 能力 | 当前实现 |
|---|---|
| gem5 R1 | 5 workloads × 4 L1D × 5 L2 的可恢复扫描；支持计划模式、并行执行、超时和逐点状态 |
| 同步语义 ROI | FFT、CHOLESKY、STREAM、MATMUL、STENCIL 均有完整 work-unit marker；记录 marker、binary、配置和 protocol SHA-256 |
| 功耗与缓存物理量 | 严格 McPAT 1.3 流程；缓存面积、长宽比和访问时间以 McPAT 内嵌 CACTI-P 记录为权威，独立 CACTI 不再覆盖正式结果 |
| 3D 布局与热分析 | fixed-bin 与 CLIP-3D 布局、模块功耗守恒、HotSpot 稳态验证、热约束持续频率和布局可视化 |
| 热代理梯度诊断 | 对同一批合法 L2 位置只求解一次 HotSpot，并报告各代理变体的温差方向、排序趋势、选点 regret 与频率状态；exploratory 配置用 fixed-bin HotSpot 共同偏置避免频率项错误停在 thermal-headroom |
| gem5 R2 | 从最终布局生成缓存/仲裁/TSV/线延迟向量并回标 gem5；支持结果身份校验和安全复用 |
| 配对比较 | fixed-bin/CLIP-3D 配对 runner、断点续跑、完整 provenance 复验；语义协议以 `work_units_per_ns` 排名并执行 IPC same-trace gate |
| 瞬态研究分支 | 时间窗口 McPAT/HotSpot、five-state 和 POD-ROM 代码路径已经接入；均明确标记为 exploratory/non-formal |
| 完整性检查 | 对配置、工具 binary、输入、stats、marker、缓存记录、布局和 R2 向量进行哈希绑定；协议或证据混用时 fail closed |

## 尚未完成或尚未具备正式结论

- 尚未完成新的 semantic-work 100 点 R1 网格，以及对应 fixed-bin/CLIP-3D R2
  全量配对；旧 100/200 点结果不能替代它。
- STREAM 的正式 10M 数组开销很高，当前 `warmup=3, measure=16` 仍需先做成本和
  方差定标，再决定是否启动完整网格。
- 论文未公开的热堆叠、代理权重和线延迟权重尚未全部取得可推广的正式标定；带
  `exploratory`、`operational`、`rejected` 的配置均不是 paper-equivalent 配置。
- 热代理是启发式而非精确 HotSpot 替代品。五个最差点的共同位置扫描支持论文
  \(L_c=\) die 半宽的质心核作为当前 operational 默认，但不支持按 workload/L2 尺寸
  调整 \(L_c\)，也不支持将矩形面积求积提升为共享默认；详见
  [docs/thermal_proxy_gradient_validation_zh.md](docs/thermal_proxy_gradient_validation_zh.md)。
- 尚未形成论文各表格/图的端到端复现实验包，因此不能声称已经复现论文报告的收益。
- five-state/POD-ROM 虽有实现与测试，但仍缺完整真实 workload 验证；它们是研究扩展，
  不是论文原方法的已验证替代品。

## 快速开始

所有命令从项目根目录运行：

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh
./scripts/check_tools.sh
python3 -m unittest discover -s tests -v
```

工具源码和构建结果位于 `tools/src/` 与 `tools/build/`。下载和构建说明见
[docs/DOWNLOAD_TOOLS.md](docs/DOWNLOAD_TOOLS.md)。

## 主入口与运行命令

仓库不为每次实验保留一份一次性 `run_*.sh`。改变 workload、缓存点、并发数和输出
目录时，直接给以下主入口传参；实验输出写入 ignored 的 `runs/`。

### 1. 构建带语义 marker 的五个 benchmark

```bash
source scripts/env.sh
python3 scripts/build_semantic_benchmarks.py
```

该命令会应用受控 patch、编译五个 workload，并生成
`benchmarks/bin/semantic_manifest.json`。

### 2. 生成或执行 semantic R1 计划

只生成 100 点计划，不启动 gem5：

```bash
python3 scripts/run_r1_sweep.py \
  --experiment configs/experiments/r1_semantic_cache_sweep.json \
  --output-root runs/architecture_sweep/r1
```

执行或恢复扫描：

```bash
python3 scripts/run_r1_sweep.py \
  --experiment configs/experiments/r1_semantic_cache_sweep.json \
  --output-root runs/architecture_sweep/r1 \
  --jobs 1 --execute
```

同一命令重复运行会验证并跳过成功点；只有明确要替换成功证据时才使用 `--rerun`。

### 3. 单点 lifting 与 R2

先生成 fixed-bin 的物理/热结果和 R2 向量：

```bash
python3 -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/semantic/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/lifting_semantic/matmul_32kB_512kB/fixed-bin \
  --config configs/experiments/clip3d_proxy_anchor_paper_lc_diagnostic.json \
  --layout-method fixed-bin
```

将 `--layout-method` 改为 `clip3d`、输出到独立目录即可生成优化布局。确认物理/热结果后
加 `--run-r2` 才会启动耗时的第二次 gem5；不要让两种布局共用输出目录。

在启动 R2 前，可用共同 HotSpot 位置诊断检查该点的启发式热梯度。它只接受已生成的
`modules.json`，不会重新运行 gem5 或 McPAT：

```bash
python3 -m workflow.thermal.diagnose_proxy_gradient \
  --modules runs/lifting_semantic/matmul_32kB_512kB/fixed-bin/modules.json \
  --config configs/experiments/clip3d_proxy_anchor_paper_lc_diagnostic.json \
  --output-dir runs/proxy_gradient/matmul_32kB_512kB \
  --variant paper-center=center,0.5 \
  --variant paper-area=area-quadrature,0.5 \
  --grid-points 3 --workers 2
```

只应据此比较温差方向和选点 regret；报告性温度与性能仍以最终布局的 HotSpot/R2 为准。

### 4. 批量 lifting、配对 R2 与汇总

批量物理流程的主入口是：

```bash
python3 -m workflow.run_lifting_sweep \
  --r1-root runs/architecture_sweep/r1/semantic \
  --r1-experiment configs/experiments/r1_semantic_cache_sweep.json \
  --r1-profile semantic \
  --output-root <fixed-or-clip-root> \
  --config <experiment-config.json> \
  --layout-method <fixed-bin-or-clip3d> \
  --jobs 4
```

已有与该 semantic 网格及实验配置哈希一致的预声明 selection 后，使用：

```bash
python3 -m workflow.r2.run_paired_sweep \
  --r1-root <semantic-r1-root> \
  --fixed-root <fixed-root> --clip-root <clip-root> \
  --selection <semantic-selection.json> \
  --config <experiment-config.json> \
  --status-root <paired-status-root> --jobs 4

python3 -m workflow.analysis.summarize_paired_sweep \
  --fixed-root <fixed-root> --clip-root <clip-root> \
  --selection <semantic-selection.json> \
  --config <experiment-config.json> \
  --csv <paired-results.csv> --output <paired-summary.json>
```

现有 `balanced50_traffic_weighted.json` 绑定的是 legacy R1 配置，不可直接冒充新的
semantic selection。完整执行顺序见
[docs/balanced50_experiment_zh.md](docs/balanced50_experiment_zh.md)，但其中 legacy 路径
必须按语义协议说明替换后才可用于新结论。

## 目录结构

```text
CLIP/
├── benchmarks/semantic_roi/   # marker 公共代码与受控 benchmark patches
├── configs/                   # gem5、工具配置和实验身份
├── docs/                      # 协议、方法、执行和验证说明
├── scripts/                   # 环境、构建和 R1 主入口
├── tests/                     # 单元、合同和小型端到端测试
├── tools/                     # gem5/McPAT/CACTI/HotSpot 源码与构建结果
├── workflow/                  # lifting、布局、热、R2、瞬态和汇总实现
├── runs/                      # 本地实验输出，不提交
└── results/                   # 经审查的结果；原始 run 不应直接提交
```

稳态流水线细节见 [docs/clip3d_pipeline_zh.md](docs/clip3d_pipeline_zh.md)，旧正式执行文档见
[docs/formal_reproduction_zh.md](docs/formal_reproduction_zh.md)。若二者与语义 ROI 文档冲突，
以 [docs/semantic_roi_protocol_zh.md](docs/semantic_roi_protocol_zh.md) 的新测量口径为准。

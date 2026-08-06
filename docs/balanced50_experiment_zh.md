# Balanced-50 流量加权实验运行手册

本文是已完成 canonical R1 的 Balanced-50 后续实验手册。它只使用既有的
100 个 R1 `paper` 点；**R1 永不再次执行**。所有命令都应从项目根目录
运行（合并到主分支后为 `/home/zyjiang/Agenticflow/CLIP`），并先加载环境：

```bash
source scripts/env.sh

R1=runs/architecture_sweep/r1/paper
CFG=configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json
SELECT=configs/experiments/balanced50_traffic_weighted.json
ROOT=runs/operational_balanced50_traffic_weighted
FIXED=$ROOT/fixed_bin
CLIP3D=$ROOT/clip3d
STATUS=$ROOT/paired_r2_status
RESULTS=results/operational_balanced50_traffic_weighted
```

## 范围、分类与限制

这是 `operational-exploratory-traffic-weighted` 的非正式探索性证据：
`non_formal=true`、`paper_equivalent=false`、
`shared_parameter_accepted=false`。通信权重来自完成的 R1 共享 L2
`demandAccesses` 计数器；R2 仍只能接收一个共享 L2XBar 延迟，不能把该权重
解释为每核因果 IPC 敏感度。

本实验不运行瞬态热仿真，也不运行 strict-P1 全局界工具；不把结果作为论文等价、
formal，或共享参数接受证据。布局阶段只产生物理/热/延迟证据，不能把
`IPC1 × f_sus` 当成 R2 结果或最终配对报告中的代理值。

canonical R1 网格严格是 5 个 workload × 4 个 L1D × 5 个 L2，即 100 个点。
`stencil/l1d_32kB/l2_512kB.corrupt_duplicate_20260731T011500_CST` 是已知的
非 canonical 重复目录，审计应排除它而不是把它计入 100 点。

`$SELECT` 在读取结果之前固定 50 个架构点（每个 workload 10 点）；不可根据
结果替换或新增选择点。若 50 个已选 CLIP-3D 延迟向量中有 `K` 个与对应
fixed-bin 向量的完整 `gem5_overrides` 不同，则总共会有 `50+K` 次真实物理
R2 运行：每一对固定先运行一次，只有完整 overrides 和 provenance 都完全匹配时
才复用 fixed-bin R2 到 CLIP-3D。

## 先验证代码，再接触实验输出

以下检查不启动 gem5、McPAT、CACTI 或 HotSpot。必须在任何审计、布局或 R2
命令前成功完成：

```bash
source scripts/env.sh
python -m unittest discover -s tests -v
python -m compileall -q workflow scripts tests
git diff --check
```

预期是全部单元测试通过、编译命令退出码为零，以及 `git diff --check` 没有输出。

## 1. 仅刷新并审计 R1 计划（绝不执行 R1）

先建立审计目录，然后运行下列精确命令：

```bash
mkdir -p results/operational_balanced50_traffic_weighted/r1_audit
python -m workflow.analysis.refresh_r1_plan \
  --root runs/architecture_sweep/r1/paper \
  --experiment configs/experiments/r1_cache_sweep.json \
  --profile paper \
  --output results/operational_balanced50_traffic_weighted/r1_audit
python -m workflow.analysis.audit_r1 \
  --root runs/architecture_sweep/r1/paper \
  --experiment configs/experiments/r1_cache_sweep.json \
  --profile paper \
  --output results/operational_balanced50_traffic_weighted/r1_audit/audit.json \
  --expected-points 100 \
  --require-complete
```

`refresh_r1_plan` 只重建 `planned_jobs.json`；其内部调用刻意没有 `--execute`，并在
刷新前后哈希全部 canonical 的 `status.json`、`r1_metadata.json` 与 `stats.txt`。
它是 plan-only、hash-protected 操作，不是重新运行 R1 的授权。预期为 100 个
canonical valid success、一个排除的非 canonical 重复、100 个 planned job、一个
instruction scope，且刷新前后 canonical 哈希完全一致。

审计输出在 `$RESULTS/r1_audit/`：`canonical_r1_before.sha256.json`、
`canonical_r1_after.sha256.json`、`refresh_report.json` 与 `audit.json`。
R1 原始的逐点 `stdout.log`、`stderr.log`、`stats.txt`、`status.json` 保持在
`$R1/<workload>/l1d_<size>/l2_<size>/`，不可改写。

## 2. 生成或恢复 100 个 layout-only fixed-bin 点

初始并发建议为 4 个 worker：

```bash
python -m workflow.run_lifting_sweep \
  --r1-root "$R1" \
  --r1-experiment configs/experiments/r1_cache_sweep.json \
  --r1-profile paper \
  --output-root "$FIXED" \
  --config "$CFG" \
  --layout-method fixed-bin \
  --jobs 4
```

预期 `discovered=100`、没有失败点，且 `$FIXED/sweep_status.json` 的
`contains_r2=false`、每个 `pipeline_summary.json` 都没有 R2 值。逐点输出为
`$FIXED/<workload>/l1d_<size>/l2_<size>/`；其中包括布局/HotSpot 输入、
`r2_latency.json`、`performance.json` 和 `pipeline_summary.json`。

## 3. 生成或恢复 100 个 layout-only CLIP-3D 点

同样以 4 个 worker 开始：

```bash
python -m workflow.run_lifting_sweep \
  --r1-root "$R1" \
  --r1-experiment configs/experiments/r1_cache_sweep.json \
  --r1-profile paper \
  --output-root "$CLIP3D" \
  --config "$CFG" \
  --layout-method clip3d \
  --jobs 4
```

预期 `discovered=100`、没有失败点，并且 `$CLIP3D/sweep_status.json` 仍显示没有
R2 值。每点还必须有 `optimizer_report.json` 与 `layout_selection.json`，作为
优化器和最终布局选择证据；其余逐点输出路径与 fixed-bin 相同，只是根目录为
`$CLIP3D`。

## 4. 预检并只执行一个已选配对（smoke）

两套 100 点 layout-only 根目录完整后，先以单并发运行唯一的真实配对 smoke：

```bash
python -m workflow.r2.run_paired_sweep \
  --r1-root "$R1" --fixed-root "$FIXED" --clip-root "$CLIP3D" \
  --selection "$SELECT" --config "$CFG" --status-root "$STATUS" \
  --jobs 1 --limit 1
```

`--limit 1` 不是随机抽样；它总是选择 manifest 的第一个条目，并在
`$STATUS/status.json` 中记录 `limited_run=true`。成功只证明这个首项的 fixed 和
CLIP BIPS2 都有效（可能仅在完整 overrides/provenance 匹配时复用）；它不等同于
50 对完成，也不能产生最终汇总报告。

## 5. 恢复/执行全部预声明 50 对

确认 smoke 后，以 4 个 worker 执行或恢复完整选择集：

```bash
python -m workflow.r2.run_paired_sweep \
  --r1-root "$R1" --fixed-root "$FIXED" --clip-root "$CLIP3D" \
  --selection "$SELECT" --config "$CFG" --status-root "$STATUS" \
  --jobs 4
```

完成条件是 `$STATUS/status.json` 的 `complete=true`：50 个成功配对、0 个失败、
每个 pair 的 provenance 通过验证，并记录独立 CLIP R2 数 `K` 与
`physical_r2_runs=50+K`。总状态文件是 `$STATUS/status.json`；逐对状态文件是
`$STATUS/<workload>/l1d_<size>/l2_<size>/pair_status.json`。真正执行的本地 R2
结果位于相应 layout 点的 `gem5_r2/`，包括 `status.json`、`r2_result.json`、
`stats.txt`、`gem5.log`。复用的 CLIP 点改为保存 `r2_reuse.json`，并在
`pipeline_summary.json` 记录其已验证来源。

## 6. 生成严格的配对 CSV/JSON

仅在完整 50 对成功后运行：

```bash
mkdir -p "$RESULTS"
python -m workflow.analysis.summarize_paired_sweep \
  --fixed-root "$FIXED" --clip-root "$CLIP3D" \
  --selection "$SELECT" --config "$CFG" \
  --csv "$RESULTS/paired_results.csv" \
  --output "$RESULTS/paired_summary.json"
```

预期为 50 行、每个 workload 10 行、没有代理值；每个复用行都有有效 reuse 证据，
`paired_summary.json` 中为 `complete=true`。最终交付文件是
`$RESULTS/paired_results.csv` 与 `$RESULTS/paired_summary.json`。

## 恢复、重跑与故障处理

对同一根目录重复执行以上相同命令会验证并跳过已完成的 layout 点或完整配对，
因此是正常的恢复方式。不要为了恢复而加 `--rerun`。`--rerun` 是显式 opt-in：它
会请求替换已有结果，只应在确认需要重新取得该证据时使用；对于 layout sweep
不要把 `--rerun-r2`、`--run-r2` 或 `--reuse-r2-root` 加入第 2/3 步，因为它们必须
保持 layout-only。任何失败都先检查对应 `sweep_status.json`、pair status、
`gem5_r2/gem5.log` 与逐点 JSON，再在不改变 `$R1` 的前提下恢复。

运行产生的 `$ROOT/` 和 `$RESULTS/` 是实验输出，不随本手册提交；除非仓库跟踪策略
明确要求小型最终摘要，否则不要 `git add runs/` 或 `git add results/`。

# 可配置时间窗瞬态热仿真旁路

该功能是论文稳态复现之外的可选验证。默认关闭，不改变原来的：

```text
R1 → McPAT → 布局 → 稳态HotSpot → f_sus → R2
```

开启后，稳态流水线完成后再执行独立旁路：

```text
专用瞬态R1（按 --sample-ms 设置的周期累计统计）
→ 相邻快照相减得到窗口统计
→ 每窗口McPAT
→ 按最终布局栅格化多行功耗
→ 一次HotSpot瞬态求解
```

## 直接从主入口运行

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh

R1=runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB
OUT=runs/pilot/matmul_32k_512k_fixed_transient
CFG=configs/experiments/clip3d_constrained_5p0.json

python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --layout-method fixed-bin \
  --transient true \
  --transient-sample-ms 2 \
  --transient-initial-temperature steady
```

若未给出`--transient-r1-dir`，程序会在：

```text
$OUT/transient/r1/
```

单独重跑一个带周期统计的R1。它使用新包装器
`configs/gem5/clip_r1_transient.py`，不会修改或覆盖正式R1目录。

## 分开运行和复用瞬态R1

先只生成瞬态R1：

```bash
python -m workflow.transient.run_transient_r1 \
  --source-r1-dir "$R1" \
  --output-dir runs/transient_r1/matmul_32k_512k \
  --sample-ms 2
```

再由主入口复用：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --layout-method fixed-bin \
  --transient true \
  --transient-sample-ms 2 \
  --transient-r1-dir runs/transient_r1/matmul_32k_512k
```

也可以在已有稳态输出上直接运行旁路：

```bash
python -m workflow.transient.run_transient_pipeline \
  --source-r1-dir "$R1" \
  --steady-output-dir "$OUT" \
  --output-dir "$OUT/transient" \
  --config "$CFG" \
  --sample-ms 2 \
  --transient-r1-dir runs/transient_r1/matmul_32k_512k
```

## 主要输出

```text
$OUT/transient/
├── r1/                              # 独立周期统计R1（如未外部提供）
├── windows/gem5/
│   ├── windows_manifest.json
│   └── window_XXXX/stats.txt        # 按所选采样间隔生成增量统计
├── windows/mcpat/
│   ├── power_windows.json
│   └── window_XXXX/                 # XML、McPAT输出和模块功耗
├── hotspot/
│   ├── power_transient.ptrace
│   ├── transient.ttrace
│   ├── transient_summary.csv
│   └── transient_result.json
└── transient_pipeline_summary.json
```

默认使用原稳态结果的`steady.txt`作为瞬态初始温度，避免把短ROI误解为
从25°C冷启动。`--transient-initial-temperature ambient`可额外生成冷启动结果。

当前旁路不做温度相关漏电反馈，也不执行动态DVFS；它用于验证平均稳态模型
是否掩盖时间变化的峰值温度。

## MATMUL 双布局 operational 验证

下面命令复现实验室当前的 MATMUL 32kB L1D / 512kB L2、2 ms 瞬态比较。
这是 `operational` 验证，明确是 `non-formal`：它不构成论文等价复现、严格
复现或正式参数证据。规范 R1 和已经完成的两个稳态 pilot 都是只读输入；不要
覆盖它们，也不要因为这个瞬态分支重跑 R2。

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh

python -m workflow.transient.run_dual_layout_validation \
  --source-r1-dir runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB \
  --fixed-steady-dir runs/operational_raw_power_p1/pilot_direct_20260731/fixed-bin \
  --clip3d-steady-dir runs/operational_raw_power_p1/pilot_direct_20260731/clip3d \
  --output-root runs/transient_validation/matmul_32kB_512kB_2ms_20260803 \
  --config configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json \
  --sample-ms 2
```

输出根目录中的 `shared_r1/` 是为本次比较新建的周期统计 R1；fixed-bin 与
CLIP-3D 共享该 R1 和同一份逐窗口功耗轨迹。因此它与规范 R1
`runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB` 不同，后者及两个
已完成的 pilot 仍保持只读。

先读取状态和总摘要；`status.json` 的 `state` 必须为 `success`，失败时其中会
记录异常类型和消息。`experiment_summary.json` 记录共享 R1、两个布局分支、
实际时长及比较产物路径。

```bash
python -m json.tool \
  runs/transient_validation/matmul_32kB_512kB_2ms_20260803/status.json
python -m json.tool \
  runs/transient_validation/matmul_32kB_512kB_2ms_20260803/experiment_summary.json
python -m json.tool \
  runs/transient_validation/matmul_32kB_512kB_2ms_20260803/comparison/transient_comparison.json
```

解读时以 `comparison/transient_comparison.json` 的一致性检查为准：两种布局必须
引用同一份共享 R1、相同采样间隔、窗口数、真实 ROI 时长和功耗轨迹身份。再阅读
其中 fixed-bin 与 CLIP-3D 的稳态峰值、轨迹内峰值、最终峰值及
`clip_minus_fixed` 温差，并同时检查功耗峰值到温度峰值的时间滞后、最后窗口填充
时长和模型限制。该差异只说明此 operational、non-formal 条件下的瞬态热响应，
不能外推为正式论文结论。

## 采样间隔与历史比较

采样间隔由 `--sample-ms`（独立旁路）或 `--transient-sample-ms`（主入口）设置，默认值
仍为 10 ms。此前的比较使用 10 ms；上面的当前实验使用 2 ms，以获得更高的时间分辨率。
此前比较的结果根目录为
`runs/transient_validation/matmul_32kB_512kB_10ms_20260803`。历史设计和计划文档
保留其原始 10 ms 记录，作为当时实验的溯源，不应改写。

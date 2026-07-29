# 10 ms瞬态热仿真旁路

该功能是论文稳态复现之外的可选验证。默认关闭，不改变原来的：

```text
R1 → McPAT → 布局 → 稳态HotSpot → f_sus → R2
```

开启后，稳态流水线完成后再执行独立旁路：

```text
专用瞬态R1（10 ms累计统计）
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
  --transient-sample-ms 10 \
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
  --sample-ms 10
```

再由主入口复用：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --layout-method fixed-bin \
  --transient true \
  --transient-sample-ms 10 \
  --transient-r1-dir runs/transient_r1/matmul_32k_512k
```

也可以在已有稳态输出上直接运行旁路：

```bash
python -m workflow.transient.run_transient_pipeline \
  --source-r1-dir "$R1" \
  --steady-output-dir "$OUT" \
  --output-dir "$OUT/transient" \
  --config "$CFG" \
  --sample-ms 10 \
  --transient-r1-dir runs/transient_r1/matmul_32k_512k
```

## 主要输出

```text
$OUT/transient/
├── r1/                              # 独立周期统计R1（如未外部提供）
├── windows/gem5/
│   ├── windows_manifest.json
│   └── window_XXXX/stats.txt        # 每10 ms增量统计
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

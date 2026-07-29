# CLIP-3D 正式复现执行顺序

本文只列正式实验命令。所有命令都从项目根目录执行：

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh
```

`scripts/env.sh` 会优先使用项目内 `.venv`；当前固定版本为 NumPy 2.4.6、SciPy 1.15.3，CLIP-3D 布局器因此使用论文指定的 L-BFGS-B。

## 0. 等正式 R1 完成并审计

查看当前状态但不要求完成：

```bash
python -m workflow.analysis.audit_r1 \
  --root runs/architecture_sweep/r1/paper \
  --output results/checkpoints/r1_audit_latest.json
```

在进入下游前执行严格检查：

```bash
python -m workflow.analysis.audit_r1 \
  --root runs/architecture_sweep/r1/paper \
  --output results/checkpoints/r1_audit_final.json \
  --expected-points 100 \
  --require-complete
```

当前正在运行的 `paper` 扫描使用统一的 `cpu0` 指令停止锚点。不要在同一结果根目录混入 `paper_all_cores`。如果决定采用论文文字的“每核心均达到目标”口径，必须从头运行独立的 `paper_all_cores` 扫描；不能只重跑其中一部分。

## 1. 基线冷却：100点固定布局及唯一的正式R2

```bash
python -m workflow.run_lifting_sweep \
  --r1-root runs/architecture_sweep/r1/paper \
  --output-root runs/architecture_sweep/lifting_baseline_3p5 \
  --config configs/experiments/clip3d_pipeline.json \
  --layout-method fixed-bin \
  --run-r2 \
  --jobs 4
```

这一步执行 `Rconv=3.5 K/W` 的100个 HotSpot 点和100次正式 R2。每点的 R2 有独立 `status.json`，中断后重新执行同一命令会续跑。

## 2. 受限冷却：复用相同延迟的100个R2

```bash
python -m workflow.run_lifting_sweep \
  --r1-root runs/architecture_sweep/r1/paper \
  --output-root runs/architecture_sweep/lifting_constrained_5p0 \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method fixed-bin \
  --reuse-r2-root runs/architecture_sweep/lifting_baseline_3p5 \
  --jobs 4
```

固定布局只改变冷却热阻，不改变 CACTI、TSV 或线延迟。因此程序先逐点比较完整 `gem5_overrides`，完全相同才允许复用 IPC2；不匹配会拒绝继续。

严格生成两种冷却条件的表 IV 诊断：

```bash
python -m workflow.analysis.summarize_sweep \
  --root runs/architecture_sweep/lifting_baseline_3p5 \
  --csv results/tables/table4_baseline_points.csv \
  --output results/tables/table4_baseline.json \
  --expected-points 100

python -m workflow.analysis.summarize_sweep \
  --root runs/architecture_sweep/lifting_constrained_5p0 \
  --csv results/tables/table4_constrained_points.csv \
  --output results/tables/table4_constrained.json \
  --expected-points 100
```

正式汇总默认不允许用 `IPC1×f_sus` 冒充 BIPS2；任何点缺少 R2 都会报错。

## 3. 表III：闭式频率验证

论文的七个锚点已列在：

```text
configs/experiments/table3_anchors.example.json
```

确认正式点路径后运行：

```bash
python -m workflow.thermal.run_anchor_validation \
  --manifest configs/experiments/table3_anchors.example.json \
  --output results/tables/table3_frequency_validation.json
```

程序为每个锚点在 `0.5/1.0/2.0 GHz` 下按照论文均匀 γ 规则缩放功耗并真实调用 HotSpot；发生降频的锚点还会在求得的 `f_sus` 再验证一次安全温度。

## 4. 受限冷却下的80点四方法布局实验

固定布局结果已经包含在第2步。其余三种方法只运行论文指定的四个工作负载：

```bash
python -m workflow.run_lifting_sweep \
  --r1-root runs/architecture_sweep/r1/paper \
  --output-root runs/layout_sweep/cool3d_standard_5p0 \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method cool3d-standard \
  --workloads fft matmul stencil stream \
  --run-r2 --jobs 4

python -m workflow.run_lifting_sweep \
  --r1-root runs/architecture_sweep/r1/paper \
  --output-root runs/layout_sweep/sa_lambda_5p0 \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method sa-lambda \
  --workloads fft matmul stencil stream \
  --run-r2 --jobs 4

python -m workflow.run_lifting_sweep \
  --r1-root runs/architecture_sweep/r1/paper \
  --output-root runs/layout_sweep/clip3d_5p0 \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method clip3d \
  --workloads fft matmul stencil stream \
  --run-r2 --jobs 4
```

`cool3d-standard` 和 `sa-lambda` 各自对三个候选执行真实 HotSpot。由于论文没有公开这两个对照方法的源码和数值参数，本项目的有限网格与定种子 SA 是显式的独立复现，所有候选和搜索参数均写入 `layout_search/search_report.json`。

`clip3d` 对解析优化器提出的候选和 fixed-bin 各执行一次真实 HotSpot，再按真实热约束 BIPS、离散平均线延迟和 Tmax 依次选择。代理候选更差时会回退 fixed-bin；`layout_selection.json` 会保留两次实测和回退原因。该保护是防止未公开代理系数造成假收益的复现措施，不应描述成论文原算法。由于配置现在包含修正后的 CACTI/Table-II 几何与 `8.72` 热阻校准，旧配置生成的下游结果会被续跑检查判为过期。

生成表 V、VI、VII 的机器可读结果：

```bash
python -m workflow.analysis.summarize_layout_study \
  --fixed-root runs/architecture_sweep/lifting_constrained_5p0 \
  --cool3d-root runs/layout_sweep/cool3d_standard_5p0 \
  --sa-root runs/layout_sweep/sa_lambda_5p0 \
  --clip3d-root runs/layout_sweep/clip3d_5p0 \
  --csv results/tables/layout_study_points.csv \
  --output results/tables/tables5_6_7.json \
  --expected-points 80
```

## 5. 最终验收

必须同时满足：

- R1审计为 `100/100` 且只有一种指令窗口口径；
- 两个固定布局扫描各100点，且所有点都有真实或经向量一致性验证复用的IPC2；
- 三个非固定布局扫描各80点；
- 每点的 HotSpot 热阻、布局方法、TSV、线延迟和求解器记录完整；
- 每个 CLIP-3D 点都有 fixed/optimized 两次 HotSpot 验证以及明确的最终策略；
- 表III锚点有21次基础频率运行及必要的 `f_sus` 验证；
- 表IV至表VII不含 smoke 点或代理 BIPS2。

正式 gem5 R2 与 R1 使用同一指令窗口，耗时会远大于 McPAT、CACTI 和 HotSpot。建议先用 `--jobs 1` 验证一个正式成功点，再根据节点资源提高并行度。

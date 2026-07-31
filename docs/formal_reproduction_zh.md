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

## 6. Raw-power P1 的非正式运行配置

`configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json` 只用于运行实验，不是严格/正式复现结果。它保留严格 P1 的原始报告 `results/parameter_studies/raw_power_strict_20260730/proxy_train_16/calibration_report.json` 作为不可修改的证据；该严格报告仍因原有 `0.8` 空间 Spearman 门槛而被拒绝。

运行前用独立评估器生成非正式结论，评估器不会改写配置或源报告：

```bash
python -m workflow.analysis.evaluate_operational_proxy \
  --proxy-report results/parameter_studies/raw_power_strict_20260730/proxy_train_16/calibration_report.json \
  --config configs/experiments/clip3d_constrained_5p0_raw_power_p1_operational.json \
  --output results/parameter_studies/raw_power_strict_20260730/operational_proxy_report.json
```

该配置固定使用报告中的 `alpha=1.5643788695171585`、`beta=0.0` 和 `cross_tier_weight=0.995`。接受条件是内部验证及独立 STREAM 目标验证的 RMSE 和中心化 RMSE 都优于默认值，两个空间 Spearman 均不低于 `0.5`，交层权重严格位于 `(0, 1)`，并且 beta 状态为 `fixed_unidentifiable_under_p1`；配置参数必须与源报告在 `1e-12` 内一致。全部 leave-one-workload-out 排名仅作为诊断，最低值 `0.28571428571428575` 不参与该运行许可门槛。

输出会明确标记 `mode=operational` 和 `non_formal=true`，且动作文本为 `operational use permitted; non-formal and not promotable`。`formal_validation.promotion` 明确禁止提升该配置，`promote_validated_config` 也会在读取任何报告前拒绝 `strict_p1` 不为 true 的候选。`lambda_wire=0.0` 仍是待匹配 R2 延迟研究校准前的非正式零值，不能作为已校准线延迟参数引用。

### 6.1 受保护的 operational pilot 和完整运行

只能使用 `scripts/run_operational_raw_power_p1.sh` 启动此非正式配置。启动器先运行独立评估器并要求其建议被接受，再记录配置和不可修改源报告的 SHA-256；它不启动 R1，也不执行任何正式提升。`pilot` 只使用 MATMUL 的 `l1d_32kB/l2_512kB` R1 点，顺序运行一次 fixed-bin 和一次 clip3d，并且两次都运行 R2：

```bash
tmux new-session -d -s clip_operational_pilot "cd /home/zyjiang/Agenticflow/CLIP && flock -n /tmp/clip-operational-pilot.lock bash scripts/run_operational_raw_power_p1.sh pilot"
tmux attach -t clip_operational_pilot
```

默认输出根目录是新的 `runs/operational_raw_power_p1/pilot`；若该目录已经存在，启动器会拒绝执行，避免混合或覆盖结果。可将第二个参数替换为一个不存在的自定义输出根目录。

完整运行在启动两个 100 点 lifting sweep 之前，必须确认 R1 根目录恰有 100 个 `status.json`，并且每一个的状态都是 `success`。因此在 100 个 R1 点仍有未完成、失败或缺失时，它会拒绝运行：

```bash
tmux new-session -d -s clip_operational_full "cd /home/zyjiang/Agenticflow/CLIP && flock -n /tmp/clip-operational-full.lock bash scripts/run_operational_raw_power_p1.sh full"
tmux attach -t clip_operational_full
```

完整模式默认写入新的 `runs/operational_raw_power_p1/full/{fixed-bin,clip3d}`，两个 sweep 都固定使用 `--jobs 1 --run-r2`。所有这些输出都必须保持 operational/non-formal 标签；尤其 `lambda_wire=0.0` 仍是等待匹配 R2 证据的未校准值。

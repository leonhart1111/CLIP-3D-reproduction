# 瞬态热 ROM（探索性）操作指南

`transient-rom` 是一个可选的、经真实 HotSpot 最终复核的瞬态热优化流程。它不会替换默认的稳态流程：未指定 `--thermal-mode transient-rom` 时，仍执行原有稳态 Eq. (13) 流程并在稳态汇总中报告 `bips2`。

本流程固定标记为 `non_formal=true`、`paper_equivalent=false`、`thermal_mode="transient-rom"`。它是受控的 raw-power 方法扩展，不能作为论文稳态结果或正式复现实验的证据。

## 前置条件

在 CLIP 根目录准备并检查工具链：gem5（R1/R2）、McPAT、CACTI、HotSpot、Python 3、NumPy；单 tier 的 `[1]` 合法域还需要 SciPy 的 Delaunay 支持。

```bash
cd /home/zyjiang/Agenticflow/CLIP
source ./scripts/env.sh
./scripts/check_tools.sh
```

输入 R1 必须是完整且可审计的 canonical R1。ROM 模式会先在 `OUTPUT/steady_preflight/` 生成或复用固定 bin、**未运行 R2** 的稳态预检；预检的模块、CACTI 产物、热堆栈、冷却、网格和 raw-power provenance 必须与当前配置一致。不要把 `--transient true` 与 ROM 模式合用。

使用随仓库提供的探索性配置：

```bash
CONFIG=configs/experiments/clip3d_transient_rom_exploratory.json
R1=runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB
OUT=runs/transient_rom/matmul_32kB_512kB
```

## 首次校准与后续复用

首次运行须创建并通过校准包。省略 `--transient-rom-r1-dir` 时，流程会在 `OUT/transient_rom/r1/` 创建 2 ms 周期统计 R1；运行 `--run-r2` 才会得到可报告的最终 `bips2_trans`。

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" --output-dir "$OUT" --config "$CONFIG" \
  --thermal-mode transient-rom --transient-rom-calibrate --run-r2
```

已接受的包可在相同科学 identity 下复用。包可位于本次输出默认位置，或通过只读的显式路径指定；复用仍会对选出的布局执行最终真实 HotSpot 验证，并在指定 `--run-r2` 时重新测量该布局的 R2。

```bash
PACKAGE=/absolute/path/to/accepted-rom-package
PERIODIC_R1=/absolute/path/to/periodic-r1
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" --output-dir "$OUT-reuse" --config "$CONFIG" \
  --thermal-mode transient-rom --transient-rom-r1-dir "$PERIODIC_R1" \
  --transient-rom-package-dir "$PACKAGE" --run-r2
```

接受包会绑定 canonical R1 元数据、窗口功耗、模块与布局几何、配置、HotSpot 二进制、网格、热堆栈、冷却、允许的 L2 tier，以及完整校准设计（8 个锚点、2 个留出点、插值域和 Delaunay simplices）的哈希/identity。复用只读取包内 `anchors.json`，不会用当前代码重新生成设计；`pod_model.npz` 中的 B_L2 anchor ID、拟合 case 顺序、训练输入/温度哈希、转换阈值和设计哈希必须与 `fit_report.json` 完全一致。复用还会重新读取 `calibration_cases.json` 与 `validation_report.json`，校验 8+2 case 的点/布局映射、包内 artifact 哈希、全部留出 gate 和误差阈值，并保留历史留出 RMSE/峰温误差。case artifact 路径只允许相对于包根目录，源 `modules.json` 的原始字节也会复制进包，因此完整包可整体移动后复用；路径逃逸、symlink、证据缺失、清单缺失或清单哈希陈旧都会拒绝复用。不得只复制或修改接受标记。

## 固定的 8+2 调用及产物

`8+2` 不是可调的采样预算：8 个训练锚点 HotSpot 作业用 PRBS 功耗窗口拟合 POD 连续状态空间模型；2 个独立留出点 HotSpot 作业只用于验收。它们合计固定为 10 次校准调用，优化器内部的 HotSpot 调用必须为 0，最终候选的 HotSpot 频率/PSS 搜索另行计数。

典型输出树如下（实际文件还带有哈希和非正式分类）：

```text
OUT/
├── steady_preflight/                 # 固定 bin、R2-disabled 预检
└── transient_rom/
    ├── r1/                            # 可选生成的周期统计 R1
    ├── windows/mcpat/power_windows.json
    ├── rom_package/
    │   ├── modules.json
    │   ├── calibration_manifest.json
    │   ├── anchors.json
    │   ├── training/<anchor-id>/
    │   ├── holdout/<holdout-id>/
    │   ├── pod_model.npz
    │   ├── fit_report.json
    │   ├── validation_report.json
    │   └── rom_acceptance.json
    ├── optimization/{optimization_report.json,proposed_layout.json}
    ├── final_hotspot_validation/transient_sustainable_frequency.json
    ├── r2_latency.json
    ├── gem5_r2/r2_result.json          # 仅 --run-r2
    └── transient_rom_summary.json
```

合法 L2 tier 域只能来自配置的 `layout_optimizer.allowed_l2_tiers`。`[1]` 是单 tier 域（8 个训练点加 2 个留出点均在 tier 1，使用 Delaunay 域）；`[0, 1]` 是双 tier 域（每 tier 4 个训练锚点，留出点覆盖两个 tier，使用双线性矩形域）。两种情形都不允许把候选移动到域外或擅自扩展 tier。

## 质量门与失败语义

默认参数为 2 ms 采样、64 个 PRBS 窗口、20 个周期重复、PSS 末周期容差 `0.01 C`、频率细化容差 `0.01 GHz`。离散到连续转换还有三个显式数值门：增广离散矩阵条件数不超过 `1e12`、SciPy `logm` error estimate 不超过 `1e-8`、独立计算的 `||expm(logm(M))-M||_1/||M||_1` 不超过 `1e-8`；任何一个超限都会中止校准，而不是只写入诊断。两个留出点都必须通过以下门：几何/功耗/频率/HotSpot trace/温度网格 identity 相同，ROM 和 HotSpot 均达到 PSS，整张最终周期网格 RMSE 不超过 `0.75 C`，峰温绝对误差不超过 `1.0 C`，且安全/不安全分类一致。PSS 峰值包含最终周期的初始状态，并比较整个网格而非单一热点。

任一训练或留出作业失败、哈希不符、PSS 不收敛、RMSE/峰温阈值越界、分类不一致，都会使 `rom_acceptance.json` 不被接受并阻止优化。最终真实 HotSpot 在任何频率发生执行/I/O 错误、返回畸形 trace/搜索证据、或 PSS 不收敛时，汇总状态为 `rom_final_validation_failed`，`final_validation_failure.category` 分别记录 `tool_error`、`validation_contract_error` 或 `pss_nonconvergence`；调用前即可发现的缺失输入和非空输出目录则直接作为前置条件错误拒绝。失败时 ROM 预测仍保留，但不会启动 R2，`f_sus_trans_hotspot_ghz` 和 `bips2_trans` 均为 `null`。若所有真实 HotSpot 点均已收敛、但最低频率仍不安全，则状态为 `thermally_infeasible`、分类为 `true_thermal_infeasible`，明确区别于工具失败。不要以稳态回退或 ROM 预测替代这些结果。

## 汇总字段：预测与验证必须分开

阅读 `transient_rom_summary.json` 时，应明确区分：

- `f_sus_trans_rom_pred_ghz` 与 `bips1_trans_rom_pred`：ROM 在优化内给出的预测，仅用于选择候选和诊断，不是最终 HotSpot 验证值。
- `f_sus_trans_hotspot_ghz`：最终候选经真实 HotSpot 周期稳态验证后的频率。
- `ipc2_trans` 与 `bips2_trans`：当前候选布局的真实 R2 IPC，以及仅当真实 HotSpot 验证和 R2 都成功时计算的验证后瞬态 BIPS。
- `training_hotspot_calls`、`holdout_hotspot_calls`、`calibration_hotspot_calls`：包内历史校准证据，复用时仍为 `8`、`2`、`10`；对应的 `*_this_invocation` 字段在纯复用调用中均为 `0`。
- `bips2`：只属于独立稳态汇总；ROM 汇总中故意不存在这个含糊字段，不能把稳态 `bips2` 当作 `bips2_trans`。

因此，报告应同时保留预测、留出误差、最终 HotSpot 结果、R2 结果、调用次数和所有 identity。即便所有门通过，结果仍是 non-formal、paper-inequivalent 的探索性扩展，不能提升为正式/论文等价结论。

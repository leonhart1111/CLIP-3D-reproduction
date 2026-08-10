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

若要验证与当前稳态 5 点实验一致的“非零线长权重 + 通信权重 + nearest 整数周期分区”流程，应使用下面这个独立配置，而不是上面的 λ=0 隔离配置：

```text
configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json
```

它固定使用 `lambda_wire=0.0020119160767721133`、`wire_objective=discrete-partition`、`wire_aggregation=traffic-weighted`、`wire_rounding=nearest`、41×41 网格、固定布局基线和 `allowed_l2_tiers=[1]`。该 λ 来自仓库内可追溯但**未通过正式验收**的 FFT 拟合报告：R²、单调性和跨工作负载迁移门并未全部通过。因此该配置只能证明瞬态 ROM、整数分区和双分支实测流程可运行，不能据此宣称参数已被论文或跨负载实验验证。

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

## 复用现有 R1 的非零 λ 配对实验

下面的命令不会重跑 canonical R1，也不会重跑已经完成的 2 ms 周期统计 R1。它会准备一次共享功耗窗口，执行固定的 8 个训练瞬态、2 个留出点平均功耗稳态初始化和 2 个留出点瞬态验证，使用 ROM 搜索 L2 布局，然后分别用真实 HotSpot 和各自的 gem5 R2 验证 fixed-bin 与 CLIP-3D。必须使用新的空输出目录；不要覆盖历史证据。

```bash
cd /home/zyjiang/Agenticflow/CLIP/.worktrees/transient-rom
source scripts/env.sh

R1=/home/zyjiang/Agenticflow/CLIP/runs/architecture_sweep/r1/paper/matmul/l1d_32kB/l2_512kB
PERIODIC_R1=/home/zyjiang/Agenticflow/CLIP/runs/transient_validation/matmul_32kB_512kB_lambda0020119_2ms_precision6_20260806_030743/transient/shared_r1
CFG=configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json
OUT=runs/transient_rom/matmul_32kB_512kB_lambda0020119_discrete_$(date +%Y%m%d_%H%M%S)

python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$OUT" \
  --config "$CFG" \
  --thermal-mode transient-rom \
  --transient-rom-r1-dir "$PERIODIC_R1" \
  --transient-rom-calibrate \
  --run-r2
```

公共入口会在复用稳态预检之前校验离散搜索控制；偶数网格、未包含 fixed-bin、非法 tier 或不受支持的通信聚合会直接拒绝。fixed-bin 与 CLIP 两个优化候选各自输出的 `r2_wire_cycles` 都必须与本分支 `r2_latency.json` 的 `components_cycles.layout_wire` 及 `layout_delays.traffic_weighted_wire_cycles` 完全相等，否则两次 R2 都不会启动，并在不一致的分支留下 `integer_cycle_identity` 失败记录。

接受包会绑定 canonical/周期 R1 各自的 `status.json`、`r1_metadata.json`、`stats.txt` 六份原始输入哈希，以及窗口功耗、模块与布局几何、配置、HotSpot 二进制、网格、热堆栈、冷却、允许的 L2 tier 和完整校准设计（8 个锚点、2 个留出点、插值域和 Delaunay simplices）的哈希/identity。复用只读取包内 `anchors.json`，不会用当前代码重新生成设计；`pod_model.npz` 中的 B_L2 anchor ID、拟合 case 顺序、训练输入/温度哈希、转换阈值和设计哈希必须与 `fit_report.json` 完全一致。复用还会重新读取 `calibration_cases.json` 与 `validation_report.json`，校验训练/留出 case 的点与布局映射、两份留出稳态初值、包内 artifact 哈希、全部留出 gate 和误差阈值，并保留历史留出 RMSE/峰温误差。case artifact 路径只允许相对于包根目录，源 `modules.json` 的原始字节也会复制进包，因此完整包可整体移动后复用；路径逃逸、symlink、证据缺失、清单缺失或清单哈希陈旧都会拒绝复用。不得只复制或修改接受标记。旧的 10-call 环境温度起步包不满足新契约，必须重新校准。

## 固定的 8+2+2 调用及产物

校准预算固定为 12 次 HotSpot 调用：8 个训练锚点使用环境温度起步的 PRBS 瞬态轨迹拟合 POD 连续状态空间模型；2 个独立留出点各先调用一次稳态 HotSpot，根据该点同一布局、同一频率缩放功耗轨迹、同一堆栈和冷却条件求平均功耗稳态初值，再各调用一次带 `-init_file` 的周期瞬态 HotSpot。优化器内部的 HotSpot 调用仍必须为 0。

最终候选的每个频率点同样需要两次调用：一次匹配的平均功耗稳态初始化和一次周期瞬态验证。这样消除了从环境温度缓慢升温造成的伪 PSS 不收敛，但没有放宽 `0.01 C` 的全网格周期末差值门槛。

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
    │   ├── holdout_<holdout-id>/
    │   │   ├── steady_initialization.json
    │   │   ├── initialization.steady.txt
    │   │   ├── initialization.grid.steady.txt
    │   │   └── transient.ttrace
    │   ├── pod_model.npz
    │   ├── fit_report.json
    │   ├── validation_report.json
    │   └── rom_acceptance.json
    ├── optimization/{optimization_report.json,proposed_layout.json}
    ├── final_hotspot_validation/        # λ=0 continuous 旧单分支模式
    │   └── transient_sustainable_frequency.json
    ├── final_validation/                # discrete-partition 新配对模式
    │   ├── fixed_bin/
    │   │   ├── frequency_*/{steady_initialization.json,initialization.steady.txt,transient.ttrace}
    │   │   ├── transient_sustainable_frequency.json
    │   │   ├── r2_latency.json
    │   │   ├── gem5_r2/r2_result.json   # 仅 --run-r2
    │   │   └── branch_summary.json
    │   ├── clip3d/
    │   │   ├── frequency_*/{steady_initialization.json,initialization.steady.txt,transient.ttrace}
    │   │   ├── transient_sustainable_frequency.json
    │   │   ├── r2_latency.json
    │   │   ├── gem5_r2/r2_result.json   # 仅 --run-r2
    │   │   └── branch_summary.json
    │   ├── paired_comparison.json       # 仅两边 HotSpot+R2 均完整时生成
    │   └── paired_comparison.csv
    └── transient_rom_summary.json
```

合法 L2 tier 域只能来自配置的 `layout_optimizer.allowed_l2_tiers`。`[1]` 是单 tier 域（8 个训练点加 2 个留出点均在 tier 1，使用 Delaunay 域）；`[0, 1]` 是双 tier 域（每 tier 4 个训练锚点，留出点覆盖两个 tier，使用双线性矩形域）。两种情形都不允许把候选移动到域外或擅自扩展 tier。

## 质量门与失败语义

默认参数为 2 ms 采样、64 个 PRBS 窗口、20 个周期重复、PSS 末周期容差 `0.01 C`、频率细化容差 `0.01 GHz`。离散到连续转换还有三个显式数值门：增广离散矩阵条件数不超过 `1e12`、SciPy `logm` error estimate 不超过 `1e-8`、独立计算的 `||expm(logm(M))-M||_1/||M||_1` 不超过 `1e-8`；任何一个超限都会中止校准，而不是只写入诊断。ROM 评价不再从零温升开始，而是先对持续时间加权的平均功耗求解 `A x_ss = -B(l) u_bar`；状态矩阵条件数不得超过配置的 `max_condition_number`，归一化残差不得超过 `1e-10`，且没有回退到环境温度的路径。两个留出点都必须通过以下门：几何/功耗/频率/HotSpot trace/温度网格 identity 相同，ROM 和 HotSpot 均达到 PSS，整张最终周期网格 RMSE 不超过 `0.75 C`，峰温绝对误差不超过 `1.0 C`，且安全/不安全分类一致。PSS 峰值包含最终周期的初始状态，并比较整个网格而非单一热点。

任一训练或留出作业失败、哈希不符、PSS 不收敛、RMSE/峰温阈值越界、分类不一致，都会使 `rom_acceptance.json` 不被接受并阻止优化。最终真实 HotSpot 的失败会区分为 `steady_initialization_tool_failure`、`steady_initialization_contract_failure`、`transient_hotspot_tool_failure`、`transient_hotspot_contract_failure`、`validation_contract_error` 或 `pss_nonconvergence`。失败时 ROM 预测仍保留，但不会启动 R2，`f_sus_trans_hotspot_ghz` 和 `bips2_trans` 均为 `null`。若所有真实 HotSpot 点均已收敛、但最低频率仍不安全，则状态为 `thermally_infeasible`、分类为 `true_thermal_infeasible`，明确区别于工具失败。不要以稳态回退或 ROM 预测替代这些结果。

## 汇总字段：预测与验证必须分开

阅读 `transient_rom_summary.json` 时，应明确区分：

- `f_sus_trans_rom_pred_ghz` 与 `bips1_trans_rom_pred`：ROM 在优化内给出的预测，仅用于选择候选和诊断，不是最终 HotSpot 验证值。
- `f_sus_trans_hotspot_ghz`：最终候选经真实 HotSpot 周期稳态验证后的频率。
- `ipc2_trans` 与 `bips2_trans`：当前候选布局的真实 R2 IPC，以及仅当真实 HotSpot 验证和 R2 都成功时计算的验证后瞬态 BIPS。
- `training_hotspot_calls`、`holdout_initialization_hotspot_calls`、`holdout_transient_hotspot_calls`、`calibration_hotspot_calls`：包内历史校准证据，固定为 `8`、`2`、`2`、`12`；对应的 `*_this_invocation` 字段在纯复用调用中均为 `0`。
- `final_initialization_hotspot_calls`、`final_transient_hotspot_calls`、`final_validation_hotspot_calls`：最终频率搜索的稳态初值、周期瞬态及二者总调用数。总数必须等于前两项之和。
- `bips2`：只属于独立稳态汇总；ROM 汇总中故意不存在这个含糊字段，不能把稳态 `bips2` 当作 `bips2_trans`。

因此，报告应同时保留预测、留出误差、最终 HotSpot 结果、R2 结果、调用次数和所有 identity。即便所有门通过，结果仍是 non-formal、paper-inequivalent 的探索性扩展，不能提升为正式/论文等价结论。

对于新的离散配对模式，顶层汇总不再把 CLIP 单分支结果伪装成整个实验结果，而是使用三个命名空间：

- `predicted.clip3d`：ROM 的候选选择预测；只解释为什么选中该布局。
- `validated.fixed_bin` 与 `validated.clip3d`：两个布局各自经过真实瞬态 HotSpot 得到的持续频率及验证分类。
- `measured.fixed_bin` 与 `measured.clip3d`：两个布局各自真实 gem5 R2 的 IPC2，以及 `BIPS2_trans = IPC2 × f_sus_trans_hotspot`。

首要科学比较文件是 `transient_rom/final_validation/paired_comparison.json`（CSV 同目录）。它只在两边都存在真实 HotSpot 持续频率和真实 R2 IPC2 时生成；其中的提升率为 `(CLIP BIPS2_trans - fixed BIPS2_trans) / fixed BIPS2_trans × 100%`。ROM predicted 值永远不能替代这个 measured 配对结果，也不能在缺失一边时发布提升率。

## MATMUL/STENCIL 双点验证入口

双点验证固定选择稳态 5 点结果中的 MATMUL 和 STENCIL `64kB/512kB`：前者是正提升点，后者是负提升点。该选择能检查瞬态模型是否会改变稳态排序，而不是只挑选有利样本。它仍使用非零 `lambda_wire`、通信权重和 nearest 整数周期分区，分类始终是探索性、非论文等价。

先生成两个互相独立、不会覆盖 canonical R1 的 2 ms 周期统计 R1：

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh

time python -m workflow.transient.run_transient_r1 \
  --source-r1-dir runs/architecture_sweep/r1/paper/matmul/l1d_64kB/l2_512kB \
  --output-dir runs/transient_r1/balanced2_2ms_20260811/matmul_64kB_512kB \
  --sample-ms 2

time python -m workflow.transient.run_transient_r1 \
  --source-r1-dir runs/architecture_sweep/r1/paper/stencil/l1d_64kB/l2_512kB \
  --output-dir runs/transient_r1/balanced2_2ms_20260811/stencil_64kB_512kB \
  --sample-ms 2
```

这些是“新输出目录”命令。若某目录已有中断尝试，默认不得覆盖，应换一个新目录；
`--rerun` 只用于用户明确决定放弃该目录旧证据后的恢复，不属于正常可恢复流程。

两个 `status.json` 均为 `success` 后，先只运行 ROM 校准、布局搜索和真实 HotSpot 热验收，不启动 R2：

```bash
python -m workflow.experiments.transient_rom_balanced2 \
  --output-root runs/transient_rom_balanced2/validation_20260811
```

只有根目录 `summary.json` 的状态为 `thermal_validated`，才运行第二阶段：

```bash
python -m workflow.experiments.transient_rom_balanced2 \
  --output-root runs/transient_rom_balanced2/validation_20260811 \
  --run-r2
```

第二条命令会先重新校验两个热 checkpoint，然后在新的 `r2/` 子目录中复用各自的 12-call ROM 包，分别执行 fixed-bin 和 CLIP-3D 的真实 HotSpot/R2。checkpoint 绑定 selection、科学配置、稳态基线、canonical/周期 R1、ROM acceptance/manifest、对应 thermal package 和 paired branches 的路径与 SHA-256；复用时还逐项重验 manifest inventory、acceptance 的配置/R1 身份，以及每次真实 `transient_result.json` 的完整 HotSpot 命令。任一输入字节或 live artifact 变化都会拒绝复用。任何已有 point output、状态或日志都视为一次既有尝试，除非完整 checkpoint 重新验证成功，否则不得覆盖，必须换新的输出根目录。

当前通用验证器兼容早期已完成的 12-call 包：若旧 `calibration_cases.json` 尚未显式列出 `initial.steady.txt` 或 `transient_result.json`，验证器只会从同一 case 目录读取这两个文件，并要求它们已被完整 package manifest 绑定。由于这类旧包尚未记录六份原始 R1 哈希，Balanced-2 严格 checkpoint 入口不会复用它们；可重新校准生成带 R1 绑定的新包。旧 10-call ambient-start 包仍然拒绝。

主要产物为：

```text
validation_20260811/
├── thermal/{matmul,stencil}_64kB_512kB/   # 不含 R2 的热验收
├── r2/{matmul,stencil}_64kB_512kB/        # 仅 --run-r2 后存在
├── status/                                 # 每点、每阶段状态
├── logs/                                   # 公共入口 stdout/stderr
├── status.json                             # 根阶段状态
├── summary.json                            # 全精度、带哈希的权威汇总
└── summary.csv                             # 便于汇报的确定性表格
```

热阶段的表格只包含两个布局的真实瞬态持续频率，IPC/BIPS 和提升率保持为空。R2 阶段完成后才计算 `BIPS2_trans = IPC2 × f_sus_trans_hotspot`，并同时列出稳态提升、瞬态提升及二者的百分点变化。周期 R1 通常每点需要数小时；R2 还需对每点两个布局分别运行 gem5，因此完整双点测试仍可能需要十几小时以上，入口默认串行以避免争用节点 CPU 和混淆日志。

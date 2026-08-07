# 可选瞬态热 ROM 优化：设计规范

**状态：已确认设计，尚未实现。**  
**日期：2026-08-07**  
**适用范围：当前 CLIP-3D 复现实验的固定 2-tier 堆叠、固定散热条件、固定核心簇、仅 L2 可移动场景。**

## 1. 目标与边界

本设计在不改变论文原始稳态流程的前提下，增加一个可选的瞬态热优化模式。它的目的不是把
HotSpot 放进每次布局迭代，而是用少量、可审计的 HotSpot 校准轨迹训练一个降阶模型
（reduced-order model, ROM）。优化器在内环中用 ROM 计算布局的瞬态持续频率和预测 BIPS；
最终选中的布局仍必须由真实 HotSpot 和 gem5 R2 验证。

首版严格固定以下物理范围：

- 固定 die 外形、2-tier 堆叠、封装和 cooling 配置；
- 固定核心簇及其他非 L2 模块的位置；
- 仅共享 L2 的 `(x_mm, y_mm, tier)` 可变；
- 一个 ROM 只对应一个 workload、一个架构点和一条窗口化功耗轨迹；
- 固定电压；动态功耗与频率线性相关；给定温度的漏电功耗保持不变；
- 使用既有窗口化 McPAT 功耗和 HotSpot 配置，不能修改 HotSpot 源码。

以下事项明确不属于首版：跨 workload 泛化、跨堆叠/冷却泛化、可移动核心簇、温度相关漏电、
DVFS 电压变化，以及把 ROM 结果标为论文等价的正式复现结果。

## 2. 与现有流程的关系

原稳态路径完全保留：

```text
R1 -> McPAT + CACTI -> 稳态代理布局搜索 -> 最终 HotSpot -> Eq. (13) f_sus -> R2 -> BIPS2
```

其默认行为不变。新增的 `transient-rom` 路径与之并列：

```text
周期统计 R1 -> 窗口 McPAT
                |
                +-> 8 次短 HotSpot 校准轨迹 -> POD/状态空间 ROM
                +-> 2 次真实 HotSpot 留出验证 -> 质量门控
                                                    |
候选 L2 布局 -> ROM PSS 递推 -> f_sus,trans^ROM -> BIPS1,trans^ROM
                                                    |
                                        一个最终布局的真实 HotSpot PSS
                                                    |
                                           R2 -> BIPS2,trans
```

现有 `--transient true` 的含义仍是“完成窗口化 McPAT/HotSpot 的真实瞬态观察”，不得改变。
它不自动训练 ROM，也不自动把一个短轨迹 `Tmax` 当作持续频率。ROM 模式必须显式请求。

## 3. 用户接口和配置

`workflow.run_lifting_pipeline` 将新增：

```text
--thermal-mode {steady,transient-rom}
--transient-rom-dir PATH
--transient-rom-calibrate
--transient-rom-r1-dir PATH
```

- `--thermal-mode steady` 为默认值，等价于当前流程；它不读取 ROM 文件。
- `--thermal-mode transient-rom` 要求已存在且已通过门控的 ROM 包，或同时指定
  `--transient-rom-calibrate` 来建立该 ROM 包。
- `--transient-rom-dir` 是独立输出根目录；不得与原 R1、R2、稳态或真实瞬态输出重叠。
- `--transient-rom-r1-dir` 可复用既有周期统计 R1；若省略，遵循现有瞬态分支的“在独立目录
  新建专用 R1”逻辑，绝不修改源 R1。

实验 JSON 将新增可选 `transient_rom` 对象。默认不存在；因此历史配置仍按稳态执行。
首版默认值为：

```json
{
  "enabled": false,
  "sample_interval_ms": 2.0,
  "calibration_runs": 8,
  "validation_runs": 2,
  "pod_energy_threshold": 0.999,
  "max_pod_rank": 16,
  "max_logm_condition_number": 1000000000000.0,
  "max_logm_error_estimate": 1e-8,
  "max_exp_log_reconstruction_error": 1e-8,
  "calibration_windows": 64,
  "pss_period_repeats": 20,
  "pss_tolerance_c": 0.01,
  "frequency_tolerance_ghz": 0.01,
  "max_holdout_peak_error_c": 1.0,
  "max_holdout_grid_rmse_c": 0.75
}
```

任何非默认数值都必须写入 `calibration_manifest.json` 和最终 summary；不得存在隐藏的阈值或
“自动放宽”逻辑。

## 4. ROM 数学模型

令完整 HotSpot 网格的相对环境温升为
\(\boldsymbol\theta=\mathbf T-T_{\rm amb}\mathbf1\)。在固定物理堆叠内，热方程为线性时不变系统：

\[
\mathbf C\dot{\boldsymbol\theta}+\mathbf G\boldsymbol\theta=\mathbf P(t).
\]

通过校准轨迹的 SVD/POD 得到正交温度基 \(\mathbf U\)，并以
\(\boldsymbol\theta\approx\mathbf U\mathbf x\) 表示低维热状态。拟合的连续时间状态空间模型为：

\[
\dot{\mathbf x}=
\mathbf A_c\mathbf x+
\mathbf B_{\rm fixed}\mathbf u_{\rm fixed}(t)+
\mathbf B_{\rm L2}(p)u_{\rm L2}(t).
\]

其中：

- \(\mathbf u_{\rm fixed}\) 是按固定模块名称排列的窗口功耗向量；
- \(u_{\rm L2}\) 是 L2 的窗口功耗输入；
- \(\mathbf B_{\rm L2}(p)\) 是位置相关输入矩阵；
- \(p=(x,y,\mathrm{tier})\) 是合法 L2 布局。

对第 \(k\) 个窗口，在频率 \(f=sf_0\) 下：

\[
\Delta t_k(s)=\Delta t_k^0/s,
\qquad
\mathbf P_k(s)=\mathbf P_{\rm leak,k}+s\mathbf P_{\rm dyn,k}.
\]

因此 ROM 使用精确离散化：

\[
\mathbf A_{d,k}(s)=\exp\left(\mathbf A_c\Delta t_k^0/s\right),
\]

\[
\mathbf x_{k+1}=\mathbf A_{d,k}(s)\mathbf x_k+
\mathbf B_{d,k}(s,p)\mathbf u_k(s).
\]

其中 \(\mathbf B_{d,k}\) 由连续模型的矩阵积分得到；实现不得显式求逆不稳定矩阵。该递推既保留了
低频下同一工作量耗时变长，也保留了动态功耗缩放和漏电能量随时间增加的效应。

具体地，先在名义采样间隔 \(\Delta t^0\) 上用正则化最小二乘拟合离散模型
\(\mathbf x_{k+1}=\mathbf A_d\mathbf x_k+\mathbf B_d\mathbf u_k\)。随后通过增广矩阵对数得到连续模型：

\[
\begin{bmatrix}\mathbf A_c&\mathbf B_c\\\mathbf0&\mathbf0\end{bmatrix}
=\frac{1}{\Delta t^0}
\log\left(
\begin{bmatrix}\mathbf A_d&\mathbf B_d\\\mathbf0&\mathbf I\end{bmatrix}
\right).
\]

不同频率的离散矩阵再由同一增广矩阵指数得到：

\[
\exp\left(
\begin{bmatrix}\mathbf A_c&\mathbf B_c\\\mathbf0&\mathbf0\end{bmatrix}
\frac{\Delta t_k^0}{s}
\right)
=
\begin{bmatrix}\mathbf A_{d,k}(s)&\mathbf B_{d,k}(s)\\\mathbf0&\mathbf I\end{bmatrix}.
\]

这避免了 \(\mathbf A_c^{-1}\) 或 \(\mathbf F^{-1}\) 的显式求逆；若矩阵对数的虚部、条件数或
重构误差超过配置门槛，校准必须失败。

每个频率把 ROI 功耗周期反复施加，直到相邻周期末全部 ROM 网格重构温度的最大差不超过
`pss_tolerance_c`。最终周期及其初态的全网格峰温用于安全判断：

\[
f_{\rm sus,trans}^{\rm ROM}=f_0\max\{s:T^{\rm ROM}_{\max,\mathrm{PSS}}(p;s)\le T_{\rm safe}\}.
\]

频率搜索沿用验证器的非全局单调安全策略：先扫描配置网格，再仅在相邻安全/不安全边界内局部细化。
这避免把未经证实的全局温度单调性作为假设。

## 5. 十次 HotSpot 校准设计

总预算严格为十次新的 HotSpot 调用：八次训练和两次留出验证。每次训练采用 64 个
2 ms 窗口的持久激励 PRBS（伪随机二值序列），从环境温度开始；它产生丰富时间样本，
不是只提供一个峰温标量。

### 5.1 八个训练布局

八个训练锚点始终遵循当前配置的 `layout_optimizer.allowed_l2_tiers`，不会扩大 L2 的合法 tier
范围：若允许 tier 为 `[0, 1]`，每个 tier 各构造四个非重叠锚点，即该 tier 合法区域中 `(x,y)` 的
左下、右下、左上、右上极值；若只允许一个 tier，则在该 tier 构造八个确定性、非重叠锚点（四角、
四条边的中点，必要时向可行域内部收缩）。锚点必须：

- 位于 L2 边界约束内；
- 与固定模块无面积重叠；
- 通过既有 `check_geometry`；
- 在 `anchors.json` 中记录产生规则、坐标、碰撞面积、tier 与几何哈希。

若双 tier 模式的任一 tier 无法产生四个互不重复的合法锚点，或单 tier 模式无法产生八个互不重复的
合法锚点，校准失败，而不是减少样本悄悄继续。

每个固定模块名称和 L2 输入均使用不同、零均值且有界的 PRBS 振幅；功耗始终为非负，且每一个
模块输入均保留其原始平均功耗量级。故每个输入通道保留其真实空间功耗分布，而非把所有固定模块
错误压缩为一个热源。训练在全温度网格输出上进行。

### 5.2 两个留出验证

两个未被训练采用的、合法的内部 L2 点由确定性规则生成：双 tier 模式每个 tier 一个点；单 tier 模式
在该 tier 生成两个点，尽量远离全部训练锚点。第一个在 \(f_0\)，第二个在
`0.6 f0 + 0.4 fmin`（裁剪到频率范围）运行真实的周期稳态 HotSpot。
这两个作业使用真实窗口 McPAT 功耗，而不是 PRBS，以分别验证空间插值、真实 workload
功耗以及频率时间缩放。两者都必须有 PSS 证据。

## 6. 位置插值、POD 与辨识

双 tier 模式中，每个 tier 独立地对四个锚点的 \(\mathbf B_{\rm L2}\) 进行受限双线性插值。单 tier
模式中，对八个锚点进行确定性的二维 Delaunay 三角剖分，并在包含该候选的三角形内使用重心插值。
候选点必须位于对应 tier 训练锚点凸包的闭包；对于凸包外候选、退化单元或负插值权重，布局优化器
将其视为非法候选并记录原因。首版不进行外推。

POD 基只由八次训练温度轨迹构成。选择不超过 `max_pod_rank` 的最小 rank，使累计奇异值能量
达到 `pod_energy_threshold`；若在 `max_pod_rank` 内无法达到该能量，校准失败。若训练重构误差未达标，
校准失败。所有奇异值、最终 rank、基矩阵校验和与训练轨迹哈希必须被记录。

状态矩阵使用稳定性约束拟合：\(\mathbf A_c\) 的实部特征值必须为非正；不满足则拒绝模型。
离散状态和输入快照通过正则化最小二乘识别，正则系数、条件数、残差和稳定性检查写入
`fit_report.json`。ROM 不得通过任意修正温度、偏移峰温或按验证集调参来获得通过。

## 7. 质量门控与错误处理

ROM 只有满足全部条件才能被优化入口消费：

1. 八个训练 HotSpot 作业均成功且输入/输出哈希匹配；
2. POD 重构和状态辨识均有限、稳定，矩阵维度一致；
3. 两个留出 HotSpot 作业均达到 PSS；
4. 两个留出点的最终周期全网格 RMSE 不超过 `max_holdout_grid_rmse_c`；
5. 两个留出点的峰温绝对误差不超过 `max_holdout_peak_error_c`；
6. ROM 与 HotSpot 对每个留出点给出相同的安全/不安全结论；
7. 几何、窗口功耗、频率模型、HotSpot 可执行文件和配置哈希一致。

这些条件由流水线和直接 ROM 优化入口共用同一个包验证器执行。训练/留出 case 的模块、布局、功耗和温度 artifact 必须使用包根相对路径，且不得逃逸或指向 symlink；源模块 JSON 按原始字节复制到包内。因而完整接受包可整体移动，但不能脱离其 case、拟合、留出和 inventory 证据单独移动接受标记。命令行校准入口和流水线校准入口都必须在验收后写同样的 `rom_artifact_manifest.json`。

任何失败都必须中止 `--thermal-mode transient-rom`，指出失败的作业、误差和可定位的 artifact
路径。禁止自动改用稳态 Eq. (13)，禁止把未验证 ROM 的预测写为 `BIPS2_trans`。

最终候选要经过真实 HotSpot PSS 验证；若真实验证失败，结果状态为 `rom_final_validation_failed`。
该情况保留 ROM 预测以便诊断，但不得报告成功的瞬态优化效果。
执行/I/O 失败记为 `tool_error`，畸形 trace 或搜索证据记为 `validation_contract_error`，PSS 未收敛记为 `pss_nonconvergence`；可在调用前确定的输入/输出前置条件错误不包装成工具失败。

## 8. 新的模块与产物

计划新增以下职责明确的模块：

```text
workflow/transient/rom/
  calibration_design.py       # 锚点、留出点和 PRBS 的确定性生成
  materialize_calibration.py  # 训练/留出 HotSpot 输入与清单
  pod_state_space.py          # POD、连续状态空间辨识和稳定性检查
  layout_rom.py               # L2 输入矩阵插值、PSS 温度/频率评价
  calibrate_rom.py            # 组织 8+2 作业、质量门控、写 ROM 包
  optimize_transient_layout.py# 用 ROM 搜索 L2 布局；无 HotSpot 内环
  validate_rom_layout.py      # 仅最终候选的真实 HotSpot PSS 验证
```

计划在 `workflow/transient/` 增加公共验证和 I/O 辅助，而不是改动稳态
`workflow/thermal/sustainable_frequency.py`。稳态优化器也不接受 ROM 模型对象。

每次 ROM 校准的独立目录至少包含：

```text
transient_rom/
  modules.json                              # 与身份哈希一致的包内源模块字节
  calibration_manifest.json
  anchors.json
  training/{anchor-id}/...              # 输入、轨迹、HotSpot 日志、哈希
  holdout/{holdout-id}/...              # 同上，含 PSS 证据
  pod_model.npz                         # U, A_c, B, 元数据
  fit_report.json
  validation_report.json
  rom_acceptance.json
```

一次 ROM 优化输出至少包含：

```text
transient_rom_optimization/
  optimization_report.json
  proposed_layout.json
  predicted_frequency.json
  final_hotspot_validation.json
  transient_rom_summary.json
```

所有 ROM 产物和汇总均必须带有：

```json
{
  "non_formal": true,
  "paper_equivalent": false,
  "thermal_mode": "transient-rom"
}
```

`BIPS2_trans` 仅在真实最终 HotSpot PSS 和 R2 都成功时写入。它不得覆盖稳态 `bips2`。

## 9. 测试与验收

新增单元和集成测试必须至少覆盖：

- 锚点/留出点的确定性、边界与碰撞拒绝；
- PRBS 有界、非负、可复现，且固定与 L2 输入不共线；
- POD 截断、重构误差、稳定状态空间拒绝；
- 矩阵时间缩放在解析一阶 RC 合成数据上的正确性；
- L2 输入矩阵的 tier 内插值、凸包外拒绝、tier 不混合；
- PSS 首状态包含、全网格峰值、安全边界局部细化；
- 缺失哈希、错误工作负载、错误 sample interval、未通过留出验证的 ROM 均被拒绝；
- `steady` 默认路径的既有回归不变；
- 小型合成 HotSpot fixture 上验证 ROM 能通过门控并完成一次优化；
- 最终真实验证失败时不得生成 `bips2_trans`。

真实 MATMUL 试验的成功标准不是“必须提升 BIPS”，而是：ROM 先通过留出门控，优化内不调用
HotSpot，最终 HotSpot/R2 结果与全部预测、误差和 provenance 可审计地并存。只有在此基础上，
才能讨论布局带来的瞬态 BIPS 改善。

## 10. 可比性与报告规则

稳态与瞬态实验必须使用同一 R1 架构点、相同模块几何、相同 L2 合法域、相同 R2 延迟模型和
相同最终验证政策。报告至少同时列出：

- `f_sus_steady`、`bips2`；
- `f_sus_trans_rom_pred`、ROM 留出误差；
- `f_sus_trans_hotspot`、`bips2_trans`；
- 校准调用数（固定为 10）、最终验证调用数和优化内 HotSpot 调用数（必须为 0）；
- `non_formal` 与 `paper_equivalent` 标记。

结论不得将 ROM 瞬态结果表述为论文稳态 Eq. (13) 的复现实验；它是经真实 HotSpot 复核的、
受控范围内的方法扩展。

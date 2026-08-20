# 热代理修正：问题分离与验证准则

## 目标与边界

CLIP-3D 的 Equation (14) 不是 HotSpot 的替代品。它只需为 L2 给出有用的
“离开热区”方向；最终温度和 BIPS 仍由 HotSpot 与 R2 给出。因此本验证不以
绝对温度、严格全排序，或 HotSpot top-3 与代理 top-3 重合为验收条件。

一个代理变体至少应在**同一 HotSpot 物理合同**下检查：相对 fixed-bin 的温差
方向、代理选点的 HotSpot selection regret、以及是否真的跨越了频率门限。

## 已区分的四类问题

1. **物理合同漂移。** 64x64 标定曾在预测阶段丢失 `input_granularity`、
   `compact_trace` 和 `ptrace_precision`，从 module-input 静默退回 grid-cell；更早的
   operational 配置还把这套未缩放参数与 `local_resistance_scale=8.72` 的热堆叠混用。
   这已经修正；任何后续诊断必须记录这三个字段、网格、完整热栈和 `R_conv`。
2. **频率模型的适用条件被混淆。** Equation (14) 的绝对温度只是一阶启发式；
   原始代理的共同偏置可把候选送入错误的 95 C 分支。即使门限正确，Equation
   (13) 的单一全局 \(\gamma\) 还假定动态/漏电功耗在空间上近似同比例。non-formal
   operational 诊断允许一份同合同 fixed-bin HotSpot 作为共同锚点，且只将
   代理的候选间 delta 送入 Equation (13)。这不改变排序或梯度；对于明显非均匀
   \(\gamma\)，则按论文的两点 HotSpot 回退做单独验证，而不重新拟合代理系数。
3. **宏块几何被压缩为质心。** HotSpot 使用经过 CACTI 长宽比和 McPAT 面积
   归一化的 L2 矩形；代理若把它压成一个点，会漏掉有限矩形的源/受体积分。
   `area-quadrature` 是允许的最小修正，且不改变 Eq. (14) 的核或共享 `L_c`。
4. **峰值网格/边界效应。** HotSpot 的峰值可在相邻 grid cell 或不同模块间
   切换，而简单 `max` 核没有完整的 3D Green 函数。曾试验将 Eq. (14) 全面
   栅格化为 `grid-field`；在首个 512kB 差点上，方向一致率由 5/8 降为 3/8，
   故不纳入主线。这个结果支持保留简单论文核，而不是将代理扩展成隐式
   HotSpot 替代品。

| 候选问题 | 最小可证伪测试 | 当前状态与可作出的结论 |
| --- | --- | --- |
| HotSpot 输入合同在标定与预测间漂移 | 对同一 `modules.json` 比对 grid、module/grid-cell 输入、ptrace 精度、stack 和 `R_conv`；只复用合同完全匹配的探针 | 已定位并修复 `run_one()` 漏传三项 materialization 选项的问题；并证实旧 `stream/16/1024` 的 32×32/grid-cell、`local_resistance_scale=8.72` 结果为 97.597166 C，而同布局的未缩放 64×64/module 合同为 81.776555 C。两者都不能与同一个绝对代理温度混用；后续诊断统一使用未缩放 module-input 合同。 |
| Eq.(14) 的绝对偏置让 Equation (13) 进入错误门限，或 module-level \(\gamma\) 使 uniform-\(\gamma\) 近似失效 | 同一 fixed-bin 的 HotSpot 温度只作为候选间共同偏置，比较 raw 与 anchor 后的频率状态；另选真实高功耗 FFT 锚点，并以分离 dynamic/leakage ptrace 在 1.0 GHz 和闭式 \(f_{sus}\) 重跑 HotSpot | FFT fixed-bin 已证实闭式热限频链路可进入 1.15 GHz；低功耗 stencil 在真实 HotSpot 下全部 2 GHz 是不可观测，而非频率公式失败。FFT 的 1.0 GHz 分离功耗 HotSpot 为 90.603891 C，而 Eq.(11) 的 global-\(\gamma\) 预测为 90.919710 C，误差 −0.315819 C，远超 0.02 C，故 global-\(\gamma\) 被拒绝。Eq.(9) two-point 解为 1.159790386 GHz；其独立、逐 cell 分离功耗 HotSpot 复核为 95.000000 C（`layer_1_g1622`），安全误差 0 C，满足 0.02 C 严格门限。 |
| L2 有限矩形被质心化，或把核长度错误地随 L2 尺寸改变 | 固定共享参数与同一 HotSpot 网格，比较 center 与 area-quadrature；保持 `L_c=die/2` 的几何消融与冻结拟合候选分开 | 首个 L2 留出点中面积积分的冻结候选方向较好。不可把 L2 长宽代入 `L_c`：论文把 `L_c` 设为 die half-width，L2 尺寸仅应出现在面积积分。完整留出汇总仍在运行。 |
| HotSpot peak 的 cell/边界切换使极小温差不可靠 | 以 `|ΔT|>=0.02 C` 才计 sign；对接近选点用更高 grid 重跑 | `grid-field` 扩展在首点更差，未采用。当前不把亚阈值的反向符号计为代理失败；高分辨率复核只在完整留出证据显示真实选择差异时执行。 |

## 关于 Lc 与热绑定

`L_c` 是横向热扩散核的共享长度，不是 L2 的半宽/半长。同一 die、封装和材料
下，它不能按 workload 或单个 cache 容量重新设定；真正依赖 L2 尺寸的是面积
积分。严格论文复现采用 `L_c = die-side / 2`；工程版若拟合其它共享值，必须在
workload、缓存容量和空间位置留出集上验证，不能以训练点误差推广。

这里需要把三个层面严格分开。论文在 Equation (14) 后明确将 `L_c` 设为 die
half-width，并先把每个 tile 的子块功耗汇聚到质心；因此本诊断的 `paper-center`
复现的是**论文披露的几何部分**，`paper-area` 只是“保持论文的共享 `L_c`、但把
有限 L2 矩形作积分”的单因素几何消融。论文没有公开 `alpha`、`beta` 和
`w_cross` 的数值，故这两个 `paper-*` 变体仍保留冻结的诊断系数，不能称为完整的
论文数值复现。`fitted-*` 变体同样不是论文默认值：它们只是在固定 64x64
module-input 合同下检验一个冻结的、跨点共享的诊断候选。尤其
不能因为 L2 的 CACTI 长宽比不同就把 `L_c` 改成 L2 半宽或半长；那会同时改变核的
全局扩散长度和 L2 自身几何，失去可辨识性。

论文还说明 `alpha` 和 `beta` 是一次选定的单位换算常数，用来让 `T_hat` 落在
80--110 C、从而让 Equation (13) 有非平凡响应；它们并不是按每个 workload 重调的
HotSpot 回归参数。因此低功耗点的原始代理若全部落在 95 C 以下，不能据此声称
“频率项坏了”。应先用同合同的高功耗锚点确认 Equation (13) 的热限频分支，再只用
低功耗点判断空间梯度是否让 L2 定性离开热点。

梯度诊断同时输出 raw Equation-(14) 与 fixed-bin-anchor 后的频率状态及其各自同
HotSpot 的状态一致率。前者用来暴露单位换算/绝对偏置造成的错误门限；后者仅用来
验证“在已知热包络内，梯度能否导向更冷的位置”。两者都不参与温度排序的计算。
报告还分别给出三者的可持续频率范围及 `*_frequency_varies`。频率验证还记录
module-level \(\gamma\) 的功率加权值、范围和跨度；只有该范围足够窄、或分离
dynamic/leakage HotSpot 在 \(f_{sus}\) 的安全误差不超过 **0.02 C**（论文 Table III
的验证量级）时，global-\(\gamma\) 闭式频率才可作为频率证据。该阈值在频率验证
输出的 `frequency_settings.max_safe_error_c` 中显式记录，不能以“接近 95 C”的
度量级误差替代。温度落入热限频状态本身并不足以让布局优化受益：只有不同
合法位置的可持续频率确实有差异，Equation (13) 才对 L2 的位置提供热梯度。

若 global-\(\gamma\) 检验未通过，验证器保留该失败；不会改调 \(\alpha\)、\(L_c\)，
也不会放宽阈值。它从同一布局的 nominal \(f_0\) 热图与一份低频、逐 cell 分离
dynamic/leakage 热图，逐 cell 重建 Equation (9) 的
\(A_i+(f/f_0)B_i\)，再以最先达到 \(T_{safe}\) 的 cell 求 two-point
`two_point_affine_frequency`。这是论文说明的“一次额外 HotSpot”回退；结果仅作
验证/校准，绝不把 HotSpot 调用加入 Equation (14) 的优化内环。只有随后在这个
闭式频率运行的独立 HotSpot 仍满足 0.02 C 安全误差时，才可把该回退报告为通过。
因此 global-\(\gamma\) 的正式接受同时要求：分离功耗参考点的空间温度误差不超过
0.02 C，且其闭式 \(f_{sus}\) 安全复核通过；不能只因后者偶然落在 95 C 就接受前者。

不能把这个回退偷换成“将 Equation (14) 的模块 dynamic/leakage 分量分别求和”
的无 HotSpot 版本。一个隔离的候选 worktree 在同一 FFT fixed-bin anchor 上作了
这一测试：其 component-affine proxy 为 `1.148779 GHz`，不仅没有逼近真实两热图
解 `1.159790 GHz`，还比 global-\(\gamma\) 的 `1.150022 GHz` 略差。因此该候选
不合并；它说明真正缺失的是 **HotSpot 的动态/漏电空间热图**，不是再给 Eq.(14)
增加一个模块功耗权重即可弥补的参数问题。

频率项是否活动首先由真实 fixed-bin 热状态决定。一个点即使是“代理较差点”，
也可能在目标冷却包络下始终低于 `T_safe`；此时频率在所有位置都是 2 GHz，热
代理无法也不应制造性能增益。优化器报告中的
`observability_diagnostics.sampled_thermal_frequency_term_active` 与
`sampled_thermal_frequency_term_varies` 显式记录这一区别。
共同 HotSpot 网格诊断还报告每个变体的
`hotspot_frequency_observability.hottest_headroom_to_safe_c`：为正且没有跨阈值时，
当前 L2 可移动范围在物理上不足以激活频率项；这不是低相关度排名或 \(L_c\)
参数的证据。

## 分阶段实验

1. 先以语义 ROI 的 `stencil / L1D=128kB / L2=512kB` 在未缩放、64x64、
   module-input、`R_conv=5` 合同下比较 `fitted/paper × center/area` 四个变体。
2. 只有在首点确认可重复的方向结果后，才扩展到其余四个差点；每一点使用相同
   原始 McPAT/CACTI 模型、同一候选格和相同 HotSpot 合同。
   若长时 HotSpot 任务中断，以同一命令加 `--resume` 重启；它只复用
   `calibration_sample.json` 与当前合同完全一致的已完成探针，避免把部分目录
   误当成有效结果。
3. 单独选择高功耗 FFT 锚点验证 Equation (13) 的频率链路及 uniform-\(\gamma\)
   前提。低功耗差点只验证温度梯度，不被错误地当作频率测试。
4. 对接近的选点以更高 HotSpot 网格复核；小于数值/网格不确定度的温差应记为
   tie，而非代理失败或性能提升。

## 当前证据（2026-08-20，四个架构留出点已完成）

四个架构留出点已在各自 3x3 个合法 top-tier 位置上完成。下表为冻结的
`fitted-area` 候选；`sign` 仅在相对 fixed-bin 温差绝对值不少于 0.02 C 时计入，
`headroom` 是采样位置中最热的真实 HotSpot 到 95 C 的余量。

| L2 留出点 | Spearman | sign | selection regret | headroom |
| --- | ---: | ---: | ---: | ---: |
| stencil / 128kB / 512kB | 1.000 | 6/6 | 0.000 C | 5.356 C |
| stencil / 128kB / 256kB | 0.950 | 5/5 | 0.000 C | 6.196 C |
| stream / 16kB / 1024kB | 0.983 | 8/8 | 0.000 C | 13.002 C |
| stream / 64kB / 128kB | 1.000 | 3/3 | 0.000 C | 14.101 C |

合计为 22/22 个可比较方向正确，平均 Spearman 为 0.983，四个点的 selection
regret 都是 0。四个 3x3 合法位置范围也均未跨越 95 C；距门限最近的采样点仍有
5.356 C 余量（四点最热位置余量均值为 9.664 C）。因此这些差点上的频率项在真实
HotSpot 中物理上不可观测，不能以“代理未使频率变化”作为参数或梯度失败的证据。
系列汇总将这一 `hottest_headroom_to_safe_c` 和是否跨门限显式写入，以免今后把
headroom 情况误报为频率模型失败。

首个 `stencil / 128kB / 512kB` 点还完成了几何消融。normal（`R_conv=1.042`）和
stressed（`R_conv=5`）合同给出相同选择/方向结论；下表列 stressed 的结果。

| 共享几何变体 | Spearman | sign | selection regret |
| --- | ---: | ---: | ---: |
| fitted area-quadrature, `L_c/die=0.0586007` | 1.000 | 6/6 | 0.000 C |
| fitted center, `L_c/die=0.0586007` | 0.883 | 6/6 | 0.000 C |
| paper center, `L_c/die=0.5` | 0.350 | 3/6 | 0.050 C |
| paper area-quadrature, `L_c/die=0.5` | 0.500 | 3/6 | 0.033 C |

因此，当前证据支持把 `area-quadrature + L_c/die=0.0586007` 作为**待继续扩展验证的
共享诊断候选**；它不是按 workload/cache 调参，也尚未被提升为论文等价默认值。
同一高功耗/低功耗锚点还表明频率项的分界正确：`stencil / 128kB / 2048kB`
在 stressed 合同下为 93.762 C（仍有 2 GHz headroom），而
`fft / 64kB / 1024kB` 为 118.118 C（进入 Equation (13) 的热限频区）。后者只
验证内环的频率可观测性：其原始 `fitted area-quadrature` 代理给出
117.460 C / 1.168320 GHz，故 Equation (14) 到 Equation (13) 的启发式频率项在
高功耗合同下确实活动。其全局 `gamma=0.415837` 闭式值 1.150022 GHz 则未通过
分离 dynamic/leakage HotSpot 的严格验证，最终频率应使用已验证的 Eq.(9) two-point
值 1.159790386 GHz；五个低功耗差点继续只评价空间梯度。

### 拟合/留出边界

上述共享参数来自 `alpha_lc_unscaled_identification_reduced.json` 的 2026-08-17
reduced identification：训练设计只包含 `L1D={16,128}kB`、`L2={128,2048}kB`，并在
训练期固定为未缩放、64x64、module-input 合同。故五个差点中，
`stencil/128/512`、`stencil/128/256` 与 `stream/16/1024` 是 L2 架构留出，
`stream/64/128` 是 L1D 架构留出；`stencil/16/2048` 落在该 reduced 设计内，只能作为
实现回归/空间位置复核，**不得**计入共享参数的泛化结论。所有五点均只评估冻结参数，
绝不按测试结果重新拟合。此前的 32x32 `paper-L_c` 结论见
`thermal_proxy_gradient_validation_zh.md`，它已被明确标为历史诊断，不能与当前 64x64
物理合同证据混用。

完成多个点后，用 `workflow/thermal/summarize_proxy_gradient_series.py` 仅汇总已完成的
`gradient_diagnostic.json`。调用方必须将留出与拟合集内点作为不同 `LABEL=PATH` 传入；
它拒绝混用不同 config/HotSpot/grid/变体合同，使用可比较 sign 的总计数而非平均每点比例，
且不触发重拟合或新的 HotSpot 求解。

该文档只定义诊断和决策规则；在五点完成前，不将任何拟合的 `alpha`、`L_c` 或
面积积分变体标为论文等价或共享已验收参数。

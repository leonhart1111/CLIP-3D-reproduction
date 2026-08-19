# 热代理梯度诊断与当前 operational 修正

## 结论与边界

截至 2026-08-20，新建的 non-formal
`configs/experiments/clip3d_proxy_anchor_paper_lc_diagnostic.json` 将 Equation (14) 的
`lc_die_side_ratio` 显式设为 `0.5`，即论文所述的 die half-width；空间核保持论文兼容的
模块质心实现。不要把 \(L_c\) 改为 L2 宏块的半宽/半长，也不要按 workload 或单个缓存
大小分别拟合它。

该修正只证明简单代理在当前五个最差点上能提供有用的离开热区趋势，并不声称精确预测
HotSpot 温度、完整排序或论文报告的性能提升。所有报告性温度仍必须来自最终布局的
HotSpot 求解。

## 已确认的问题

1. 旧 exploratory 配置将 \(L_c\) 设为 `0.0586007389 * die side`。论文 Eq. (14) 明确
   指定 \(L_c\) 为 die half-width，即 `0.5 * die side`。前者在一个点会产生反向梯度。
2. 代理绝对温度曾低于 HotSpot 约 15--18 C，使所有候选被归类为 `thermal_headroom`，
   Equation (13) 的频率项恒为 2 GHz。`fixed-bin-hotspot` anchor 只加共同偏置，不改变
   候选间温差或排序，因此用于 operational 诊断来恢复正确的频率区间。
3. `calibrate_proxy.py` 过去在预测时没有传递配置中的 \(L_c\)，会隐式退回 die 半宽，
   而运行期却使用配置值；该合同错误已修正并有单元测试。
4. HotSpot 使用真实矩形面积与网格功耗分布；旧质心代理没有使用矩形范围。面积求积已
   实现并测试，但在五点间没有稳定优于质心核，因此不是当前默认。HotSpot 峰值网格单元
   在位置间切换也是简单 `max` 代理无法精确排名的原因。

## 五个最差点的共同位置扫描

每点固定同一份 raw McPAT/CACTI `modules.json`、热栈、32×32 HotSpot 合同和 top-tier L2
约束。先求解固定 bin anchor，再求解 3×3 的所有合法 L2 位置；同一批 HotSpot 温度同时
评估三个代理，不按结果重新拟合参数。`sign` 只统计相对 fixed-bin 温差绝对值大于
0.02 C 的候选；`regret` 是代理最低温候选相对该网格 HotSpot 最低温的温差。

| 点 | 合法位置 | 旧小 \(L_c\) 中心核：rho / sign / regret | paper \(L_c\) 中心核：rho / sign / regret | paper \(L_c\) 面积求积：rho / sign / regret |
|---|---:|---:|---:|---:|
| stencil, L1D 128kB, L2 512kB | 9 | -0.150 / 0.000 / 1.213 C | 0.667 / 0.625 / 0.031 C | 0.517 / 0.500 / 0.028 C |
| stencil, L1D 16kB, L2 2048kB | 8 | 1.000 / 1.000 / 0.000 C | 0.881 / 1.000 / 0.000 C | 0.333 / 0.857 / 0.619 C |
| stream, L1D 16kB, L2 1024kB | 9 | 0.967 / 1.000 / 0.102 C | 0.400 / 0.714 / 0.157 C | -0.067 / 0.571 / 0.639 C |
| stencil, L1D 128kB, L2 256kB | 9 | 0.417 / 0.167 / 0.054 C | 0.367 / 0.833 / 0.032 C | 0.567 / 0.667 / 0.032 C |
| stream, L1D 64kB, L2 128kB | 9 | 0.900 / 1.000 / 0.000 C | 0.633 / 0.667 / 0.004 C | 0.683 / 0.667 / 0.004 C |

五点非加权汇总：旧小 \(L_c\) 的平均有效 sign 为 `0.633`、平均 regret 为 `0.274 C`；
paper \(L_c\) 质心核分别为 `0.768` 与 `0.045 C`；面积求积分别为 `0.652` 与 `0.264 C`。
旧小 \(L_c\) 的平均 rho 可被两个局部拟合良好的 stream/大 L2 点抬高，但它在
stencil/512kB 完全反向且选择损失最大，因此不适合作为共享 operational 参数。

原始诊断输出位于本地 ignored `runs/thermal_proxy_gradient_20260820/phase2/`；每个目录都
包含 `gradient_diagnostic.json`，记录布局、峰值单元、原始/锚定代理温度和频率状态。

## 当前实现与使用

`workflow.thermal.diagnose_proxy_gradient` 是主交互诊断入口。它将 HotSpot 结果与代理评估
分离，故可在不运行 gem5、McPAT 或 R2 的前提下比较多种 \(L_c\)/几何变体。建议先用
`paper-center=center,0.5`；`paper-area=area-quadrature,0.5` 只用于研究对照，不能据此修改
共享配置。

该新诊断配置启用了 `thermal_anchor_policy=fixed-bin-hotspot`。严格 P1 配置
显式拒绝该选项，因此该额外一次 HotSpot 求解不会被误写为论文 Algorithm 1 的单次终验。

下一步是在 64×64 HotSpot 网格对少量代表性候选复核 32×32 的梯度结论，并对更新后的
同合同数据重新标定 \(\alpha\) 与 \(w_{cross}\)。这些工作完成前，当前修正仍不可提升为
paper-equivalent 或 shared formally accepted 参数。

# CLIP-3D 未公开参数的实验标定

本文记录公式（14）的 `alpha`、`beta`、跨层耦合权重，以及公式（15）的
`lambda_wire` 在本地复现环境中的标定过程。论文没有公开这些数值，因此以下
结果是可审计的本地复现增强，不应写成论文原始参数。

> 2026-07-29 复核结论：下文 `alpha=1.526` 的旧拟合虽然降低了绝对误差，
> 但不满足论文“代理落在 80--110 C”的选值条件，而且 21 个合法样本全部让
> L2 留在 tier 1，不能辨识 `beta` 的跨层惩罚。旧报告现在只保留为诊断历史，
> 不再作为论文严格配置。严格配置使用位置排序实验支持的
> `cross_tier_weight=0.965`、范围约束下的 `alpha=0.3, beta=0`，并要求在正式
> 扫描前继续做跨工作负载验证。

## 1. 标定对象

当前热代理为：

\[
\hat T_{\max}=T_{amb}+R_{conv}P_{tot}
+\alpha\max_i\sum_jP_jK(d_{ij})w(z_i,z_j)
+\beta\sum_{j\in bottom}P_j .
\]

本实验固定论文/配置已经确定的 `T_amb`、`R_conv` 和
`L_c=die_width/2`，拟合：

- `alpha`：中心点热耦合项的摄氏度/瓦转换系数；
- `beta`：底层总功耗的额外惩罚系数；
- `cross_tier_weight`：跨层模块之间的耦合衰减；
- `lambda_wire`：公式（15）中线延迟周期到 BIPS 损失的转换系数。

这里的 `alpha` 是公式（14）的热代理系数，不是论文前面分段线性 DVFS
公式中曾使用、随后被公式（13）替代的同名斜率。

## 2. HotSpot 实验设计

使用三个与当前正式流程一致的模块模型：

1. FFT，L1D=64 kB、L2=1 MB；
2. FFT，L1D=32 kB、L2=256 kB；
3. 正式 MATMUL R1，L1D=64 kB、L2=1 MB，经当前 McPAT 功耗校准和
   CACTI/Table-II 几何重新生成。

每个模型把共享 L2 放在归一化 3x3 坐标上，剔除和 NoC 或固定模块重叠的
位置，共得到 21 个合法布局。当前三个模型的底层均没有足够连续空白容纳 L2，
因此合法样本都位于 tier 1；这也是 `beta` 可辨识性弱的重要原因。

每个样本都经过矩形-网格精确重叠功耗分配，并由 detailed-3D HotSpot 求解。
训练使用 16x16 热网格。网格收敛检查如下：

| 同一 fixed FFT 布局 | Tmax |
|---|---:|
| 8x8 | 123.25 C |
| 16x16 | 124.48 C |
| 32x32 | 124.39 C |

16x16 与目标 32x32 只差 0.09 C，因此用它降低扫点成本；两个已有的 32x32
布局另外作为完全不参与拟合的外部确认样本。

## 3. 为什么不能只最小化绝对温度误差

如果仅最小化 `proxy_Tmax - HotSpot_Tmax`，求解器会得到一个绝对温度很准、
但把同一模型所有位置预测成相同温度的退化解。这样的模型不能指导布局。

因此拟合残差同时包含：

\[
r_{abs}=\hat T-T,
\]

以及每个模型内部去掉均值后的位置残差：

\[
r_{spatial}=(\hat T-\overline{\hat T})-(T-\overline T).
\]

联合目标为 `r_abs` 与 `10*r_spatial` 的平方和。权重 10 表示约 0.1 C 的位置
温差误差与 1 C 的绝对温度误差具有相近重要性。

由于公式中的 `max_i` 会让跨层权重附近出现热点接收者切换，响应不是适合直接
连续拟合的光滑曲面。脚本在训练集上拟合 `alpha/beta`，以 0.005 步长在留出集
选择跨层权重，再固定该权重用全部样本重估 `alpha/beta`。

## 4. 热代理结果

最终结果：

```text
alpha             = 1.526065981689432
beta              = 7.23e-22（数值上等于 0）
cross_tier_weight = 0.965
```

| 指标 | 原占位参数 | 拟合参数 |
|---|---:|---:|
| 留出集绝对温度 RMSE | 13.488 C | 0.700 C |
| 留出集空间中心化 RMSE | 0.125 C | 0.0196 C |
| 留出集空间 Spearman | 0.543 | 1.000 |
| 全部21点绝对温度 RMSE | 13.413 C | 0.732 C |
| 全部21点空间 Spearman | 0.933 | 0.924 |

32x32外部确认中，fixed 到旧 optimized 的真实温差为 +0.18 C：

- 原参数预测温差约 +0.000053 C，位置敏感度约低 3400 倍；
- 拟合参数预测温差约 +0.297 C，方向正确，但幅度约为真实值 1.65 倍；
- 两点绝对温度 RMSE 从 12.943 C 降到 1.148 C。

因此它比占位参数明显更适合搜索，但仍不是 HotSpot 的精确替代。`beta` 落在零
边界，说明在当前“核心固定于底层、L2只能合法放在顶层”的设计空间里，底层
功耗项与其他项高度相关；不能声称 `beta=0` 是通用物理结论。

## 5. lambda_wire 标定

使用同一 FFT 架构的 matched R2 对：fixed 的线延迟为 2 周期，中心候选为
1 周期，IPC2 分别为 3.413775829 和 3.417361953。于是：

\[
\frac{\Delta IPC_2}{\Delta cycle}=0.00358612444,
\]

并在该受限点 `f_sus=0.932477281 GHz`、`IPC1=3.474870777` 下得到：

\[
\lambda_{wire}
=\frac{f_{sus}(\Delta IPC_2/\Delta cycle)}{IPC_1}
=0.000962332063.
\]

原占位值 0.02 约大 20.8 倍，会让连续线长在目标函数中过度压制温度收益。
该数值目前只有一个工作负载、一个周期差的局部样本，因此其置信度低于热参数；
正式结论应在更多工作负载和 0/1/2/3 周期点上重复 R2 灵敏度实验。

## 6. 单点闭环检查

在 FFT 64 kB/1 MB 点上，最终 32x32 目标网格结果为：

| 方法 | L2位置 (mm) | HotSpot Tmax | f_sus | BIPS1热代理 | 离散平均线周期 |
|---|---|---:|---:|---:|---:|
| fixed | (0, 0) | 124.39 C | 0.932477 GHz | 3.240238 | 1 |
| 拟合热参数 + 拟合lambda | (2.631, 8.177) | 124.06 C | 0.940947 GHz | 3.269671 | 1 |

完全拟合后的候选相对 fixed 降温 0.33 C，热约束 BIPS1 提高约 0.908%。二者
的平均线延迟最终都舍入为 1 周期，因此该点不需要新的 R2；IPC2 可复用同一
离散延迟向量。16x16诊断中，若只拟合热参数却继续使用旧 `lambda=0.02`，
候选反而比 fixed 高 0.29 C、BIPS1 低 0.794%，证明两个参数组必须共同标定。
这个结果只完成了一个 pilot 点验证，还不能外推为论文表 VI 的平均布局收益。

## 7. 复现实验

热参数扫描：

```bash
source scripts/env.sh

python -m workflow.thermal.calibrate_proxy \
  --model fft_64k_1m=runs/validation/fft_64k_1m_geometry_v4/clip3d_5p0/modules.json \
  --model fft_32k_256k=runs/validation/fft_32k_256k_geometry_v4/fixed_3p5/modules.json \
  --model matmul_formal=results/calibration/models/matmul_64k_1m_formal/modules.json \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --output-dir results/calibration/thermal_proxy_5p0_3x3_20260728 \
  --grid-points 3 \
  --workers 1 \
  --hotspot-grid-size 16 \
  --spatial-weight 10 \
  --cross-weight-step 0.005 \
  --external-case fft_64k_1m_grid32=runs/validation/fft_64k_1m_geometry_v4/clip3d_5p0/layout_validation/fixed-bin/hotspot \
  --external-case fft_64k_1m_grid32=runs/validation/fft_64k_1m_geometry_v4/clip3d_5p0/layout_validation/optimized/hotspot
```

已有样本会按配置签名复用；只有加 `--force` 才重新运行 HotSpot。

线权重标定：

```bash
python -m workflow.r2.calibrate_lambda_wire \
  --baseline-result runs/pilot/fft_64k_1m_baseline_3p5/gem5_r2/r2_result.json \
  --candidate-result runs/pilot/fft_64k_1m_clip3d_5p0/gem5_r2/r2_result.json \
  --baseline-latency runs/pilot/fft_64k_1m_baseline_3p5/r2_latency.json \
  --candidate-latency runs/pilot/fft_64k_1m_clip3d_5p0/r2_latency.json \
  --ipc1 3.474870776816248 \
  --frequency-ghz 0.9324772810897376 \
  --output results/calibration/thermal_proxy_5p0_3x3_20260728/lambda_wire_report.json \
  --input-config results/calibration/thermal_proxy_5p0_3x3_20260728/suggested_config.json \
  --output-config results/calibration/thermal_proxy_5p0_3x3_20260728/fitted_config.json
```

主要输出：

- `calibration_report.json`：训练、留出、留一模型和32x32外部验证；
- `samples.csv`：21个布局的真实温度、默认预测和拟合预测；
- `lambda_wire_report.json`：R2有限差分及公式；
- `fitted_config.json`：可用于下一轮 pilot 的完整配置，不覆盖正式配置；
- `post_fit_validation.json`：拟合前后单点闭环对比。

下一步应选择 FFT、MATMUL、STENCIL、STREAM 各至少一个正式模块模型，运行
相同的低分辨率校准位置集，并补充每个工作负载至少两个离散延迟等级的 R2。
如果跨工作负载参数漂移仍然较大，应放弃单组全局系数，改为 HotSpot 热导矩阵
或矩形面积积分响应面，而不是继续增大 `alpha`。

## 8. 表 III 多锚点物理参数复核（2026-07-29）

单独用 FFT 64 kB/1 MB 把 `local_resistance_scale` 调到 8.72 会得到
124.39 C（目标 124.4 C），但独立锚点暴露出明显过拟合：

| 锚点 | 论文 P / gamma / Tmax | 旧流程 P / gamma / Tmax |
|---|---|---|
| MATMUL 128 kB/2 MB, Rconv=3.5 | 16.68 / 0.443 / 93.7 C | 15.47 / 0.497 / 94.83 C |
| FFT 32 kB/256 kB, Rconv=3.5 | 15.32 / 0.444 / 94.6 C | 15.79 / 0.442 / 99.64 C |
| FFT 64 kB/1 MB, Rconv=5.0 | 16.24 / 0.446 / 124.4 C | 16.24 / 0.446 / 124.39 C |

这证明功耗标定、硅横向扩散和 TIM 垂直热阻不能混进一个单点乘数。新增两个
工具：

- `thermal/calibrate_paper_anchors.py`：只用论文公开的 P、gamma、固定布局
  Tmax 拟合全局/按工作负载 McPAT 缩放，并从
  `Tlocal=Tmax-Tamb-Rconv*P` 给出共同热栈的一阶估计；
- `thermal/calibrate_stack_components.py`：分别对有效硅热阻和 TIM 热阻作
  +10% HotSpot 探针，构造 `dT/dlog(rho)` 灵敏度矩阵，再做最小二乘更新。

第一轮功耗结果为：全局后备 `dynamic_scale=1.046854`、
`leakage_scale=1.034221`；FFT 两锚点共同拟合为 `0.988417/1.045096`；
MATMUL 单锚点为 `1.199977/1.015652`。MATMUL 数值仅有一个锚点，必须标注为
局部校准，不能外推为论文参数。

运行命令及全部残差保存在：

```text
results/calibration/table3_crosscheck_20260729/
```

参数接受标准是三个固定布局锚点的温度与功耗误差，而不是优化布局相对 fixed 的
收益。这样可避免为了得到“显著提升”而反向调物理参数。

# 未缩放稳态热代理参数辨识

## 目的与证据边界

本实验为 strict-P1 稳态热代理辨识 `alpha` 与平面热特征长度 `Lc`。参考真值仅来自本地
CACTI、McPAT 与 HotSpot，不使用论文面积、温度或期望 BIPS 作为拟合标签。已有 gem5 R1
只读复用；本阶段不运行 R1、R2，也不拟合 `lambda_wire`。

## 物理输入合同

- 缓存面积和访问时间来自本地 45 nm CACTI characterization；
- 核逻辑和互连面积、动态功耗、亚阈值与栅极漏电来自未缩放 McPAT 输出；
- `global_scaling=none`，功耗无后处理乘数；
- HotSpot `local_resistance_scale=1.0`；
- 45 个工作点的原始 `r1_metadata.json` 和 `stats.txt` 在模型准备前后 SHA-256 不变。

## 相对温差与 beta 的处理

同一 strict-P1 工作点只移动 L2 的 `(x,y)`，模块功耗和层分配保持不变。以左下角布局
`x0` 为参考：

```text
Delta T_HS(x) = Tmax_HS(x) - Tmax_HS(x0)
Delta T_hat(x) = alpha * [H(x; Lc, wcross) - H(x0; Lc, wcross)]
```

环境温度、`Rconv * Ptotal` 和 `beta * Pbottom` 在差分中消去。因此本实验不拟合 beta，
但这不表示 beta 等于零；识别 beta 需要额外的跨层交换实验。

## 空间代理

```text
K(d; Lc) = 1 / sqrt(1 + (d/Lc)^2)
H(x; Lc, wcross) = max_i sum_j Pj * K(dij; Lc) * w(zi,zj)
```

同层 `w=1`，跨层 `w=wcross`。模块矩形使用二阶面积求积，不压缩为中心点。alpha 必须
非负，Lc 在 `0.02–4.0` 倍 die side 的对数空间搜索。

## HotSpot 输入修复与网格收敛

历史流程把功耗先离散成网格单元，再把这些单元作为 HotSpot floorplan blocks。该方式在
32×32 下与原生模块 floorplan 的 Tmax 完全一致，但会让 64/128 的 ptrace 过长，也增加
冗余。参数辨识现使用原生模块矩形与模块功耗，并用零功耗 whitespace 精确填满每层 die；
HotSpot 自身的 `grid_rows/grid_cols` 决定内部热网格。

在 MATMUL、L1D 64 kB、L2 512 kB 的六个布局上，以 128×128 为参考：

| Grid | 最大绝对 Tmax 差 | 排序一致 | 结论 |
|---:|---:|:---:|:---:|
| 32×32 | 0.274614 °C | 是 | 拒绝 |
| 64×64 | 0.068396 °C | 是 | 接受 |
| 128×128 | 0 °C | 参考 | 参考 |

预先冻结的阈值为 0.10 °C，因此后续正式案例使用 64×64。128×128 的六案例耗时约两小时，
只作为一次性收敛参考。

## 实验设计

1. 单位功率跨层响应：三种缓存尺寸、core/L2 两种源形状、上下两层、六个位置，共 72
   个 HotSpot 案例；用跨层/同层温升比的中位数确定 `wcross`，并进行整案例 bootstrap。
2. 真实功耗空间实验：五个负载与九个缓存组合，共 45 个工作点；每点 13 个确定性合法
   L2 布局，共 585 个 HotSpot 案例。
3. 对固定 `Lc` 用 Huber 损失拟合非负 alpha；对 `Lc` 做对数扫描与黄金分割精修。
4. 按整工作点 bootstrap 1000 次。置信区间采用冻结的 161 点对数 Lc 网格；正式点估计
   与留出折仍连续精修。
5. 验收仅使用整负载留出、整缓存组合留出和空间角点留出预测，不使用训练误差。

## 当前状态（2026-08-13）

- 45/45 个未缩放 CACTI/McPAT 模型已准备；
- 32/64/128 网格收敛已完成，选择 64；
- 原生模块级 64×64 的 72 案例单位响应正在运行；
- 585 案例真实功耗实验将在新的 `wcross` 通过稳定性门后恢复；
- alpha/Lc 尚未给出，避免在数据未完成或验收未通过时提前宣称参数可用。

结果根目录：

```text
results/parameter_studies/unscaled_alpha_lc_20260813/
```


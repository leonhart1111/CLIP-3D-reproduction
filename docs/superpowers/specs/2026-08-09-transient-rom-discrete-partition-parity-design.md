# 瞬态 ROM 与非零 λ 整数周期流程对齐设计

日期：2026-08-09  
状态：已获用户设计批准，等待实施计划  
分支：`feature/transient-rom`

## 1. 目标

在保留论文原始稳态流程和现有 `λ=0` 热模型隔离配置的前提下，新增一条非正式的瞬态 ROM 对比流程。该流程与当前稳态 5 点离散分区实验共享相同的非热设计变量、候选空间、通信权重、整数取整规则和 R2 延迟映射，只把稳态闭式热代理替换为经 8+2 个真实 HotSpot 作业校准和留出验证的瞬态 ROM。

最终实验必须对 fixed-bin 和 ROM 选出的 CLIP-3D 布局分别执行真实瞬态 HotSpot 持续频率验证，并在请求 R2 时分别执行或严格复用与各自延迟向量一致的 gem5 R2。只有两个分支都具备真实 HotSpot 和真实 gem5 证据时，才允许报告配对的 `BIPS2_trans` 改进百分比。

## 2. 科学分类与限制

本功能是研究性扩展，不是论文等价结果：

- `non_formal=true`；
- `paper_equivalent=false`；
- `shared_parameter_accepted=false`；
- 不得通过正式配置提升入口；
- 不得把 ROM 预测温度、预测频率或目标函数值表述为最终实测结果。

采用的

```text
lambda_wire = 0.0020119160767721133
```

来自 FFT 局部匹配 R2 的实验拟合。该值的原始报告已明确拒绝其正式或跨工作负载推广，原因包括 `R² < 0.95`、一次单调性违例以及尚无跨工作负载迁移验证。本设计使用它的目的仅是验证非零线长权重、通信加权、整数周期搜索和瞬态 ROM 能否构成闭合且可审计的流程。

## 3. 与稳态 5 点配置的对齐边界

新增配置应以当前稳态配置

```text
configs/experiments/
clip3d_constrained_5p0_raw_power_p1_lambda0020119_
traffic_weighted_discrete_partition_exploratory.json
```

为非热参数基准，并加入已验证的 `transient_rom` 设置。以下字段必须保持相等：

```text
frequency.f0_ghz                         = 2.0
frequency.fmin_ghz                      = 0.4
frequency.tsafe_c                       = 95.0
frequency.ambient_c                     = 25.0
physical.grid_size                      = 32
physical.utilization                    = 0.7
physical.r_convec_k_per_w               = 5.0
layout_optimizer.r_convec_k_per_w       = 5.0
layout_optimizer.alpha                  = 1.5643788695171585
layout_optimizer.beta                   = 0.0
layout_optimizer.cross_tier_weight      = 0.995
layout_optimizer.lambda_wire            = 0.0020119160767721133
layout_optimizer.allowed_l2_tiers       = [1]
layout_optimizer.wire_objective         = discrete-partition
layout_optimizer.partition_grid_steps   = 41
layout_optimizer.include_fixed_baseline = true
delay.wire_rounding                     = nearest
delay.wire_aggregation                  = traffic-weighted
```

`alpha`、`beta` 和 `cross_tier_weight` 保留在配置和 provenance 中，用于证明与稳态配置的来源一致，但不得进入 ROM 瞬态温度计算。ROM 取代稳态热代理是两条流程有意保留的核心差异。

虽然物理域仍限制 L2 位于 tier 1，此配置不能声明 `formal_validation.strict_p1=true`，因为 traffic-weighted 扩展和未验收的非零 λ 均不属于正式 strict-P1 方法。

## 4. 方案选择

### 4.1 采用方案

抽取一个共享的整数周期分区搜索内核，供稳态优化器和瞬态 ROM 优化器共同使用。共享内核负责：

1. `41×41` 坐标网格生成；
2. allowed tier 域遍历；
3. L2 几何构造和重叠判定；
4. fixed-bin 候选显式加入；
5. 连续线延迟到 gem5 整数周期的统一映射；
6. 以整数 R2 周期为键的分区；
7. 分区内最优候选和全局候选的确定性排序；
8. 搜索计数、分区和拒绝候选的审计记录。

稳态与 ROM 只提供各自的热频率评估回调：稳态回调使用当前闭式热代理，ROM 回调使用 POD/ROM 的瞬态持续频率预测。

### 4.2 未采用方案

- 不在 ROM 优化器中复制一份完整稳态搜索代码，以免后续取整或并列规则再次漂移。
- 不只重评估稳态代理筛出的少量代表点，因为同一整数周期分区中，稳态最优位置不保证也是瞬态 ROM 最优位置。
- 不在每次优化迭代中调用 HotSpot。校准固定为 8 个训练点和 2 个留出点，优化器内部 HotSpot 调用数必须为零。

## 5. 共享候选和整数周期定义

对每个允许 tier 和网格坐标构造 L2 候选：

```text
x_i = x_max * i / (partition_grid_steps - 1)
y_j = y_max * j / (partition_grid_steps - 1)
i,j = 0,...,40
```

候选与固定模块重叠面积大于 `1e-8 mm²` 时拒绝。fixed-bin 位置不依赖网格命中，必须额外加入并通过相同几何检查。

每个合法候选首先用共享的布局延迟函数计算四个核心到共享 L2 的连续线延迟，再使用 canonical R1 中的共享 L2 demand-access 通信计数归一化得到权重 `w_i`：

```text
sum_i w_i = 1
w_i >= 0
```

traffic-weighted 连续周期为：

\[
C_{\mathrm{wire,cont}}=\sum_{i=0}^{3} w_i C_{\mathrm{wire},i}.
\]

整数 R2 周期只能由共享 `round_wire_cycles(..., "nearest")` 计算：

\[
N_{\mathrm{wire}}=
\operatorname{round}_{\mathrm{nearest}}(C_{\mathrm{wire,cont}}).
\]

不得在 ROM 优化器、延迟向量构造器或报告代码中各自实现新的取整公式。

## 6. ROM 目标函数和分区选择

ROM 对每个合法候选预测瞬态持续频率
`f_sus_trans_rom(x,y,z)`，目标函数为：

\[
J_{\mathrm{ROM}}(x,y,z)=
-\mathrm{IPC}_1 f_{\mathrm{sus,trans}}^{\mathrm{ROM}}(x,y,z)
+\lambda_{\mathrm{wire}}\mathrm{IPC}_1N_{\mathrm{wire}}(x,y,z).
\]

搜索不是对不连续函数使用 L-BFGS-B。所有合法网格候选按整数
`N_wire` 分区，每个分区保留目标值最小的候选。确定性排序键为：

```text
(objective_loss,
 continuous_selected_wire_cycles,
 tier,
 y_mm,
 x_mm)
```

fixed-bin 与所有分区代表共同进入全局选择。这样 ROM 预测目标下的选中候选不会比 fixed-bin 候选更差；但最终真实 HotSpot 仍可能由于 ROM 误差改变两者的真实排序，报告必须保留并说明这种差异。

## 7. 完整数据流

1. 读取 canonical R1，并校验工作负载、缓存参数、四核心范围及 stats provenance。
2. 复用与 canonical R1 匹配的 2 ms 周期统计 R1；不得改动或重跑 canonical R1。
3. 从周期 R1 生成逐窗口 McPAT 原始动态功耗和漏电功耗。
4. 运行 8 个训练布局/功耗 HotSpot 作业，拟合 POD 连续状态空间模型。
5. 运行 2 个独立留出 HotSpot 作业，检查峰温误差、全网格 RMSE、PSS 和频率语义。
6. 只有 ROM 包通过全部门控后，才执行共享整数周期分区搜索。
7. 记录 fixed-bin 预测候选、全部整数周期分区、ROM 选中候选和搜索 identity。
8. 对 fixed-bin 和 ROM 选中布局分别运行最终真实瞬态 HotSpot PSS/持续频率搜索。
9. 分别从两个最终布局通过标准 R2 构造器生成延迟向量。
10. 对每个分支断言优化/报告中的整数周期与延迟向量使用的周期完全相等。
11. 仅在请求 `--run-r2` 时，分别运行或通过现有严格 provenance 校验复用 gem5 R2。
12. 两个分支均具备真实 HotSpot 和真实 R2 后，生成配对报告。

## 8. 配对指标

每个分支的最终分数为：

\[
BIPS2_{\mathrm{trans}}
=IPC2_{\mathrm{gem5}}\times
f_{\mathrm{sus,trans}}^{\mathrm{HotSpot}}.
\]

配对提升为：

\[
\Delta BIPS2_{\mathrm{trans}}(\%)=
\frac{BIPS2_{\mathrm{trans,CLIP3D}}
-BIPS2_{\mathrm{trans,fixed}}}
{BIPS2_{\mathrm{trans,fixed}}}\times100\%.
\]

ROM 预测值只用于解释搜索决策，不得代替公式中的真实 HotSpot 持续频率。lambda 目标函数值也不得被报告成 BIPS。

## 9. 产物和证据分层

建议在瞬态 ROM 输出根目录保留：

```text
steady_preflight/
transient_rom/
  windows/
  rom_package/
  optimization/
    partition_search.json
    proposed_layout.json
    optimization_report.json
  final_validation/
    fixed_bin/
      hotspot/
      r2_latency.json
      gem5_r2/                 # 仅 --run-r2
      branch_summary.json
    clip3d/
      hotspot/
      r2_latency.json
      gem5_r2/                 # 仅 --run-r2
      branch_summary.json
  paired_comparison.json       # 两个分支均完整时
  paired_comparison.csv        # 两个分支均完整时
  pipeline_summary.json
```

字段命名必须区分三类证据：

- `predicted_*`：ROM 搜索预测；
- `validated_*`：真实 HotSpot 验证；
- `measured_*`：gem5 R2 测量。

## 10. 失败门控

以下任一条件失败时，流程不得发布配对提升：

1. 缺少或无法归一化四核心共享 L2 通信计数；
2. 配置与被声明的稳态基准在关键非热字段上不一致；
3. `partition_grid_steps` 不是大于等于 3 的奇数；
4. `include_fixed_baseline` 不为 true；
5. ROM 8+2 数量、identity、artifact 哈希或留出门限不匹配；
6. ROM 优化阶段发生 HotSpot 调用；
7. fixed-bin 候选缺失或非法；
8. 优化器整数周期与 R2 延迟向量不一致；
9. 最终任一分支 HotSpot 工具失败、PSS 不收敛或频率搜索证据不完整；
10. 请求配对 BIPS 时任一分支缺少合法 R2；
11. 报告尝试用 predicted 值填充 validated/measured 字段。

失败输出应保留阶段、异常类型、输入 identity 和已有证据，不得留下看似成功的旧配对文件。

## 11. λ provenance 的可移植性

当前测试读取 `.gitignore` 覆盖的本地
`results/parameter_studies/.../lambda_wire_report.json`，导致独立 worktree 无法运行完整测试。

实施时应把该原始报告的不可变副本放入受 Git 跟踪的 provenance 目录，并保留原报告字段、拒绝结论和来源说明。非零 λ 的稳态探索配置、新 ROM 配置和对应测试统一引用该受跟踪证据。不得删除用户现有 `results/` 文件，也不得把拒绝状态改写为已验收。

## 12. 测试策略

所有行为修改遵循测试驱动开发：先写失败测试，确认因缺少目标行为而失败，再写最小实现。

至少覆盖：

1. 新 ROM 配置与稳态基准配置的关键字段逐项相等；
2. ROM 接受 `discrete-partition`，拒绝偶数网格、过小网格和禁用 fixed baseline；
3. 稳态和 ROM 调用同一个候选网格、重叠检查和整数周期分区内核；
4. 合成四核心通信计数下，traffic-weighted 连续值和 nearest 整数值正确；
5. 分区搜索确定性、fixed-bin 始终存在、并列规则稳定；
6. ROM 搜索结果的整数周期与标准 R2 延迟向量逐项一致；
7. ROM 优化器不导入或调用 HotSpot/subprocess 工具路径；
8. 最终 fixed-bin 和 CLIP-3D 都执行真实 HotSpot 验证；
9. 缺少任一真实 R2 时不生成最终提升百分比；
10. 两个真实分支完整时 `BIPS2_trans` 和改进百分比计算正确；
11. 默认稳态入口和现有 `λ=0` ROM 隔离配置行为不变；
12. 受跟踪 λ provenance 在独立 worktree 中可用；
13. ROM 专项、瞬态专项、主流程和完整测试发现集通过。

## 13. 验收标准

实现完成必须同时满足：

- 新配置明确为非正式且不能被正式提升；
- 非零 λ、traffic weights、nearest 取整、41×41 网格和 fixed baseline 均出现在运行证据中；
- 稳态与 ROM 共用整数周期分区实现；
- 优化器内部 HotSpot 调用数为零；
- 校准 HotSpot 固定为 8+2，最终双布局真实验证另计；
- fixed-bin 和 CLIP-3D 的最终持续频率均来自真实 HotSpot；
- 最终 IPC2 均来自 gem5 R2 或被严格验证的同向量缓存；
- 优化周期与 R2 周期 identity 完全一致；
- 配对报告不混用预测、验证和测量字段；
- 原稳态与 `λ=0` ROM 流程未被替换；
- 完整测试在独立 worktree 中无失败。

## 14. 非目标

本轮不处理：

- 重新拟合或正式提升 `lambda_wire`；
- 重跑 canonical R1；
- 允许核心模块移动；
- 扩展 L2 到 tier 0；
- 把 HotSpot 放入每个优化候选的内循环；
- 宣称瞬态 ROM 是论文原方法；
- 用单个 MATMUL 点证明跨工作负载收益。

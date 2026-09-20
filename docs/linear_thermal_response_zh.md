# LogicFolding 阶段一：非 ML 线性热响应流程

## 1. ML 在搜索流程中的真实位置

ML 不只负责“第 6 步排序和剪枝”。proposal 中 ML thermal predictor 的主要职责是：

```text
第 2 步：预测温度图并通过 autograd 生成 sensitivity map
第 5 步：用 sensitivity 估计候选动作的热代价
第 6 步：根据性能、能耗和热代价排序、剪枝
第 7 步：对保留候选快速重算温度；最终关键候选仍运行 HotSpot
```

阶段一完全不使用 ML。它用 HotSpot 有限差分建立同样的接口：

```text
HotSpot baseline
        ↓
module-power central finite difference
        ↓
linear response H
        ↓
temperature prediction + sensitivity map
        ↓
candidate ranking / Pareto pruning
        ↓
full floorplan + HotSpot queue
```

ML 后续只替换“快速 thermal predictor”，不会替换 architecture simulator、action generator、Pareto search 或最终 HotSpot。

## 2. 线性模型

固定物理合同下，阶段一使用：

\[
\mathbf T=\mathbf b+\mathbf H\mathbf P.
\]

其中：

- `P` 是按固定顺序排列的模块功耗向量；
- `T` 是两个 active tier 上的完整 32×32 温度向量；
- `H` 的单位是 K/W；
- `b` 是固定边界条件形成的温度截距。

对模块 `i` 运行正负功耗扰动：

\[
\mathbf H_{:,i}\approx
\frac{\mathbf T(\mathbf P+\delta P_i)-
      \mathbf T(\mathbf P-\delta P_i)}{2\delta P_i}.
\]

模块增加或减少的功耗按照与 CLIP 功耗栅格化相同的面积交叠权重分配到 32×32 grid，保证扰动总功耗严格等于 `delta_w`。

## 3. 已实现代码

| 文件 | 作用 |
|---|---|
| `workflow/thermal/linear_response.py` | 线性响应、平滑峰值、K/W sensitivity、保存和 hash 校验 |
| `workflow/thermal/build_linear_response.py` | 对既有 HotSpot baseline 执行模块级正负有限差分 |
| `workflow/thermal/linear_search.py` | 候选温度预测、热约束判断、Pareto front 和快速排序 |
| `workflow/thermal/run_linear_search.py` | 从候选 JSON 生成 top-K 完整评估队列 |
| `configs/thermal/linear_response_mvp.json` | 固定 2 tier、32×32、95 C、delta 和验收合同 |

## 4. 构建线性响应

线性响应实现使用 NumPy。请使用已经安装 `requirements-layout.txt` 的项目虚拟环境，例如：

```bash
source /home/zyjiang/Agenticflow/CLIP/.venv/bin/activate
```

首先需要一个已经运行完成的 HotSpot baseline case，至少包含：

```text
layout.json
power_grid.json
power.ptrace
hotspot_manifest.json
hotspot.config
stack.lcf
materials.txt
bottom.flp
top.flp
grid.steady.txt
```

然后运行：

```bash
cd /home/zyjiang/Agenticflow/CLIP/.worktrees/thermal-sensitivity

python3 -m workflow.thermal.build_linear_response \
  --baseline-case <completed-hotspot-case> \
  --output-dir runs/thermal_sensitivity/linear_response/<case-name> \
  --source-mode module \
  --delta-w 0.1 \
  --hotspot /home/zyjiang/Agenticflow/CLIP/tools/src/hotspot/hotspot \
  --workers 2
```

输出：

```text
linear_response.npz
linear_response.json
build_manifest.json
source_0000/positive/
source_0000/negative/
...
```

阶段一只支持 `module` source mode。完整 2048-cell response 需要处理大量零功耗 cell 的单边扰动，不在当前 MVP 中伪装成已完成能力。

## 5. 候选动作文件

architecture simulator 和 power adapter 后续应生成：

```json
{
  "schema_version": 1,
  "candidates": [
    {
      "id": "pe_plus_one_group_tier1",
      "architecture_action": "increase_pe",
      "tier_mapping": "new_pe_on_tier1",
      "power_w": [1.2, 0.8, 0.4],
      "performance": 112.0,
      "energy": 4.8,
      "area": 8.2,
      "communication": 1.7
    }
  ]
}
```

`power_w` 必须严格遵循 `linear_response.json` 中的 `source_names` 顺序。若 architecture action 新增了一个尚未存在于响应基中的模块，必须先把它映射到既有 module group，或者重新构建更完整的 source basis；不能静默追加维度。

## 6. 排序和剪枝

运行：

```bash
python3 -m workflow.thermal.run_linear_search \
  --response runs/thermal_sensitivity/linear_response/<case-name>/linear_response.json \
  --candidates <candidate-actions.json> \
  --config configs/thermal/linear_response_mvp.json \
  --output runs/thermal_sensitivity/search/<round>/linear_search.json
```

排序过程执行：

1. 计算候选模块功耗变化；
2. 用 `H` 预测完整温度图；
3. 计算 `Tmax` 和 `Tsoft`；
4. 以 95 C 判断预测热可行性；
5. 对 performance、energy、area、communication 和 Tsoft 做 Pareto 筛选；
6. Pareto candidates 优先进入 top-K；
7. top-K 被标记为必须运行完整 floorplan 和 HotSpot，而不是直接认定为最终结果。

## 7. 当前完成边界

已经完成：

- HotSpot 模块级正负扰动；
- 面积守恒的 module-to-grid perturbation；
- 线性响应矩阵；
- 平滑峰值敏感度；
- architecture action 热代价预测接口；
- 热约束筛选；
- Pareto front 和 top-K 队列；
- response 数组和合同 hash。

尚未完成：

- 实际 Gemmini/Chipyard counter；
- Gemmini power adapter；
- 自动生成 PE/SRAM/bandwidth action；
- 对新增模块自动生成 floorplan/tier mapping；
- top-K 到完整 floorplanner/HotSpot 的自动 orchestration；
- 真实 HotSpot sensitivity 实验结果；
- ML predictor。

## 7.1 保留候选的 HotSpot 验收

`run_linear_search.py` 只负责快速预测和剪枝。对被选中的候选，必须调用
`validate_linear_response.py`：它复制 baseline 的几何、封装、材料、边界和
HotSpot 配置，只替换候选功耗 trace，然后重新运行完整 3-D HotSpot。程序会
逐点比较预测温度向量和 HotSpot 的 `grid.steady.txt`，并报告：

- 温度图 MAE 和最大绝对误差；
- `Tmax` 误差；
- 平滑峰值 `Tsoft` 误差；
- 是否满足验收门限。

例如对排序报告中的候选执行：

```bash
python3 -m workflow.thermal.validate_linear_response \
  --response runs/thermal_sensitivity/linear_response/<case>/linear_response.json \
  --candidates runs/thermal_sensitivity/search/<round>/linear_search.json \
  --output runs/thermal_sensitivity/validation/<round> \
  --hotspot /home/zyjiang/Agenticflow/CLIP/tools/src/hotspot/hotspot \
  --workers 4
```

`--candidates` 可以是原始 candidate document，也可以是
`run_linear_search.py` 生成的报告；两者均使用同一个 `candidates` 数组。对于
search report，验证器默认只读取其中的
`selected_for_full_evaluation_ids`；只有显式加 `--all-candidates` 才会把被剪枝
的候选也作为诊断重跑。
验证输出中的每个候选目录保留了实际 HotSpot 输入和输出，便于复核。该步骤
仍不是新的 floorplan 结果：如果候选改变了模块面积、坐标、tier、封装或
网格合同，必须先由上游 floorplanner 生成新的 case，并重新构建线性响应。

所以当前实现完成的是非 ML 搜索核心，而不是完整的 Gemmini Architecture DSE。

## 8. 后续与 ML 的接口

当前接口为：

```text
power vector → temperature map → sensitivity → candidate score
```

未来 ML 版本保持相同接口：

```text
power map + masks → U-Net temperature map → autograd sensitivity → candidate score
```

因此 `linear_search.py` 的 candidate、Pareto 和 pruning 逻辑无需重写；只需要把 `LinearThermalResponse` 替换为经过验收的 ML predictor adapter。

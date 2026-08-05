# CLIP-3D lifting 流水线：脚本、公式与运行方法

本文说明 gem5 R1 之后还需要做什么，以及本项目中新补齐的程序如何对应论文方法。这里的 “lifting（提升）” 可以理解为：把架构模拟器中的抽象缓存容量、访问次数和 IPC，转换成有面积、有功耗、有物理位置、有温度和实际延迟的三维芯片模型。

## 1. 为什么不能在 gem5 R1 后直接结束

gem5 只知道“这条指令用了多少模拟周期”，并不知道模块在硅片上的面积、发热位置和层间连接。论文的完整闭环还需要：

1. 将 `stats.txt` 转成 McPAT 活动统计，得到模块面积、动态功耗和泄漏功耗；
2. 用 CACTI 得到每种 L1/L2 的物理访问时间；
3. 把核心、L1 和 L2 放在两层芯片上；
4. 把不规则模块矩形切分到规则网格，生成 HotSpot 能读取的 `.flp`、`.ptrace`、`.lcf`；
5. 用 HotSpot 得到额定 2 GHz 下的最高温度；
6. 用闭式公式算可持续频率；
7. 从最终布局推导 TSV 跳数和 Bakoglu–Meindl 面内线周期，与 CACTI、仲裁延迟一起写回 gem5，运行 R2 得到 `IPC2`；
8. 计算最终 `BIPS2 = IPC2 × f_sus`，再做架构排名。

## 2. 最简单的端到端命令

先用已成功的 smoke 点验证整条流水线：

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh

python -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/smoke/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/lifting_smoke/matmul_baseline
```

这条命令不会再次运行 R1，也不会默认启动耗时的 R2。它依次完成 McPAT、CACTI、布局、32×32 分块、HotSpot、持续频率和 R2 参数生成。当前机器上，主要耗时是正式 32×32 五层 HotSpot 求解，单点约需一到两分钟。

输出摘要在：

```text
runs/lifting_smoke/matmul_baseline/pipeline_summary.json
```

如果要使用公式(14)-(15)的解析式布局搜索，而非固定分箱：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/smoke/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/lifting_smoke/matmul_optimized \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method clip3d
```

如果还要立即运行第二次 gem5：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/smoke/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/lifting_smoke/matmul_closed_loop \
  --run-r2
```

正式论文点使用 1 亿指令预热和 5 亿指令测量，R2 会和 R1 一样耗时；建议在 `tmux` 或作业调度器中运行。

## 3. 每一步在做什么

### 3.1 gem5 → McPAT

程序：

```text
workflow/mcpat/gem5_to_mcpat.py
workflow/mcpat/parse_mcpat.py
```

转换器读取 `r1_metadata.json` 和 `stats.txt`，为四个核心分别填写周期数、指令类型、分支、L1 访问和 L1 未命中，并为共享 L2 填写访问和未命中。生成的是 45 nm、2 GHz、四发射、192 项 ROB 的四核心 XML。x86 活动量使用 gem5 的 `commitStats0.numOps`（微操作），不再使用不存在的 `opsCommitted`；ROB、rename、dispatch、wakeup 和物理寄存器访问优先使用 gem5 O3CPU 直接计数器。

论文没有公开 McPAT XML。正式配置把 320 K、ITRS-HP、long-channel、conservative interconnect、`opt_for_clk=0` 明确记录为本地建模假设。`mcpat/mcpat.json` 直接保存 McPAT 报告的逐模块动态功耗和泄漏功耗，后续流程不使用论文结果拟合的功耗乘数。

当前实际 McPAT 详细结果通常被拆成 34 个物理模块：

- 每个核心拆为 IFU、rename、LSU、MMU、execution、other，并保留独立的
  L1I、L1D，共 32 个；
- 一个共享 L2；
- 一个互连块。

若简化 McPAT 输出缺少这些详细标题，解析器才回退为每核心一个聚合逻辑块。

每个模块都有：

```text
area_mm2
dynamic_power_w
subthreshold_leakage_w
gate_leakage_w
leakage_power_w
total_power_w
```

核心逻辑功耗通过“核心总量减 L1I、L1D”得到，所以不会重复计算缓存功耗。所有 McPAT 字段映射和不可直接观测的假设都保存在 `mcpat/mapping_report.json`。

### 3.2 面积校准

程序：`workflow/floorplan/build_module_model.py`

论文将参考设计（4 核、32 kB L1D、512 kB L2、45 nm）校准为 150 mm²。本项目现在只使用本地 CACTI 运行得到的缓存面积和延迟，非缓存逻辑面积采用 McPAT。参考原始面积为 `41.783078+3.9707715872=45.7538495872 mm²`，因此面积统一乘以：

```text
scale = 150 / 45.7538495872 = 3.278412665892
```

只缩放面积，不缩放 McPAT 功耗。L1/L2 保留本机 CACTI 输出的面积和长宽比；因此 L2 不再被错误地强制为正方形，也不再套用论文表 II 面积。70% 是后续布局利用率，不能先把 150 mm² 乘以 70%；否则会重复应用利用率。面积来源、缩放前面积和宏块长宽均写入 `modules.json`，后续若更换 McPAT 模板或 CACTI 配置，必须重新计算，而不能继续沿用。

### 3.3 CACTI 延迟

程序：`workflow/cacti/characterize_cache.py`

脚本为每个容量单独生成配置，固定 45 nm、64 B cache line、L1 两路、L2 八路，然后在 CACTI 源码目录中执行。必须从源码目录执行，因为这个版本按相对路径读取 `tech_params/45nm.dat`；从输出目录运行会段错误。缓存访问时间、面积、长宽、读写能量和漏电功耗均来自本地 CACTI 标准输出；项目中已经删除论文表 II 覆盖分支。

CACTI 输出的 ns 延迟按 2 GHz（每周期 0.5 ns）转换为周期：

```text
cycles_float = access_time_ns / 0.5
cycles_int   = floor(cycles_float + 0.5)
```

即四舍五入到最近整数，且至少为 1 周期。

### 3.4 自动芯片边长和 P1 固定分箱

程序：`workflow/floorplan/generate_hotspot_inputs.py`

模块先按 P1 规则分层：

- tier 0（底层）：4 个核心逻辑及其私有 L1；
- tier 1（顶层）：共享 L2 和互连。

设每一层模块面积之和为 `A_z`，利用率为 `u=0.70`，方形芯片边长为：

```text
die_side = sqrt(max(A_0, A_1) / u)
```

四个核心簇放进底层四个象限；共享 L2 从顶层左下角开始进行确定性 shelf packing，与论文布局实验的固定基线一致。模块没有覆盖的区域仍由网格硅单元填满，但功耗为零；这就是论文所说的“死硅填充”，它保留横向导热，而不是绝热空洞。

### 3.5 模块分块与功耗栅格化

这是用户特别提到的步骤，也是 `generate_hotspot_inputs.py` 的核心。

每层被切成 32×32 个完全相同的网格。一个模块往往同时覆盖多个网格，某个网格分到的功耗等于“重叠面积占模块面积的比例”乘模块总功耗：

```text
cell_power += overlap(module, cell) / module_area × module_power
```

例如，一个 2 W 模块有 25% 面积落在某网格，该网格就从它获得 `2×25%=0.5 W`。动态功耗、泄漏功耗和总功耗分别执行同一套计算。

脚本强制检查每层：

```text
sum(cell_power) == sum(module_power)
```

误差阈值为 `1e-10 W`；不守恒会直接报错，而不会继续给 HotSpot 错误输入。检查结果写入 `hotspot/hotspot_manifest.json`。

### 3.6 HotSpot 五层输入

生成文件：

```text
hotspot/bottom.flp
hotspot/top.flp
hotspot/power.ptrace
hotspot/power_dynamic.ptrace
hotspot/power_leakage.ptrace
hotspot/stack.lcf
hotspot/hotspot.config
hotspot/materials.txt
```

`stack.lcf` 是五层：被动中介层、底层有源硅、TIM、顶层有源硅、顶部 TIM。有源硅厚度为论文给出的 50 μm。`.ptrace` 的列顺序与两个有源 `.flp` 的模块顺序完全相同，避免 HotSpot 常见的模块数量或列错位问题。

论文只说明“Cool-3D 默认封装”，没有给出全部层材料数值。`physical.thermal_stack.local_resistance_scale` 因此作为显式的有效局部热阻参数写入配置和 `hotspot_manifest.json`；它只影响硅/TIM 层热阻，不修改 McPAT 报告的动态功耗或泄漏功耗。该参数属于热堆叠假设，必须和功耗来源分开报告。

`workflow/thermal/run_hotspot.py` 用 `grid + detailed_3D` 模式求稳态，并从所有报告单元中取最大温度。

### 3.7 泄漏比例和持续频率

程序：`workflow/thermal/sustainable_frequency.py`

泄漏比例为：

```text
gamma = total_leakage / (total_leakage + total_dynamic)
```

若 2 GHz 下 `Tmax <= Tsafe`，持续频率仍是 2 GHz。否则使用论文公式(13)：

```text
f_sus = max(f_min,
            f0/(1-gamma) × ((T_safe-T_amb)/(Tmax(f0)-T_amb) - gamma))
```

默认 `f0=2 GHz`、`f_min=0.4 GHz`、`T_safe=95°C`、`T_amb=25°C`。程序同时保留截断前的频率和状态：有热余量、热限制或碰到最低频率。

### 3.8 R2 延迟向量

程序：

```text
workflow/r2/build_latency_vector.py
workflow/r2/run_r2.py
```

论文公式(6)中的附加项为：

```text
L = CACTI cache cycles + (Ncores-1) + 2×TSV_hops + 1
```

布局协同实验还加入单独的面内项：

```text
wire_delay = 0.69 × R × C × ManhattanDistance²
```

四核心时，L2 仲裁是 3 周期；TSV 跳数由最终核心/L2层号推导，每跳2周期；L1流水线是1周期。面内延迟按配置的策略离散成整数周期。生成的 `r2_latency.json` 保存每核心距离、浮点线周期、离散周期和最终 `gem5_args`。R2具有独立状态文件，会检查四个核心的指令和周期均有效。最终：

```text
BIPS2 = IPC2 × f_sus_GHz
```

因为 IPC 的单位是“指令/周期”，GHz 是“十亿周期/秒”，两者相乘正好是“十亿指令/秒”。

### 3.9 解析式布局器

程序：`workflow/floorplan/optimize_layout.py`

布局器枚举 L2 在两个 tier 上的选择，并从左下、中心、右上三个初值搜索 `(x,y)`。目标包含：

- 公式(14)的平滑热代理；
- 公式(13)的闭式持续频率；
- 45 nm Bakoglu–Meindl 线延迟；
- 非重叠约束。

项目内 `.venv` 已固定安装 SciPy 1.15.3，正式配置要求论文的 L-BFGS-B；若环境未正确激活会直接报错，而不会静默更换求解器。`optimizer_report.json` 记录求解器和每个候选。

热代理只用于提出候选。由于论文没有公开代理系数，而且实测表明代理梯度可能与 detailed-3D HotSpot 不一致，流水线会对 optimized 和 fixed-bin 各运行一次真实 HotSpot。选择时先比较 `IPC1×f_sus`；仅在数值相同时再比较 R2 真正使用的离散平均线延迟，最后比较真实 Tmax。若 optimized 更差则回退 fixed-bin，并在 `layout_selection.json` 中记录原因。该保护是复现工程的防回归措施，不宣称为论文原算法；它避免把代理失配误报成布局收益。

## 4. 正式扫描与排名

正式流程包括两种冷却条件、严格R2、表III锚点和四种布局方法，且第二种固定布局冷却条件可以在延迟向量完全相同时复用第一次R2。为避免把代理BIPS、不同热阻或不同布局方法混在一起，完整命令统一列在 `docs/formal_reproduction_zh.md`。

## 5. 输出文件含义

| 文件 | 含义 |
|---|---|
| `mcpat/input.xml` | 从 gem5 统计生成的四核心 McPAT 输入 |
| `mcpat/mcpat.json` | 可机读的逐模块面积和功耗 |
| `cacti/cacti_characterization.json` | CACTI ns、周期、能耗和面积 |
| `modules.json` | 统一模块模型、IPC1、面积校准和 gamma |
| `hotspot/layout.json` | 每个模块的 `(x,y,w,h,tier)` |
| `hotspot/power_grid.json` | 每个单元的动态/泄漏/总功耗 |
| `hotspot/hotspot_manifest.json` | HotSpot 文件路径、论文参数、假设和守恒残差 |
| `hotspot/thermal_result.json` | Tmax、峰值单元和求解命令 |
| `performance.json` | 公式(13)、持续频率和 BIPS |
| `r2_latency.json` | 公式(6)分项与可直接使用的 gem5 参数 |
| `pipeline_summary.json` | 单点最终摘要 |

## 6. 测试

```bash
source scripts/env.sh
python -m unittest discover -s tests -v
```

当前测试包括公式(13)频率公式、McPAT/CACTI解析、本地CACTI几何传递、固定布局、面积重叠功耗守恒、布局延迟、冷却一致性、配置变更续跑守卫、严格R2汇总守卫、三候选对照布局，以及真实HotSpot频率缩放案例。

## 7. 必须明确的复现边界

论文没有公开原始源码，因此以下内容是显式复现假设，而不是声称来自论文：

- gem5 计数器到少数没有一对一计数器的 McPAT 活动字段的估算关系；
- McPAT 320 K 器件工作点和未公开 XML 字段的活动映射；
- 本机 CACTI 版本和组织参数与论文未公开设置之间可能存在差异；
- 当前 McPAT 版本为避免 CACTI-P 无有效阵列组织而保留的 32-bit 地址宽度；
- 被动中介层厚度 100 μm；
- 由表 III 内部温升锚定的有效局部热阻系数；
- 公式(14)-(15)未公开的 `alpha`、`beta`、跨层权重和 `lambda_wire` 数值；
- 将拓扑罚时放在 gem5 xbar `forward_latency` 的具体反标方法。
- Cool3D-standard 与 SA+λ 的论文实现和搜索参数未公开，本项目提供确定性、完整留痕的独立对照复现；
- 当前正在运行的正式 R1 使用 CPU0 指令停止锚点；另提供独立的 `paper_all_cores` 协议，但两种口径不能混合。

这些假设全部写入配置或每点 JSON。论文明确给出的 45 nm、核心数、频率、关联度、网格、利用率、硅厚、温度限制、冷却热阻和 TSV 周期则单独标为 paper parameters。

最后，smoke 结果只证明程序连通。正式复现论文数值仍需完成 100 个 R1 点和相应 R2；短 smoke 工作负载的功耗和温度不能与论文表格直接比较。

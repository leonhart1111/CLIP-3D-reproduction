# Gemmini 类 systolic-array testbench 选择与构建说明

## 1. 这一步解决什么问题

LogicFolding 的第一版热敏感度实验不能直接从完整 transformer 开始。需要先固定一个可重复的 accelerator family、workload、功耗输出格式和 HotSpot 映射方式。

本分支首先冻结以下 MVP：

| 项目 | 当前选择 |
|---|---|
| accelerator family | Gemmini 类二维 systolic array |
| 生成器 | Berkeley/UC Berkeley 的 Gemmini Chisel generator |
| 系统集成 | Chipyard + Rocket 核 + Gemmini RoCC accelerator |
| 默认 mesh | 16×16 PE |
| dataflow | output-stationary |
| scratchpad | 256 KiB |
| DMA bus | 16 B |
| 热网格 | 两个 active tier，各 32×32 |
| 第一条 smoke workload | 64×64×64 GEMM |
| 主要敏感度 workload | 256×256×256 GEMM |
| 可选 workload | 小型 3×3 convolution |
| transformer | 暂定为自定义 tiled-GEMM chain，尚未声称有官方 binary |

配置文件是：

```text
configs/accelerators/gemmini_logicfolding_mvp.json
```

它只描述 testbench 合同，不包含 Gemmini 或 Chipyard 源码。第三方工具仍然作为外部依赖，避免把上游仓库复制到 CLIP 中。

## 2. Gemmini 到底是什么

Gemmini 不是一个已经固定好的商用芯片，而是一个可以生成不同硬件实例的 Chisel generator。它通常被接入 Chipyard，作为 Rocket 核的 RoCC accelerator。

一个具体 Gemmini instance 由以下参数决定：

- PE array 的行数和列数；
- systolic dataflow，例如 output-stationary 或 weight-stationary；
- scratchpad 容量和 bank 数量；
- accumulator 容量；
- DMA bus 宽度；
- 数据类型，例如 int8、int16 或更高精度；
- 时钟频率和 memory hierarchy。

因此本项目中的 `16×16 + 256 KiB scratchpad` 是一个 architecture instance，不是“Gemmini 的唯一架构”。

## 3. 推荐的开源仓库

官方仓库：

- Gemmini：<https://github.com/ucb-bar/gemmini>
- Chipyard：<https://github.com/ucb-bar/chipyard>

Gemmini 负责：

- 生成 systolic-array RTL；
- 生成 scratchpad、accumulator 和 DMA 结构；
- 提供 Gemmini C/assembly API；
- 提供 `gemmini-rocc-tests` 等测试程序。

Chipyard 负责：

- 将 Gemmini 接入 Rocket/BOOM 等处理器；
- 配置 SoC 结构；
- 生成 Verilator、VCS 等仿真目标；
- 提供 RISC-V 工具链和系统级仿真环境。

二者的关系是：

```text
Gemmini generator
        ↓
生成 accelerator RTL
        ↓
Chipyard 集成 Rocket + Gemmini
        ↓
Verilator/VCS 仿真
        ↓
Gemmini workload binary
        ↓
cycles / counters / activity
```

不要把 Gemmini 当成像 gem5 一样的独立 cycle simulator。Gemmini 本身主要是硬件生成器；需要 Chipyard 和 Verilator/VCS 才能形成完整的 RTL 仿真流程。

## 4. 如何生成 testbench build plan

在新分支 worktree 中运行：

```bash
cd /home/zyjiang/Agenticflow/CLIP/.worktrees/thermal-sensitivity

python3 -m workflow.accelerator.testbench \
  --manifest configs/accelerators/gemmini_logicfolding_mvp.json \
  --suite smoke \
  --external-root /home/zyjiang/Agenticflow/CLIP/external \
  --output runs/accelerator/gemmini_smoke_selection.json \
  --print-plan
```

也可以使用脚本入口：

```bash
python3 scripts/select_gemmini_testbench.py \
  --manifest configs/accelerators/gemmini_logicfolding_mvp.json \
  --suite sensitivity_mvp \
  --external-root /home/zyjiang/Agenticflow/CLIP/external \
  --output runs/accelerator/gemmini_sensitivity_selection.json \
  --print-plan
```

这个命令只做三件事：

1. 校验 accelerator 和 workload 合同；
2. 选择指定 suite 中的 workload；
3. 输出带 manifest hash 的 build plan。

它不会自动下载或修改第三方仓库。

## 5. smoke suite 与 sensitivity suite

### smoke

```text
gemm_64
```

目的只是确认：

- Chipyard 能生成指定 Gemmini 配置；
- RISC-V binary 可以启动；
- Gemmini 指令可以执行；
- 能够记录 cycle 和 instruction counter；
- 能够将结果转换成 CLIP 的 module/power contract。

### sensitivity_mvp

```text
gemm_64
gemm_256
conv2d_8x8
```

其中：

- `gemm_64`：重复 smoke，检查小规模 workload 的稳定性；
- `gemm_256`：主要 compute-heavy thermal sensitivity workload；
- `conv2d_8x8`：检查 memory traffic、reuse 和 compute 的差异。

### transformer_planned

```text
transformer_mlp_tiny
```

这项目前只在 manifest 中占位，不能直接运行。Gemmini 没有一个跨版本、通用、可以直接假设存在的 transformer binary。正确做法是把 transformer 拆成 tiled GEMM 序列，例如：

```text
Q = X × Wq
K = X × Wk
V = X × Wv
Attention = softmax(QKᵀ) × V
FFN1 = X × W1
FFN2 = GELU(FFN1) × W2
```

第一版可以先实现 QKV 和 FFN 的 GEMM 部分；softmax、GELU 和 layer normalization 暂时作为 CPU-side reference 或单独的 non-systolic kernel。每一个阶段都必须有独立的 cycle、功耗和结果校验，不能把一个未验证的 transformer 程序直接当成 Gemmini benchmark。

## 6. 上游构建的一般流程

由 build plan 输出的上游流程通常如下：

```bash
mkdir -p /home/zyjiang/Agenticflow/CLIP/external
cd /home/zyjiang/Agenticflow/CLIP/external

git clone --recursive https://github.com/ucb-bar/chipyard.git chipyard
git clone --recursive https://github.com/ucb-bar/gemmini.git gemmini

cd chipyard
./build-setup.sh
source env.sh

# 具体 make 目标和 CONFIG 名称必须以所 pin 的 Chipyard 版本为准
make -C sims/verilator CONFIG=GemminiRocketConfig
```

Gemmini 软件测试通常位于 Chipyard/Gemmini 集成树的 `gemmini-rocc-tests` 目录。不同 commit 的目录和 build target 可能变化，因此本项目的 manifest 只保存逻辑 entrypoint，例如 `matmul` 和 `conv`，不会把某个未经验证的 binary 文件名写死。

正式实验之前必须记录：

```bash
git -C /home/zyjiang/Agenticflow/CLIP/external/chipyard rev-parse HEAD
git -C /home/zyjiang/Agenticflow/CLIP/external/gemmini rev-parse HEAD
```

并把两个 commit 写入 testbench selection manifest。

## 7. 如何与 CLIP 热敏感度流程连接

Gemmini/Chipyard 仿真本身不直接输出 HotSpot 温度。数据链路应该是：

```text
Gemmini/Chipyard workload
        ↓
cycle / instruction / activity counters
        ↓
power adapter
        ↓
module-level dynamic/leakage/total power
        ↓
two-tier floorplan
        ↓
32×32 power grid per active tier
        ↓
HotSpot baseline
        ↓
finite-difference thermal sensitivity
        ↓
architecture/floorplan candidate ranking
```

Gemmini 输出的 activity 必须先经过独立 power adapter，不能把 cycle 数直接当成瓦特，也不能把 Gemmini 的 scratchpad 容量直接当成面积。面积、功耗和模块几何都必须注明来源：

- RTL/counter-derived：来自仿真计数器；
- analytical：来自 Accelergy、CACTI 或参数模型；
- measured/calibrated：只有有外部锚点时才能使用这个标签。

## 8. 当前明确没有实现的部分

本次 testbench 骨架暂时没有完成：

- Gemmini/Chipyard 的具体 commit pinning；
- RISC-V 工具链安装；
- Verilator binary 的实际构建；
- `gemm_64`、`gemm_256` 的真实 counter 采集；
- transformer custom tiled-GEMM binary；
- Gemmini counter 到 dynamic/leakage power 的正式适配器；
- GPU/NPU/CPU 与 Gemmini 的多 endpoint 并行仿真。

因此当前分支可以验证 testbench 选择、合同和 build plan，但还不能宣称 Gemmini 已经接入完整 CLIP 热闭环。

## 9. 下一步建议

下一步应按以下顺序推进：

1. 选择并固定 Chipyard 和 Gemmini commit；
2. 实际生成 `GemminiRocketConfig` Verilator target；
3. 只跑 `gemm_64` smoke；
4. 记录 cycle、instruction 和 accelerator counter；
5. 建立一个最小 Gemmini power adapter；
6. 运行 `gemm_256` 并生成 32×32 双 tier 功耗图；
7. 对同一功耗图做 HotSpot finite-difference sensitivity；
8. 最后再实现 transformer GEMM chain。

这样做的原因是：如果一开始就加入 transformer、多个 mesh、多个 tier 和多个 operating point，出现问题时无法区分到底是 Gemmini binary、Chipyard 配置、功耗适配器还是 HotSpot 映射出了错误。

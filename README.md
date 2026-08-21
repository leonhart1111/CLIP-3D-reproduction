# CLIP-3D 论文复现工程

本目录用于复现 CLIP-3D 的架构—功耗—缓存—热—布局闭环。四个基础工具的源码不会混放在工作流代码中，而是统一下载到 `tools/src/`。

## 快速开始

```bash
cd /home/zyjiang/Agenticflow/CLIP
source ./scripts/env.sh
./scripts/check_tools.sh
python3 -m unittest discover -s tests -v
```

工具源码和可执行文件已经位于 `tools/src/`。上述命令检查环境并运行小规模测试，
不会启动耗时的正式 100 点 gem5 扫描。

详细说明见 [docs/DOWNLOAD_TOOLS.md](docs/DOWNLOAD_TOOLS.md)。

## 目录结构

```text
CLIP/
├── benchmarks/                 # 工作负载源码、输入和编译结果
│   ├── src/{splash2,stream,matmul,stencil}/
│   ├── bin/
│   └── inputs/
├── configs/                    # 工具配置和论文实验参数
│   ├── {gem5,mcpat,cacti,hotspot}/
│   ├── architecture/
│   └── experiments/
├── data/                       # 原始、中间和清洗后数据
│   ├── raw/
│   ├── intermediate/
│   └── processed/
├── docs/                       # 下载、构建和复现文档
├── manifests/                  # 版本、命令、配置和结果清单
├── results/                    # 最终表格、图、布局和检查点
│   ├── {tables,figures,layouts,checkpoints}/
├── runs/                       # smoke test、架构扫描和布局扫描
│   ├── {smoke,architecture_sweep,layout_sweep}/
│   ├── logs/
│   └── tmp/
├── scripts/                    # 下载、环境和检查脚本
├── tests/                      # 单元测试和小规模端到端测试
├── tools/
│   ├── src/{gem5,mcpat,cacti,hotspot}/
│   ├── build/{gem5,mcpat,cacti,hotspot}/
│   ├── install/bin/
│   └── versions/               # 实际下载的 commit 记录
└── workflow/                   # CLIP-3D 自己的流水线实现
    ├── mcpat/
    ├── cacti/
    ├── floorplan/
    ├── thermal/
    ├── r2/
    └── analysis/
```

## 四个工具在流程中的角色

| 工具 | 本项目中的用途 | 源码目录 |
|---|---|---|
| gem5 | R1/R2 架构模拟，输出 IPC 和活动统计 | `tools/src/gem5` |
| McPAT | 从活动统计估计模块功耗和面积 | `tools/src/mcpat` |
| CACTI | 表征 L1/L2 访问时间和面积 | `tools/src/cacti` |
| HotSpot | 求解两层三维堆叠的稳态温度，以及可选的瞬态温度轨迹 | `tools/src/hotspot` |

## 单点 lifting 快速运行

```bash
cd /home/zyjiang/Agenticflow/CLIP
python3 -m workflow.run_lifting_pipeline \
  --r1-dir runs/architecture_sweep/r1/smoke/matmul/l1d_32kB/l2_512kB \
  --output-dir runs/lifting_smoke/matmul_baseline
```

完整方法说明见 [docs/clip3d_pipeline_zh.md](docs/clip3d_pipeline_zh.md)，严格正式执行顺序见 [docs/formal_reproduction_zh.md](docs/formal_reproduction_zh.md)，可选10 ms瞬态热仿真见 [docs/transient_thermal_zh.md](docs/transient_thermal_zh.md)。热代理和线延迟参数的独立验证方法见 [docs/surrogate_parameter_validation_zh.md](docs/surrogate_parameter_validation_zh.md)。若上述文档与 [docs/semantic_roi_protocol_zh.md](docs/semantic_roi_protocol_zh.md) 的测量口径冲突，以后者为准。

### HotSpot 功耗输入粒度

稳态主流程可显式选择 HotSpot 的功耗输入方式：

```bash
# 论文兼容默认：先按面积重叠栅格化为每层 32×32 功耗格
python3 -m workflow.run_lifting_pipeline ... \
  --hotspot-input-granularity grid-cell

# 工程诊断：保留真实模块矩形，交由 HotSpot 映射到其内部热网格
python3 -m workflow.run_lifting_pipeline ... \
  --hotspot-input-granularity module
```

两种方式的结果不能混合。严格 P1 配置固定为 `grid_size=32` 与
`input_granularity=grid-cell`；`module` 是非论文严格的工程选项，必须与它自己的
参数辨识、HotSpot 合同和验收结果一起报告。

## 热代理梯度诊断（non-formal）

`workflow.thermal.diagnose_proxy_gradient` 是热代理的主交互入口。它固定一个
`modules.json` 与 HotSpot 物理合同，在合法 L2 位置上运行共同的 HotSpot 网格，并比较
Equation (14) 变体的基础温度梯度；它不拟合参数、不运行 gem5/McPAT/R2，也不把结果称为
论文等价优化。最终布局温度仍应使用完整流水线的 HotSpot 结果。

```bash
cd /home/zyjiang/Agenticflow/CLIP
source ./scripts/env.sh
python -m workflow.thermal.diagnose_proxy_gradient \
  --modules <existing-modules.json> \
  --config configs/experiments/clip3d_unscaled_stressed_proxy_gradient_diagnostic.json \
  --output-dir runs/proxy_gradient/<case-name> \
  --variant fitted-area=area-quadrature,0.058600738908940346 \
  --variant fitted-center=center,0.058600738908940346 \
  --variant paper-center=center,0.5 \
  --variant paper-area=area-quadrature,0.5 \
  --grid-points 3 --workers 2
```

若命令在 HotSpot 求解期间中断，以**完全相同的参数**增加 `--resume`；仅合同匹配的完整
探针会被复用。多个点完成后使用下列入口汇总，必须将拟合集内点与架构留出点标为不同标签：

```bash
python -m workflow.thermal.summarize_proxy_gradient_series \
  --report l2-holdout=runs/proxy_gradient/<held-out-case>/gradient_diagnostic.json \
  --report in-fit-regression=runs/proxy_gradient/<in-fit-case>/gradient_diagnostic.json \
  --output runs/proxy_gradient/series_summary.json
```

汇总器拒绝混用不同的 config、HotSpot、网格、允许 tier 或代理变体。诊断的科学边界、已知
问题和当前实验结论见 [docs/thermal_proxy_repair_zh.md](docs/thermal_proxy_repair_zh.md)。

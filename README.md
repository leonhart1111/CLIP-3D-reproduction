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

完整方法说明见 [docs/clip3d_pipeline_zh.md](docs/clip3d_pipeline_zh.md)，严格正式执行顺序见 [docs/formal_reproduction_zh.md](docs/formal_reproduction_zh.md)，可选10 ms瞬态热仿真见 [docs/transient_thermal_zh.md](docs/transient_thermal_zh.md)。热代理和线延迟参数的独立验证方法见 [docs/surrogate_parameter_validation_zh.md](docs/surrogate_parameter_validation_zh.md)。

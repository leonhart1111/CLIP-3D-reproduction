# Workflow modules

这里是 CLIP-3D 自己的实现，不包含四个第三方工具的源码。
- `run_lifting_pipeline.py`：一个 R1 点的端到端流水线；
- `run_lifting_sweep.py`：发现、续跑、并行处理所有成功 R1 点；
- `mcpat/`：gem5 到四核 McPAT XML 的转换和详细输出解析；
- `cacti/`：隔离配置、45 nm 表征和离散周期查找表；
- `floorplan/`：模块模型、P1 固定分箱、公式(5)功耗栅格化、公式(14)-(15)布局器；
- `thermal/`：HotSpot 稳态求解和公式(13)持续频率；
- `transient/`：默认关闭的周期统计、逐窗 McPAT 和 HotSpot 瞬态验证旁路；
- `r2/`：公式(6)延迟向量和 gem5 R2 回标运行；
- `analysis/`：扫描汇总与 Kendall-τ 排名诊断。

正式复现还包括两种冷却配置、四种布局方法、布局相关R2延迟、表III锚点验证和表V--VII汇总。完整命令见 `docs/formal_reproduction_zh.md`。

所有程序都可以用 `python3 -m workflow.<模块>` 从项目根目录运行。
完整用法见 `docs/clip3d_pipeline_zh.md`。
10 ms 瞬态旁路见 `docs/transient_thermal_zh.md`。

当前受限冷却配置是 `configs/experiments/clip3d_constrained_5p0.json`。
动态功耗和漏电功耗直接来自 McPAT，不使用论文结果拟合的乘数。

参数研究工具：

- `thermal/calibrate_stack_components.py`：硅与TIM热阻有限差分可辨识性检查；
- `thermal/calibrate_proxy.py`：布局位置代理拟合，并显式拒绝没有跨层样本时对
  beta 的伪辨识。这些工具不会修改 McPAT 输出功耗。

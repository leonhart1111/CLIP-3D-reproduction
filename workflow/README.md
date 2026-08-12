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
可配置采样间隔的瞬态旁路见 `docs/transient_thermal_zh.md`。
其中 MATMUL 的 fixed-bin/CLIP-3D 双布局命令使用
`run_dual_layout_validation`，读取既有 operational 稳态输出、生成一个共享的
周期统计 R1，并在 `comparison/transient_comparison.json` 中报告差异；它是
non-formal operational 验证，不是论文正式复现。

当前受限冷却配置是 `configs/experiments/clip3d_constrained_5p0.json`。
动态功耗和漏电功耗直接来自 McPAT，不使用论文结果拟合的乘数。

参数研究工具：

- `thermal/calibrate_stack_components.py`：硅与TIM热阻有限差分可辨识性检查；
- `thermal/calibrate_proxy.py`：布局位置代理拟合，并显式拒绝没有跨层样本时对
  beta 的伪辨识。这些工具不会修改 McPAT 输出功耗。

可选的通信频率加权扩展由
`configs/experiments/clip3d_constrained_5p0_raw_power_p1_lambda0020119_traffic_weighted_exploratory.json`
启用。它直接复用完成的 R1 中每个 CPU 的共享 L2 `demandAccesses`，令
`q_i=A_i/sum(A)`，并以 `sum(q_i*tau_i)` 替代公式(15)和 R2 中的算术平均线延迟。
`modules.json` 会保留原始计数、精确 counter 名、权重、统计文件和测量窗口。
这是非正式研究扩展；R2 仍把结果作为一个标量写入共享 `L2XBar`，不能解释为
每核心独立延迟或 IPC 因果敏感度。

瞬态热约束的 `--thermal-mode transient-rom` 默认使用轻量五状态闭式代理：
Core0--Core3 与共享 L2 各保留一个一阶热惯性状态，按窗口递推
`T_next=a*T+(1-a)*T_eq`，优化器内部 HotSpot 调用数为 0。后端由
`transient_rom.backend` 配置，默认值是 `five-state`；原矩阵 POD-ROM 仍可用
`pod-rom` 选择。`transient_rom.five_state.tau_core_s`、`tau_l2_s` 和
`parameter_status` 会写入每次报告；默认时间常数是 provisional 控制值，不是已完成的
参数测量。

five-state 流程示例：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" --output-dir "$OUT" \
  --config configs/experiments/clip3d_transient_rom_lambda0020119_traffic_weighted_discrete_partition_exploratory.json \
  --thermal-mode transient-rom \
  --transient-rom-r1-dir "$PERIODIC_R1" \
  --run-r2
```

两种后端都必须经过最终真实 HotSpot 周期验证；five-state 结果是 non-formal、
paper-inequivalent 研究扩展。

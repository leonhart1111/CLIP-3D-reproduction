# gem5 同步语义 ROI 测量协议

## 为什么不再用“某个核心的指令数”结束测量

旧 `paper`/`paper_all_cores` 协议分别用 CPU0 或最后达到阈值的核心结束窗口。
这能产生确定的 gem5 退出事件，但不能保证 fixed-bin 与 CLIP-3D 完成相同数量的
workload 工作：cache/xbar 延迟改变线程到达 barrier、锁和任务队列的时序，因而会
改变各核等待指令以及窗口内落入的迭代数。即使两个窗口都声称“500M instructions”，
它们也可能截断在不同的算法边界；反之，即使完成同样工作，逐核动态指令数也可能
因同步路径不同而略有差异。

因此新的正式入口是独立配置
`configs/experiments/r1_semantic_cache_sweep.json`，scope 为
`semantic-work`。旧 `configs/experiments/r1_cache_sweep.json` 保持不变，只用于
读取和审计历史 instruction-window 结果。

## 窗口定义

每个 workload 在完整 work-unit 边界执行以下顺序：

1. 四个 worker 启动并完成固定数量的 warmup work units；
2. 所有参与线程同步；指定线程调用 `m5_work_begin(1, 0)`；
3. gem5 收到 `workbegin` 后重置统计；
4. workload 完成固定数量的完整 work units；
5. 所有参与线程再次到达同步边界；指定线程调用 `m5_work_end(1, 0)`；
6. gem5 收到 `workend` 后 dump 统计并结束本次仿真。

程序必须在 end boundary 后仍暴露至少一个 sentinel unit。它不进入测量窗口，也不
需要真的执行；它保证 `workend` 位于合法的下一工作边界，而不是程序退出之后。
marker work ID 固定为 1，顺序必须严格为 `workbegin`、`workend`。

## 五个 work unit 与正式计数

| workload | work unit | warmup | measure | 默认总量/参数 |
|---|---|---:|---:|---|
| FFT | 一次四线程 forward FFT | 16 | 80 | `-m16 -p4 -r100` |
| CHOLESKY | 一次 numeric factorization | 16 | 80 | `-p4 -r100` |
| STREAM | 一轮 Copy/Scale/Add/Triad | 3 | 16 | `NTIMES=20` |
| MATMUL | 四核各完成一行的 balanced row batch | 40 | 200 | `-n 1024 -r 1 -t 4`，共 256 units |
| STENCIL | 一次完整 Jacobi iteration | 80 | 400 | `-n 2048 -i 500 -t 4` |

FFT、MATMUL、STENCIL 在 marker 前后使用线程 barrier；CHOLESKY 的 marker 位于前一
factorization 的 worker 全部结束、下一批 worker 创建之前；STREAM 的 marker 位于
前一轮四个 OpenMP kernel 的隐式 barrier 全部结束之后。

## 构建与运行

先构建带 marker 的五个 binary：

```bash
source scripts/env.sh
python scripts/build_semantic_benchmarks.py
```

构建脚本检查外部 SPLASH-2/STREAM 源码和 Makefile 必须同时完成 patch，并生成
`benchmarks/bin/semantic_manifest.json`。每个 R1 protocol ID 绑定 workload binary、
gem5 binary 和 gem5 config 的 SHA-256；R2 重新计算 workload binary SHA，不匹配即
拒绝运行。

生成独立的 100 点计划：

```bash
python scripts/run_r1_sweep.py \
  --experiment configs/experiments/r1_semantic_cache_sweep.json
```

建议先对五个 workload 各执行一个短 marker smoke，再启动正式网格。正式执行示例：

```bash
python scripts/run_r1_sweep.py \
  --experiment configs/experiments/r1_semantic_cache_sweep.json \
  --execute --jobs 1
```

每点除原有文件外还必须有 `roi_events.json`。`status.json` 记录 marker 间
`completion_ticks`、按 `cpu_clock/simFreq` 换算的全局 `completion_cycles`、逐核
instruction vector 和 `work_units_per_cycle`。catalogue、R2 cache、attachment 和配对
汇总都会重新验证这些证据，不能只依赖 `state=success`。

## 性能比较规则

稳态布局分支的主指标为：

```text
work_units_per_cycle = measured_work_units / completion_cycles
work_units_per_ns = work_units_per_cycle * sustainable_frequency_GHz
```

`completion_cycles` 来自两个全局 marker 的 tick 距离，不能用
`max(cpuN.numCycles)` 代替。fixed-bin/CLIP-3D 的 work-unit type、数量、protocol ID 和
binary SHA 必须完全相同。

IPC 仍保留为诊断量。只有两次 R2 的逐核动态指令向量逐项完全相等时，
`ipc_comparison_allowed=true`，才允许称为 same-trace IPC/BIPS 比较；只要任一核不同，
IPC/BIPS 改进值必须为空，排名继续使用 fixed-work throughput。同步造成的指令数差异
不会否定相同语义工作量，但会否定“相同指令轨迹”的说法。

## 历史结果隔离

此前 CPU0/最慢核心 instruction-window 协议得到的 100/200 点结果属于只读 legacy
证据。它们可用于复盘旧流程，但不能与 `semantic-work` 点拼接、补点、计算新协议的
平均提升或作为新 Balanced-50 的输入。新旧协议使用独立实验配置、输出 profile、
protocol identity 和 score definition；汇总器检测到混合协议会直接失败。

## 2026-08-19 实际 gem5 验证

五类 marker 路径均观察到严格的 `workbegin → reset → workend`，marker tick 差与
最终 `stats.txt` 的 `simTicks` 完全一致，且四核均有正指令数：

| workload | measured units | completion cycles | instruction vector |
|---|---:|---:|---|
| FFT (`-m10 -p4 -r4`) | 2 | 313,846 | `[558068, 557557, 557873, 558758]` |
| CHOLESKY (`-p4 -r4`) | 2 | 16,294,320 | `[18860473, 12881343, 13480629, 13334215]` |
| STREAM (`ARRAY_SIZE=100000`, `NTIMES=20`) | 1 | 2,822,929 | `[342987, 354090, 502973, 507535]` |
| MATMUL (`-n32 -r1 -t4`) | 4 | 13,602 | `[19504, 19434, 19414, 19444]` |
| STENCIL (`-n32 -i4 -t4`) | 2 | 4,837 | `[4124, 4550, 4092, 4582]` |

FFT、CHOLESKY、MATMUL、STENCIL 使用正式构建 binary。STREAM marker smoke 使用同一
源码和相同 `NTIMES=20`，仅把编译时数组缩小到 100K；正式 10M 数组 binary 的成本
探针运行约 25 分钟仍未完成第一个 work unit，随后主动停止以免继续占用共享节点。
因此 STREAM 的 `warmup=3, measure=16` 仍是待成本/方差试验确认的初始预算，不能在
完整 20 点启动前视为已定标。

MATMUL 另执行了两次 R2，只把 xbar forward latency 从 7 cycles 改为 12 cycles。
两边都完成相同 4 个 balanced-row-batch units，但指令向量分别为：

```text
fixed: [19530, 19478, 19476, 19462], IPC2=5.508551
clip:  [19532, 19460, 19462, 19466], IPC2=5.417883
```

最大逐核相对指令差为 0.0924%。新比较器正确给出
`same_trace=false`、`ipc_comparison_allowed=false` 和
`ipc_improvement_percent=null`；相同 2 GHz 频率下，主 fixed-work score 的变化为
-2.6350%。这证明即使语义工作量相同，延迟引起的线程同步时序也会改变动态指令数，
旧的指令阈值/IPC 主比较不能回答 fixed-bin 与 CLIP-3D 的工作完成时间问题。

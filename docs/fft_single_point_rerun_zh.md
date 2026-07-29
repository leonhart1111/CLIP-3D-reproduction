# FFT 64 kB / 1 MB 单点修正后的重跑方法

## 哪些阶段需要重跑

不需要重跑工具编译、工作负载编译或 gem5 R1。以下阶段必须从现有 R1 重新生成：

1. gem5 统计到 McPAT XML 的活动映射；
2. McPAT 模块功耗和泄漏比例；
3. CACTI 表 II 有效值及本机 CACTI 审计值；
4. 模块模型、布局、32×32 功耗网格；
5. HotSpot、闭式持续频率和 R2 延迟向量；
6. CLIP-3D 的 L2 自动布局。

修正几何后，fixed-bin 和 CLIP-3D 候选的平均线延迟都四舍五入为 1 周期，关键路径都为 12 周期。旧 fixed-bin 是 13 周期，不能复用；旧 CLIP-3D 点恰好也是同一组 12 周期 `gem5_overrides`，因此本单点的三个最终结果都可以复用旧 CLIP-3D 的 IPC2。复用入口会逐字段检查向量，不一致时会拒绝。

CLIP-3D 现在会对 optimized 与 fixed-bin 各运行一次真实 HotSpot。若解析热代理提出的布局使真实热约束 BIPS 下降，程序会回退 fixed-bin，并把候选、实测温度和原因写入 `layout_selection.json`。

## 完整命令

所有命令均从项目根目录运行：

```bash
cd /home/zyjiang/Agenticflow/CLIP
source scripts/env.sh

R1=runs/architecture_sweep/r1/paper/fft/l1d_64kB/l2_1024kB
REUSE_R2=runs/pilot/fft_64k_1m_clip3d_5p0
ROOT=runs/pilot_v4/fft_64k_1m
BASE3=$ROOT/fixed_3p5
FIXED5=$ROOT/fixed_5p0
CLIP5=$ROOT/clip3d_5p0

mkdir -p "$ROOT" results/tables
```

先运行回归测试：

```bash
python -m unittest discover -s tests -v
```

重跑 3.5 K/W fixed-bin，并复用已验证为同一 12 周期向量的旧 CLIP-3D R2：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$BASE3" \
  --config configs/experiments/clip3d_pipeline.json \
  --layout-method fixed-bin \
  --reuse-r2-dir "$REUSE_R2"
```

重跑 5.0 K/W fixed-bin；其布局延迟与 3.5 K/W 相同：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$FIXED5" \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method fixed-bin \
  --reuse-r2-dir "$REUSE_R2"
```

重跑 5.0 K/W CLIP-3D 自动布局。保护逻辑确定最终布局后，复用入口才会核对 R2 向量：

```bash
python -m workflow.run_lifting_pipeline \
  --r1-dir "$R1" \
  --output-dir "$CLIP5" \
  --config configs/experiments/clip3d_constrained_5p0.json \
  --layout-method clip3d \
  --reuse-r2-dir "$REUSE_R2"
```

两个 fixed-bin 命令各运行一次 32×32 detailed-3D HotSpot；CLIP-3D 命令运行两次，用于真实验证 fixed 与 optimized。`--reuse-r2-dir` 只省略耗时数小时的 gem5 R2，不会复用温度或布局。

汇总三个新点：

```bash
python -m workflow.analysis.summarize_sweep \
  --root "$ROOT" \
  --csv results/tables/pilot_v4_fft_64k_1m.csv \
  --output results/tables/pilot_v4_fft_64k_1m.json \
  --expected-points 3

column -s, -t results/tables/pilot_v4_fft_64k_1m.csv

python -m json.tool "$CLIP5/layout_selection.json"
```

## 应检查的结果

修正后的 McPAT 锚点应为：

```text
total_power_w = 16.24 W
gamma         = 0.446
```

重新校准后的热锚点为：

```text
R_conv=5.0: Tmax=124.39 C, f_sus=0.932477 GHz
R_conv=3.5: Tmax=100.00 C, f_sus=1.759326 GHz
```

修正后的 CLIP-3D 候选与最终选择为：

```text
optimized L2  = tier 1, (2.671, 4.147) mm
optimized     = 124.57 C, 0.927881 GHz, mean wire=0.633→1 cycle
fixed-bin     = 124.39 C, 0.932477 GHz, mean wire=1.032→1 cycle
最终策略       = fixed-bin（真实热约束 BIPS 保护触发回退）
IPC2          = 3.417362（复用且验证了相同延迟向量的旧 R2）
BIPS2         = 3.186612
关键路径       = 12 cycles
```

解析代理预测 optimized 为 `111.538°C`，真实 HotSpot 却为 `124.57°C`，说明未公开的代理系数不能可靠代表当前 detailed-3D 温度梯度。fixed 与 optimized 的平均线延迟虽然分别为 1.032 和 0.633 周期，但二者写入 gem5 后都是 1 周期；因此也没有可报告的离散时延收益。

根本限制仍是当前 McPAT 分布中共享 L2 仅占总功耗约 2.40%，底层不可移动模块承载约 97.44% 功耗；只移动 L2 很难改变底层热点。论文未公开的逐模块 McPAT XML/功耗分布与热代理系数是复现其大幅布局收益的主要不确定项。不能任意放大 L2 功耗或改用最坏路径周期来制造接近论文的结果。

热阻系数 `8.72` 是在缓存面积真正进入布局后重新校准的；同一功耗图在两种对流热阻下均已真实运行 HotSpot，而不是用温差外推。

第二个 FFT 32 kB/256 kB 锚点已用新几何和 `8.72` 重新实跑：当前为 `15.786 W、γ=0.4418、99.64 C`，论文为 `15.32 W、γ=0.444、94.6 C`。功耗高约 3%，温度高 5.04 C；即使按总功耗线性归一，仍约高 2.84 C。这证明单一有效热阻只能命中校准架构，不能补偿不同缓存配置下未公开的模块功耗密度与 Cool-3D 材料细节。该点又位于 95 C 边界附近，正式扫描必须把分类差异列为复现不确定性，而不能继续调一个全局系数去同时强行拟合所有点。

关键文件如下：

- `mcpat/mapping_report.json`：gem5 到 McPAT 的直接计数器及假设；
- `mcpat/mcpat.json`：`raw_power`、全局校准系数和校准后逐模块功耗；
- `cacti/cacti_characterization.json`：表 II 有效值及 `measured_*` 本机值；
- `hotspot/hotspot_manifest.json`：五层参数和有效局部热阻校准；
- `optimizer_report.json`：CLIP-3D 的所有候选与最终 L2 坐标；
- `r2_latency.json`：用于判定 R2 能否复用的完整延迟向量；
- `pipeline_summary.json`：最终温度、频率、IPC2 和 BIPS2。

## 仍然属于复现假设的部分

论文没有发布 McPAT XML、完整 CACTI 组织参数、Cool-3D 层材料表或热代理系数。因此 320 K McPAT 工作点、全局功耗校准、表 II 查表模式和 `8.72` 有效局部热阻系数都被明确记录。它们能重建该公开锚点，但正式论文结论仍应通过其余表 III 锚点交叉验证，不能把单锚点吻合当成全部 80/100 点已经复现。

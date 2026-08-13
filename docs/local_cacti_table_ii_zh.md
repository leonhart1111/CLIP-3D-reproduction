# 本地 CACTI 的 Table II 等价测量

## 结论

稳态流程不再读取论文 Table II 的缓存面积或延迟，也不再把面积缩放到
150 mm²。下表全部由本机 CACTI 对当前 gem5 缓存结构重新测量，属于本项目的
“Table II 等价表”，而不是论文数值的复制。

| 缓存 | 容量 | CACTI访问时间/ns | 2 GHz原始周期 | 向上取整周期 | 面积/mm² |
|---|---:|---:|---:|---:|---:|
| L1D | 16 kB | 0.424380 | 0.848760 | 1 | 0.129604 |
| L1D | 32 kB | 0.451611 | 0.903222 | 1 | 0.183589 |
| L1D | 64 kB | 0.535344 | 1.070688 | 2 | 0.290901 |
| L1D | 128 kB | 0.732799 | 1.465598 | 2 | 0.562112 |
| L2 | 128 kB | 1.383840 | 2.767680 | 3 | 1.176263 |
| L2 | 256 kB | 1.498170 | 2.996340 | 3 | 1.572845 |
| L2 | 512 kB | 1.711120 | 3.422240 | 4 | 2.502063 |
| L2 | 1024 kB | 2.092980 | 4.185960 | 5 | 4.836591 |
| L2 | 2048 kB | 2.721960 | 5.443920 | 6 | 9.266143 |

周期换算严格使用：

\[
L_{cache}=\max\left(1,\left\lceil t_{access} f_0\right\rceil\right),
\quad f_0=2\ \mathrm{GHz}.
\]

仅在数值与整数的差小于 `1e-12` 时消除浮点误差；真实的非整数延迟一定向上
取整。例如2.02变成3，而精确的3.0仍为3。

## McPAT/CACTI对齐条件

当前缓存契约来自R1/gem5结构和实验配置：

- 45 nm；
- 64-byte cache line；
- L1二路组相联，L2八路组相联；
- 每个gem5 Cache对象对应一个UCA bank；
- L2XBar宽度为64 byte/cycle，因此缓存输出宽度为512 bit；
- 320 K，ITRS-HP器件，conservative interconnect；
- normal access、ECC开启、4 cores。

流水线先运行独立CACTI，再将同一记录中`cycle_time`与`access_time`分别向上取整，
写入McPAT XML的throughput与latency字段；R2读取其中的access latency。因此
McPAT功耗、floorplan几何和R2延迟共享同一个characterization ID，而不是使用
人为的10-cycle占位值。

## 面积规则

- L1I、L1D、L2：直接采用本地CACTI的面积、宽和高；
- 非缓存逻辑和互连：直接采用McPAT面积；
- McPAT报告的缓存面积仅保留为诊断字段；
- 面积、边长和功耗均不乘全局校准系数；
- HotSpot die尺寸由这些真实模块面积和配置的floorplan utilization自然推导。

因此新结果不能与历史的150 mm²缩放结果混用。已有R1保持有效，但McPAT、
CACTI、模块模型、布局、HotSpot、持续频率、R2延迟和R2均需重新生成。

## 复现命令

在项目根目录执行：

```bash
source scripts/env.sh
python -m scripts.characterize_local_table_ii \
  --cacti tools/src/cacti/cacti \
  --base-config tools/src/cacti/cache.cfg \
  --output-dir data/cacti/local_45nm_table_ii_equivalent_artifacts \
  --report-dir data/cacti
```

表格输出：

- `data/cacti/local_45nm_table_ii_equivalent.json`
- `data/cacti/local_45nm_table_ii_equivalent.csv`

逐行证据位于 `data/cacti/local_45nm_table_ii_equivalent_artifacts/`，包括每个
容量的生成配置、CACTI原始输出、配置哈希、输出哈希和完整表征JSON。当前工具
来自归档而非独立Git checkout，因此Git revision为null；可执行文件、基础配置
和源归档仍分别由SHA-256记录。

CACTI在32 kB配置下偶尔会把未初始化的亚正规数（约`1e-307`）打印到cells、
row logic或column logic内部泄漏分项；这些分项不参与本流程提取的总泄漏、延迟
或面积。为使证据可重复，stdout保存前会去除行尾空白，并且只把上述三个已知
字段中绝对值小于`1e-300`的数规范为0；规范化规则写入
`provenance.raw_output_normalization`，其他文本不改动。characterization ID由
内容和文件SHA-256构成，不包含输出目录或工具路径，因此相同工具与参数在不同
目录运行会得到相同ID。

论文Table II只允许在报告中作为外部对照。其数值不得导入JSON、缓存查找代码或
正式配置，也不得通过搜索CACTI参数来刻意逼近。

## 2026-08-13单点贯通验证

复用了既有MATMUL 32 kB/512 kB R1，仅运行R1之后的稳态阶段，不运行长耗时
gem5 R2。验证输出位于临时审计目录
`/tmp/clip-cacti-unscaled-steady-final-smoke`，结果为：

- 未缩放模块总面积：45.753293 mm²；
- 70% floorplan utilization对应die边长：7.850327 mm；
- L1I/L1D/L2访问周期：1/1/4；
- McPAT、CACTI、modules和R2 latency的characterization ID完全相同；
- fixed-bin稳态Tmax：141.769206 °C；
- 持续频率：0.481082 GHz。

高温不是重新引入面积缩放的理由。它表明历史150 mm²缩放曾降低功率密度，且
基于旧几何得到的热代理参数与`local_resistance_scale=8.72`不再具有正式验收
资格。后续必须在未缩放几何上重新进行HotSpot热参数辨识；本次修复不通过调热
参数掩盖该物理变化。

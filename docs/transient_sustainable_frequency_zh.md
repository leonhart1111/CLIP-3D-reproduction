# 瞬态热约束下的持续频率：数学推导与 CLIP-3D 扩展设计

> **状态：设计记录，尚未接入优化器。**
>
> 本文档给出在不把 HotSpot 放入布局优化内循环的前提下，将 CLIP-3D 的稳态
> 持续频率模型扩展为瞬态模型的严格推导。它不改变已经完成的稳态复现、R1、R2
> 或瞬态验证结果。文中的“建议流程”须在独立分支和独立输出目录中实现、校准和验证。

## 1. 要解决的问题

CLIP-3D 的原始方法在早期三维芯片设计阶段，以闭式频率模型代替对每一个候选
布局反复调用 HotSpot。优化器因此能直接搜索实际吞吐率（BIPS）最优的布局，而不是
搜索人为加权的“温度 + 线长”代理目标。

当前项目已有的瞬态旁路为：

```text
专用周期统计 R1 → 窗口统计 → 每窗口 McPAT → HotSpot 瞬态温度轨迹
```

它回答的是“在名义频率 `f0` 下，给定功耗轨迹会产生什么温度波动”。它**尚未**计算
瞬态持续安全频率，也不能把轨迹中的一个 `Tmax(f0)` 直接代入稳态 Equation (13)。

本设计的目标是定义并快速评价：

\[
f_{\mathrm{sus,trans}}(p)
\]

即布局 \(p\) 在给定工作负载功耗轨迹下、长期运行且始终不超过热安全阈值时可持续的
最高恒定频率。优化器内只运行快速数学模型；HotSpot 仅用于离线校准和最终验证。

## 2. 记号与范围

| 记号 | 含义 |
|---|---|
| \(p\) | 布局：模块/核心簇的平面坐标和层号 |
| \(f_0\) | R1、McPAT 功耗统计的名义频率 |
| \(f\) | 待评估的恒定工作频率 |
| \(s=f/f_0\) | 归一化频率，\(s\in[s_{\min},1]\) |
| \(\mathbf T\) | 所有 HotSpot 热网格单元的温度向量 |
| \(\boldsymbol\theta=\mathbf T-T_{\rm amb}\mathbf1\) | 相对环境温度的温升向量 |
| \(\mathbf C,\mathbf G\) | 热容矩阵与热导/散热矩阵 |
| \(\mathbf l_k,\mathbf d_k\) | 第 \(k\) 个窗口的模块级漏电、名义频率动态功耗 |
| \(\Delta t_k^0\) | 第 \(k\) 个窗口在 \(f_0\) 下的真实时间长度 |
| \(\mathbf H(p)\) | 模块功耗到 HotSpot 网格的面积守恒栅格化映射 |
| \(T_{\rm safe}\) | 热安全阈值 |

第一版模型使用与论文和当前项目一致的边界：固定电压、每周期活动不随频率变化、
动态功耗与频率线性相关、McPAT 漏电在给定温度下固定。温度相关漏电和电压 DVFS
属于后续的非线性扩展，见第 11 节。

## 3. 论文稳态闭式的来源

稳态热方程为：

\[
\mathbf G\boldsymbol\theta=\mathbf P,
\qquad
\boldsymbol\theta=\mathbf R_\theta\mathbf P,
\qquad
\mathbf R_\theta=\mathbf G^{-1}.
\tag{1}
\]

在固定电压下，模块/网格功耗可按动态与漏电拆分：

\[
\mathbf P(f)=\mathbf P_{\rm leak}+s\mathbf P_{\rm dyn}.
\tag{2}
\]

代入 Equation (1)：

\[
\mathbf T(f)=T_{\rm amb}\mathbf1+
\mathbf R_\theta\mathbf P_{\rm leak}+
s\mathbf R_\theta\mathbf P_{\rm dyn}.
\tag{3}
\]

因此稳态的每个网格温度都关于 \(s\) 仿射。若进一步假设每个网格的漏电比例相同，
漏电比例为 \(\gamma\)，论文得到：

\[
T_{\max}(f)-T_{\rm amb}
=
\left[\gamma+(1-\gamma)s\right]
\left[T_{\max}(f_0)-T_{\rm amb}\right].
\tag{4}
\]

令 \(T_{\max}(f)=T_{\rm safe}\) 并解出 \(f\)，就是论文 Equation (13)：

\[
f_{\rm sus}=
\frac{f_0}{1-\gamma}
\left[
\frac{T_{\rm safe}-T_{\rm amb}}
{T_{\max}(f_0)-T_{\rm amb}}-\gamma
\right],
\tag{5}
\]

随后再限制在 \([f_{\min},f_0]\)。这正是当前
`workflow/thermal/sustainable_frequency.py` 的计算依据。

Equation (5) 是**稳态模型**的结果：它只有一个功耗图，没有时间顺序、热容或初始
温度。它不能直接推广到瞬态轨迹。

## 4. 为什么瞬态 `Tmax(f0)` 不足以决定频率

瞬态热模型是：

\[
\mathbf C\frac{d\boldsymbol\theta(t)}{dt}+
\mathbf G\boldsymbol\theta(t)=\mathbf P(t).
\tag{6}
\]

与稳态 Equation (1) 相比，多出了热容项 \(\mathbf C\dot{\boldsymbol\theta}\)。因此温度
取决于完整的功耗历史 \(\{\mathbf P(\tau):0\leq\tau\leq t\}\)，而不只取决于一个峰值或平均功耗。

两个轨迹即使满足：

\[
T^{A}_{\max}(f_0)=T^{B}_{\max}(f_0),
\]

也可能有不同的功耗脉冲位置、脉冲顺序、热点网格或初始热状态。它们在另一个频率下
不一定有相同峰温，更不一定有相同安全频率。故：

\[
T^{\rm transient}_{\max}(f_0)
\not\Rightarrow f_{\rm sus,trans}.
\tag{7}
\]

特别地，若仅将动态功耗乘以 \(s\)，却继续用原来的 2 ms 或 10 ms HotSpot 时间步，
等价于在低频时执行了更少的周期/指令。这不是同一段程序在低频下的热行为。

## 5. 频率缩放下的功耗与时间拉伸

当前瞬态 R1 在 \(f_0\) 下生成窗口；第 \(k\) 个窗口包含固定数量的架构周期/活动。
在频率 \(f=sf_0\) 下，保持同一段计算工作量时，应有：

\[
\mathbf P_k(p;s)=
\mathbf H(p)\left[\mathbf l_k+s\mathbf d_k\right],
\tag{8}
\]

\[
\Delta t_k(s)=\frac{\Delta t_k^0}{s}.
\tag{9}
\]

Equation (8) 表示动态功耗随频率缩放而漏电功耗近似不变；\(\mathbf H(p)\) 确保模块
矩形与网格面积交叠后的功耗守恒。Equation (9) 表示同一窗口的周期活动在低频下需要
更长真实时间。

由此可见动态能量近似保持不变：

\[
\left(s\mathbf d_k\right)\frac{\Delta t_k^0}{s}
=\mathbf d_k\Delta t_k^0.
\tag{10}
\]

而漏电能量随降频增大：

\[
\mathbf E_{{\rm leak},k}(s)
=\mathbf l_k\frac{\Delta t_k^0}{s}.
\tag{11}
\]

这正是瞬态模型必须考虑时间拉伸的物理原因。

### 5.1 仅作对照：固定绝对时间波形的仿射特例

若功耗波形由外部真实时间驱动，所有 \(\Delta t_k\) 与频率无关，且初始温度也与频率无关，
则可以写成：

\[
\boldsymbol\theta_k(s)=\mathbf a_k+s\mathbf b_k.
\tag{12}
\]

并以漏电和动态两组递推分别计算 \(\mathbf a_k\)、\(\mathbf b_k\)。此时可直接取每个
网格、每个时刻给出的频率上界之最小值。

这是一种有用的两基向量验证方法，但它**不适用于当前以 gem5 周期/指令活动记录的
benchmark 轨迹**，因为它忽略了 Equation (9)。它不能作为本项目的持续频率定义。

## 6. 正确的离散瞬态递推公式

定义：

\[
\mathbf F=\mathbf C^{-1}\mathbf G.
\tag{13}
\]

在一个窗口内将功耗视为常量，Equation (6) 的精确离散解为：

\[
\boldsymbol\theta_{k+1}(p;s)=
\mathbf A_k(s)\boldsymbol\theta_k(p;s)+
\mathbf B_k(s)\mathbf P_k(p;s),
\tag{14}
\]

其中：

\[
\mathbf A_k(s)=
\exp\left(-\mathbf F\frac{\Delta t_k^0}{s}\right),
\tag{15}
\]

\[
\mathbf B_k(s)=
\mathbf F^{-1}\left[\mathbf I-\mathbf A_k(s)\right]\mathbf C^{-1}.
\tag{16}
\]

联立 Equations (8)、(14)–(16)，得到建议使用的瞬态频率评价式：

\[
\boxed{
\boldsymbol\theta_{k+1}(p;s)=
\mathbf A_k(s)\boldsymbol\theta_k(p;s)+
\mathbf B_k(s)\mathbf H(p)
\left[\mathbf l_k+s\mathbf d_k\right].
}
\tag{17}
\]

这就是瞬态对论文 Equation (13) 的正确替代。它仍是解析的状态转移方程，但一般不能
再化简为仅依赖一个 \(T_{\max}(f_0)\) 的标量表达式。

实现时不应显式求矩阵逆；应使用稳定的矩阵指数、线性方程求解或降阶后的等价运算。

## 7. 初始条件：采用周期稳态而非 `f0` 稳态温度

现有瞬态验证以对应布局的 `steady.txt` 作为初温，即：

\[
\mathbf T(0)=\mathbf T_{\rm steady}(f_0).
\]

这适合观察“从 \(f_0\) 的平均功耗稳态出发”的短时间热波动，但不同频率下它不是
一致的初始条件。若研究对象是“持续安全频率”，推荐将一个 ROI 功耗轨迹视作周期性
重复负载，并使用**周期稳态**。

设一个 ROI 有 \(N\) 个窗口。定义：

\[
\mathbf u_k(p;s)=\mathbf B_k(s)\mathbf P_k(p;s),
\]

\[
\boldsymbol\Phi(s)=
\mathbf A_{N-1}(s)\cdots\mathbf A_1(s)\mathbf A_0(s),
\tag{18}
\]

\[
\mathbf q(p;s)=
\sum_{r=0}^{N-1}
\left[
\mathbf A_{N-1}(s)\cdots\mathbf A_{r+1}(s)
\right]
\mathbf u_r(p;s).
\tag{19}
\]

其中空乘积为单位矩阵。一个周期结束时：

\[
\boldsymbol\theta_N=
\boldsymbol\Phi(s)\boldsymbol\theta_0+
\mathbf q(p;s).
\tag{20}
\]

周期稳态要求 \(\boldsymbol\theta_N=\boldsymbol\theta_0\)，故：

\[
\boxed{
\boldsymbol\theta^{\rm PSS}_0(p;s)=
\left[\mathbf I-\boldsymbol\Phi(s)\right]^{-1}
\mathbf q(p;s).
}
\tag{21}
\]

以 Equation (21) 为初值，用 Equation (17) 遍历一个完整周期，就能获得长期重复运行时
的温度轨迹。若研究有限任务而不是持续运行，应改用真实给定的初温；结果必须称为
“该初温、该任务时长下的安全频率”，不可泛称持续频率。

## 8. 新的持续频率定义与安全状态

定义布局 \(p\)、频率比例 \(s\) 的热违规函数：

\[
g_p(s)=
\max_{0\leq k\leq N,\,i}
\left[T_{\rm amb}+\theta_{i,k}^{\rm PSS}(p;s)\right]
-T_{\rm safe}.
\tag{22}
\]

安全频率比例及频率为：

\[
\boxed{
s^\star(p)=
\max\{s\in[s_{\min},1]:g_p(s)\leq0\},
\qquad
f_{\rm sus,trans}(p)=f_0s^\star(p).
}
\tag{23}
\]

若 \(g_p(1)\leq0\)，则 \(f_{\rm sus,trans}=f_0\)。若：

\[
g_p(s_{\min})>0,
\]

则该设计应记录为：

```text
thermally_infeasible_at_fmin
```

而不能仅因为系统有 \(f_{\min}\) 就把它标为“安全”。如需保留旧流程的可比性，可同时
报告下限频率处的 BIPS，但必须将它与安全频率结果分开。

不能未经实验证明就假定 \(g_p(s)\) 单调。较高频率降低了执行时间但提高动态功耗，
较低频率降低动态功耗却拉长漏电暴露时间。稳妥的求解方式为：

1. 在确定性频率网格上评价 \(g_p(s)\)；
2. 找到安全/不安全区间边界；
3. 对每个边界做 bracket + refine；
4. 取最大的安全 \(s\)。

只有在跨工作负载和布局验证了单调性后，才可以把二分搜索作为安全的加速手段。

## 9. 如何避免在优化内环运行 HotSpot

### 9.1 HotSpot 校准的瞬态 ROM

完整网格的 \(\mathbf C\)、\(\mathbf G\) 状态维度较高，不适合每次 L-BFGS-B 或无梯度
搜索评价时直接做完整矩阵计算。使用降阶基 \(\mathbf V\) 近似：

\[
\boldsymbol\theta\approx\mathbf V\mathbf z,
\]

并定义：

\[
\mathbf C_r=\mathbf V^T\mathbf C\mathbf V,
\qquad
\mathbf G_r=\mathbf V^T\mathbf G\mathbf V.
\tag{24}
\]

降阶状态 \(\mathbf z\) 满足：

\[
\mathbf C_r\dot{\mathbf z}+
\mathbf G_r\mathbf z=
\mathbf V^T\mathbf P.
\tag{25}
\]

它具有与 Equation (17) 相同的形式，只是矩阵维度远小于完整网格：

\[
\mathbf z_{k+1}=
\mathbf A_{r,k}(s)\mathbf z_k+
\mathbf B_{r,k}(s)\mathbf V^T\mathbf H(p)
\left[\mathbf l_k+s\mathbf d_k\right].
\tag{26}
\]

然后由：

\[
\widehat{\mathbf T}_k=T_{\rm amb}\mathbf1+\mathbf V\mathbf z_k
\tag{27}
\]

恢复网格温度并计算 Equation (22)。这使每个候选布局只需进行低维递推和频率搜索，
无需启动 HotSpot 子进程。

### 9.2 ROM 的两种构建方式

1. **直接 RC 状态模型（优先的长期方案）**：从与 HotSpot 完全一致的网格、材料和
   封装参数构建/导出 \(\mathbf C,\mathbf G\)，再通过模型降阶得到 \(\mathbf V\)。其理论
   保真度最高，但必须用 HotSpot 逐项验证边界条件、层定义和数值步进一致性。
2. **黑盒 HotSpot 校准 ROM（建议的首个实现）**：对固定 die 尺寸、层叠和散热条件，
   用若干空间功耗脉冲/阶跃作为激励运行 HotSpot，记录热响应并用 POD、ERA 或平衡截断
   得到低阶状态模型。无需修改 HotSpot 源码，但必须用未参与拟合的布局和功耗轨迹验证。

在当前“模块与死硅均视为相同硅材料、die 外形固定”的模型下，\(\mathbf C\)、\(\mathbf G\)
对布局近似不变；布局只改变 \(\mathbf H(p)\)。若将来允许移动操作改变 die 外形、材料或
层厚，则 ROM 必须按新的物理堆叠重新校准，不能直接复用。

## 10. 与布局优化和 R2 的连接

使用瞬态频率替换当前稳态 `closed_form_frequency()` 后，优化器的热-性能目标可写为：

\[
\mathcal L(p)=
-\mathrm{IPC1}\cdot f_0s^\star(p)
+\lambda_{\rm wire}\,\mathrm{IPC1}\,\tau_{\rm wire}(p)
+\mathcal P_{\rm legal}(p).
\tag{28}
\]

其中 \(\mathcal P_{\rm legal}\) 处理不重叠、边界和层容量约束。\(\lambda_{\rm wire}\)
仍是线延迟性能权衡项；原先稳态热代理的 \(\alpha\)、\(\beta\)、\(w_{\rm cross}\) 不再作为
温度预测参数使用，因为热响应应由 HotSpot 校准的 ROM 给出。

对“核心也可移动”的扩展，决策变量应是完整核心簇与 L2 的位置/层号：

\[
p=\{(x_c,y_c,z_c)\}_{c=0}^{3}\cup\{(x_{L2},y_{L2},z_{L2})\}.
\]

不建议在第一版中让 IFU、LSU、执行单元等核心子模块彼此独立移动；应保持每个核心
簇内部相对布局固定，避免产生不真实的微架构 floorplan。

最终评价保持两阶段：

\[
\mathrm{BIPS2}_{\rm trans}(p)=
\mathrm{IPC2}(p)\cdot f_{\rm sus,trans}(p).
\tag{29}
\]

R2 只在 fixed 与最终优化布局上运行，不为每个候选布局或每个频率重新运行。Equation (29)
继承了论文“IPC 近似不随频率变化、硬件流水级数固定”的假设。若设计允许随频率重定时
缓存/互连级数，则应在最终 \(f_{\rm sus,trans}\) 处重新将物理秒延迟映射为周期并运行一次
频率一致的 R2；该问题必须与论文口径分开报告。

网格化的硬矩形交叠会使 \(\mathbf H(p)\) 对坐标分段变化。若使用 L-BFGS-B，ROM 内应
采用可微平滑 footprint，并在最终 HotSpot 验证时恢复真实面积交叠；或者使用适合非光滑
目标的 pattern search/CMA-ES。

## 11. 模型边界：电压 DVFS 与温度相关漏电

第一版中 \(P_{\rm dyn}\propto f\) 是固定电压假设。真实 DVFS 若同时改变供电电压，
动态功耗应近似为：

\[
\mathbf P_{\rm dyn}(s)=
s\left(\frac{V(s)}{V_0}\right)^2\mathbf P_{\rm dyn}(1).
\tag{30}
\]

漏电还会依赖电压和温度：

\[
\mathbf P_{\rm leak}=\mathbf P_{\rm leak}(V,\mathbf T).
\tag{31}
\]

这使 Equation (17) 成为非线性热-功耗耦合递推。ROM 依旧可以取代 HotSpot 内循环，
但每个候选频率需要温度/漏电固定点迭代或数值时间推进；不能再声称拥有一个单行闭式
频率公式。必须先获得明确的电压-频率表和温度相关漏电模型，不能用人为缩放“拟合”结果。

## 12. 最终 HotSpot 验证要求

ROM 仅用于快速搜索。任何报告的最终数字仍须由 HotSpot 验证。对每个 fixed/optimized
最终布局，验证应至少包括：

1. 在 \(f_0\)、预测 \(f_{\rm sus,trans}\)、\(f_{\min}\)（若可行）运行瞬态 HotSpot；
2. 动态功耗按 Equation (8) 缩放，HotSpot 时间步按 Equation (9) 拉伸；
3. 不使用 `steady.txt(f0)` 作为不同频率的最终初温；重复功耗周期直到周期首尾温度差
   小于预设容差，或以可信的 PSS 温度场初始化；
4. 比较 ROM 与 HotSpot 的逐时刻峰温、全程峰温、热点位置和预测安全频率；
5. 在未参与 ROM 校准的布局、工作负载和功耗轨迹上报告误差与排序一致性。

若 ROM 预测频率 \(s^\star\) 的 HotSpot 验证未落在安全阈值附近，不能通过调节隐藏系数
使它“看起来正确”；应扩充训练激励、提高 ROM 阶数、修正功耗/时间映射，或缩小 ROM 的
适用范围。

## 13. 当前 2 ms 实验的含义与限制

现有 MATMUL 2 ms 运行记录的时长加权总功耗约为 14.0591 W，范围约为
14.0033–14.1100 W，峰谷约为平均值的 0.759%。已有输出中的峰温波动约为 0.05 °C，
且部分历史结果生成于温度只输出到 0.01 °C 的版本。

该实验说明了“周期统计 R1 → 窗口 McPAT → 栅格功耗 → HotSpot 瞬态”的数据链路可运行。
但它不能证明：

- 瞬态频率模型已经被验证；
- 瞬态必然带来明显布局收益；
- 当前的 \(f_0\) 初温能够代表任意降频后的长期状态；
- 现有稳态 \(\alpha,\beta\) 代理可迁移到核心可移动的空间。

尤其是这个 MATMUL 点的功耗波动较小，适合做链路正确性检查，不适合单独作为瞬态
优化收益的证据。

## 14. 建议实施和验证顺序

```text
1. 固定电压、固定核心簇、固定 die 外形的模型边界
2. 复用/生成一次高时间分辨率 transient R1 与窗口 McPAT 功耗轨迹
3. 针对每种物理 stack/die/cooling 条件校准 ROM
4. 用保留的布局和轨迹检验 ROM 的温度、频率和排序误差
5. 将 Equation (23) 接入独立的布局优化入口；优化内零次 HotSpot
6. 对 fixed 与最终布局做 PSS 条件下的 HotSpot 瞬态复验
7. 仅对最终布局运行 R2，并输出 Equation (29)
```

任何阶段的输出均应记录模型版本、stack、网格、时间步、功耗轨迹哈希、ROM 校准集、
验证集、频率搜索网格和可行性状态，以保证后续结果可审计、可复现。

## 15. 与现有文件的关系

| 现有位置 | 作用 | 与本设计的关系 |
|---|---|---|
| `workflow/thermal/sustainable_frequency.py` | 论文稳态 Equation (13) | 保留为稳态基线，不应原地替换 |
| `workflow/floorplan/optimize_layout.py` | 现有 L2 优化和稳态空间热代理 | 未来新增独立瞬态优化入口，不破坏该入口 |
| `workflow/transient/run_transient_r1.py` | 专用周期统计 R1 | 提供 \(\Delta t_k^0\) 与功耗活动来源 |
| `workflow/transient/stats_windows.py` | 将累计统计拆为连续窗口 | 提供窗口边界和实际时长 |
| `workflow/transient/run_windowed_mcpat.py` | 每窗口 McPAT | 提供 \(\mathbf l_k,\mathbf d_k\) |
| `workflow/transient/run_hotspot_transient.py` | 当前 \(f_0\) HotSpot 瞬态验证 | 保留；未来新增频率缩放/PSS 验证包装器 |
| `workflow/transient/run_dual_layout_validation.py` | 两个已有布局的瞬态对比 | 继续作为验证，不把结果误作新频率模型 |

本文件提出的是独立扩展，不应修改冻结的稳态复现目录或正在运行的 R1/R2 实验。

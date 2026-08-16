# 四个工具的下载方法

## 1. 官方源码地址

| 工具 | 官方/上游仓库 | 本地位置 |
|---|---|---|
| gem5 | <https://github.com/gem5/gem5.git> | `tools/src/gem5` |
| McPAT | <https://github.com/HewlettPackard/mcpat.git> | `tools/src/mcpat` |
| CACTI | <https://github.com/HewlettPackard/cacti.git> | `tools/src/cacti` |
| HotSpot | <https://github.com/uvahotspot/HotSpot.git> | `tools/src/hotspot` |

论文译文没有给出四个仓库的精确 commit。工程脚本默认使用：

- gem5：`v23.1.0.0`，使用稳定标签减少接口漂移；
- McPAT：`master`；
- CACTI：`master`；
- HotSpot：`master`。

下载后，脚本会把每个仓库的实际 commit 写入 `tools/versions/` 和 `manifests/tool_versions.tsv`。开始大规模实验后，不应再随意更新源码。

## 2. 推荐方法：运行项目脚本

```bash
cd /home/zyjiang/Agenticflow/CLIP
./scripts/download_tools.sh
```

脚本具有以下行为：

- 使用浅克隆减少下载量；
- 初始化所需的 Git submodule；
- 已存在有效 Git 仓库时不会覆盖；
- 目标目录非空但不是 Git 仓库时停止，避免覆盖用户文件；
- 下载后记录 commit、描述字符串和远程地址。

检查结果：

```bash
./scripts/check_tools.sh
```

加载路径变量：

```bash
source ./scripts/env.sh
```

## 3. 手动下载命令

如果不使用脚本，可以逐个执行：

```bash
cd /home/zyjiang/Agenticflow/CLIP

git clone --depth 1 --branch v23.1.0.0 --recurse-submodules \
  https://github.com/gem5/gem5.git tools/src/gem5

git clone --depth 1 --branch master --recurse-submodules \
  https://github.com/HewlettPackard/mcpat.git tools/src/mcpat

git clone --depth 1 --branch master --recurse-submodules \
  https://github.com/HewlettPackard/cacti.git tools/src/cacti

git clone --depth 1 --branch master --recurse-submodules \
  https://github.com/uvahotspot/HotSpot.git tools/src/hotspot
```

本工程已经提前建立了四个空目标目录。Git 允许克隆到现有的空目录；如果其中已经存在文件，应先检查文件来源，不要直接覆盖。

### HotSpot 温度输出精度补丁与重建

HotSpot 新下载完成后，应用仓库跟踪的补丁以保留温度输出的六位小数。该补丁只改变
文本温度输出的格式精度，不改变 HotSpot 的热模型、方程或内部 `double` 计算。由于
`tools/src/` 不受 Git 跟踪，必须在每次重新下载 HotSpot 后重新应用此补丁并重建。

使用下列三路检查和重建。反向 dry-run 成功时，源码已经打过补丁；反向检查
失败时，必须再由正向 dry-run 确认源码确实兼容后才可应用。两种 dry-run 都
失败表示源码已部分修改或版本不兼容，此时必须中止，不能让 `make` 使用该源码。
无论补丁原先已应用还是刚成功应用，都必须重建，以免使用旧的 `hotspot` 二进制。

```bash
patch_file=patches/hotspot/0001-six-decimal-temperature-output.patch

if patch -d tools/src/hotspot -p1 --dry-run -R < "$patch_file"; then
  echo "HotSpot six-decimal patch is already applied."
elif patch -d tools/src/hotspot -p1 --dry-run < "$patch_file"; then
  patch -d tools/src/hotspot -p1 < "$patch_file" || exit 1
else
  echo "HotSpot source is partially patched or incompatible; aborting." >&2
  exit 1
fi

make -C tools/src/hotspot hotspot
```

### McPAT 内嵌 CACTI-P 指标补丁与重建

McPAT 已内嵌 CACTI-P；本工程不再为校正后的流程单独运行 CACTI。每次重新下载
McPAT 后，使用下列脚本应用受版本约束的补丁并执行干净重建：

```bash
scripts/build_mcpat.sh
```

脚本会先以反向 dry-run 检测已经应用的补丁；否则仅在正向 dry-run 成功时应用
`patches/mcpat/0001-emit-embedded-cacti-p-metrics.patch`。补丁只在 McPAT print level 5
输出已计算的内嵌 CACTI-P `local_result`，并以 `CLIP_MCPAT_CACTI_P_V1` 记录 L1I、L1D
和共享 L2 的访问时间、周期时间与阵列尺寸；它不改变 CACTI-P 的输入、代价函数或
缓存构建。脚本运行 `make clean` 后重建，并为实际二进制和补丁 SHA-256、构建命令及
UTC 时间戳写入 `tools/build/mcpat/build_provenance.json`。

上游 McPAT makefile 默认强制 `-m32`。脚本默认通过 `CXX=g++ CC=gcc` 进行本机构建，
以避免未安装 32 位开发头文件的主机失败；如需另一套编译器，可在调用前设置
`MCPAT_CXX` 和 `MCPAT_CC`。脚本仅修改不受仓库跟踪的 `tools/src/mcpat` 工具树，绝不
修改 CLIP 仓库中受跟踪的源码。

## 4. 指定其他版本

下载脚本接受环境变量。例如：

```bash
GEM5_REF=v24.1.0.0 \
MCPAT_REF=master \
CACTI_REF=master \
HOTSPOT_REF=master \
./scripts/download_tools.sh
```

也可以把 `REF` 设置为确认存在的 tag 或 branch。若需要使用一个任意 commit，应先正常克隆，再执行：

```bash
git -C tools/src/gem5 fetch --depth 1 origin <commit-sha>
git -C tools/src/gem5 checkout --detach <commit-sha>
```

之后重新运行：

```bash
./scripts/check_tools.sh --write-manifest
```

## 5. 下载前的最低要求

至少需要：

- Git
- 可访问 GitHub 的网络
- 支持 C/C++17 的编译器
- Python 3
- SCons，供 gem5 使用
- GNU Make

下载本身只要求 Git；编译依赖将在下一阶段单独配置。gem5 源码和构建目录会占用较大磁盘空间，建议为整个项目预留至少 20–30 GB。

## 6. 网络或代理环境

如果机器通过代理访问 GitHub，可临时设置：

```bash
export HTTPS_PROXY=http://proxy-host:proxy-port
export HTTP_PROXY=http://proxy-host:proxy-port
./scripts/download_tools.sh
```

不要把包含用户名或密码的代理地址写入仓库文件。

## 7. 下载后的验收标准

以下命令都应成功：

```bash
git -C tools/src/gem5 rev-parse HEAD
git -C tools/src/mcpat rev-parse HEAD
git -C tools/src/cacti rev-parse HEAD
git -C tools/src/hotspot rev-parse HEAD
```

并且存在：

```text
manifests/tool_versions.tsv
tools/versions/gem5.version
tools/versions/mcpat.version
tools/versions/cacti.version
tools/versions/hotspot.version
```

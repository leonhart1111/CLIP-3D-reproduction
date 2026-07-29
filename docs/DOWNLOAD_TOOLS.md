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


# Tool version records

`scripts/download_tools.sh` 在下载完成后为每个工具生成一个 `.version` 文件，并在 `manifests/tool_versions.tsv` 中记录完整清单。

不要手工伪造这些文件；它们应来自实际 Git checkout。


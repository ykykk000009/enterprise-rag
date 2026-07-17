# Document RAG v0.2.1

## 新增

- 每个版本同时发布标准在线版和离线完整版；
- 离线版内置 BGE 中文嵌入、INT8 BGE Reranker、Qwen3 Q8 GGUF；
- 内置 llama.cpp CPU 推理运行时和 libarchive `bsdtar`；
- 离线安装自动选择离线更新包，标准安装选择标准更新包；
- 发布包增加第三方许可证和模型文件 SHA-256 清单。

## 优化

- 标准版默认启用 BGE Reranker，首次使用下载模型后可离线运行；
- 更新器可对离线模型和工具执行备份、替换与失败回滚；
- RAR/7Z 在无独立 `bsdtar` 时可回退到 Windows 系统 `tar.exe`；
- README、安装、技术架构、构建和更新文档统一为 Qwen3 与双版本流程。

## 下载

- `DocQA-v0.2.1-win-x64.zip`：标准在线版；
- `DocQA-v0.2.1-win-x64-offline.zip`：离线完整版。

两个包均为完整首次安装包，解压后运行 `DocQA.exe`。升级时永久保留 `user-data`。

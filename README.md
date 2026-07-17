# Document RAG

本地优先的企业文档智能检索与引用问答软件。它把文件内容、文件名、路径和压缩包内
文件统一建立索引，提供精确字符串查找、混合模糊检索、原文验证与 RAG 问答，并返回
可追溯的文件路径、页码、章节、匹配段落和上下文。

整个检索和问答链在本机运行，不调用商业大模型 API，也不需要 API Key。

## Windows 下载

每个正式版本同时提供两个完整安装包：

| 版本 | Release 文件 | 适合场景 |
|---|---|---|
| 标准在线版 | `DocQA-vX.Y.Z-win-x64.zip` | 包体较小；首次索引或问答时下载 BGE、Reranker 和 Qwen3，之后可离线使用 |
| 离线完整版 | `DocQA-vX.Y.Z-win-x64-offline.zip` | 内置 BGE、INT8 Reranker、Qwen3 Q8 GGUF、llama.cpp 和 libarchive；解压后无需联网 |

从 [GitHub Releases](https://github.com/ykykk000009/enterprise-rag/releases/latest)
下载对应 ZIP，完整解压后双击 `DocQA.exe`。无需安装 Python。

> 两个包都是首次安装可用的完整程序，不是增量补丁。发布包不包含 `user-data`，
> 因此不会携带开发者或其他用户的文档、数据库和模型缓存。

详见 [Windows 下载安装说明](docs/INSTALL_WINDOWS.zh-CN.md)。

## 主要能力

- 一个知识库授权多个目录，支持增量扫描、修改/删除检测、失败重试和知识库删除；
- 解析 PDF、DOC/DOCX、PPT/PPTX、XLS/XLSX、TXT、Markdown、ZIP、RAR、7Z、
  TAR 和 GZ，扫描型文档可使用 RapidOCR；
- 将正文、文件名、文件路径和压缩包内文件名共同用于检索；
- SQLite FTS5 关键词召回 + BGE 向量召回 + RRF 融合 + BGE Cross-Encoder 重排；
- 精确字符串查找与混合模糊检索；
- Qwen3 根据命中块及连续前后文生成答案，并给出原文引用；
- 检索结果按文档和内容去重，提供匹配段落、上下文和在线预览；
- SQLite 与 Qdrant Local 本地持久化，程序更新永久保留 `user-data`。

## 技术摘要

```text
授权目录
  -> 解析 / OCR / 压缩包展开
  -> 结构感知切块（约 420 tokens，40 tokens 重叠）
  -> SQLite 元数据 + FTS5
  -> BGE 向量 -> Qdrant Local

问题
  -> FTS5 + BGE 混合召回
  -> RRF 融合与内容去重
  -> BGE Reranker
  -> 命中块 ±2 连续块
  -> Qwen3 生成与证据审查
  -> 答案 + 引用 + 文件路径 + 原文上下文
```

默认模型：

- 向量：`BAAI/bge-small-zh-v1.5`
- 重排：`BAAI/bge-reranker-base`
- 回答：`Qwen/Qwen3-0.6B`
- 离线回答格式：官方 `Qwen3-0.6B-Q8_0.gguf`，由 `llama.cpp` 在 CPU 上运行

完整实现和安全边界见 [技术架构与模型说明](docs/TECHNICAL_GUIDE.zh-CN.md)。

## 源码启动

要求 Python 3.11 或 3.12。

```powershell
git clone https://github.com/ykykk000009/enterprise-rag.git
cd enterprise-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
New-Item -ItemType Directory -Force knowledge, data, models\huggingface
uvicorn enterprise_document_rag.main:app --host 127.0.0.1 --port 8000 --workers 1
```

打开 `http://127.0.0.1:8000`；OpenAPI 文档位于 `/docs`。

## 验证与发布

```powershell
pytest
ruff check src windows_launcher.py updater.py scripts
```

双版本构建、离线资产准备和 Release 文件清单见
[构建与部署](docs/DEPLOYMENT.zh-CN.md) 与 [软件更新与发布](docs/UPDATES.zh-CN.md)。

## 数据与隐私

`.env`、模型、业务文档、数据库、向量索引、真实评测语料、`tests/` 和构建产物均被
Git 排除。应用仅监听 `127.0.0.1`，当前版本没有完整的组织登录系统，不应直接暴露到
公网。第三方模型和工具许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

本项目代码采用 [MIT License](LICENSE)。

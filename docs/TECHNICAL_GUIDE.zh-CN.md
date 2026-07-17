# 技术架构与模型说明

## 1. 定位

Document RAG 是单进程、本地优先的企业文档检索增强生成应用。它只读访问用户明确
授权的目录，增量解析与索引文档，并提供可追溯到文件、压缩包成员、页码、章节和原文
段落的检索与问答。

## 2. 架构

```text
浏览器
  -> FastAPI / Uvicorn
     -> 知识库与多目录授权
     -> 后台扫描任务
        -> 文件解析 / OCR / 压缩包成员解析
        -> 结构感知切块
        -> SQLite WAL + FTS5
        -> BGE Embedding + Qdrant Local

查询
  -> 知识库与文档范围过滤
  -> FTS5 + BGE Dense Retrieval
  -> Reciprocal Rank Fusion
  -> 文档/内容去重
  -> BGE Cross-Encoder Reranker
  -> 命中块及连续前 2 / 后 2 块
  -> Qwen3 生成
  -> 通用证据审查、引用校验
```

| 层 | 实现 |
|---|---|
| Web/API | FastAPI、Uvicorn、原生 HTML/CSS/JavaScript |
| 元数据/全文检索 | SQLite WAL、FTS5 |
| 向量索引 | Qdrant Local |
| 文档解析 | PyMuPDF、python-docx、openpyxl、xlrd、Office COM、libarchive |
| OCR | RapidOCR、ONNX Runtime |
| 向量 | `BAAI/bge-small-zh-v1.5` |
| 重排 | `BAAI/bge-reranker-base` |
| 回答 | `Qwen/Qwen3-0.6B` |

## 3. 检索和问答链路

索引会把正文、文件名、规范化路径和压缩包内成员名共同写入可检索内容。文本按文档
结构优先切分，并受 `420 tokens` 目标长度、`40 tokens` 重叠、`180-650 tokens`
边界约束。表格按工作表、行和逻辑区域生成可引用文本。

查询阶段同时执行：

1. SQLite FTS5 关键词召回；
2. BGE 向量语义召回；
3. RRF 融合；
4. 近重复内容和同一文档重复命中合并；
5. BGE Cross-Encoder 对候选重新排序。

“原文验证”直接展示检索证据。“提交问题”会为命中块补齐连续前后各两个块，再将核心
证据交给 Qwen3；生成后执行证据完整性审查、引用编号修正和授权范围校验。证据不足时
拒绝作答，不把模型常识伪装成文档结论。

## 4. 在线版与离线版模型

两种版本使用相同模型家族和检索策略：

| 组件 | 标准在线版 | 离线完整版 |
|---|---|---|
| Embedding | 首次从 Hugging Face 下载 BGE | 内置 BGE safetensors |
| Reranker | 首次下载 BGE CrossEncoder | 内置官方 ONNX 动态 INT8 转换 |
| Qwen3 | Transformers 本地加载 | 官方 Q8_0 GGUF + llama.cpp |
| 网络 | 首次使用需要 | 强制 `HF_HUB_OFFLINE=1` |

离线量化是为了让程序、三套模型和工具保持在 GitHub 单 Release 附件 2 GB 上限内。
模型来源、许可证、文件大小和 SHA-256 写在包内 `MODEL_MANIFEST.json`。

项目不调用云端大模型 API，不读取商业模型 API Key。可将 `LLM_BACKEND` 改为
`extractive` 关闭生成模型；此时仅摘录证据，资源占用更低，但归纳能力明显下降。

## 5. 格式与解析边界

支持 PDF、DOC/DOCX、PPT/PPTX、XLS/XLSX/XLSM、TXT、Markdown、ZIP、RAR、7Z、
TAR 和 GZ。压缩包处理设有成员数、单成员大小、总解压大小和压缩比限制，并拒绝路径
穿越。旧版 `.doc/.ppt` 在 Windows 上使用本机 Office 转换；不分发 Microsoft Office。
离线版内置 BSD 许可的 libarchive `bsdtar` 用于 RAR/7Z。

## 6. 持久化与更新

所有用户状态位于 `user-data`：

- `data/agent.db`：知识库、文档、任务、分块和 FTS5；
- `data/qdrant/`：向量索引；
- `models/huggingface/`：标准版下载缓存；
- `knowledge/`：可选本地资料目录；
- `updates/` 与 `backups/`：更新包、日志和回滚数据。

更新程序文件时永久保留 `user-data`。若模型、切块规则或向量索引版本改变，保留知识库
和目录配置，只重建向量索引。

## 7. 配置与安全

配置由 `pydantic-settings` 从环境变量或 `.env` 读取，默认仅监听 `127.0.0.1` 且
Qdrant Local 只允许单进程访问。源码部署应保持 `--workers 1`。

当前版本面向本机单用户，没有完整的账号、租户和公网认证体系。不要直接暴露到公网。
生产环境还需要反向代理 HTTPS、身份认证、目录白名单、最小权限账号、备份、磁盘监控
和依赖漏洞管理。

第三方模型和二进制许可见
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

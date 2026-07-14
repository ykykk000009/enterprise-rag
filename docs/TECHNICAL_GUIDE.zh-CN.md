# 技术架构与大模型调用说明

## 1. 项目定位

本项目是单进程、本地优先的企业文档 RAG（检索增强生成）应用。它读取管理员明确授权的目录，增量解析和索引文档，然后为自然语言检索与问答提供可追溯引用。源文件只读，索引状态可以持久化并在重启后恢复。

## 2. 总体架构

```text
浏览器 / API 客户端
        |
     FastAPI
        |
        +-- 数据源扫描 -> 后台工作线程 -> 解析/OCR -> 分块 -> BGE 向量化
        |                                      |          |
        |                                   SQLite     Qdrant Local
        |                                  元数据/FTS5    向量
        |
        +-- 查询 -> ACL/知识库过滤 -> 向量检索 + FTS5 -> 可选重排
                                                       |
                                                  本地 Qwen 生成
                                                       |
                                               答案 + 原文引用
```

主要组件如下：

| 层 | 实现 | 用途 |
|---|---|---|
| Web/API | FastAPI + Uvicorn | 页面、健康检查、知识库、数据源、索引、检索、问答 API |
| 元数据与关键词检索 | SQLite（WAL、FTS5） | 文档、版本、任务、分块、ACL 范围与全文检索 |
| 向量库 | Qdrant Local | 本地持久化语义向量 |
| 文档解析 | PyMuPDF、python-docx、openpyxl、xlrd | PDF、DOCX、Excel、文本和 ZIP |
| OCR | RapidOCR + ONNX Runtime | 扫描型 PDF 或低文本量文档图片识别 |
| 向量模型 | `BAAI/bge-small-zh-v1.5` | 文档分块和查询向量化 |
| 重排模型 | `BAAI/bge-reranker-base` | 可选，对混合检索候选重新排序 |
| 生成模型 | `Qwen/Qwen2.5-0.5B-Instruct` | 基于检索证据生成带编号引用的答案 |

## 3. 大模型/模型使用位置与 API Key

| 位置 | 配置项 | 默认方式 | 是否需要 API Key | 是否必须 |
|---|---|---|---|---|
| 向量化 | `EMBEDDING_BACKEND=bge` | Sentence Transformers 在本机加载 BGE | 否 | 默认语义检索需要；可改为测试用 `hash` |
| 答案生成 | `LLM_BACKEND=qwen_transformers` | Transformers 在本机加载 Qwen | 否 | 否；可改为 `extractive`，直接摘录证据 |
| 检索重排 | `RERANKER_ENABLED=true` | CrossEncoder 在本机加载 BGE reranker | 否 | 否，默认关闭 |
| OCR | `OCR_ENABLED=true` | RapidOCR/ONNX 本地推理 | 否 | 否，但扫描件通常需要 |

结论：**当前代码没有调用云端大模型 API，也没有读取任何大模型 API Key。** 默认模型首次使用时会从 Hugging Face Hub 下载，因此首次部署通常需要互联网，但下载公开模型通常不要求 Hugging Face Token。若网络受限，可在可联网机器预下载模型目录，再将模型缓存复制到部署机并设置 `HUGGINGFACE_HOME`。对于受限/需授权模型，Hugging Face 可能要求 `HF_TOKEN`，但本项目的三个默认模型均不是通过商业推理 API 调用。

若设置 `LLM_BACKEND=extractive`，问答只取最相关证据的首句，不加载 Qwen，资源占用和首次下载量会降低，但回答的归纳能力也会下降。

## 4. 数据处理链路

1. 用户创建知识库并登记数据源目录。
2. 扫描器只遍历登记的数据源，按文件指纹判断新增、修改或删除。
3. 后台线程逐文件解析；低文本量 PDF/DOCX 可触发 OCR。
4. 文本按配置的长度和重叠窗口分块，写入 SQLite，并批量生成向量写入 Qdrant Local。
5. 查询时先限定知识库和允许访问的文档，再合并向量召回与 FTS5 召回结果。
6. 可选重排模型计算问题—段落相关性。
7. 问答模块仅把检索证据交给本地 Qwen，并校验引用对应当前可见文档分块；证据不足时拒绝作答。

## 5. 支持格式与持久化目录

支持 `.pdf`、`.docx`、`.xlsx`、`.xlsm`、`.xls`、`.txt`、`.md` 和 `.zip`。ZIP 只读取受支持的内部文件，并带有成员数、单文件大小、总解压大小和压缩比限制。

默认持久化内容：

- `data/agent.db`：SQLite 元数据、任务和 FTS5 索引；
- `data/qdrant/`：Qdrant Local 向量数据；
- `models/huggingface/`：模型缓存（配置 `HUGGINGFACE_HOME` 后）；
- `knowledge/`：示例授权文档根目录，部署时应只读挂载。

这些内容均不应提交到 GitHub。备份时应停止应用，或使用项目的 `scripts/backup.py`，确保 SQLite 与向量数据状态一致。

## 6. 关键配置

配置由 `pydantic-settings` 从环境变量或 `.env` 读取。完整默认值见 `src/enterprise_document_rag/config.py`，常用项见 `.env.example`。

特别注意：

- `AUTHORIZED_ROOTS` 可填写逗号分隔的多个目录，但当前版本尚未在创建数据源接口中强制校验该白名单，见下方安全边界；
- `DATABASE_URL` 当前应使用 SQLite URL；
- `QDRANT_PATH` 必须是可持久化、可写目录；
- CPU 是默认设备，可在具备兼容 PyTorch 环境时将模型设备改为 `cuda`；
- 同一个 `data/qdrant` 目录不要由多个应用进程同时打开，所以生产启动也必须保持 `--workers 1`。

## 7. API 概览

- `GET /health/live`、`GET /health/ready`：存活和就绪检查；
- `/api/v1/knowledge-bases`：知识库管理；
- `/api/v1/sources`、`/api/v1/sources/{id}/scan`：数据源与扫描；
- `/api/v1/documents`：索引状态、失败重试、重建和逻辑删除；
- `POST /api/v1/search`：混合检索；
- `POST /api/v1/query`：RAG 问答与引用；
- `POST /api/v1/field-search`：字段在文档中的精确查找。

启动后可在 `/docs` 查看 OpenAPI 交互文档。

## 8. 当前边界与生产化建议

这是 MVP：应用已有查询阶段的知识库/文档范围过滤，但没有完整的组织账号、登录、租户和反向代理认证。另一个重要边界是：`AUTHORIZED_ROOTS` 已定义在配置中，但 `POST /api/v1/sources` 当前只验证提交的路径是一个真实目录，并未验证它属于该白名单。也就是说，能够调用该接口的人可以登记应用进程有权读取的任意目录。

因此不要直接裸露到公网。Docker 部署应只挂载明确授权的宿主机目录；源码部署应使用权限受限的专用系统账号。生产使用还应增加：根目录白名单强制校验、反向代理 HTTPS、身份认证、网络访问控制、定期备份、磁盘容量监控、日志采集和依赖漏洞更新。多实例横向扩展需要先把 Qdrant Local 替换为 Qdrant Server，并重新设计任务协调机制。

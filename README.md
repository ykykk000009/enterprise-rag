<img width="2480" height="1350" alt="image" src="https://github.com/user-attachments/assets/92363b93-f29c-4ecc-9b84-b3854ce80094" /># Enterprise Document Local RAG Agent

本地优先的企业文档检索与引用问答应用。支持 PDF、DOCX、XLS/XLSX、TXT、Markdown 和 ZIP，可返回文件、页码、章节与原文引用。

默认在本机运行 BGE 向量模型和 Qwen2.5 生成模型，**不需要商业大模型 API Key**。模型首次使用时从 Hugging Face 下载，之后可离线运行。

## Windows 下载

从 [GitHub Releases](https://github.com/ykykk000009/enterprise-rag/releases/latest) 下载 `EnterpriseDocumentRAG-windows-x64-online.zip`，完整解压后双击 `EnterpriseDocumentRAG.exe`。无需安装 Python。

详见 [Windows 下载安装说明](docs/INSTALL_WINDOWS.zh-CN.md)。

## 主要能力

- 授权目录的增量扫描、更新与删除检测；
- PDF/DOCX/Excel/文本解析及扫描件 OCR；
- SQLite FTS5 与 Qdrant Local 混合检索；
- 可选 BGE 重排、本地 Qwen 证据问答；
- 文件、页码、章节、引用片段定位；
- 单进程、CPU 默认、本地持久化。

## 网页展示
<img width="2480" height="1350" alt="image" src="https://github.com/user-attachments/assets/2e74e01e-22fc-4893-9c4c-94d02d022aeb" />
  
## 源码启动

要求 Python 3.11 或 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
New-Item -ItemType Directory -Force knowledge, data, models\huggingface
uvicorn enterprise_document_rag.main:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000`；OpenAPI 文档位于 `/docs`。

## 验证

```powershell
pytest
ruff check src tests windows_launcher.py
```

## 文档

- [技术架构与模型说明](docs/TECHNICAL_GUIDE.zh-CN.md)
- [Windows 下载安装](docs/INSTALL_WINDOWS.zh-CN.md)
- [构建与部署](docs/DEPLOYMENT.zh-CN.md)
- [评测说明](docs/T08_EVALUATION.md)

## 数据与安全

`.env`、模型、业务文档、数据库、索引、真实评测语料和构建产物均被 Git 排除。当前 MVP 没有完整登录系统，不能直接暴露到公网。详见技术文档中的安全边界。

本项目采用 [MIT License](LICENSE)。

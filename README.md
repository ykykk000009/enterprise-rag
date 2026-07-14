# Enterprise Document Local RAG Agent

一个本地优先的企业文档检索与有引用问答 MVP。应用可索引管理员授权目录中的 PDF、DOCX、XLS/XLSX、TXT、Markdown 和 ZIP 文档，通过混合检索返回文件、页码、章节和原文引用。

默认方案在本机运行 BGE 向量模型和 Qwen2.5 生成模型，**不需要 OpenAI、阿里云或其他商业大模型 API Key**。首次启动相关功能时需要联网从 Hugging Face 下载模型；也可以预先下载模型后离线运行。

## 普通用户下载

Windows 10/11 x64 用户请从 [GitHub Releases](https://github.com/ykykk000009/enterprise-rag/releases/latest) 下载 `EnterpriseDocumentRAG-windows-x64-online.zip`。无需安装 Python，解压后双击 EXE 即可使用。

完整步骤见：[Windows 下载安装说明](docs/INSTALL_WINDOWS.zh-CN.md)。

## Windows 下载即用版

本项目的用户界面本来就是网页，但可以打包成 Windows 本地桌面应用：用户解压 ZIP 后双击 `EnterpriseDocumentRAG.exe`，启动器会运行本机服务并自动打开浏览器。用户无需安装 Python，文件和模型不会上传到云端。

构建在线版（首次使用模型时联网下载）：

```powershell
.\packaging\windows\build_windows.ps1
```

构建携带现有模型缓存的离线完整版：

```powershell
.\packaging\windows\build_windows.ps1 -IncludeModels -ModelCachePath ..\models\huggingface
```

ZIP 产物位于 `dist/windows/`。详细说明见[Windows 应用打包与交付](docs/WINDOWS_APP_PACKAGING.zh-CN.md)。

## 快速启动

要求 Python 3.11 或 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
New-Item -ItemType Directory -Force knowledge, data, models\huggingface
uvicorn enterprise_document_rag.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`，API 文档位于 `http://127.0.0.1:8000/docs`。把待索引文档放入 `knowledge/`，在页面中新建知识库、添加该目录为数据源并执行扫描。

Linux/macOS 的激活命令为 `source .venv/bin/activate`，复制配置可使用 `cp .env.example .env`。

## 文档

- [技术架构与大模型调用说明](docs/TECHNICAL_GUIDE.zh-CN.md)
- [Windows 下载安装说明](docs/INSTALL_WINDOWS.zh-CN.md)
- [GitHub 发布、打包与部署指南](docs/DEPLOYMENT.zh-CN.md)
- [仓库上传内容说明](docs/REPOSITORY_CONTENTS.zh-CN.md)
- [评测说明](docs/T08_EVALUATION.md)

## 测试

```powershell
python -m pip install -e ".[dev]"
pytest
```

## 安全提醒

不要提交 `.env`、`data/`、`models/`、`knowledge/` 或真实业务文档。部署到局域网或公网前，应在应用前增加 HTTPS、身份认证和访问控制；当前 MVP 本身没有完整的用户登录系统。

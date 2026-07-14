# Windows 应用打包与交付

## 1. 应该做成 EXE，还是网站？

推荐做成“**EXE 启动器 + 本机网页界面**”。这不是部署在公网的网站：FastAPI 服务运行在用户自己的电脑上，只监听 `127.0.0.1`；EXE 启动后自动打开默认浏览器。这样既保留当前成熟的 Web UI，也能达到普通用户双击使用、无需安装 Python的目标。

不建议把所有内容强塞进单个 EXE。PyTorch、Transformers、OCR 和文档解析依赖很多，单文件模式启动时还要把大量文件解压到临时目录，启动慢且容易触发杀毒软件。项目采用 PyInstaller `--onedir` 便携目录，再整体压缩成 ZIP。

## 2. 用户体验

用户拿到 ZIP 后：

1. 完整解压；
2. 双击 `EnterpriseDocumentRAG.exe`；
3. 等待浏览器自动打开；
4. 创建知识库、选择文档目录并扫描；
5. 关闭启动器窗口即可停止服务。

便携包带有 `portable.mode` 标记，运行数据保存在 EXE 同目录的 `user-data/`。因此把整个解压目录放在 E 盘，程序、模型缓存和索引都会留在 E 盘。升级时应保留 `user-data/`，或者先做备份。删除 `portable.mode` 后，启动器会改用 `%LOCALAPPDATA%\EnterpriseDocumentRAG`。启动器会在 8765—8784 中选择一个空闲端口，仅允许本机访问。

## 3. 两种发布包

### 在线轻量版

“轻量”是相对于携带模型而言，Python、PyTorch 和 OCR 依赖仍会让 ZIP 比普通桌面程序大。用户第一次执行索引时下载 BGE，第一次智能问答时下载 Qwen；之后模型缓存在用户目录，可以离线运行。

```powershell
Set-Location E:\findfileagent\codex_agent_mvp
.\packaging\windows\build_windows.ps1
```

产物：`dist/windows/EnterpriseDocumentRAG-windows-x64-online.zip`。

### 离线完整版

把已经下载好的 Hugging Face 缓存放入发行包，目标电脑不需要联网。当前工作区的缓存默认位于 `E:\findfileagent\models\huggingface`：

```powershell
Set-Location E:\findfileagent\codex_agent_mvp
.\packaging\windows\build_windows.ps1 `
  -IncludeModels `
  -ModelCachePath E:\findfileagent\models\huggingface
```

产物：`dist/windows/EnterpriseDocumentRAG-windows-x64-offline.zip`。离线包可能达到数 GB，适合通过网盘、对象存储或移动硬盘交付，不适合直接提交到 GitHub 仓库。

## 4. 构建环境

构建机要求：

- Windows 10/11 x64；
- Python 3.11 或 3.12 x64，并已加入 PATH；也可以用 `-BootstrapPython C:\Python311\python.exe` 指定解释器；
- 首次构建可访问 PyPI；
- 足够的磁盘空间，建议至少预留 15 GB；
- 离线版还需完整的模型缓存。

构建脚本会创建独立的 `.venv-package`、安装项目与 PyInstaller、运行全部测试、生成 `--onedir` 程序目录，并压缩成 ZIP。不要在 Linux 上构建 Windows EXE；PyInstaller 应在目标操作系统上构建。

## 5. EXE 与模型的关系

EXE 只是启动器和 Python 应用的封装，并不意味着模型必须塞进 EXE：

| 内容 | 在线版 | 离线完整版 |
|---|---|---|
| Python 与项目依赖 | 包含 | 包含 |
| 应用前端与后端 | 包含 | 包含 |
| BGE/Qwen 模型 | 首次使用下载 | 随包携带 |
| API Key | 不需要 | 不需要 |
| 用户业务文档 | 不包含 | 不包含 |

## 6. 发布到 GitHub

GitHub 仓库只提交源码和构建脚本。构建完成后，在 GitHub Release 中上传 ZIP，用户从 Releases 页面下载，而不是让用户克隆源码。

建议发布两个资产：

- `EnterpriseDocumentRAG-windows-x64-online.zip`：适合可以联网的用户；
- 离线完整版放到支持大文件的平台，并在 Release 说明中给出下载链接和 SHA-256。

发布前应在一台没有安装 Python的干净 Windows 电脑或 Windows Sandbox 中验证：解压、启动、模型下载、文档索引、问答、退出和再次启动。还应对 ZIP 做病毒扫描，并核对 PyTorch、Qwen、BGE、RapidOCR 等依赖及模型许可证是否符合你的分发场景。

## 7. 后续可选：制作安装程序

如果需要桌面快捷方式、开始菜单、卸载入口和版本升级，可以在便携目录外再套一层 Inno Setup 或 WiX 安装器。第一版建议先发布 ZIP 便携版，稳定后再做安装器；核心 EXE 和应用代码无需改变。

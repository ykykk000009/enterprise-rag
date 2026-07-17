# Windows 下载安装

## 选择版本

从 [GitHub Releases](https://github.com/ykykk000009/enterprise-rag/releases/latest)
下载以下任一完整包：

| 版本 | 文件名 | 是否需要联网 |
|---|---|---|
| 标准在线版 | `DocQA-vX.Y.Z-win-x64.zip` | 首次建立索引和首次问答需要下载模型 |
| 离线完整版 | `DocQA-vX.Y.Z-win-x64-offline.zip` | 不需要；模型和解析工具已内置 |

标准版和离线版都包含 `DocQA.exe`、前端、OCR、解析器、更新器和全部程序依赖，均可
用于首次安装。离线版额外包含 BGE、Reranker、Qwen3 和本地推理/压缩包工具。

## 安装

1. 下载 ZIP 以及同名 `.sha256` 文件。
2. 计算 ZIP 的 SHA-256，并与校验文件比较。
3. 将 ZIP 完整解压到 E 盘，例如 `E:\DocQA`，不要在压缩包内直接运行。
4. 双击 `DocQA.exe`，等待浏览器自动打开。
5. 创建知识库，授权一个或多个文档目录并扫描。
6. 关闭启动器窗口即可退出服务。

适用于 Windows 10/11 x64，无需安装 Python或申请大模型 API Key。程序只监听本机
`127.0.0.1`。扫描型 PDF 和 Office 文档 OCR 会比普通文本解析慢。

## 文件和数据

```text
DocQA\
  DocQA.exe
  Updater.exe
  _internal\
  user-data\
    data\agent.db
    data\qdrant\
    models\huggingface\   # 标准版首次使用时下载
    knowledge\
    launcher.log
```

离线完整版还包含安装目录级的 `models\`、`tools\`、`licenses\`、
`MODEL_MANIFEST.json` 和 `offline.mode`。这些是只读程序资产；用户数据仍只保存在
`user-data`。

## 更新与迁移

程序启动后后台检查 GitHub Release。标准版自动选择标准更新包，离线版自动选择离线
完整更新包。更新器会先备份 SQLite 和旧程序，永久保留 `user-data`，失败时自动回滚。

手工迁移也很简单：

1. 关闭旧版。
2. 备份旧目录中的整个 `user-data`。
3. 将新版解压到新目录。
4. 把旧版 `user-data` 移到新版目录。
5. 启动新版，确认知识库正常后再删除旧程序目录。

不要把新 ZIP 解压到仍在运行的旧目录，也不要在升级前删除 `user-data`。

## 常见问题

- **未知发布者**：当前版本没有商业代码签名，请只从本项目 Release 下载并核对 SHA-256。
- **标准版模型下载失败**：检查 Hugging Face 网络连接；也可改用离线完整版。
- **离线版仍请求联网**：确认运行目录中存在 `offline.mode` 和完整的 `models/tools`。
- **浏览器未打开**：保持启动器运行，点击“打开应用”，并查看 `user-data/launcher.log`。
- **RAR/7Z 解析失败**：离线版使用内置 libarchive；标准版优先使用系统 `tar/bsdtar`。
- **卸载**：关闭程序后删除整个目录；需要保留知识库时先备份 `user-data`。

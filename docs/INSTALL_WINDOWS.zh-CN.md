# Windows 下载安装说明

## 下载

进入项目的 [GitHub Releases 页面](https://github.com/ykykk000009/enterprise-rag/releases/latest)，下载：

`EnterpriseDocumentRAG-windows-x64-online.zip`

该版本适用于 Windows 10/11 x64。ZIP 约 351.4 MB，不需要另外安装 Python，也不需要申请大模型 API Key。

## 安装与使用

1. 将 ZIP 完整解压到 E 盘，例如 `E:\EnterpriseDocumentRAG`。不要直接在压缩包内运行 EXE。
2. 双击 `EnterpriseDocumentRAG.exe`。
3. 等待启动窗口显示服务已启动，默认浏览器会自动打开应用。
4. 在页面中新建知识库。
5. 添加需要检索的文档目录，然后执行扫描。
6. 索引完成后，可以使用字段定位、文档检索和带引用问答。
7. 使用完毕后关闭启动器窗口，程序即退出。

## 特点

- 用户不需要安装 Python；
- 不需要 OpenAI、阿里云或其他商业大模型 API Key；
- 服务只监听本机 `127.0.0.1`，不是公网网站；
- 程序、数据库、模型缓存和索引均保存在解压目录的 `user-data`；
- 把程序解压到 E 盘后，默认不会占用 C 盘存放模型和索引；
- 当前发布包为在线版，约 351.4 MB；
- 第一次索引和智能问答时需要联网下载 BGE/Qwen 本地模型；
- 模型下载完成后可以离线使用；
- 发布前已实际启动 EXE，并验证本机健康检查接口正常。

## 第一次使用为什么较慢

首次使用可能需要下载并加载本地模型。普通文本型 PDF/DOCX 通常较快；没有文字层的扫描 PDF 需要逐页 OCR，在 CPU 电脑上会明显更慢。请保持启动器窗口开启，并避免电脑进入睡眠。

## 数据目录

默认目录结构：

```text
EnterpriseDocumentRAG/
├─ EnterpriseDocumentRAG.exe
├─ _internal/
├─ portable.mode
└─ user-data/
   ├─ data/                 数据库和向量索引
   ├─ knowledge/            默认知识目录
   ├─ models/huggingface/   本地模型缓存
   └─ launcher.log          启动日志
```

升级程序前，请备份 `user-data`。更新时可以替换其他程序文件，但应保留原来的 `user-data` 文件夹。

## 常见问题

### Windows 提示未知发布者

当前个人发布包没有商业代码签名证书。请确认文件来自本项目的 GitHub Releases，并核对 Release 页面提供的 SHA-256 后再运行。

### 浏览器没有自动打开

确认启动器窗口仍在运行，然后点击“打开应用”。也可以检查 `user-data/launcher.log`。

### 首次模型下载失败

检查网络是否能访问 Hugging Face，关闭程序后重新启动并重试。已经完整下载的模型会保留在 `user-data/models/huggingface`，不会每次重复下载。

### 如何彻底卸载

先关闭启动器，再删除整个 `EnterpriseDocumentRAG` 文件夹即可。删除前如需保留知识库索引，请备份 `user-data`。

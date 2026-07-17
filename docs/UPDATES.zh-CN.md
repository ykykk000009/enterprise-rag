# 软件更新与发布

应用通过 `ykykk000009/enterprise-rag` 的最新正式 GitHub Release 检查更新。启动检查
异步执行，每 24 小时最多一次；网络失败不影响本地功能。

## 每个版本的 Release 文件

```text
DocQA-vX.Y.Z-win-x64.zip
DocQA-vX.Y.Z-win-x64.zip.sha256
DocQA-vX.Y.Z-win-x64-offline.zip
DocQA-vX.Y.Z-win-x64-offline.zip.sha256
```

两个 ZIP 都是可首次安装的完整包。标准安装自动选择标准更新包；安装目录存在
`offline.mode` 时自动选择离线更新包。GitHub Release 附件中不包含 `user-data`。

## 发布步骤

1. 更新 `src/enterprise_document_rag/__init__.py` 中的三段式版本号。
2. 更新 Release Notes、模型/索引/切块版本说明。
3. 准备 `.offline-assets`。
4. 运行双版本构建脚本。
5. 检查隐私、许可证、文件清单、SHA-256 和离线启动。
6. 提交并推送源码，创建 `vX.Y.Z` 标签和正式 Release。
7. 上传四个附件，不要手工修改打包后的 ZIP。

## 更新执行

用户点击“立即更新”后：

1. 下载与当前安装类型匹配的 ZIP；
2. 同时验证 GitHub asset digest 和 `.sha256`；
3. 备份 `user-data/data/agent.db`；
4. 将独立 `Updater.exe` 复制到临时目录；
5. 关闭主程序；
6. 备份旧程序；离线更新同时备份旧模型和工具；
7. 替换程序、图标和对应版本资产；
8. 保留整个 `user-data`；
9. 启动新版，失败则恢复旧程序、模型、工具和 SQLite。

日志位于 `user-data/updates/updater.log`，回滚文件位于
`user-data/updates/vX.Y.Z/rollback`。

## 兼容性规则

- 只更新页面、回答模型或普通功能：继续使用原 SQLite 和 Qdrant；
- 更换向量模型、切块规则或索引格式：保留知识库和授权目录，只重建向量；
- 更新程序图标：ZIP 中的 `docqa.ico` 与新 EXE 一起替换，并刷新 Windows 图标缓存；
- `user-data` 永不进入 Release，也永不被更新器覆盖。

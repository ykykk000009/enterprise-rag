# 软件更新与发布

应用通过 GitHub 仓库 `ykykk000009/enterprise-rag` 的最新正式 Release 检查更新。
启动检查在后台运行，每 24 小时最多执行一次；设置窗口也可以手动检查。

## 发布新版本

1. 修改 `src/enterprise_document_rag/__init__.py` 中的 `__version__`，使用三段式版本号。
2. 运行测试和 Windows 打包脚本：

   ```powershell
   .\packaging\windows\build_windows.ps1
   ```

3. 在 GitHub 创建标签为 `vX.Y.Z` 的正式 Release。
4. 上传以下两个文件：

   ```text
   DocQA-vX.Y.Z-win-x64.zip
   DocQA-vX.Y.Z-win-x64.zip.sha256
   ```

ZIP 内含主程序、独立更新器、依赖目录、前端资源、`portable.mode` 和
`version.json`、当前版本的 `docqa.ico`，不含 `user-data`。主程序图标也会在
打包时嵌入 `DocQA.exe`。不要手工修改打包后的文件，否则 GitHub
附件 digest 和 `.sha256` 将无法通过双重校验。

## 用户数据保护

更新器从 Windows 临时目录运行，并执行以下操作：

- 等待旧主程序退出；
- 在 `user-data/updates/vX.Y.Z/rollback/program` 保存旧程序；
- 备份 SQLite 到 `user-data/backups/agent-before-vX.Y.Z.db`；
- 替换程序文件，但跳过 `user-data`、`portable.mode` 和旧版外置 `models`；
- 替换 `DocQA.exe` 和 `docqa.ico` 后通知 Windows 刷新图标缓存；
- 新版启动失败时恢复旧程序和 SQLite；
- 日志写入 `user-data/updates/updater.log`。

Qdrant、模型、知识库配置和授权目录不会被更新包覆盖。更换嵌入模型、切块规则
或索引格式时，应在版本说明中明确要求用户重建向量索引。

## 首次启用说明

已经发布的 v0.1.0 安装包不含自动更新代码和 `Updater.exe`，因此该版本需要用户
手动安装一次包含本功能的新版本。此后的版本可以在应用内完成更新。

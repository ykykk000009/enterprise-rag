# Document RAG v0.2.3

## 修复

- 修复离线版 Qwen3 回答中混入 llama.cpp 启动 Logo、模型信息、英文命令和性能统计的问题；
- 对长上下文、换行被规范化的提示词增加稳定的回答截取逻辑；
- 增加真实 llama.cpp 启动输出的回归测试；
- 更新内置 GitHub 仓库地址为 `ykykk000009/DocQA-APP`。

本版本不更换 BGE、Reranker、Qwen3 模型、切块规则、数据库结构或向量索引格式，已有
知识库和向量索引可以继续使用。

## 下载

- `DocQA-v0.2.3-win-x64.zip`：标准在线版；
- `DocQA-v0.2.3-win-x64-offline.zip`：离线完整版。

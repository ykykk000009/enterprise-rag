# Prompt to start implementation

Open this repository and build the lightweight local enterprise document RAG MVP.

Before editing code, read `AGENTS.md`, `SPEC.md`, `spec.json`, `tasks.json`, and `acceptance.json`. Treat `AGENTS.md` as mandatory constraints. Implement `tasks.json` sequentially from T01. Do not introduce any dependency listed in `spec.json.forbidden_mvp_dependencies`.

For each task:

1. State the intended files and tests.
2. Implement the smallest complete change.
3. Run the task checks plus existing tests.
4. Fix failures before continuing.
5. Report completed acceptance evidence and remaining limitations.

Start with T01 only. Do not implement the knowledge graph until T01–T08 pass.
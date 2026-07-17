# Document RAG v0.3.0

## Online Transformers package

This release replaces the previous standard/offline package strategy with a
single online-models package intended for new installations and future
updates.

The package includes:

- the complete `DocQA.exe` application runtime;
- `BAAI/bge-small-zh-v1.5` in Transformers format;
- `BAAI/bge-reranker-base` in Transformers format;
- OCR, Office and archive parsing tools;
- the updater and application documentation.

The package deliberately does not include Qwen3, user documents, SQLite
metadata, Qdrant vectors, chunk results, local test data, `.env` files or the
`tests` directory.

On first launch, the application downloads `Qwen/Qwen3-0.6B` from the official
Hugging Face repository:

https://huggingface.co/Qwen/Qwen3-0.6B

The model is stored under `user-data/models/huggingface`. The download is
shown in the web interface, can be retried, and does not block document
parsing or exact string search. RAG question answering remains disabled until
the model is complete.

Existing releases v0.2.2 and v0.2.3 are retired. Future application updates
use `DocQA-vX.Y.Z-win-x64.zip` and preserve `user-data`, including downloaded
models, SQLite metadata and the Qdrant local index.

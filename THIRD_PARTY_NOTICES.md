# Third-Party Notices

Document RAG is distributed under the MIT License. The application and its
Windows packages also contain or can download third-party components under
their own licenses.

## Complete offline package

| Component | Purpose | License | Upstream |
|---|---|---|---|
| BAAI/bge-small-zh-v1.5 | Chinese text embeddings | MIT | https://huggingface.co/BAAI/bge-small-zh-v1.5 |
| BAAI/bge-reranker-base | Cross-encoder reranking; distributed as a dynamic INT8 ONNX conversion of the official ONNX model | MIT | https://huggingface.co/BAAI/bge-reranker-base |
| Qwen/Qwen3-0.6B-GGUF | Local answer generation, Q8_0 GGUF | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-0.6B-GGUF |
| llama.cpp | Local GGUF inference runtime | MIT | https://github.com/ggml-org/llama.cpp |
| libarchive / bsdtar | RAR and 7z archive reading | BSD-2-Clause | https://libarchive.org |

The offline package contains the applicable license texts in `licenses/` and
an exact file/source/hash inventory in `MODEL_MANIFEST.json`.

The Windows `bsdtar` runtime is built from the Anaconda `defaults` packages and
uses separate open-source runtime libraries: bzip2, GNU libiconv, libxml2, LZ4,
OpenSSL, XZ Utils, zlib and Zstandard. Their exact versions, package URLs,
SHA-256 values and license identifiers are recorded in
`licenses/libarchive-dependencies/CONDA_PACKAGES.json`. License texts are
included beside that manifest. Some of these components are available under
LGPL/GPL terms; see `THIRD_PARTY_SOURCE_OFFER.md` for corresponding source.

## Python dependencies

The packaged application also includes open-source Python libraries such as
FastAPI, Uvicorn, PyTorch, Transformers, Sentence Transformers, ONNX Runtime,
Qdrant Client, PyMuPDF, python-docx, openpyxl, RapidOCR and their transitive
dependencies. Their copyright and license metadata remain available in the
packaged `_internal` distribution metadata.

No Microsoft Office binaries, proprietary RAR binaries, user documents,
databases, credentials or private environment files are included.

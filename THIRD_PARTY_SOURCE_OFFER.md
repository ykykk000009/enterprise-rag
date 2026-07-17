# Third-Party Source Availability

The complete offline package contains unmodified open-source model files,
native executables and dynamically linked libraries. Exact binary package
URLs, versions and SHA-256 values are recorded in `MODEL_MANIFEST.json` and
`licenses/libarchive-dependencies/CONDA_PACKAGES.json`.

Corresponding upstream source is available from:

- Qwen3: https://github.com/QwenLM/Qwen3
- llama.cpp: https://github.com/ggml-org/llama.cpp
- FlagEmbedding / BGE: https://github.com/FlagOpen/FlagEmbedding
- libarchive: https://github.com/libarchive/libarchive
- bzip2: https://sourceware.org/bzip2/
- GNU libiconv: https://www.gnu.org/software/libiconv/
- libxml2: https://gitlab.gnome.org/GNOME/libxml2
- LZ4: https://github.com/lz4/lz4
- OpenSSL: https://github.com/openssl/openssl
- XZ Utils: https://github.com/tukaani-project/xz
- zlib: https://github.com/madler/zlib
- Zstandard: https://github.com/facebook/zstd
- Anaconda package recipes: https://github.com/AnacondaRecipes

The source version corresponding to each distributed binary is the version
listed in the package manifests. Recipients may replace the separate runtime
DLLs with compatible modified builds as permitted by the applicable licenses.

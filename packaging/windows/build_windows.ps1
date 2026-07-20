param(
    [string]$OfflineAssetsPath = ".offline-assets",
    [string]$OnlineModelAssetsPath = ".online-assets",
    [switch]$OnlineOnly,
    [switch]$OnlineModels,
    [string]$BootstrapPython = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BuildVenv = Join-Path $ProjectRoot ".venv-package"
$Python = Join-Path $BuildVenv "Scripts\python.exe"
$DistRoot = Join-Path $ProjectRoot "dist\windows"
$AppDir = Join-Path $DistRoot "DocQA"
$PackageTemp = Join-Path $ProjectRoot ".tmp-package"
$PyInstallerCache = Join-Path $ProjectRoot ".pyinstaller-cache"
$UpdaterDist = Join-Path $PackageTemp "updater-dist"
$IconPath = Join-Path $PSScriptRoot "assets\docqa.ico"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force $PackageTemp, $PyInstallerCache | Out-Null
$env:TEMP = $PackageTemp
$env:TMP = $PackageTemp
$env:PYINSTALLER_CONFIG_DIR = $PyInstallerCache

$ExistingUserData = Join-Path $AppDir "user-data"
if (Test-Path $ExistingUserData) {
    throw "user-data exists in the dist build directory. Back it up or move it to a separate installation before packaging."
}
if (Test-Path $AppDir) {
    Remove-Item $AppDir -Recurse -Force
}

if (-not (Test-Path $Python)) {
    & $BootstrapPython -m venv $BuildVenv
}

& $Python -m pip install --upgrade pip
& $Python -m pip install ".[dev]" pyinstaller
& $Python -m pytest

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onefile `
    --windowed `
    --icon $IconPath `
    --name Updater `
    --distpath $UpdaterDist `
    --workpath (Join-Path $ProjectRoot "build\updater") `
    --specpath (Join-Path $ProjectRoot "build") `
    updater.py

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name DocQA `
    --icon $IconPath `
    --distpath $DistRoot `
    --workpath (Join-Path $ProjectRoot "build\pyinstaller") `
    --specpath (Join-Path $ProjectRoot "build") `
    --collect-all enterprise_document_rag `
    --collect-all huggingface_hub `
    --collect-all sentence_transformers `
    --collect-all transformers `
    --collect-all rapidocr_onnxruntime `
    --collect-all qdrant_client `
    --collect-submodules onnxruntime `
    windows_launcher.py

Copy-Item (Join-Path $PSScriptRoot "QUICK_START.txt") (Join-Path $AppDir "QUICK_START.txt") -Force
Copy-Item (Join-Path $PSScriptRoot "portable.mode") $AppDir -Force
Copy-Item (Join-Path $ProjectRoot "README.md") $AppDir -Force
Copy-Item (Join-Path $ProjectRoot "THIRD_PARTY_NOTICES.md") $AppDir -Force
Copy-Item (Join-Path $ProjectRoot "THIRD_PARTY_SOURCE_OFFER.md") $AppDir -Force
Copy-Item (Join-Path $UpdaterDist "Updater.exe") (Join-Path $AppDir "Updater.exe") -Force
Copy-Item $IconPath (Join-Path $AppDir "docqa.ico") -Force

$InternalRoot = Join-Path $AppDir "_internal"
$BundledTestDirectories = Get-ChildItem -LiteralPath $InternalRoot -Recurse -Directory |
    Where-Object { $_.Name.ToLowerInvariant() -in @("test", "tests") } |
    Sort-Object { $_.FullName.Length } -Descending
foreach ($Directory in $BundledTestDirectories) {
    Remove-Item -LiteralPath $Directory.FullName -Recurse -Force
}
Get-ChildItem -LiteralPath $InternalRoot -Recurse -File |
    Where-Object { $_.Name -like "test_*.py*" -or $_.Name -like "*_test.py*" } |
    Remove-Item -Force

$Version = (& $Python -c "from enterprise_document_rag import __version__; print(__version__)").Trim()

function Write-VersionMetadata([string]$Edition) {
    $VersionMetadata = @{
        product_name = "Document RAG"
        executable = "DocQA.exe"
        icon_file = "docqa.ico"
        app_version = $Version
        edition = $Edition
        reranker_enabled = $false
        database_schema_version = "t05"
        embedding_model = "BAAI/bge-small-zh-v1.5"
        reranker_model = "BAAI/bge-reranker-base"
        answer_model = "Qwen/Qwen3-0.6B"
        qwen_download_required = ($Edition -eq "online-models")
        qwen_download_url = "https://huggingface.co/Qwen/Qwen3-0.6B"
        chunking_rule_version = "token-aware-v1"
        vector_index_version = "v1"
    } | ConvertTo-Json
    Set-Content -LiteralPath (Join-Path $AppDir "version.json") -Value $VersionMetadata -Encoding UTF8
}

function Write-ZipPackage([string]$ZipName) {
    $ZipPath = Join-Path $DistRoot $ZipName
    if (Test-Path $ZipPath) {
        Remove-Item $ZipPath -Force
    }
    if (Test-Path (Join-Path $AppDir "user-data")) {
        throw "user-data was found in the build output. Packaging stopped."
    }
    Compress-Archive -Path $AppDir -DestinationPath $ZipPath -CompressionLevel Optimal
    $Hash = (Get-FileHash $ZipPath -Algorithm SHA256).Hash
    $ChecksumPath = "$ZipPath.sha256"
    Set-Content -LiteralPath $ChecksumPath -Value "$($Hash.ToLower())  $ZipName" -Encoding ASCII
    Write-Host "Build completed: $ZipPath"
    Write-Host "SHA-256: $Hash"
}

Write-VersionMetadata "online"

if ($OnlineModels) {
    $ResolvedOnlineAssets = (Resolve-Path $OnlineModelAssetsPath).Path
    $RequiredOnlineAssets = @(
        "models\embedding-bge-small-zh-v1.5\model.safetensors",
        "models\reranker-bge-base\model.safetensors",
        "licenses",
        "MODEL_MANIFEST.json"
    )
    foreach ($RequiredAsset in $RequiredOnlineAssets) {
        if (-not (Test-Path (Join-Path $ResolvedOnlineAssets $RequiredAsset))) {
            throw "Online model asset is missing: $RequiredAsset"
        }
    }
    Copy-Item (Join-Path $ResolvedOnlineAssets "models") $AppDir -Recurse -Force
    Copy-Item (Join-Path $ResolvedOnlineAssets "licenses") $AppDir -Recurse -Force
    Copy-Item (Join-Path $ResolvedOnlineAssets "MODEL_MANIFEST.json") $AppDir -Force
    Set-Content -LiteralPath (Join-Path $AppDir "online-models.mode") -Value "transformers" -Encoding ASCII
    Write-VersionMetadata "online-models"
    Write-ZipPackage "DocQA-v$Version-win-x64.zip"
    exit 0
}

Write-ZipPackage "DocQA-v$Version-win-x64.zip"

if (-not $OnlineOnly) {
    $ResolvedOfflineAssets = (Resolve-Path $OfflineAssetsPath).Path
    $RequiredOfflineAssets = @(
        "models\embedding-bge-small-zh-v1.5",
        "models\reranker-bge-base-int8\model.int8.onnx",
        "models\qwen3\Qwen3-0.6B-Q8_0.gguf",
        "tools\llama.cpp\llama-cli.exe",
        "tools\libarchive\bsdtar.exe",
        "licenses",
        "MODEL_MANIFEST.json"
    )
    foreach ($RequiredAsset in $RequiredOfflineAssets) {
        if (-not (Test-Path (Join-Path $ResolvedOfflineAssets $RequiredAsset))) {
            throw "Offline asset is missing: $RequiredAsset"
        }
    }
    Copy-Item (Join-Path $ResolvedOfflineAssets "models") $AppDir -Recurse -Force
    Copy-Item (Join-Path $ResolvedOfflineAssets "tools") $AppDir -Recurse -Force
    Copy-Item (Join-Path $ResolvedOfflineAssets "licenses") $AppDir -Recurse -Force
    Copy-Item (Join-Path $ResolvedOfflineAssets "MODEL_MANIFEST.json") $AppDir -Force
    Set-Content -LiteralPath (Join-Path $AppDir "offline.mode") -Value "complete" -Encoding ASCII
    Write-VersionMetadata "offline-complete"
    Write-ZipPackage "DocQA-v$Version-win-x64-offline.zip"
}

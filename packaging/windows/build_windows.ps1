param(
    [switch]$IncludeModels,
    [string]$ModelCachePath = "..\models\huggingface",
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
    --collect-all sentence_transformers `
    --collect-all transformers `
    --collect-all rapidocr_onnxruntime `
    --collect-all qdrant_client `
    --collect-submodules onnxruntime `
    windows_launcher.py

Copy-Item (Join-Path $PSScriptRoot "QUICK_START.txt") (Join-Path $AppDir "QUICK_START.txt") -Force
Copy-Item (Join-Path $PSScriptRoot "portable.mode") $AppDir -Force
Copy-Item (Join-Path $ProjectRoot "README.md") $AppDir -Force
Copy-Item (Join-Path $UpdaterDist "Updater.exe") (Join-Path $AppDir "Updater.exe") -Force
Copy-Item $IconPath (Join-Path $AppDir "docqa.ico") -Force

$Version = (& $Python -c "from enterprise_document_rag import __version__; print(__version__)").Trim()
$VersionMetadata = @{
    product_name = "Document RAG"
    executable = "DocQA.exe"
    icon_file = "docqa.ico"
    app_version = $Version
    database_schema_version = "t05"
    embedding_model = "BAAI/bge-small-zh-v1.5"
    chunking_rule_version = "token-aware-v1"
    vector_index_version = "v1"
} | ConvertTo-Json
Set-Content -LiteralPath (Join-Path $AppDir "version.json") -Value $VersionMetadata -Encoding UTF8

if ($IncludeModels) {
    $ResolvedModels = (Resolve-Path $ModelCachePath).Path
    $ModelTarget = Join-Path $AppDir "models\huggingface"
    New-Item -ItemType Directory -Force $ModelTarget | Out-Null
    Copy-Item (Join-Path $ResolvedModels "*") $ModelTarget -Recurse -Force
}

$ZipName = if ($IncludeModels) {
    "DocQA-v$Version-win-x64-offline.zip"
} else {
    "DocQA-v$Version-win-x64.zip"
}
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

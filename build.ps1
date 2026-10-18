# ===== llama.cpp build with this fork's web UI (Windows) =====
# Builds build\bin\Release\llama-server.exe and the tools, with the web UI built from
# tools\ui (voice, TTS and STT settings), never the upstream UI from the Hugging Face bucket.
#   .\build.ps1             incremental build      (or: build.bat)
#   .\build.ps1 -Clean      delete build\ first
#   .\build.ps1 -Cpu        without CUDA
#   .\build.ps1 -Generator "Visual Studio 18 2026"
# CUDA: $env:CUDA_PATH when it holds nvcc, else the newest toolkit under
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA. GPU architecture: what nvidia-smi
# reports (12.0 -> 120), else "native". OpenSSL (https): C:\Program Files\OpenSSL.
# Node >= 20.19 builds the UI; with an older or no Node a pinned Node is fetched once into
# %LOCALAPPDATA%\llama.cpp-build.
param(
    [switch]$Clean,
    [switch]$Cpu,
    [string]$Generator = "Visual Studio 17 2022"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$NodeVersion = "v24.21.0"
function Die($message) { Write-Host "[ERROR] $message" -ForegroundColor Red; exit 1 }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Die "cmake not found" }

# ===== Node for the UI build
function Test-Node {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { return $false }
    $v = (node -p "process.versions.node").Split(".") | ForEach-Object { [int]$_ }
    return ($v[0] -gt 22) -or ($v[0] -eq 22 -and $v[1] -ge 12) -or ($v[0] -eq 20 -and $v[1] -ge 19)
}
if (-not (Test-Node)) {
    $name = "node-$NodeVersion-win-x64"
    $dir = Join-Path $env:LOCALAPPDATA "llama.cpp-build\$name"
    if (-not (Test-Path "$dir\node.exe")) {
        Write-Host "[Setup] Node is missing or too old for the UI: fetching Node $NodeVersion"
        $tmp = Join-Path ([IO.Path]::GetTempPath()) "$name.zip"
        Invoke-WebRequest "https://nodejs.org/dist/$NodeVersion/$name.zip" -OutFile $tmp
        $sums = (Invoke-WebRequest "https://nodejs.org/dist/$NodeVersion/SHASUMS256.txt").Content
        $want = ($sums -split "`n" | Where-Object { $_ -match " $name\.zip$" }) -split " " | Select-Object -First 1
        if ((Get-FileHash $tmp -Algorithm SHA256).Hash -ne $want.ToUpper()) { Die "checksum of $name.zip does not match" }
        Expand-Archive $tmp -DestinationPath (Split-Path $dir) -Force
        Remove-Item $tmp
    }
    $env:PATH = "$dir;$env:PATH"
}
Write-Host "[OK] node $(node --version), npm $(npm.cmd --version)"

# ===== CUDA
$cudaArgs = @()
if (-not $Cpu) {
    $cuda = $null
    if ($env:CUDA_PATH -and (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
        $cuda = $env:CUDA_PATH
    } else {
        $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if (Test-Path $root) {
            $cuda = Get-ChildItem $root -Directory | Where-Object { Test-Path "$($_.FullName)\bin\nvcc.exe" } |
                Sort-Object { [version]($_.Name.TrimStart("v")) } | Select-Object -Last 1 -ExpandProperty FullName
        }
    }
    if ($cuda) {
        $env:CUDA_PATH = $cuda
        $env:CUDA_HOME = $cuda
        $env:PATH = "$cuda\bin;$env:PATH"
        $arch = "native"
        try { $arch = ((nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1) -replace "[.\s]", "") } catch {}
        if (-not $arch) { $arch = "native" }
        $env:TORCH_CUDA_ARCH_LIST = $arch.Insert($arch.Length - 1, ".")
        Write-Host "[OK] CUDA at $cuda, architecture $arch"
        $cudaArgs = @(
            "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=$arch",
            "-DGGML_CUDA_FA_QUANTS=all", "-DGGML_CUDA_FORCE_MMQ=ON", "-DGGML_CUDA_FORCE_CUBLAS=OFF"
        )
    } else {
        Write-Host "[Info] no CUDA toolkit found: CPU build"
    }
}

$sslArgs = @()
if (Test-Path "C:\Program Files\OpenSSL") { $sslArgs = @("-DOPENSSL_ROOT_DIR=C:\Program Files\OpenSSL") }

# ===== Configure and build
if ($Clean -and (Test-Path build)) { Remove-Item -Recurse -Force build }
# a dist\ in the UI sources wins over the npm build; it is only ever a stale copy
if (Test-Path tools\ui\dist) { Remove-Item -Recurse -Force tools\ui\dist }

cmake -B build -G $Generator -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=ON @cudaArgs @sslArgs
if ($LASTEXITCODE) { Die "cmake configure failed" }
cmake --build build --config Release --parallel
if ($LASTEXITCODE) { Die "build failed" }

# ===== Check what was built
$voice = Get-ChildItem build\tools\ui\dist -Recurse -File -ErrorAction SilentlyContinue |
    Select-String -Pattern ttsVoice -List | Select-Object -First 1
if (-not $voice) { Die "the embedded web UI has no voice settings: the UI build failed (see 'UI:' lines above)" }
if (-not (& build\bin\Release\llama-server.exe --help 2>$null | Select-String -- "--asr-url")) {
    Die "build\bin\Release\llama-server.exe has no --asr-url option"
}
Write-Host "[OK] build\bin\Release\llama-server.exe with this fork's web UI (voice, TTS, STT settings)" -ForegroundColor Green

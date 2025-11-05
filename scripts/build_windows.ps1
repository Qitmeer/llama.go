# llama_build.ps1
$ErrorActionPreference = "Stop"

if (-Not (Test-Path "./core" -PathType Container)) {
    Write-Host "Must run in the root directory of the project"
    exit 1
}

if (-Not (Test-Path "./core/llama.cpp/src" -PathType Container)) {
    git submodule update --init --recursive
    Write-Host "Update llama.cpp"
}

# Apply memory loading patch if not already applied
if (-Not (Test-Path "core/llama.cpp/.patch_applied")) {
    Write-Host "Applying memory loading patch..."
    Push-Location core/llama.cpp
    $patchCheck = git apply --check ..\patches\memory-loading.patch 2>&1
    if ($LASTEXITCODE -eq 0) {
        git apply ..\patches\memory-loading.patch
        New-Item -ItemType File -Path ".patch_applied" -Force | Out-Null
        Write-Host "Patch applied successfully"
    } else {
        Write-Host "Warning: Patch already applied or cannot be applied"
    }
    Pop-Location
}

cmake --version

$coreDir = (Get-Location).Path + "\core"
$buildDir = (Get-Location).Path + "\build"

Write-Host "core dir: $coreDir"
Write-Host "build dir: $buildDir"

cmake -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" -S $coreDir -B $buildDir
cmake --build $buildDir --target llama_core -- -j 9

# Go
go version

$GITVER = git rev-parse --short=7 HEAD
$GITDIRTY = if (git diff --quiet) { "" } else { "-dirty" }
$GITVERSION = "${GITVER}${GITDIRTY}"
$versionBuild = "github.com/Qitmeer/llama.go/version.Build=dev-$GITVERSION"
$env:CGO_ENABLED = "1"
$env:LD_LIBRARY_PATH = "$buildDir/lib"

cd ./cmd/llama
go build -ldflags "-X $versionBuild" -o $buildDir/bin/llama.exe

Write-Host "Output executable file: $buildDir/bin/llama.exe"
& "$buildDir/bin/llama.exe" --version

# Build modelembed
Write-Host "Building modelembed..."
cd ../modelembed
go build -o $buildDir/bin/modelembed.exe

Write-Host "Output executable file: $buildDir/bin/modelembed.exe"








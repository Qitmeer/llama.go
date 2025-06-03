#!/usr/bin/env bash

set -e

if [ ! -d "./core" ]; then
    echo "Must run in the root directory of the project"
    exit 1
fi

if [ ! -d "./core/llama.cpp/src" ]; then
    git submodule update --init --recursive
    echo "Update llama.cpp"
fi

cmake --version


coreDir=$(pwd)/core
buildDir=$(pwd)/build

echo "core dir:" ${coreDir}
echo "build dir:" ${buildDir}

# cuda
cudaTag=""
cudaCmake=""
if [[ "$(uname -s)" == "Linux" ]]; then
    if [[ -d "/usr/local/cuda" ]] && command -v nvcc &> /dev/null; then
        echo "Try use CUDA"
        cudaCmake="-DGGML_CUDA=ON"
    fi
fi

cmake -DCMAKE_BUILD_TYPE=Release  $cudaCmake -G "Unix Makefiles" -S $coreDir -B $buildDir
cmake --build $buildDir --target llama_core -- -j 9

if [ -e $buildDir/lib/libggml-cuda.a ]; then
    cudaTag="-tags=cuda"
fi

# go
go version

GITVER=$(git rev-parse --short=7 HEAD)
GITDIRTY=$(git diff --quiet || echo '-dirty')
GITVERSION="${GITVER}${GITDIRTY}"
versionBuild="github.com/Qitmeer/llama.go/version.Build=dev-${GITVERSION}"

export CGO_ENABLED=1
export LD_LIBRARY_PATH=$buildDir/lib
cd ./cmd/llama
go build $cudaTag -ldflags "-X ${versionBuild}" -o $buildDir/bin/llama

echo "Output executable file:${buildDir}/bin/llama"
$buildDir/bin/llama --version



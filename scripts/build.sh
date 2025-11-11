#!/usr/bin/env bash

set -e

# Build C++ libraries first
bash ./scripts/build_cpp.sh

buildDir=$(pwd)/build

# cuda tag for go build
cudaTag=""
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

# Build modelembed
echo "Building modelembed..."
cd ../modelembed
go build -o $buildDir/bin/modelembed

echo "Output executable file:${buildDir}/bin/modelembed"



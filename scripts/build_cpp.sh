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

# Apply memory loading patch if not already applied
if [ ! -f "./core/llama.cpp/.patch_applied" ]; then
    echo "Applying memory loading patch..."
    cd ./core/llama.cpp
    if git apply --check ../patches/memory-loading.patch 2>/dev/null; then
        git apply ../patches/memory-loading.patch
        touch .patch_applied
        echo "Patch applied successfully"
    else
        echo "Warning: Patch already applied or cannot be applied"
    fi
    cd ../..
fi

cmake --version

coreDir=$(pwd)/core
buildDir=$(pwd)/build

echo "core dir:" ${coreDir}
echo "build dir:" ${buildDir}

# cuda
cudaCmake=""
if [[ "$(uname -s)" == "Linux" ]]; then
    if [[ -d "/usr/local/cuda" ]] && command -v nvcc &> /dev/null; then
        echo "Try use CUDA"
        cudaCmake="-DGGML_CUDA=ON"
    fi
fi

cmake -DCMAKE_BUILD_TYPE=Release $cudaCmake -G "Unix Makefiles" -S $coreDir -B $buildDir
cmake --build $buildDir --target llama_core -- -j 9

echo "C++ libraries built successfully"

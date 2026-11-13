#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
revision="$(cat scripts/llama_cpp_revision.txt)"
if [[ ! -d tools/llama.cpp/.git ]]; then
    git clone --no-checkout https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
fi
git -C tools/llama.cpp fetch --depth 1 origin "$revision"
git -C tools/llama.cpp checkout --detach "$revision"
# CUDA 12.4 on this host supports GCC 13; override these for another toolkit.
export CC="${CC:-gcc-13}"
export CXX="${CXX:-g++-13}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-g++-13}"
.venv/bin/cmake -S tools/llama.cpp -B tools/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-86}" \
    -DGGML_NATIVE=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DCMAKE_MAKE_PROGRAM="$project_root/.venv/bin/ninja"
.venv/bin/cmake --build tools/llama.cpp/build --target llama-server -j "${BUILD_JOBS:-4}"


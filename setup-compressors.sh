#!/bin/bash

RED='\033[0;31m'
BOLDRED='\033[1;31m' 
GRAY='\033[0;37m'
NOCOLOR='\033[0m'

# cusz-stock and cusz-interp
echo "\n${BOLDRED}setting up stock cuSZ and cuSZ-Hi (this work)...${GRAY}"
cmake -S cusz-Hi -B cusz-Hi/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz-Hi/build -- -j

# fzgpu
echo "\n${BOLDRED}setting up FZ-GPU...${GRAY}"
pushd fzgpu
make -j
popd


# cuszp
echo "\n${BOLDRED}setting up cuSZp...${GRAY}"
cmake -S cuszp -B cuszp/build \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cuszp/build -- -j

# zfp-cuda
echo "\n${BOLDRED}setting up ZFP-CUDA...${GRAY}"
cmake -S zfp-cuda -B zfp-cuda/build \
    -D ZFP_WITH_CUDA=on \
    -D CUDA_SDK_ROOT_DIR=$(dirname $(which nvcc))/.. \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build zfp-cuda/build -- -j

echo -e "${NOCOLOR}"
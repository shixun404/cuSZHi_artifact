#!/bin/bash

RED='\033[0;31m'
BOLDRED='\033[1;31m' 
GRAY='\033[0;37m'
NOCOLOR='\033[0m'

# cusz-stock and cusz-interp
echo "\n${BOLDRED}setting up stock cuSZ-Hi (this work)...${GRAY}"
cmake -S cusz-Hi -B cusz-Hi/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz-Hi/build -- -j

# cusz-i
echo "\n${BOLDRED}setting up stock cuSZ-i ...${GRAY}"
cmake -S cusz_24_2d -B cusz_24_2d/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz_24_2d/build -- -j


echo "\n${BOLDRED}setting up stock cusz-interp_16_4steps ...${GRAY}"
cmake -S cusz-interp_16_4steps -B cusz-interp_16_4steps/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz-interp_16_4steps/build -- -j

echo "\n${BOLDRED}setting up stock cusz-interp_16_4steps_reorder ...${GRAY}"
cmake -S cusz-interp_16_4steps_reorder -B cusz-interp_16_4steps_reorder/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz-interp_16_4steps_reorder/build -- -j

echo "\n${BOLDRED}setting up stock cusz-interp_16_4steps_reorder_att_balance ...${GRAY}"
cmake -S cusz-interp_16_4steps_reorder_att_balance -B cusz-interp_16_4steps_reorder_att_balance/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz-interp_16_4steps_reorder_att_balance/build -- -j


# cusz-stock
echo "\n${BOLDRED}setting up stock cuSZ ...${GRAY}"
cmake -S cusz -B cusz/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="75;80;86" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cusz/build -- -j

# fzgpu
echo "\n${BOLDRED}setting up FZ-GPU...${GRAY}"
pushd fzgpu
make -j
popd


# cuszp
echo "\n${BOLDRED}setting up cuSZp...${GRAY}"
cmake -S cuszp -B cuszp2/build \
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
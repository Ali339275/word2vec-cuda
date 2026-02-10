cmake_minimum_required(VERSION 3.18)
project(word2vec_cuda LANGUAGES CXX CUDA)

# -----------------------------
# Global settings
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# -----------------------------
# Common sources
# -----------------------------
set(COMMON_SOURCES
    dataset.cpp
    utils.cu
)

# Common include directories
set(COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# =====================================================
# Base CUDA executable (reference implementation)
# =====================================================
add_executable(w2v_base_cuda_train
    ${COMMON_SOURCES}
    w2v_base_cuda/main.cu
    w2v_base_cuda/kernels.cu
)

target_include_directories(w2v_base_cuda_train PRIVATE
    ${COMMON_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/w2v_base_cuda
)

set_target_properties(w2v_base_cuda_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# =====================================================
# Optimized CUDA executable (fused kernel)
# =====================================================
add_executable(w2v_opt_cuda_train
    ${COMMON_SOURCES}
    w2v_optimized_cuda/main.cu
    w2v_optimized_cuda/kernels.cu
)

target_include_directories(w2v_opt_cuda_train PRIVATE
    ${COMMON_INCLUDES}
    ${CMAKE_CURRENT_SOURCE_DIR}/w2v_optimized_cuda
)

set_target_properties(w2v_opt_cuda_train PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
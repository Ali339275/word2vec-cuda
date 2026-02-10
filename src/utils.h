// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------
// Device math helpers (header-defined)
// -----------------------------------------

__device__ __forceinline__
float sigmoid(float x) {
    // Clamp for numerical stability
    x = fmaxf(fminf(x, 10.0f), -10.0f);
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__
float log_sigmoid(float x) {
    x = fmaxf(fminf(x, 10.0f), -10.0f);
    return -logf(1.0f + expf(-x));
}

// -----------------------------------------
// Embedding initialization kernel
// -----------------------------------------
__global__
void init_embeddings(
    float* W_in,
    float* W_out,
    int vocab_size,
    int emb_dim
);

#endif // UTILS_H
// utils.cu
#include "utils.h"
#include <curand_kernel.h>


// -----------------------------------------
// Embedding initialization kernel
// -----------------------------------------

__global__
void init_embeddings(
    float* W_in,
    float* W_out,
    int vocab_size,
    int emb_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = vocab_size * emb_dim;

    if (idx >= total) return;

    // Simple deterministic initialization:
    // W_in ~ U(-0.5/emb_dim, 0.5/emb_dim)
    // W_out = 0

    int word = idx / emb_dim;
    int dim  = idx % emb_dim;

    // Hash-based pseudo-random init (no CURAND needed here)
    unsigned int seed = word * 1315423911u + dim * 2654435761u;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);

    float rnd = (seed & 0xFFFFFF) / float(0xFFFFFF); // [0,1)
    float scale = 0.5f / emb_dim;

    W_in[idx]  = (rnd * 2.0f - 1.0f) * scale;
    W_out[idx] = 0.0f;
}
// src/w2v_optimized_cuda/kernels.cu
#include "kernels.h"
#include "../utils.h"
#include <cuda_runtime.h>

// --------------------------------------------------
// Shared-memory reduction (sum) for dot product
// --------------------------------------------------
__device__ __forceinline__
float block_reduce_sum(float* sh, int tid, int n) {
    __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    return sh[0];
}

// --------------------------------------------------
// Tier-1 epoch kernel: trains ALL batches of an epoch
// Grid is 2D:
//   blockIdx.y = batch_id  in [0 .. batches_per_epoch-1]
//   blockIdx.x = sample_id in [0 .. batch_size-1]
// blockDim.x = emb_dim (e.g. 288)
// --------------------------------------------------
__global__ void sgns_train_epoch_kernel(
    float* W_in,
    float* W_out,
    const int* centers,     // [B * BS]
    const int* pos,         // [B * BS]
    const int* neg,         // [B * BS * K]
    float* loss_out,        // [B] : sum of sample losses per batch
    int vocab_size,
    int emb_dim,
    int num_neg,
    int batch_size,
    int batches_per_epoch,
    float lr
) {
    const int batch_id  = (int)blockIdx.y;
    const int sample_id = (int)blockIdx.x;
    const int d         = (int)threadIdx.x;

    if (batch_id >= batches_per_epoch || sample_id >= batch_size) return;
    if (d >= emb_dim) return;

    const int sample_global = batch_id * batch_size + sample_id;

    const int c = centers[sample_global];
    const int p = pos[sample_global];

    if (c < 0 || c >= vocab_size || p < 0 || p >= vocab_size) return;

    float* in_c  = W_in  + (size_t)c * emb_dim;
    float* out_p = W_out + (size_t)p * emb_dim;

    extern __shared__ float sh[]; // emb_dim floats

    // Cache center value (old snapshot for this update step)
    const float in_old = in_c[d];

    // -----------------------------
    // Positive pair
    // -----------------------------
    const float outp_old = out_p[d];
    sh[d] = in_old * outp_old;
    float pos_dot = block_reduce_sum(sh, d, emb_dim);

    // y=1 => g = sigmoid(dot) - 1
    const float g_pos = sigmoid(pos_dot) - 1.0f;

    // Center gradient accum (per dim)
    float grad_in = g_pos * outp_old;

    // Update out_p (atomic due to collisions across samples)
    atomicAdd(&out_p[d], -lr * g_pos * in_old);

    // Loss accum (thread 0)
    float sample_loss = 0.0f;
    if (d == 0) {
        sample_loss -= log_sigmoid(pos_dot);
    }
    __syncthreads();

    // -----------------------------
    // Negatives
    // neg indexing:
    // neg[(batch_id*batch_size + sample_id)*num_neg + k]
    // -----------------------------
    for (int k = 0; k < num_neg; ++k) {
        const int n = neg[(size_t)sample_global * num_neg + k];
        if (n < 0 || n >= vocab_size) continue;

        float* out_n = W_out + (size_t)n * emb_dim;
        const float outn_old = out_n[d];

        sh[d] = in_old * outn_old;
        float neg_dot = block_reduce_sum(sh, d, emb_dim);

        // y=0 => g = sigmoid(dot)
        const float g_neg = sigmoid(neg_dot);

        // accumulate center grad
        grad_in += g_neg * outn_old;

        // update negative output
        atomicAdd(&out_n[d], -lr * g_neg * in_old);

        if (d == 0) {
            sample_loss -= log_sigmoid(-neg_dot);
        }
        __syncthreads();
    }

    // -----------------------------
    // Update center embedding
    // atomic for correctness across blocks (same word id collisions)
    // -----------------------------
    atomicAdd(&in_c[d], -lr * grad_in);

    // -----------------------------
    // Write batch loss (sum of sample losses)
    // Only one atomicAdd per sample (thread 0)
    // -----------------------------
    if (d == 0) {
        atomicAdd(&loss_out[batch_id], sample_loss);
    }
}
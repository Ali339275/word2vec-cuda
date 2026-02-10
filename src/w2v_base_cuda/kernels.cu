// src/w2v_base_cuda/kernels.cu
#include "kernels.h"
#include "../utils.h"
#include <cuda_runtime.h>


// shared-memory reduction
__device__ __forceinline__
float block_reduce_sum(float* sh, int tid, int n) {
    __syncthreads();
    for (int stride = n / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }
    return sh[0];
}

// =======================================================
// Kernel 1: Positive — update ONLY W_out(pos),
//          write grad_in_pos[sample,d] = g_pos * out_p_old
// =======================================================
__global__ void sgns_positive_out_kernel(
    const float* W_in,
    float* W_out,
    const int* center_ids,
    const int* pos_ids,
    float* grad_in_pos,
    int vocab_size,
    int emb_dim,
    float lr
) {
    int sample = blockIdx.x;
    int d = threadIdx.x;
    if (d >= emb_dim) return;

    int c = center_ids[sample];
    int p = pos_ids[sample];
    if (c < 0 || c >= vocab_size || p < 0 || p >= vocab_size) return;

    const float* in_c  = W_in  + (size_t)c * emb_dim;
    float* out_p       = W_out + (size_t)p * emb_dim;

    extern __shared__ float sh[];

    const float in_old  = in_c[d];
    const float out_old = out_p[d];

    sh[d] = in_old * out_old;
    float dot = block_reduce_sum(sh, d, emb_dim);

    // y=1 => g = sigmoid(dot) - 1
    float g = sigmoid(dot) - 1.0f;

    // grad wrt center: g * out_old
    grad_in_pos[(size_t)sample * emb_dim + d] = g * out_old;

    // update output: out_p -= lr * g * in_old
    // collisions possible on W_out => atomicAdd for correctness
    atomicAdd(&out_p[d], -lr * g * in_old);
}

// =======================================================
// Kernel 2: Negative — update ONLY W_out(neg),
//          write grad_in_neg[sample,d] = sum_k g_neg * out_n_old
// One block per sample, loop over negatives inside the block
// =======================================================
__global__ void sgns_negative_out_kernel(
    const float* W_in,
    float* W_out,
    const int* center_ids,
    const int* neg_ids,    // [batch_size * num_neg]
    float* grad_in_neg,    // [batch_size * emb_dim]
    int vocab_size,
    int emb_dim,
    int num_neg,
    float lr
) {
    int sample = blockIdx.x;
    int d = threadIdx.x;
    if (d >= emb_dim) return;

    int c = center_ids[sample];
    if (c < 0 || c >= vocab_size) return;

    const float* in_c = W_in + (size_t)c * emb_dim;
    extern __shared__ float sh[];

    const float in_old = in_c[d];
    float accum = 0.0f;  // accumulate grad for center dim d

    for (int k = 0; k < num_neg; ++k) {
        int n = neg_ids[sample * num_neg + k];
        if (n < 0 || n >= vocab_size) continue;

        float* out_n = W_out + (size_t)n * emb_dim;
        float out_old = out_n[d];

        sh[d] = in_old * out_old;
        float dot = block_reduce_sum(sh, d, emb_dim);

        // y=0 => g = sigmoid(dot)
        float g = sigmoid(dot);

        // accumulate center grad: g * out_old
        accum += g * out_old;

        // update negative output: out_n -= lr * g * in_old
        atomicAdd(&out_n[d], -lr * g * in_old);
    }

    grad_in_neg[(size_t)sample * emb_dim + d] = accum;
}

// =======================================================
// Kernel 3: Apply center update once per sample
// W_in[c,d] -= lr * (grad_in_pos + grad_in_neg)
// Atomic is recommended because same word id can appear
// in multiple samples concurrently.
// =======================================================
__global__ void sgns_apply_center_kernel(
    float* W_in,
    const int* center_ids,
    const float* grad_in_pos,
    const float* grad_in_neg,
    int vocab_size,
    int emb_dim,
    float lr
) {
    int sample = blockIdx.x;
    int d = threadIdx.x;
    if (d >= emb_dim) return;

    int c = center_ids[sample];
    if (c < 0 || c >= vocab_size) return;

    float g = grad_in_pos[(size_t)sample * emb_dim + d]
            + grad_in_neg[(size_t)sample * emb_dim + d];

    atomicAdd(&W_in[(size_t)c * emb_dim + d], -lr * g);
}

// =======================================================
// Your loss kernel can stay exactly as you already have it
// =======================================================
__global__ void sgns_loss_kernel(
    const float* W_in,
    const float* W_out,
    const int* center_ids,
    const int* pos_ids,
    const int* neg_ids,
    float* loss,
    int vocab_size,
    int emb_dim,
    int num_neg
) {
    int sample = blockIdx.x;
    int d = threadIdx.x;
    if (d >= emb_dim) return;

    int c = center_ids[sample];
    int p = pos_ids[sample];
    if (c < 0 || c >= vocab_size || p < 0 || p >= vocab_size) return;

    const float* in_c  = W_in  + (size_t)c * emb_dim;
    const float* out_p = W_out + (size_t)p * emb_dim;

    extern __shared__ float sh[];

    sh[d] = in_c[d] * out_p[d];
    __syncthreads();
    for (int stride = emb_dim / 2; stride > 0; stride >>= 1) {
        if (d < stride) sh[d] += sh[d + stride];
        __syncthreads();
    }
    float pos_dot = sh[0];

    float neg_sum = 0.0f;
    for (int k = 0; k < num_neg; ++k) {
        int n = neg_ids[sample * num_neg + k];
        if (n < 0 || n >= vocab_size) continue;
        const float* out_n = W_out + (size_t)n * emb_dim;

        sh[d] = in_c[d] * out_n[d];
        __syncthreads();
        for (int stride = emb_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) sh[d] += sh[d + stride];
            __syncthreads();
        }
        if (d == 0) neg_sum += log_sigmoid(-sh[0]);
        __syncthreads();
    }

    if (d == 0) loss[sample] = -log_sigmoid(pos_dot) - neg_sum;
}
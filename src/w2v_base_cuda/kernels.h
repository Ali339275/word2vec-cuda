// src/w2v_base_cuda/kernels.h
#pragma once
#include <cuda_runtime.h>

// Positive: update W_out(pos), write grad_in_pos
__global__ void sgns_positive_out_kernel(
    const float* W_in,
    float* W_out,
    const int* center_ids,
    const int* pos_ids,
    float* grad_in_pos,   // [batch_size * emb_dim]
    int vocab_size,
    int emb_dim,
    float lr
);

// Negative: update W_out(neg), write grad_in_neg
__global__ void sgns_negative_out_kernel(
    const float* W_in,
    float* W_out,
    const int* center_ids,
    const int* neg_ids,   // [batch_size * num_neg]
    float* grad_in_neg,   // [batch_size * emb_dim]
    int vocab_size,
    int emb_dim,
    int num_neg,
    float lr
);

// Apply center update once
__global__ void sgns_apply_center_kernel(
    float* W_in,
    const int* center_ids,
    const float* grad_in_pos,
    const float* grad_in_neg,
    int vocab_size,
    int emb_dim,
    float lr
);

// (Optional) keep your loss kernel unchanged if you want
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
);
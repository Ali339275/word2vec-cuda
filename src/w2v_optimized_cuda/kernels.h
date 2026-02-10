// src/w2v_optimized_cuda/kernels.h
#pragma once
#include <cuda_runtime.h>

__global__ void sgns_train_epoch_kernel(
    float* W_in,
    float* W_out,
    const int* centers,    // [batches_per_epoch * batch_size]
    const int* pos,        // [batches_per_epoch * batch_size]
    const int* neg,        // [batches_per_epoch * batch_size * num_neg]
    float* loss_out,       // [batches_per_epoch]  (sum of sample losses per batch)
    int vocab_size,
    int emb_dim,
    int num_neg,
    int batch_size,
    int batches_per_epoch,
    float lr
);
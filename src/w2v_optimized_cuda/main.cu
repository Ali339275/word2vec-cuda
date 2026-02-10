// src/w2v_optimized_cuda/main.cu
// Tier-1 SGNS CUDA training — epoch-level loss (Option A)

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

// ---- relative includes ----
#include "../dataset.h"
#include "../utils.h"
#include "kernels.h"

// -----------------------------
// Hyperparameters
// -----------------------------
static constexpr int EMB_DIM     = 288;
static constexpr int WINDOW_SIZE = 1;
static constexpr int NUM_NEG     = 60;
static constexpr int BATCH_SIZE  = 512;
static constexpr int EPOCHS      = 15;

static constexpr float LR0       = 0.05f;
static constexpr float LR_DECAY  = 0.90f;

// dataset path relative to w2v_optimized_cuda/
static const std::string TEXT_PATH =
    "../cpu/data/text8_500k";

// output
static const std::string OUT_EMB_FILE =
    "word_embeddings_cuda_tier1.bin";

// -----------------------------
// CUDA error checking
// -----------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: "                                      \
                      << cudaGetErrorString(err)                             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// -----------------------------
// Save embeddings
// -----------------------------
static void save_embeddings(const float* h_W,
                            int vocab_size,
                            int emb_dim,
                            const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output file");
    out.write(reinterpret_cast<const char*>(h_W),
              (size_t)vocab_size * emb_dim * sizeof(float));
    out.close();
}

int main() {
    try {
        std::cout << "CUDA SGNS — Tier 1 (epoch kernel)\n";

        // -----------------------------
        // Load dataset (CPU)
        // -----------------------------
        Dataset dataset(TEXT_PATH, WINDOW_SIZE, NUM_NEG, BATCH_SIZE);

        const int V = dataset.vocab_size();
        const int B = dataset.num_batches();

        std::cout << "Vocab size        : " << V << "\n";
        std::cout << "Batches / epoch   : " << B << "\n";
        std::cout << "Embedding dim     : " << EMB_DIM << "\n";
        std::cout << "Negative samples  : " << NUM_NEG << "\n\n";

        // -----------------------------
        // Allocate embeddings (GPU)
        // -----------------------------
        float* d_W_in  = nullptr;
        float* d_W_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_W_in,
            (size_t)V * EMB_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_out,
            (size_t)V * EMB_DIM * sizeof(float)));

        // Init embeddings
        {
            int total = V * EMB_DIM;
            int threads = 256;
            int blocks  = (total + threads - 1) / threads;

            init_embeddings<<<blocks, threads>>>(
                d_W_in, d_W_out, V, EMB_DIM);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // -----------------------------
        // Allocate epoch buffers (GPU)
        // -----------------------------
        int* d_centers = nullptr;
        int* d_pos     = nullptr;
        int* d_neg     = nullptr;
        float* d_loss  = nullptr;

        CUDA_CHECK(cudaMalloc(&d_centers,
            (size_t)B * BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pos,
            (size_t)B * BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neg,
            (size_t)B * BATCH_SIZE * NUM_NEG * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_loss,
            (size_t)B * sizeof(float)));

        std::vector<float> h_loss(B);

        // -----------------------------
        // Training loop
        // -----------------------------
        float lr = LR0;

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            dataset.reset();

            // -----------------------------
            // Build full epoch batches on CPU
            // -----------------------------
            std::vector<int> h_centers(B * BATCH_SIZE);
            std::vector<int> h_pos(B * BATCH_SIZE);
            std::vector<int> h_neg(B * BATCH_SIZE * NUM_NEG);

            for (int b = 0; b < B; ++b) {
                Batch batch = dataset.next_batch();

                std::copy(batch.center.begin(), batch.center.end(),
                          h_centers.begin() + b * BATCH_SIZE);

                std::copy(batch.pos.begin(), batch.pos.end(),
                          h_pos.begin() + b * BATCH_SIZE);

                std::copy(batch.neg.begin(), batch.neg.end(),
                          h_neg.begin() + (size_t)b * BATCH_SIZE * NUM_NEG);
            }

            // -----------------------------
            // Upload epoch data once
            // -----------------------------
            CUDA_CHECK(cudaMemcpy(
                d_centers, h_centers.data(),
                h_centers.size() * sizeof(int),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_pos, h_pos.data(),
                h_pos.size() * sizeof(int),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemcpy(
                d_neg, h_neg.data(),
                h_neg.size() * sizeof(int),
                cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaMemset(d_loss, 0, B * sizeof(float)));

            // -----------------------------
            // Launch epoch kernel (ONE launch)
            // -----------------------------
            dim3 grid(BATCH_SIZE, B);
            dim3 block(EMB_DIM);
            size_t shmem = EMB_DIM * sizeof(float);

            sgns_train_epoch_kernel<<<grid, block, shmem>>>(
                d_W_in, d_W_out,
                d_centers, d_pos, d_neg,
                d_loss,
                V, EMB_DIM, NUM_NEG,
                BATCH_SIZE, B,
                lr
            );

            CUDA_CHECK(cudaDeviceSynchronize());

            // -----------------------------
            // Compute epoch average loss
            // -----------------------------
            CUDA_CHECK(cudaMemcpy(
                h_loss.data(), d_loss,
                B * sizeof(float),
                cudaMemcpyDeviceToHost));

            double total_loss = 0.0;
            for (int i = 0; i < B; ++i) total_loss += h_loss[i];

            double avg_loss =
                total_loss / ((double)B * BATCH_SIZE);

            std::cout << "Epoch ["
                      << (epoch + 1) << "/" << EPOCHS
                      << "] avg_loss=" << avg_loss
                      << " | lr=" << lr << std::endl;

            lr *= LR_DECAY;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double seconds =
            std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\nTotal training time: "
                  << seconds << " seconds\n";

        // -----------------------------
        // Save embeddings
        // -----------------------------
        std::vector<float> h_W_in((size_t)V * EMB_DIM);

        CUDA_CHECK(cudaMemcpy(
            h_W_in.data(), d_W_in,
            h_W_in.size() * sizeof(float),
            cudaMemcpyDeviceToHost));

        save_embeddings(h_W_in.data(), V, EMB_DIM, OUT_EMB_FILE);

        std::cout << "Saved embeddings to: "
                  << OUT_EMB_FILE << "\n";

        // -----------------------------
        // Cleanup
        // -----------------------------
        cudaFree(d_W_in);
        cudaFree(d_W_out);
        cudaFree(d_centers);
        cudaFree(d_pos);
        cudaFree(d_neg);
        cudaFree(d_loss);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
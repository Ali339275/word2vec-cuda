// src/w2v_base_cuda/main.cu
// BASE CUDA SGNS training 

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

#include <cuda_runtime.h>

#include "../dataset.h"
#include "../utils.h"
#include "kernels.h"

// -----------------------------
// Hyperparameters
// -----------------------------
int EMB_DIM     = 288;
int WINDOW_SIZE = 1;
int NUM_NEG     = 60;
int BATCH_SIZE  = 512;
int EPOCHS      = 15;

static constexpr float LR0       = 0.05f;
static constexpr float LR_DECAY  = 0.90f;

// dataset path relative to w2v_base_cuda/
static const std::string TEXT_PATH =
    "../cpu/data/text8_500k";

// output
static const std::string OUT_EMB_FILE =
    "word_embeddings_cuda_base_stable.bin";

static void parse_args(
    int argc, char** argv,
    int& emb_dim,
    int& batch_size,
    int& epochs)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--emb-dim" && i + 1 < argc) {
            emb_dim = std::stoi(argv[++i]);
        }
        else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        }
        else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        }
        else if (arg == "--help") {
            std::cout
                << "Usage:\n"
                << "  --emb-dim <int>\n"
                << "  --batch-size <int>\n"
                << "  --epochs <int>\n";
            std::exit(0);
        }
    }
}

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
    out.write(reinterpret_cast<const char*>(h_W),
              static_cast<size_t>(vocab_size) * emb_dim * sizeof(float));
    out.close();
}

int main(int argc, char** argv) {
    try {
        std::cout << "CUDA SGNS BASE (Option B – stable)\n";

        // -----------------------------
        // Load dataset
        // -----------------------------
        parse_args(argc, argv, EMB_DIM, BATCH_SIZE, EPOCHS);

        //check size of Embedding which will be the size of a block
        if (EMB_DIM > 1024) {
            throw std::runtime_error("EMB_DIM > 1024 not supported (threads per block)");
        }

        // just to check parameters
        std::cout << "Using hyperparameters:\n"
          << "  EMB_DIM     = " << EMB_DIM << "\n"
          << "  BATCH_SIZE  = " << BATCH_SIZE << "\n"
          << "  EPOCHS      = " << EPOCHS << "\n\n";

        Dataset dataset(TEXT_PATH, WINDOW_SIZE, NUM_NEG, BATCH_SIZE);

        const int V = dataset.vocab_size();
        const int batches_per_epoch = dataset.num_batches();

        std::cout << "Vocab size      : " << V << "\n";
        std::cout << "Batches / epoch : " << batches_per_epoch << "\n\n";

        // -----------------------------
        // Allocate embeddings
        // -----------------------------
        float* d_W_in  = nullptr;
        float* d_W_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_W_in,
            static_cast<size_t>(V) * EMB_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_W_out,
            static_cast<size_t>(V) * EMB_DIM * sizeof(float)));

        // Init embeddings
        {
            int total = V * EMB_DIM;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;

            init_embeddings<<<blocks, threads>>>(
                d_W_in, d_W_out, V, EMB_DIM);

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // -----------------------------
        // Allocate batch buffers
        // -----------------------------
        int* d_center = nullptr;
        int* d_pos    = nullptr;
        int* d_neg    = nullptr;

        float* d_grad_pos = nullptr;
        float* d_grad_neg = nullptr;

        float* d_loss = nullptr;
        float* h_loss = new float[BATCH_SIZE];

        CUDA_CHECK(cudaMalloc(&d_center, BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_pos,    BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_neg,    BATCH_SIZE * NUM_NEG * sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_grad_pos,
            BATCH_SIZE * EMB_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grad_neg,
            BATCH_SIZE * EMB_DIM * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_loss,
            BATCH_SIZE * sizeof(float)));

        // -----------------------------
        // Training loop + timing
        // -----------------------------
        float lr = LR0;

        CUDA_CHECK(cudaDeviceSynchronize());
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            dataset.reset();

            for (int b = 0; b < batches_per_epoch; ++b) {
                Batch batch = dataset.next_batch();

                CUDA_CHECK(cudaMemcpy(
                    d_center, batch.center.data(),
                    BATCH_SIZE * sizeof(int),
                    cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(
                    d_pos, batch.pos.data(),
                    BATCH_SIZE * sizeof(int),
                    cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(
                    d_neg, batch.neg.data(),
                    BATCH_SIZE * NUM_NEG * sizeof(int),
                    cudaMemcpyHostToDevice));

                dim3 grid(BATCH_SIZE);
                dim3 block(EMB_DIM);
                size_t shmem = EMB_DIM * sizeof(float);

                // -----------------------------
                // 1) Positive (update W_out, write grad_pos)
                // -----------------------------
                sgns_positive_out_kernel<<<grid, block, shmem>>>(
                    d_W_in, d_W_out,
                    d_center, d_pos,
                    d_grad_pos,
                    V, EMB_DIM, lr);

                // -----------------------------
                // 2) Negative (update W_out, write grad_neg)
                // -----------------------------
                sgns_negative_out_kernel<<<grid, block, shmem>>>(
                    d_W_in, d_W_out,
                    d_center, d_neg,
                    d_grad_neg,
                    V, EMB_DIM, NUM_NEG, lr);

                // -----------------------------
                // 3) Apply center update once
                // -----------------------------
                sgns_apply_center_kernel<<<grid, block>>>(
                    d_W_in,
                    d_center,
                    d_grad_pos,
                    d_grad_neg,
                    V, EMB_DIM, lr);

                // -----------------------------
                // Loss monitoring
                // -----------------------------
                if ((b + 1) % 50 == 0 || (b + 1) == batches_per_epoch) {
                    sgns_loss_kernel<<<grid, block, shmem>>>(
                        d_W_in, d_W_out,
                        d_center, d_pos, d_neg,
                        d_loss,
                        V, EMB_DIM, NUM_NEG);

                    CUDA_CHECK(cudaMemcpy(
                        h_loss, d_loss,
                        BATCH_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

                    double avg_loss = 0.0;
                    for (int i = 0; i < BATCH_SIZE; ++i)
                        avg_loss += h_loss[i];
                    avg_loss /= BATCH_SIZE;

                    std::cout << "\rEpoch ["
                              << (epoch + 1) << "/" << EPOCHS
                              << "] Batch ["
                              << (b + 1) << "/" << batches_per_epoch
                              << "] avg_loss=" << avg_loss
                              << std::flush;
                }
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "\nEpoch "
                      << (epoch + 1) << "/" << EPOCHS
                      << " done | lr=" << lr << "\n";

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
        float* h_W_in =
            new float[static_cast<size_t>(V) * EMB_DIM];

        CUDA_CHECK(cudaMemcpy(
            h_W_in, d_W_in,
            static_cast<size_t>(V) * EMB_DIM * sizeof(float),
            cudaMemcpyDeviceToHost));

        save_embeddings(h_W_in, V, EMB_DIM, OUT_EMB_FILE);
        std::cout << "Saved embeddings to: "
                  << OUT_EMB_FILE << "\n";

        delete[] h_W_in;

        // -----------------------------
        // Cleanup
        // -----------------------------
        cudaFree(d_W_in);
        cudaFree(d_W_out);
        cudaFree(d_center);
        cudaFree(d_pos);
        cudaFree(d_neg);
        cudaFree(d_grad_pos);
        cudaFree(d_grad_neg);
        cudaFree(d_loss);
        delete[] h_loss;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
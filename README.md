# CUDA-Based Skip-Gram with Negative Sampling (SGNS)

This project implements the Skip-Gram with Negative Sampling (SGNS) algorithm in:

- CUDA (C++ GPU implementation)
- Python (PyTorch CPU baseline)

The goal is to compare performance and convergence between both implementations.

---

# 1️⃣ Project Structure

```
w2v_cpp_project/
│
├── src/
│   ├── cpu/                # PyTorch implementation
│   ├── w2v_base_cuda/      # CUDA kernels (baseline)
│   ├── build/              # Compiled CUDA binaries
│
├── CMakeLists.txt
├── README.md
```

---

# 2️⃣ Build Instructions

## 🔹 Build CUDA Version

From the project root:

```bash
make clean
cmake ..
make -j
```

Or manually using CMake:

```bash
mkdir -p src/build
cd src/build
cmake ..
make -j
```

---

## 🔹 Build Python Extension (if applicable)

```bash
python setup.py build_ext --inplace
```

---

# 3️⃣ Running the Implementations

## 🚀 CUDA Version

From the build directory:

```bash
cd ./src/build

srun --gres=gpu:1 --cpus-per-task=4 --mem=2GB ./w2v_base_cuda_train \
  --emb-dim 128 \
  --batch-size 512 \
  --epochs 15
```

### Parameters

- `--emb-dim` → Embedding dimension
- `--batch-size` → Batch size
- `--epochs` → Number of training epochs

---

## 🐍 PyTorch Version

```bash
cd ./src/cpu

srun --gres=gpu:1 --cpus-per-task=4 --mem=2GB \
python ./main.py \
  --embedding_dim 128 \
  --batch_size 512 \
  --epochs 15
```

### Parameters

- `--embedding_dim` → Embedding dimension
- `--batch_size` → Batch size
- `--epochs` → Number of epochs

---

# 4️⃣ Dataset

The project uses:

```
data/text8_500k
```

A 500k-token subset of the text8 corpus.

---

# 5️⃣ Output

After training:

- CUDA version outputs:

```
word_embeddings_cuda_base_stable.bin
```

- PyTorch version outputs:

```
word_embeddings.pt
```

---

# 6️⃣ Performance Comparison

The project compares:

- Training time
- Speedup
- Loss convergence

Speedup is computed as:

```
Speedup = CPU Time / GPU Time
```

---

# 7️⃣ Requirements

- CUDA Toolkit
- CMake
- GCC (with CUDA support)
- Python 3.x
- PyTorch
- SLURM (for `srun` execution)

---

# 🔁 Reproducibility Settings

To reproduce the reported results:

- Batch size = 512
- Learning rate = 0.01
- Epochs = 15
- Negative samples = 60
- Window size = 1

---

This project demonstrates how GPU parallelism using CUDA can significantly accelerate computationally intensive natural language processing tasks such as training word embeddings with SGNS.

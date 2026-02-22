# CUDA Word2Vec (Skip-gram + Negative Sampling)

## Build
```bash
python setup.py build_ext --inplace

make clean
cmake ..
make -j

## Cuda version:
cd ./src/build
srun --gres=gpu:1 --cpus-per-task=4 --mem=2GB ./w2v_base_cuda_train \
  --emb-dim 128 \
  --batch-size 512 \
  --epochs 15

## Pytorth version:
cd ./src/cpu
srun --gres=gpu:1 --cpus-per-task=4 --mem=2GB python ./main.py --embedding_dim 128 --batch_size 512 --epochs 15



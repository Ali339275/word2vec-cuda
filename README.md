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
  --emb-dim 256 \
  --batch-size 128 \
  --epochs 20

## Pytorth version:
cd ./src/cpu
srun --gres=gpu:1 --cpus-per-task=4 --mem=2GB python ./main.py



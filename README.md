# Distributional Alignment as a Principle for Task Vectors in In-Context Learning

This repository provides the official code and data for Distributional Alignment as a Principle for Task Vectors in In-Context Learning.

## Overview of LTV

[image]

## Installation
- `setup.sh` creates `.venv`, upgrades pip, and installs `requirements.txt`.
- Run the setup script:
```bash
./setup.sh
source .venv/bin/activate
```

## Experimental Setup
#### Models
We evaluate our method on these models:
- `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-3.1-8B`
- `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B`

#### Data
The `./datasets` directory contains the benchmarks used in our experiments.
(AGNEWS, DBPedia, HateSpeech18, MR, SST-2, SST-5, SUBJ, TREC)

#### Environment
To ensure the reproducibility of the results in our paper, we report the environment used for our experiments:
- Python: 3.13.2
- CUDA: 12.2
- GPU: NVIDIA RTX 6000 Ada Generation (49140 MiB)
- NVIDIA driver: 535.247.01

## Running Experiments

The main runner is `run/run_our_ltv.sh`. It calls `run/run_our_ltv.py` with the config file.

```bash
./run/run_our_ltv.sh
```

#### Config
Edit experiment settings here:
- `config/config_our_ltv.py`

Key flags to check:
- `num_shot` (number of shots (k), which ensures label balance by including the maximum possible examples per label without exceeding k)
- `run_baseline` (zero-shot/few-shot)
- `run_ltv` (LTV task vector)
- `num_train_queries` (number of training queries used to extract task vectors)
- `ridge_lambda` (regularization strength for adaptive task vectors)

# Acknowledgments
We gratefully acknowledge the following repositories, which served as the foundation for this work:
- [I2CL: Implicit In-context Learning](https://github.com/LzVv123456/I2CL)
- [Task Vector: In-Context Learning Creates Task Vectors](https://github.com/roeehendel/icl_task_vectors)

# Citation
TBD

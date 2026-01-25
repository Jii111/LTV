# Distributional Alignment as a Principle for Task Vectors in In-Context Learning

This repository provides the official code and data for Distributional Alignment as a Principle for Task Vectors in In-Context Learning.

## Overview of LTV

[image]

## Setup
- `setup.sh` creates `.venv`, upgrades pip, and installs `requirements.txt`.
- Run the setup script:
```bash
./setup.sh
source .venv/bin/activate
```

## Reproducibility
- Python: 3.13.2
- CUDA: 12.2
- GPU: NVIDIA RTX 6000 Ada Generation (49140 MiB)
- NVIDIA driver: 535.247.01

## Models Used in Paper
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`
- `meta-llama/Llama-3.1-8B`
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen3-8B`

## Data
`./datasets` contains datasets used for experiments.

## Run the Experiment

The main runner is `run/run_our_ltv.sh`. It calls `run/run_our_ltv.py` with the config file.

```bash
./run/run_our_ltv.sh
```

## Config

Edit experiment settings here:
- `config/config_our_ltv.py`

Key flags to check:
- `num_shot` (number of shots (k), which ensures label balance by including the maximum possible examples per label without exceeding k)
- `run_baseline` (zero-shot/few-shot)
- `run_ltv` (LTV task vector)
- `num_train_queries` (number of training queries used to extract task vectors)
- `ridge_lambda` (regularization strength for adaptive task vectors)

# Acknowledgments
- [I2CL](https://github.com/LzVv123456/I2CL)
- [ICL Task Vectors](https://github.com/roeehendel/icl_task_vectors)

# Citation
TBD

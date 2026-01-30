# Distributional Alignment as a Principle for Task Vectors in In-Context Learning

This repository provides the official code and data for Distributional Alignment as a Principle for Task Vectors in In-Context Learning. 

## Overview
We posit that probabilistic alignment with ICL is a key desideratum for effective TV extraction methods, and accordingly propose **d_NTP** and **Linear Task Vector (LTV)**.

### d_NTP
- A metric that quantifies Task Vector quality by measuring the **discrepancy between the predictive distribution under TV-based inference and that under ICL-based inference**, in terms of next-token probability (NTP).
- Empirically, lower d_NTP correlates strongly with higher downstream performance, making it a reliable indicator of TV quality.

### Linear Task Vector (LTV)
<p align="center">
  <img src="assets/method.png" alt="LTV method" width="750">
</p>

- It employs **a linear mapping** that estimates the effect of demonstrations and uses the closed-form solution of a regression problem to extract task vectors.
- LTV is designed to reduce dNTP, aligning TV inference with ICL.

## Installation
- `setup.sh` creates `.venv`, upgrades pip, and installs `requirements.txt`.
- Run the setup script:
```bash
./setup.sh
source .venv/bin/activate
```

## Experimental Setup
### Models
We evaluate our method on these models:
- Llama Series: `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`, `meta-llama/Llama-3.1-8B`
- Qwen Series: `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B`

### Data
The `./our_datasets` directory contains the benchmarks used in our experiments.
- AGNEWS, DBPedia, HateSpeech18, MR, SST-2, SST-5, SUBJ, TREC

### Environment
For reproducibility, we report the hardware and software environment used in our study:
- Python: 3.13.2 / CUDA: 12.2 / NVIDIA driver: 535.247.01
- GPU: NVIDIA RTX 6000 Ada Generation (49140 MiB)

## Running Experiments
The main execution script is `run/run_our_ltv.sh`. It calls `run/run_our_ltv.py` with the config file.

```bash
./run/run_our_ltv.sh
```

### Config
Adjust experiment settings in `config/config_our_ltv.py`

Key parameters to check:
- `num_shot` (number of shots (k), which ensures label balance by including the maximum possible examples per label without exceeding k)
- `run_baseline` (zero-shot/few-shot)
- `run_ltv` (LTV task vector)
- `num_train_queries` (number of training queries used to extract task vectors)
- `ridge_lambda` (regularization strength for adaptive task vectors)

## Acknowledgments
We gratefully acknowledge the following repositories, which served as the foundation for this work:
- [I2CL: Implicit In-context Learning](https://github.com/LzVv123456/I2CL)
- [Task Vector: In-Context Learning Creates Task Vectors](https://github.com/roeehendel/icl_task_vectors)

## Citation
TBD

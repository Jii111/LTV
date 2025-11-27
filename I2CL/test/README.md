# M2 & M3 Testing Guide

## Methods

- **M1 (Baseline)**: Single-layer constant task vector
- **M2**: Mask-based multi-layer constant task vector (per-layer demo masking)
- **M3**: Query-adaptive multi-layer task vector (ridge regression on MHA features)

## Implementation

### Per-Layer Delta Extraction (Core Feature)

Each layer ℓ independently computes in **1 forward pass**:
- **Main path**: r_ℓ^att (demo-attend, normal ICL output)
- **Delta path**: r_ℓ^mask (demo masked at THIS layer only)
- **Delta**: Δ_ℓ = r_ℓ^att - r_ℓ^mask

```python
outputs = model(
    input_ids=input_ids,
    demo_mask_positions=demo_end_positions,
    compute_delta=True,  # Extracts all layer deltas in 1 pass
    return_head_outputs=False  # True for M3 features
)
layer_deltas = outputs.layer_deltas  # List[Tensor(batch, seq, hidden)]
```

### M2: Mask-based Multi-layer
```
Extraction:
  Δ_ℓ(x, D) = r_ℓ^att(x,D) - r_ℓ^mask(x,D)
  v_ℓ(D) = E_x[Δ_ℓ(x, D)]

Injection:
  r_ℓ^TV = r_ℓ^0 + v_ℓ(D)
```

### M3: Query-adaptive
```
Extraction:
  Δ_ℓ(x, D)  (same as M2)
  φ_ℓ(x) = concat_h o_ℓ,h(x)  (MHA head outputs)
  B_ℓ(D) = Δ_ℓ @ Φ_ℓ^T @ inv(Φ_ℓ @ Φ_ℓ^T + λI)

Injection:
  r_ℓ^TV = r_ℓ^0 + B_ℓ(D) @ φ_ℓ(x)
```

## Quick Start

**Requirements**: GPU with ≥40GB VRAM, Python 3.8+, PyTorch 2.0+, Transformers 4.30+

### Unit Tests (No GPU)
```bash
cd /path/to/ICLTV_exp1/I2CL
python test/test_m2_small.py
python test/test_m3_small.py
```

### Run Experiments
```bash
# M2
python run_m2.py --config_path configs/config_m2.py
bash test/run_m2_test.sh

# M3
python run_m3.py --config_path configs/config_m3.py
bash test/run_m3_test.sh

# Compare all methods
bash test/run_all_methods.sh
```

**Results**: `exps/{m2,m3,taskvector_shot}/Qwen2.5-7B/{dataset}/`

## Configuration

```python
# M2: configs/config_m2.py
config['method'] = 'M2'
config['all_layers'] = True
config['demo_masking'] = True

# M3: configs/config_m3.py
config['method'] = 'M3'
config['ridge_lambda'] = 0.01
config['extract_head_outputs'] = True
```

## Results

Each experiment produces:
- `result_dict.json`: Accuracy/F1 for zero-shot, few-shot, M2/M3
- `kl_divergence.json`: KL divergence metrics
- `run_{i}_task_vectors.pt`: Extracted task vectors
- `run_{i}_kl_hist.png`: KL distribution plots

## Troubleshooting

**CUDA OOM**: Reduce batch size in config (`config['bs'] = 4`)

**Import errors**: Run from I2CL directory (`cd /path/to/ICLTV_exp1/I2CL`)

**Custom model not found**: Check `modeling_custom_qwen2.py` is in `I2CL/models/`

## Expected Performance

**Runtime (A100 40GB, AGNews 30-shot)**:
- M1: ~10 min/run (~25GB VRAM)
- M2: ~15 min/run (~28GB VRAM) - **2x faster than old implementation**
- M3: ~20 min/run (~30GB VRAM) - **2x faster than old implementation**

**Accuracy (AGNews 30-shot)**:
- Zero-shot: ~45%, Few-shot: ~75%
- M1: ~70%, M2: ~72%, M3: ~74%

**KL Divergence (lower = better)**:
- M1: ~0.08-0.12, M2: ~0.05-0.08, M3: ~0.03-0.06

## File Structure

```
I2CL/
├── models/modeling_custom_qwen2.py  # Per-layer delta extraction
├── wrapper_m2.py, wrapper_m3.py     # M2/M3 wrappers
├── utils_method.py                  # Ridge regression utilities
├── run_m2.py, run_m3.py             # Experiment runners
├── configs/config_m2.py, config_m3.py
└── test/
    ├── test_m2_small.py, test_m3_small.py  # Unit tests
    └── run_m2_test.sh, run_m3_test.sh, run_all_methods.sh
```

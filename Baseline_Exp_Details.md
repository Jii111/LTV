# Baseline Experiment Details

Reviewer-requested learned-task-vector baselines for the NeurIPS rebuttal, added
to the main runner (`run/run_our_ltv.py`) alongside our LTV. This document records
what each method is, the exact hyperparameters (original paper/code vs. ours), and
the deliberate deviations, so the rebuttal table caption can be written accurately.

Naming note: both papers call their method "LTV". To avoid collision we refer to
**our** method as **LTV**, and the two baselines as **Learned-TV** (Yang et al.)
and **Learnable-TV** (Saglam et al.).

All facts below were verified line-by-line against the released repositories:
- Learned-TV: https://github.com/HLYang2001/Learned_TV (ICLR 2026, arXiv:2509.24169)
- Learnable-TV: https://github.com/baturaysaglam/ICL-task-repr (ACL Findings 2025, arXiv:2502.05390)

---

## 1. Methods

### 1.1 LTV (ours, reference row)
Closed-form ridge regression `W` mapping the zero-shot final-layer hidden at the
label position `h_zs(x)` to the ICL–zero delta `h_icl(x) − h_zs(x)`. Inference adds
`W·h_zs(x_test)` to the last-layer hidden at the label position. Training-free,
label-free (demonstrations + unlabeled anchor queries only).

### 1.2 Learned-TV — Yang et al., ICLR 2026
A single vector `θ ∈ R^d` (the only trained parameter; LLM frozen), randomly
initialized `U(−0.1, 0.1)`, **added to the INPUT of one decoder layer** (their best
config: the middle layer) at the last-token position. The vector is meant to
**replace** demonstrations: it is trained on **zero-shot** prompts, so ICL plays no
role in its construction. This is the "learned, not extracted" thesis.

### 1.3 Learnable-TV — Saglam et al., ACL Findings 2025
The task representation IS derived from ICL activations. Per layer, the mean over
clean ICL prompts of the attention output (o_proj output) at the last token gives a
basis `P[l] ∈ R^d`. The only trained parameters are a **`(n_layers × n_heads)`
scalar mixing matrix Φ**; the injected vector is
`v[l] = P[l] ⊙ repeat_interleave(Φ[l], head_dim)` (block-wise scaling of the basis),
**added to the OUTPUT of every decoder layer** at the last-token position.
(This is the *language* variant — o_proj applied once, then block-scaled — not the
regression code's per-head-projection + sum form.)

---

## 2. Training objectives (one switch per baseline)

Each baseline is run with two objectives:

| variant | supervision | prompts | purpose |
|---|---|---|---|
| `ce`   | gold labels (first-token cross-entropy) | Learned-TV: zero-shot; Learnable-TV: label-shuffled k-shot ICL | paper-faithful "their method" row |
| `lmse` | label-free: our eq.-11 proxy `‖h_icl − h(param)‖²` vs ICL teacher hiddens | zero-shot | same information budget as LTV; our addition |

The current sweep runs **`lmse` only**; `ce` will be run after the `lmse` results
are reviewed.

The `lmse` variant is the scientifically interesting comparison: three
architectures (LTV = linear closed-form, Learned-TV = free vector, Learnable-TV =
head-mixing) optimizing the **same** L_MSE objective. It extends the paper's Table 8
mapping family (constant → linear → MLP) and tests whether more expressive mappings
reduce L_MSE / d_NTP and improve accuracy under a matched budget.

---

## 3. Is the original setting "data-heavy"? — only Saglam

| | Learned-TV (Yang) | Learnable-TV (Saglam) |
|---|---|---|
| optimizer steps (original) | ~1000 (10 epochs × 100 samples, batch 1) | **2000 iters × batch 32 = 64,000 sample-forwards** |
| data regime | light (comparable to ours) | **~80× our budget** |
| our matched-budget faithfulness | faithful (our ~800 steps ≈ their regime) | intentionally reduced — the source of undertraining |

Consequence: our matched-budget reproduction of **Learned-TV is faithful**, whereas
**Learnable-TV is trained on far less signal than the original**, which is why it can
undertrain (see §6). Any batch-size/step-count discussion applies to Learnable-TV.

---

## 4. Hyperparameters: original vs. ours

### Learned-TV (Yang et al.)
| item | ours | original (paper / code) | note |
|---|---|---|---|
| optimizer | AdamW | AdamW | match |
| lr | 1e-3 | 1e-3 (paper text) / **5e-3 (released code)** | we use paper text |
| LR schedule | none (constant) | **linear decay, per-sample step** | NOT reproduced (see §7) |
| weight decay | 0.01 | 0.01 | match |
| batch size | 1 | 1 | match |
| init | U(−0.1, 0.1) | U(−0.1, 0.1) | match |
| injection layer | mid (L/2) | mid (their best) | match |
| training steps | 8 ep × 100 = 800 | ≤10 ep × 100 = 1000 | ~match |
| early stop | patience 3, val acc | patience 2, val acc | slightly relaxed |
| prompts | zero-shot | zero-shot | match |

### Learnable-TV (Saglam et al.)
| item | ours | original (paper / code) | note |
|---|---|---|---|
| optimizer | Adam | Adam | match |
| lr | 5e-5 | 5e-5 | match |
| weight decay | 0.0 | 0.0 (plain Adam) | match |
| batch size | 1 | **32** | reduced (see §6) |
| init of Φ | **zero** | **randn (Gaussian)** | changed (see §6) |
| training steps | 8 ep × 100 = 800 | **2000 iters** | reduced (matched budget) |
| basis | mean o_proj output, cached once | recomputed from fresh clean prompts each iter | cached (see §7) |
| CE prompt shots | 30 (matches our demo budget) | 5–10 | unified to our 30-shot |
| early stop | patience 3, val | none (best by training-batch loss) | added |

---

## 5. What we unified across both baselines (matched budget)

To keep the comparison with LTV fair (both original papers use much larger data/label
budgets than LTV's 256 unlabeled anchors):

1. **Same 256-anchor query pool** as LTV. `ce` additionally consumes the gold labels
   of those anchors; `lmse` is label-free (same information budget as LTV).
2. **80/20 train/val split + patience early stopping** — a shared harness for both
   baselines (Saglam originally has no val split; selects best by training-batch loss).
3. **batch 1** — both baselines run at batch 1 (Saglam's original is batch 32).
4. **Metrics** — every method records the identical set: acc / macro-F1, d_NTP,
   L_mse_logit (label-logit space), hidden L_MSE (eq. 11 sum + per-dim), and the
   f = 0 (zero-shot) references, so all rows are directly comparable.

---

## 6. Deliberate deviations and why (Learnable-TV)

### 6.1 Zero-init instead of randn
Randn init injected into **all** decoder layers is a large, harmful perturbation at
step 0 (accuracy far below zero-shot). The original overcomes this only via the
64,000-forward budget; under our matched budget, patience-based early stopping would
fire before the vector becomes useful and ship a near-random vector — reporting the
baseline as *far below zero-shot*, which is misleading (unfairly bad).
**Zero-init** starts neutral (`v = 0`, injection is a no-op), so validation begins at
the zero-shot floor and improves monotonically, making early stopping safe and the
matched-budget number fair. `init: 'randn'` remains available for exact reproduction.

### 6.2 Stability: per-dim MSE for `lmse`
The eq.-11 **sum**-over-dim MSE (~10³–10⁴) overflows fp16 backward through the
32-layer injection (produced NaN). We train on **per-dim** MSE (identical optimum),
plus grad-norm clipping and a non-finite-step guard. The *reported* L_mse is computed
separately, so this training-scale choice does not affect the metrics.

### 6.3 Cached basis
The original recomputes the basis from freshly sampled clean prompts every iteration
(stochastic). We compute it once from our fixed 30-shot demonstration + 256 anchors
and cache it — impractical to resample per step here, and it does not change the method.

---

## 7. Known gaps still to decide

- **Learned-TV linear LR schedule** (§4): the released code steps a per-sample linear
  decay; our wrapper uses constant lr. Minor/maybe-affects; can be added.
- **Learned-TV lr** (§4): released code uses 5e-3, not the paper's 1e-3 (we chose 1e-3).
- **Learnable-TV batch/steps** (§6, §3): under matched forward budget, undertraining
  is driven by **too few optimizer updates**, not batch size. Increasing batch to 32
  at a fixed forward budget REDUCES updates (800 → 25) and makes it worse; the fix is
  **more optimizer steps** (optionally with gradient accumulation for a modest
  effective batch). Decision pending after `lmse` results.

### 7.1 REQUIRED follow-up: higher-step Learnable-TV run (option B)

The current sweep is the **matched-budget** run (batch 1, ~800 updates). Because
SST-2 shows Learnable-TV(lmse) barely moving off the zero-shot reference
(L_mse_logit 254 ≈ 255), we **must additionally run a higher-step variant** to
distinguish "the method is weak" from "the method is undertrained at our budget",
before drawing any conclusion for the rebuttal.

Plan (to run after the current `lmse` sweep completes):
- **Option B** — increase optimizer updates substantially (e.g. `epochs` and/or
  `samples_per_epoch` up to ~2000–3000 total forwards), optionally with gradient
  accumulation for an effective batch of ~8–16 (gradient stability without reducing
  the update count). This approaches Saglam's original signal volume (64k forwards)
  more closely while staying tractable.
- Report both runs side by side: **matched-budget (this sweep)** vs
  **higher-step (option B)**. If Learnable-TV still fails to reduce L_mse_logit /
  d_NTP under option B, the conclusion (LTV's closed-form solution wins) is robust;
  if it improves, the matched-budget number was a budget artifact and must be
  captioned as such.
- Cost note: option B is ~2.5–8× slower per cell depending on the forward budget;
  scope to a subset of datasets first (e.g. SST-2 + one hard task like TREC) to
  decide whether a full 8-dataset option-B sweep is warranted.

This follow-up is **not optional**: without it, a low Learnable-TV number cannot be
reported honestly (it would conflate method quality with training budget).

---

## 8. How to run

```bash
# on the server (elice-40g): /mnt/working/LTV
python run/run_our_ltv.py --config_path config/config_l31_baselines.py
```

Config `config['learned_tv']` / `config['learnable_tv']` control losses, lr, steps,
init, etc. Results land in `results/baselines_l31/<model>/<dataset>/result_dict.json`
under `test_result['learned_tv']` / `test_result['learnable_tv']`; aggregate with
`baselines_table.py` into per-dataset mean±std tables (acc / d_NTP / L_mse_logit /
L_mse_per_dim over 5 runs).

Model: `meta-llama/Llama-3.1-8B`, 8-bit, 8 datasets (SST-2, SST-5, MR, SUBJ, TREC,
HateSpeech18, AGNews, DBPedia), 500 test samples, 30-shot demos, 5 runs.

---

## 9. Preliminary result (SST-2 only, `lmse`, 5 runs, mean±std)

| method | acc (%) | d_NTP ↓ | L_mse_logit ↓ | L_mse_per_dim ↓ |
|---|---|---|---|---|
| zero-shot (ref) | 89.20 ± 0.00 | — | — | — |
| ICL 30-shot (upper bound) | 94.24 ± 1.18 | — | — | — |
| **LTV (ours)** | **90.32 ± 0.69** | **0.13 ± 0.03** | **7.49 ± 1.45** | **0.34 ± 0.04** |
| Learned-TV (lmse) | 85.28 ± 1.59 | 0.16 ± 0.02 | 34.30 ± 3.24 | 0.71 ± 0.04 |
| Learnable-TV (lmse) | 88.72 ± 0.84 | 0.21 ± 0.02 | 254.17 ± 21.62 | 3.14 ± 0.14 |

Reading (1 of 8 datasets — not final): under the shared L_MSE objective, LTV wins on
accuracy, d_NTP, and L_mse_logit. Learnable-TV's L_mse_logit (254) ≈ the zero-shot
reference (~255), i.e. its logit distribution barely moved — consistent with the
undertraining expected at ~80× less budget than the original (§3, §7).

*Status: full 8-dataset sweep running on elice-40g; table to be finalized on completion.*

# Learned-TV & Learnable-TV Baselines — Run Guide + Experimental Design

Two reviewer-requested learned-task-vector baselines, integrated into the main
runner (`run/run_our_ltv.py`) so they run **in the same loop as our LTV**, on the
same demonstrations, the same test split, and with the same metrics
(accuracy / macro-F1 / **d_NTP** / **L_MSE**) written to the same result files.

- Branch: **`feat/learned-and-learnable-tv-baselines`**
- Model used for our runs: `meta-llama/Llama-3.1-8B` (8-bit)
- Baselines:
  - **Learned-TV** — Yang et al., ICLR 2026 ([code](https://github.com/HLYang2001/Learned_TV), arXiv:2509.24169)
  - **Learnable-TV** — Saglam et al., ACL Findings 2025 ([code](https://github.com/baturaysaglam/ICL-task-repr), arXiv:2502.05390)

Naming note: both papers call their own method "LTV". Here **LTV = ours**;
the baselines are always written as **Learned-TV** and **Learnable-TV**.
All method facts below were verified line-by-line against the released repos.

---

# PART 1 — How to run

## 1.1 Setup

```bash
git clone <repo> && cd LTV
git checkout feat/learned-and-learnable-tv-baselines

python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -U "datasets==3.2.0"     # older datasets can't load trec from the 2026 Hub

export HF_TOKEN=<your_hf_token>                # Llama-3.1-8B is gated
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export PYTHONPATH=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 1.2 The one command

```bash
.venv/bin/python -u run/run_our_ltv.py --config_path config/config_our_ltv.py 2>&1 | tee run.log
```

That single run produces, for every (dataset, run): zero-shot, few-shot ICL,
**LTV (ours)**, **Learned-TV**, and **Learnable-TV** — each with accuracy,
macro-F1, d_NTP, and the L_MSE family.

## 1.3 What the config controls

`config/config_our_ltv.py`:

```python
config['models']   = ['meta-llama/Llama-3.1-8B']
config['datasets'] = ['sst2','sst5','mr','subj','trec','hate_speech18','agnews','dbpedia']
config['num_shot'] = 30      # 30-shot demonstrations (shared by every method)
config['run_num']  = 5       # 5 runs, different demonstration seed each
config['bs']       = 8       # eval batch size — see 1.6 if you hit OOM

config['run_baseline']      = True   # zero-shot + few-shot ICL (required: they are the d_NTP / L_MSE references)
config['run_ltv']           = True   # ours
config['run_learned_tv']    = True   # Yang et al.
config['run_learnable_tv']  = True   # Saglam et al.

config['learned_tv']   = {'losses': ['ce', 'lmse'], ...}
config['learnable_tv'] = {'losses': ['ce', 'lmse'], ...}
```

`losses` picks which variants to train (both run by default):
- **`ce`** — the paper's own objective (gold-label cross-entropy)
- **`lmse`** — our eq.-11 proxy against ICL teacher hidden states (label-free)

See PART 2 for why both exist.

## 1.4 Runtime

Measured on one A100-40GB MIG slice, Llama-3.1-8B in 8-bit, 500 test samples,
256 anchors, 30-shot demos:

| configuration | per (dataset, run) | full 8 datasets × 5 runs |
|---|---|---|
| `losses: ['lmse']` only | ~25 min | ~17 h |
| `losses: ['ce','lmse']` (default) | **~45 min** | **~30 h** |

Most of the cost is batch-1 gradient training (both baselines), not evaluation.

**To parallelize**, split the dataset list across GPUs — datasets are independent:

```bash
# GPU 0
sed "s/^config\['datasets'\].*/config['datasets'] = ['sst2','sst5']/" \
    config/config_our_ltv.py > config/_run_a.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u run/run_our_ltv.py --config_path config/_run_a.py
# GPU 1: ['mr','subj'] · GPU 2: ['trec','hate_speech18'] · GPU 3: ['agnews','dbpedia']
```

4 GPUs → roughly 7–8 hours wall-clock.

## 1.5 Progress / ETA logging

The runner prints, with no extra tooling:

- a labeled `tqdm` bar for every evaluation pass
  (`Eval zero-shot`, `Eval few-shot ICL`, `Eval LTV (λ=5.0)`,
  `Eval Learned-TV (ce)`, `Eval Learnable-TV (lmse)`, …) → it/s + per-pass ETA
- a `tqdm` bar per training epoch with running loss
  (`Learnable-TV/ce ep3/8 … loss=0.412`)
- per-epoch training curves:
  `[Learned-TV/ce] epoch 3/8 train_loss 0.4661 val_score 0.7115`
- **`[ETA]` lines** after every run and every (model × dataset):

```
[ETA] sst2 runs: 3/5 done | elapsed 1h13m | 24m30s/unit | remaining 49m00s (~08:58)
[ETA] model x dataset combinations: 1/8 done | elapsed 2h04m | 2h04m/unit | remaining 14h28m (~23:12)
```

So `tail -f run.log | grep "\[ETA\]"` is enough to track the sweep.

## 1.6 If a run crashes

Known failure on MIG / small-VRAM slices: `NVML_SUCCESS == r INTERNAL ASSERT
FAILED` from the CUDA allocator — this is an **OOM in disguise** (NVML queries
are blocked inside MIG, so PyTorch can't format the real OOM message).

Fix: lower the eval batch size (`config['bs'] = 4`) and rerun. Results are
written per dataset, so finished datasets are kept — delete only the
half-finished dataset directory and rerun. (`init_exp_path` intentionally
refuses to overwrite an existing results directory.)

Also note: `datasets==2.20` fails to load `trec` from the current Hub with a
`UnicodeDecodeError`; use `datasets>=3.2.0` as in 1.1.

## 1.7 Where results land

`results/ltv/<model>/<dataset>/result_dict.json`, written once all 5 runs of a
dataset finish:

```jsonc
{
  "test_result": {
    "zero_shot": [ {"acc": ..., "macro_f1": ...} ],
    "few_shot":  [ {"acc": ..., "macro_f1": ...}, ... ],           // 5 runs
    "ltv":       [ {"256_queries": {"ridge_lambda_5.0": {          // 5 runs
                     "acc": ..., "macro_f1": ...,
                     "d_NTP": ..., "d_NTP_zero_ref": ...,
                     "L_mse_logit": ..., "L_mse_logit_zero_ref": ...,
                     "L_mse": ..., "L_mse_per_dim": ..., "L_mse_zero_ref": ... }}} ],
    "learned_tv":   [ {"ce": {...same metric keys..., "train_curve": [...]},
                       "lmse": {...}} ],                            // 5 runs
    "learnable_tv": [ {"ce": {...}, "lmse": {...}} ]                 // 5 runs
  },
  "time": {"ltv": [...], "learned_tv": [...], "learnable_tv": [...]}
}
```

Every method — ours and both baselines — carries the **same metric set**:

| key | meaning |
|---|---|
| `acc`, `macro_f1` | task performance (label-restricted argmax; identical protocol for all methods) |
| `d_NTP` | KL(P_icl ‖ P_method) over label-token probabilities — our proposed criterion |
| `d_NTP_zero_ref` | the f = 0 reference: KL(P_icl ‖ P_zero-shot) |
| `L_mse_logit` | ‖z_icl − z_method‖² over label logits (the z of Prop. 5.1) |
| `L_mse_logit_zero_ref` | same, for f = 0 |
| `L_mse`, `L_mse_per_dim` | eq.-11 hidden-space MSE (sum over dim / per dim) |
| `L_mse_zero_ref` | eq.-11 for f = 0 |
| `curve` (baselines only) | per-epoch train loss + held-out diagnostic → convergence evidence |

## 1.8 Aggregating

```bash
.venv/bin/python run/compare_zero_ltv.py --model_name meta-llama/Llama-3.1-8B
```

prints per-benchmark mean ± std across the 5 runs and writes
`zero_vs_ltv_summary.json` (per-dataset means/stds, reduction %, cross-benchmark
averages).

---

# PART 2 — Why this experimental setting

## 2.1 The problem: both baselines are far more resource-hungry than we are

Reviewer 7wVq asked for these two methods because their absence makes it
*"difficult to assess the true state-of-the-art performance."* So they must be
run **properly**, not crippled. But their original settings consume far more
supervision than our method, and a raw side-by-side would compare methods that
saw wildly different amounts of data.

Verified from the released code:

| | labeled examples | gradient updates | training prompts |
|---|---|---|---|
| **LTV (ours)** | **30** (the demonstration only) | **0** (closed form) | — |
| **Learned-TV** (Yang) | ~1,000 (600 train + 400 val from a 1,000-example pool) | 1,000 (10 epochs × 100, batch 1) | **zero-shot** prompts |
| **Learnable-TV** (Saglam) | thousands (full train/test split) | **64,000** sample-gradients (2,000 iters × batch 32) | label-shuffled k-shot ICL prompts |

Our LTV solves a closed-form ridge problem from **30 labeled demonstrations +
256 _unlabeled_ anchor queries**. Learned-TV uses ~30× more labels;
Learnable-TV two to three orders of magnitude more.

## 2.2 Our answer: equalize the resources, never weaken the method

Every baseline gets **exactly the resources our LTV gets**:

- the **same 30-shot demonstration** (same seed, same run),
- the **same 256 anchor queries**,
- the same 500-example test split, the same 5 runs, the same evaluator.

We do **not** touch the baselines' own hyperparameters — optimizer, lr, weight
decay, init, injection site, batch size and selection rule are all theirs. What
we match is the *data budget*; otherwise the table would read "our method with 30
labels vs. their method with 64,000".

So the claim to make is not *"we beat them"* but **"at an equal resource budget
this is what each method achieves — and ours needs no labels and no gradients."**

## 2.3 Learned-TV is not an ICL method — hence the `lmse` variant

Yang et al. train a free vector `θ ∈ R^d` on **zero-shot prompts with gold
labels**; ICL demonstrations play no role anywhere in their pipeline (verified:
`create_sets.py` feeds `zsl_prompts` to the trainer). Their thesis is literally
that task vectors should be *learned, not extracted from ICL*.

That makes a direct "ICL task-vector" comparison awkward for both sides, so we
run **two variants**:

| variant | optimizes | supervision | why |
|---|---|---|---|
| **`ce`** | gold-label first-token CE on zero-shot prompts | 256 gold labels | their actual method, faithfully reproduced |
| **`lmse`** | our eq.-11 proxy ‖h_icl − h(θ)‖² against **30-shot ICL teacher hiddens** | **none** (label-free) | places their *architecture* in our ICL setting for a like-for-like comparison |

The `lmse` variant is our construction and is labeled as such. It answers:
*given the same ICL signal and no labels, what can a free vector do versus our
closed-form linear map?*

## 2.4 Learnable-TV is an ICL method — but a label-hungry one

Saglam et al. **do** derive their representation from ICL activations (per-layer
mean attention output over clean ICL prompts) and learn only a
`(n_layers × n_heads)` scalar mixing matrix Φ. It belongs in the ICL
task-vector family; it simply needs many labels to fit Φ.

Under our 30-shot / 256-anchor budget:

- **basis**: built from our fixed 30-shot demonstration + the 256 anchors
  (anchors appear only as unlabeled queries) → costs 30 labels, same as ours;
- **`ce`**: gold labels of the 256 anchors, with the demonstrations
  label-shuffled exactly as in their method (the corruption that forces the
  vector to carry the task);
- **`lmse`**: same architecture, our label-free objective.

Each training step draws a fresh shuffle/order of the fixed 30 demonstrations,
so the prompt differs every step even though the underlying pool is fixed.

## 2.5 Deliberate deviations (all documented in code comments too)

| deviation | why |
|---|---|
| **Learnable-TV: zero-init Φ** (paper uses `randn`) | with all-layer injection, `randn` is a large *harmful* perturbation at step 0; the paper escapes it only via its 64,000-forward budget. At a matched budget it would ship a near-random vector and report an unfairly low number. Zero-init starts neutral (`v = 0`, injection is a no-op) and improves from the zero-shot floor. `init: 'randn'` remains available. |
| **Learnable-TV: cached basis** (paper resamples per iteration) | our demonstration is fixed, so recomputation returns the same value; they resample because their prompts change every step. |
| **Learnable-TV: `lmse` trains on per-dim MSE** | the eq.-11 **sum** (~10³–10⁴) overflows fp16 backward through 32-layer injection (produced NaN). Per-dim has the identical optimum; reported L_MSE is computed separately, so metrics are unaffected. Grad-norm clipping and a non-finite-step guard are also applied. |
| **Learnable-TV: selection = lowest epoch-mean training loss** | this *is* the paper's rule — their `lowest_val_loss` is the current training-batch loss and `early_stoppage_tolerance` never breaks the loop. We compare epoch means because we run batch 1, where per-step loss is far noisier than their batch of 32. A held-out slice is still evaluated but **only logged** as a convergence diagnostic. |
| **Learned-TV: paper-text lr 1e-3** | the released code hardcodes 5e-3 while the paper text says 1e-3; we follow the paper. Their per-sample linear-decay schedule **is** reproduced. |
| Both: batch 1 | matches Learned-TV exactly; for Learnable-TV it is a reduction from batch 32 (see 2.6). |

## 2.6 Known limitation and the follow-up run that is still required

At a matched budget Learnable-TV gets ~800–2,400 updates versus the paper's
64,000 sample-gradients. In our first SST-2 sweep its `L_mse_logit` stayed at
254 while the zero-shot reference is 255 — i.e. **the logit distribution barely
moved**, which indicates undertraining rather than a weak method.

Therefore, **before this baseline's number is reported anywhere**, run a
higher-budget variant and show the convergence curve:

```python
config['learnable_tv']['epochs'] = 20            # or
config['learnable_tv']['samples_per_epoch'] = 400
```

then read `curve` in the result JSON. The goal is not to make the baseline win
or lose but to demonstrate **saturation**: if the held-out diagnostic plateaus
and the method still trails LTV, the conclusion is robust; if it keeps
improving, the matched-budget number was a budget artifact and must be labeled
as such.

Note that raising the *batch* at a fixed forward budget does the opposite of
what is needed — it reduces the number of optimizer updates (e.g. 800 → 25). The
correct knob is **more updates**, optionally with gradient accumulation for a
modest effective batch.

---

## Appendix — method summaries

**LTV (ours).** Closed-form ridge regression `W` mapping the zero-shot
final-layer hidden at the label position `h_zs(x)` to `h_icl(x) − h_zs(x)`; at
inference `W·h_zs(x_test)` is added at the last layer. No gradients, no labels
beyond the demonstration.

**Learned-TV (Yang et al., ICLR 2026).** One vector `θ ∈ R^d`, `U(−0.1, 0.1)`
init, added to the **input of a single decoder layer** (middle layer, their best
configuration) at the last-token position, LLM frozen. AdamW (lr 1e-3 paper /
5e-3 code, wd 0.01, per-sample linear decay), batch 1, ≤10 epochs × 100 samples,
patience-2 early stopping on held-out **full-vocab** accuracy — all reproduced.

**Learnable-TV (Saglam et al., ACL Findings 2025).** Per-layer basis `P[l]` =
mean attention output (o_proj output) at the last token over clean ICL prompts;
the trained parameters are only the `(n_layers × n_heads)` mixing weights Φ,
giving `v[l] = P[l] ⊙ repeat_interleave(Φ[l], head_dim)`, added to **every
decoder layer's output** at the last-token position. Adam (lr 5e-5, no weight
decay), CE of the true first answer token on label-shuffled k-shot ICL prompts.
(This is their *language* variant — `o_proj` applied once and the output
block-scaled — not the regression code's per-head projection + sum.)

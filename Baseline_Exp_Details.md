# Learned-TV, Learnable-TV & LoReFT Baselines — Run Guide + Experimental Design

Three reviewer-requested learned-task-vector baselines, integrated into the main
runner (`run/run_our_ltv.py`) so they run **in the same loop as our LTV**, on the
same demonstrations, the same test split, and with the same metrics
(accuracy / macro-F1 / **d_NTP** / **L_MSE**) written to the same result files.

- Branches: **`feat/learned-and-learnable-tv-baselines`** (Learned-TV +
  Learnable-TV, reviewer 7wVq) and **`feat/loreft-baseline`** (adds LoReFT,
  reviewer EHiD; based on the former)
- Model used for our runs: `meta-llama/Llama-3.1-8B` (8-bit)
- Baselines:
  - **Learned-TV** — Yang et al., ICLR 2026 ([code](https://github.com/HLYang2001/Learned_TV), arXiv:2509.24169)
  - **Learnable-TV** — Saglam et al., ACL Findings 2025 ([code](https://github.com/baturaysaglam/ICL-task-repr), arXiv:2502.05390)
  - **LoReFT** — Wu et al., NeurIPS 2024 ([code](https://github.com/stanfordnlp/pyreft), arXiv:2404.03592) — see 2.7

Naming note: both papers call their own method "LTV". Here **LTV = ours**;
the baselines are always written as **Learned-TV**, **Learnable-TV**, and
**LoReFT**. All method facts below were verified line-by-line against the
released repos.

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
config['run_loreft']        = True   # Wu et al. (feat/loreft-baseline only)

config['learned_tv']   = {'losses': ['lmse'], 'epochs': 8, 'samples_per_epoch': 100, ...}
config['learnable_tv'] = {'losses': ['ce'],   'epochs': 8, 'samples_per_epoch': 100, ...}
config['loreft']       = {'losses': ['ce','lmse'], 'rank': 4, 'epochs': 8, 'samples_per_epoch': 100, ...}
```

On `feat/loreft-baseline`, `run_learned_tv` / `run_learnable_tv` default to
**False**: that sweep belongs to PR #4 and rerunning it here would double the
cost. Flip them back on to run everything in one pass.

**Which variants are reported** (see PART 2 for why):

| baseline | variants run | reason |
|---|---|---|
| Learned-TV (Yang) | **`lmse`** only | their method is not ICL-based at all (they train on zero-shot prompts with gold labels), so the ICL-based variant is the only comparable row |
| Learnable-TV (Saglam) | **`ce` + `lmse`** | their method *is* ICL-based, so `ce` is the faithful row; `lmse` is added so both baselines can also be compared on the *same* objective as ours |
| LoReFT (Wu) | **`ce` + `lmse`** | reviewer EHiD asked for an *explicitly optimized* intervention; `ce` is the LoReFT-faithful supervised row (gold first-token CE), `lmse` puts it on the same label-free objective and information budget as our LTV |

Both objectives stay implemented for every baseline — edit the `losses` lists
to change the mix.

Training budget is **800 batch-1 steps** (`epochs 8 × samples_per_epoch 100`)
for every gradient baseline, so the update budget is identical across them.
Read the per-epoch `curve` in the result JSON before changing it — and for
Learnable-TV read 2.6 first, because for that method the step count is *not*
the knob that matters.

## 1.4 Runtime (single GPU)

Measured on one A100-40GB MIG slice, Llama-3.1-8B in 8-bit, 500 test samples,
256 anchors, 30-shot demos, 800 training steps per baseline:

| component | s/step | per (dataset, run) |
|---|---|---|
| LTV (ours) | closed form | **0.3 min** |
| Learned-TV `lmse` | 0.53 | **7.1 min** |
| Learnable-TV `lmse` | 0.71 | **9.5 min** |
| Learnable-TV `ce` | 0.91 | **12.1 min** |
| LoReFT `lmse` *(estimate)* | ~0.55 | **~7.5 min** |
| LoReFT `ce` *(estimate)* | ~0.5 | **~7 min** |
| evaluation (6 passes × 500) + basis / ICL-target collection | — | ~9.6 min |
| | | **≈ 38.6 min / cell** (PR #4 mix) |

`ce` costs only 1.27× `lmse` per step even though its prompts are 30-shot
(~600 tokens vs ~30): the backward pass through 32 layers of injection
dominates, not the prompt length.

LoReFT rows are estimates, not measurements: same zero-shot prompts, same
step budget and single-layer injection as Learned-TV, plus a negligible
(d×r) matmul — so per-step cost should land at Learned-TV `lmse` ± a few
percent. The first `[ETA]` line of a real run replaces these numbers.

**Full sweep (PR #4 mix): 8 datasets × 5 runs = 40 cells ≈ 25.7 h** on a
single GPU. **LoReFT-branch mix** (zero/few/LTV + LoReFT `ce`+`lmse`, Yang
and Saglam off): ~24 min/cell → **≈ 16 h**; `run_num = 3` → **≈ 9.6 h**.

Cheaper options: `run_num = 3` → **15.4 h**; sst2+sst5 only at 5 runs →
**6.4 h**. Every dataset writes its own `result_dict.json`, so the sweep can be
stopped and resumed dataset by dataset. The `[ETA]` lines report the real
per-cell time as soon as one cell completes — trust those over this table.

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
| **Learned-TV** (Yang) | ~1,000 labeled prompts (600 train + 400 val, split 60/40 from a 1,000-query pool; demos drawn from a separate 10,000-example pool) | up to 1,000 (10 epochs × 100 sampled items, batch 1, patience 2) — **per layer, repeated for all 32 layers** | **zero-shot** prompts (`train_set.pt` holds `zsl_prompts`) |
| **Learnable-TV** (Saglam) | thousands (full train/test split) | 2,000 iters × batch 32 = **64,000** sample-gradients, plus a basis resampled from 100 clean prompts *every iteration* → **≈264,000 forward passes** | label-shuffled k-shot ICL prompts |

Note what that budget is *not* buying: Learnable-TV's Φ is only
`n_layers × n_heads` (448 for their GPT-J, 1024 for our Llama-3.1-8B), and
Learned-TV's θ is a single 4096-vector. Neither is capacity-bound — see 2.6.

Our LTV solves a closed-form ridge problem from **30 labeled demonstrations +
256 _unlabeled_ anchor queries**. Learned-TV uses ~30× more labels;
Learnable-TV two to three orders of magnitude more.

## 2.2 Our answer: equalize the resources, never weaken the method

Every baseline gets **exactly the resources our LTV gets**:

- the **same 30-shot demonstration** (same seed, same run),
- the **same 256 anchor queries**,
- the same 500-example test split, the same 5 runs, the same evaluator.

We keep the baselines' own design decisions — optimizer, weight decay,
injection site, objective and selection rule are all theirs. What we match is
the *data budget*; otherwise the table would read "our method with 30 labels
vs. their method with 64,000".

The one place we deviate on a hyperparameter is **Learnable-TV's init and lr**,
and only to keep the method *functional* at the shortened schedule: its Φ scale
is set by the paper's `randn` init rather than by training, so a shortened run
with our zero-init leaves the injected vector at ~0. We re-couple `steps × lr`
to that scale instead. Full derivation, measurements and the exact reproduction
setting are in 2.6 — this is a deviation that *helps* the baseline, and it is
logged per-epoch (`phi_l2`, `phi_shift_l2`) so it can be checked rather than
taken on trust.

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
| **Learnable-TV: zero-init Φ + lr 1e-3** (paper: `randn`, lr 5e-5) | `randn` (std 1) is what sets Φ's scale in the paper — training moves it only ~2% (see 2.6) — and with all-layer injection a random draw is a large harmful perturbation we measured at 0.80 acc, below the 0.89 zero-shot floor. Zero-init starts neutral (`v = 0`, injection is a no-op), and lr is raised so `steps × lr` = 0.8 reaches the scale their init supplies for free. This lets Φ actually be *optimized* rather than perturbed. `init: 'randn', lr: 5e-5` remains available. |
| **Learnable-TV: cached basis** (paper resamples per iteration) | our demonstration is fixed, so recomputation returns the same value; they resample (100 clean prompts × 2000 iters) because their prompts change every step. This is where most of their ~264,000 forward passes go, and it is the deviation that actually shrinks our cost. |
| **Learnable-TV: `lmse` trains on per-dim MSE** | the eq.-11 **sum** (~10³–10⁴) overflows fp16 backward through 32-layer injection (produced NaN). Per-dim has the identical optimum; reported L_MSE is computed separately, so metrics are unaffected. Grad-norm clipping and a non-finite-step guard are also applied. |
| **Learnable-TV: selection = lowest epoch-mean training loss** | this *is* the paper's rule — their `lowest_val_loss` is the current training-batch loss and `early_stoppage_tolerance` never breaks the loop. We compare epoch means because we run batch 1, where per-step loss is far noisier than their batch of 32. A held-out slice is still evaluated but **only logged** as a convergence diagnostic. |
| **Learned-TV: lr 5e-3 (released code)** | the code hardcodes `AdamW(..., lr=5e-3)` while the paper text says 1e-3. We follow the **code**, which is the setting favourable to the baseline. |
| **Learned-TV: partial LR decay reproduced** | their scheduler is sized `num_epochs × len(train_set)` (= 6000) but only `samples_per_epoch` items are stepped per epoch (= 1000 steps), so their LR decays only 5e-3 → ~4.2e-3. We reproduce that *denominator* (ours ends at 51% of base) rather than decaying to 0 over the steps actually taken, which would halve the average LR and handicap the baseline. |
| **Learned-TV: single injection layer (`mid`)** ⚠️ | their `train_ltv.py` loops `for i in range(num_layer)`, training a **separate θ for every one of the 32 layers** and selecting the best afterwards. Reproducing that is 32× the cost, so we fix the middle layer, which their paper reports as the best configuration. **This is the one place where the baseline gets materially less than the original — state it in the table caption.** It plausibly contributes to Learned-TV `lmse` scoring below zero-shot on sst2. If a reviewer presses on it, the cheap answer is a one-dataset layer sweep (~3.8 h) showing `mid` is at or near the optimum. |
| Both: batch 1 | matches Learned-TV exactly; for Learnable-TV it is a reduction from batch 32 (see 2.6). |

## 2.6 Learnable-TV: the budget knob is `lr`, not the step count

Our first SST-2 sweep showed Learnable-TV's `L_mse_logit` stuck at 254 against a
zero-shot reference of 255 — the logit distribution barely moved. The obvious
reading is undertraining, and the obvious fix is more steps. **Both are wrong**,
and the arithmetic says why.

Φ is only `(n_layers × n_heads)` = **32 × 32 = 1024 scalars**, so the paper's
2000 iterations are not a capacity requirement. Adam's per-step update is
`lr·m̂/(√v̂+ε)`, whose magnitude is at most ≈ `lr`, so **total travel per entry
is bounded by `steps × lr`**. Their init is `torch.randn`, i.e. std 1
(`ltv.py:17`; the `normalized_weights` in `forward` is a misnomer — nothing is
normalized). Measured on a 32×32 tensor with maximally consistent gradients:

| setting | `steps × lr` | measured max\|ΔΦ\| | final mean\|Φ\| |
|---|---|---|---|
| `randn`, lr 5e-5, 2000 (**paper**) | 0.100 | 0.0998 | 0.790 *(init was 0.810)* |
| `zero`, lr 5e-5, 800 | 0.040 | 0.0400 | **0.039** |
| `zero`, lr 5e-5, 2400 | 0.120 | 0.1198 | **0.116** |
| `zero`, lr 1e-3, 800 (**shipped**) | 0.800 | 0.7886 | 0.650 |

Two conclusions:

1. **Under the paper's own budget, training moves Φ by ~2% of its magnitude.**
   Φ's scale — and most of its layer/head mixing — is set by the *random init*,
   not by the optimizer. The method's power lives in the basis `P[l]` (mean ICL
   attention output, resampled from 100 clean prompts every step), with Φ acting
   as a light reweighting on top.
2. **Therefore step count is nearly inert here.** Going 800 → 2400 raises the
   reachable scale from 0.04 to 0.12, both far below the 0.81 the paper's init
   supplies for free. Tripling the most expensive stage in the sweep would buy
   essentially nothing — which is why the shipped config does *not* do it.

What actually broke the baseline was our own zero-init: it removes the scale
setter, and at lr 5e-5 no step count recovers it. The shipped fix holds
`steps × lr` at the scale Φ must reach (`zero` + lr 1e-3 → 0.65 ≈ the paper's
0.81) rather than inflating steps. This is the *generous* choice for the
baseline: it lets Φ actually be optimized instead of fine-tuning a random draw.

**Before this baseline's number is reported anywhere**, check `curve` in the
result JSON, which now logs `phi_l2`, `phi_shift_l2` and `val_diagnostic` per
epoch. The bar is **saturation**, not a target step count: if the diagnostic
plateaus with `phi_l2` in the same range as the paper's init and the method
still trails LTV, the conclusion is robust. If `phi_l2` is an order of magnitude
low, the run is scale-limited and the number must not be reported.

To reproduce the paper's own combination for comparison, set
`init: 'randn', lr: 5e-5`. Note that raising the *batch* at a fixed forward
budget is counterproductive — it cuts the number of optimizer updates (800 → 25)
while leaving `steps × lr` even smaller.

## 2.7 LoReFT: the explicitly-optimized intervention reviewer EHiD asked for

Reviewer EHiD's core objection is that the "optimization-free" framing rests
on an unconvincing dichotomy, and asks for *"a comparison against at least one
explicitly optimized task-vector intervention (LoReFT-style)"* to separate
whether the linearity assumption genuinely matters or merely lets the method
be called optimization-free. So unlike the PR #4 baselines, this row is not a
competing task-vector paper — it is an **ablation of our own framing**: keep
everything about the harness fixed and swap in an explicitly optimized,
query-dependent intervention.

The operator is pyreft's `LoreftIntervention` verbatim (linear activation):

```
Φ(h) = h + Rᵀ((W h + b) − R h)        R ∈ R^{d×r} orthonormal columns, W ∈ R^{r×d}, b ∈ R^r
```

Inside the r-dim subspace spanned by R, the component of h is *replaced* by a
learned affine source; the orthogonal complement passes through untouched.
Three properties make it the right ablation partner:

| | Learned-TV (Yang) | **LoReFT (Wu)** | LTV (ours) |
|---|---|---|---|
| edit | constant vector θ | query-dependent low-rank edit Φ(h) | query-dependent linear f = W·h_zs |
| obtained by | gradient descent | gradient descent | **closed form (ridge)** |
| params | 4,096 | 32,772 (rank 4) | d² (closed-form, not "trained") |

The **harness is byte-for-byte the Learned-TV harness** — same 256 anchors,
same 800 batch-1 steps, same injection site (input of the middle decoder
layer, label position only), same `ce`/`lmse` objectives, same val-based
patience selection. The Learned-TV and LoReFT rows therefore differ **only in
the operator**, and both differ from LTV **only in how the mapping is
obtained**. That is exactly the axis the reviewer wants isolated.

Deviations from the original LoReFT recipe (all in `wrapper_loreft.py`'s
docstring too):

| deviation | why |
|---|---|
| single position (label token) instead of prefix+suffix position sets | every method in this suite intervenes at the label position only; multi-position would change the harness, not the operator |
| single `mid` layer by default (original: several layers) | matches the Learned-TV site so the operator comparison is clean; `layers: 'all'` or a list of ints reproduces the multi-layer setup at k× cost |
| `ce`/`lmse` objectives instead of seq2seq LM loss over a completion | classification-first-token CE *is* the LM loss restricted to our setting; `lmse` is the same-budget row |
| dropout 0 (original tasks: 0.0–0.05) | keeps the learned intervention deterministic at eval; config knob exists |

Training-dynamics sanity (the 2.6 lesson, re-checked): 800 × lr 9e-4 gives
per-entry travel 0.72 against init entry scales of ~1/√4096 ≈ 0.016 for both
R and W — this operator is **not** scale-limited, so the shared step budget is
a genuine optimization budget here. One genuine PyTorch pitfall was found and
fixed: `torch.nn.utils.parametrizations.orthogonal`'s householder map computes
`Q · diag(X).int()` assuming the underlying ±1 diagonal never moves; AdamW's
*decoupled weight decay* shrinks it to ±0.9999, the `.int()` truncates that to
0, and **R silently collapses to all-zeros** (verified on torch 2.7). The
rotation params are therefore in a weight-decay-free optimizer group, and the
per-epoch `curve` logs `ortho_err` with a hard assert < 1e-3. Before reporting
this baseline's number, check `curve` for: `rotate_drift` > 0 and
`source_drift` > 0 (both halves of the operator trained — these are
‖param − init‖, unlike the absolute `source_l2`), and `ortho_err` ~ 1e-7.
Travel bookkeeping: `max_travel_per_entry` = steps × peak lr = 0.72;
`scheduled_travel_per_entry` ≈ 0.36 is the tighter bound under the
warmup+decay schedule actually used — both ≫ the ~0.016 init scale.

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

**LoReFT (Wu et al., NeurIPS 2024).** Low-rank Representation Finetuning:
`Φ(h) = h + Rᵀ((Wh + b) − Rh)` with `R ∈ R^{d×r}` (orthonormal columns, kept
exactly orthonormal by torch's orthogonal parametrization) and a learned
affine source `Wh + b`, applied to the residual stream with the LLM frozen.
Operator taken verbatim from pyreft's `LoreftIntervention` (linear activation,
default inits); AdamW (lr 9e-4, linear warmup 10% then linear decay), rank 4,
input of the middle decoder layer at the label position. Requested by
reviewer EHiD as the explicitly-optimized comparison — see 2.7.

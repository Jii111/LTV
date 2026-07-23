"""
Regression experiment runner.

Evaluates on synthetic ICL regression tasks (linear / ReLU):
  - zero_shot : no context examples
  - few_shot  : n_shot in-context examples prepended to each query
  - LTV       : query-adaptive linear map W at last hidden layer,
                fit via closed-form ridge regression

Per-run protocol (following Garg et al., 2022):
  1. Sample one w for the whole run.
  2. Build demonstration (n_shot context pairs from w).
  3. Build train queries  (n_train context pairs from w, for LTV extraction).
  4. Build test queries   (n_test  context pairs from w, same w).
  5. Evaluate each method -> MSE / RMSE / MAE.
  6. Average metrics across run_num runs.

Usage:
  python run/run_regression.py --config config/config_regression.py \
                                --model meta-llama/Llama-3.1-8B --gpu 0 \
                                --dataset linear_regression
"""

import argparse
import importlib.util
import json
import os
import re
from typing import List

import numpy as np
import torch
from tqdm import tqdm

import core.utils.utils as utils
from core.wrapper_regression import RegressionLTVWrapper
from our_datasets.regression import (
    LinearRegressionTask,
    TwoLayerReLUTask,
    build_demo,
    build_query,
)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
REGRESSION_DATASETS = {
    "linear_regression": LinearRegressionTask,
    "relu_regression": TwoLayerReLUTask,
}

# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_float(text: str, fallback: float = 0.0) -> float:
    """Extract the first float from generated text."""
    m = _NUM_RE.search(text.strip())
    return float(m.group()) if m else fallback


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 15,
    batch_size: int = 1,
) -> List[float]:
    """Greedy generation -> parsed float predictions."""
    preds = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        for ids in out_ids:
            prompt_len = enc["input_ids"].shape[1]
            text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            preds.append(parse_float(text))
    return preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(preds: List[float], targets: List[float]) -> dict:
    p, t = np.array(preds), np.array(targets)
    mse = float(np.mean((p - t) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(p - t)))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main(args):
    cfg = args.config
    utils.set_seed(cfg["seed"])

    device = utils.set_device(args.gpu)
    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, device, output_hidden_states=True,
        load_in_8bit=cfg.get("load_in_8bit", False),
    )
    model_wrapper = RegressionLTVWrapper(model, tokenizer, model_config, device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dummy dataset just for d and task_fn
    dataset_cls = REGRESSION_DATASETS[args.dataset_name]
    dummy_ds = dataset_cls(task_name=args.dataset_name, d=cfg["d"], seed=cfg["seed"])
    d = dummy_ds.d

    # ── Experiment dir ──────────────────────────────────────────────────────
    save_dir = os.path.join(
        cfg["exp_name"], args.model_name.replace("/", "_"), args.dataset_name
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Method flags ─────────────────────────────────────────────────────────
    run_baseline = cfg.get("run_baseline", True)  # zero-shot + few-shot
    run_ltv = cfg.get("run_ltv", True)

    n_shot = cfg["num_shot"]
    n_train = cfg["num_train_queries"]
    n_test = cfg["test_data_num"]
    _rl = cfg["ridge_lambda"]
    ridge_lambdas = _rl if isinstance(_rl, list) else [_rl]
    bs = cfg.get("bs", 1)

    # ── Result storage ──────────────────────────────────────────────────────
    result = {}
    if run_baseline:
        result["zero_shot"] = []
        result["few_shot"] = []
    if run_ltv:
        for lam in ridge_lambdas:
            result[f"ltv_lam{lam}"] = []

    for run_id in tqdm(range(cfg["run_num"]), desc="Runs"):
        run_seed = cfg["seed"] + run_id
        print(f"\n{'='*60}\nRun {run_id+1}/{cfg['run_num']}\n{'='*60}")

        # 1. Sample w for this run (prior depends on dataset class)
        rng_w = np.random.default_rng(run_seed)
        w = dummy_ds.sample_w_for_run(d, rng_w)

        # 2. Context demonstration
        ctx_xs, ctx_ys = dummy_ds.regenerate_with_w(w, n=n_shot, seed=run_seed + 1000)
        demo = build_demo(ctx_xs, ctx_ys)

        # 3. Test queries
        test_xs, test_ys = dummy_ds.regenerate_with_w(w, n=n_test, seed=run_seed + 3000)
        test_queries = [build_query(x) for x in test_xs]
        true_ys = list(test_ys)

        # ── Zero-shot / Few-shot ────────────────────────────────────────────
        if run_baseline:
            preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
            result["zero_shot"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] Zero-shot    MSE={result['zero_shot'][-1]['mse']:.4f}")

            preds = generate_predictions(
                model, tokenizer, [demo + q for q in test_queries], batch_size=bs
            )
            result["few_shot"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] Few-shot     MSE={result['few_shot'][-1]['mse']:.4f}")

        # ── LTV ─────────────────────────────────────────────────────────────
        if run_ltv:
            train_xs, _ = dummy_ds.regenerate_with_w(w, n=n_train, seed=run_seed + 2000)
            train_queries = [build_query(x) for x in train_xs]

            for lam in ridge_lambdas:
                model_wrapper.extract_adaptive_task_vector(
                    demo, train_queries, tokenizer, batch_size=bs, ridge_lambda=lam,
                )
                with model_wrapper.inject_adaptive_task_vector():
                    preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
                key = f"ltv_lam{lam}"
                result[key].append(regression_metrics(preds, true_ys))
                print(f"[Run {run_id}] LTV(lam={lam})  MSE={result[key][-1]['mse']:.4f}")

    # ── Aggregate & save ────────────────────────────────────────────────────
    summary = {}
    for method, runs in result.items():
        mse_vals = [r["mse"] for r in runs]
        rmse_vals = [r["rmse"] for r in runs]
        mae_vals = [r["mae"] for r in runs]
        summary[method] = {
            "mse_mean": float(np.mean(mse_vals)),
            "mse_std": float(np.std(mse_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "mae_mean": float(np.mean(mae_vals)),
        }

    print("\n===== Summary =====")
    for method, s in summary.items():
        print(f"  {method:15}  MSE={s['mse_mean']:.4f} ± {s['mse_std']:.4f}"
              f"  RMSE={s['rmse_mean']:.4f}  MAE={s['mae_mean']:.4f}")

    out_path = os.path.join(save_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_run": result}, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config .py file")
    parser.add_argument("--model", required=True, dest="model_name")
    parser.add_argument("--dataset", required=True, dest="dataset_name",
                        choices=list(REGRESSION_DATASETS.keys()))
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("cfg", args.config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args.config = mod.config

    main(args)

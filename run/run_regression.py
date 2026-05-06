"""
Regression experiment runner (arxiv 2501.09240).

Evaluates on synthetic linear / sinusoidal ICL regression:
  - zero_shot    : no context examples
  - few_shot     : n_shot in-context examples prepended to each query
  - LTV : query-adaptive linear map W at last hidden layer

Per-run protocol (following Garg et al., 2022):
  1. Sample one w for the whole run.
  2. Build demonstration  (n_shot  context pairs from w).
  3. Build val queries    (n_val   context pairs from w, for layer selection).
  4. Build train queries  (n_train context pairs from w, for TV extraction).
  5. Build test queries   (n_test  context pairs from w, same w).
  6. Evaluate each method → MSE / RMSE / MAE.
  7. Average metrics across run_num runs.

Usage:
  python run_regression.py --config configs/config_regression.py \
                           --model meta-llama/Llama-3.1-8B  --gpu 0 \
                           --dataset linear_regression
"""

import argparse
import importlib.util
import json
import os
import re
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from baukit import TraceDict
from tqdm import tqdm

import utils
import utils_method as um
from fv_utils.extract_utils import compute_universal_function_vector
from my_datasets.regression import (
    TwoLayerReLUTask,
    build_demo,
    build_query,
)
from wrapper_m2 import M2AdaptiveWrapper

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
REGRESSION_DATASETS = {
    "linear_regression_easy": LinearRegressionEasy,
    "relu_regression": TwoLayerReLUTask,
}

# ---------------------------------------------------------------------------
# Default sub-configs passed to wrapper methods
# ---------------------------------------------------------------------------
I2CL_CFG = {
    "module":           ["mlp", "attn"],
    "layer":            "all",
    "inject_method":    "add",
    "inject_pos":       "last",
    "tok_pos":          "last",
    "init_value":       [0.1],
    "add_noise":        False,
    "noise_scale":      0.0,
    "post_fuse_method": "mean",
}

TV_CFG = {
    "module":           ["hidden"],
    "layer":            "all",
    "tok_pos":          "last",
    "post_fuse_method": "mean",
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
# Generation (with optional prefill-only hook)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 15,
    batch_size: int = 1,
) -> List[float]:
    """Greedy generation → parsed float predictions."""
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
# Prefill-only injection context managers
# (KV-cache decoding → seq_len == 1 → hook is skipped automatically)
# ---------------------------------------------------------------------------

@contextmanager
def _prefill_hook(layer_module, hook_fn):
    """Helper: register hook_fn, ensure removal after context."""
    handle = layer_module.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def inject_m2_prefill(model_wrapper, delta: Optional[torch.Tensor] = None,
                      W: Optional[torch.Tensor] = None):
    """M2 / M2-Adaptive injection at last hidden layer, prefill only."""
    assert (delta is None) != (W is None), "Provide exactly one of delta or W."
    device = model_wrapper.device
    delta = delta.to(device) if delta is not None else None
    W     = W.to(device)     if W     is not None else None

    layer_idx   = model_wrapper.num_layers - 1
    layer_module = model_wrapper._get_nested_attr(
        model_wrapper._get_arribute_path(layer_idx, "hidden")
    )

    def hook(_module, _inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.shape[1] <= 1:
            return outputs
        last = hidden.shape[1] - 1
        if delta is not None:
            hidden[:, last, :] = hidden[:, last, :] + delta
        else:
            h = hidden[:, last, :]
            if W.shape[1] == h.shape[1] + 1:  # bias-augmented: W is (D, D+1)
                W_mat, b = W[:, :-1], W[:, -1]
                hidden[:, last, :] = h + (W_mat @ h.T).T + b.unsqueeze(0)
            else:
                hidden[:, last, :] = h + (W @ h.T).T
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    with _prefill_hook(layer_module, hook):
        yield


@contextmanager
def inject_i2cl_prefill(model_wrapper, context_vector_dict: Dict, cfg: Dict):
    """
    I2CL-style injection at last token position, prefill only.

    Adds cfg['init_value'][0] * context_vector to the output of each
    (layer, module) pair, only during the prefill phase.
    """
    device   = model_wrapper.device
    strength = float(cfg["init_value"][0])
    handles  = []

    for layer in model_wrapper.inject_layers:
        for module in cfg["module"]:
            cv = context_vector_dict[layer][module].to(device)   # (D,)

            def make_hook(cv):
                def hook(_mod, _inputs, outputs):
                    hidden = outputs[0] if isinstance(outputs, tuple) else outputs
                    if hidden.shape[1] <= 1:
                        return outputs
                    last = hidden.shape[1] - 1
                    hidden[:, last, :] = hidden[:, last, :] + F.relu(
                        torch.tensor(strength, device=device)
                    ) * cv
                    if isinstance(outputs, tuple):
                        return (hidden,) + outputs[1:]
                    return hidden
                return hook

            lm = model_wrapper._get_nested_attr(
                model_wrapper._get_arribute_path(layer, module)
            )
            handles.append(lm.register_forward_hook(make_hook(cv)))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def replace_tv_prefill(model_wrapper, context_vector_dict: Dict,
                       target_layer: int, cfg: Dict):
    """
    ICL-TV style: replace last-position hidden state at target_layer,
    prefill only.
    """
    device  = model_wrapper.device
    module  = cfg["module"][0]                        # always 'hidden'
    cv      = context_vector_dict[target_layer][module].to(device)  # (D,)

    layer_module = model_wrapper._get_nested_attr(
        model_wrapper._get_arribute_path(target_layer, module)
    )

    def hook(mod, inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.shape[1] <= 1:
            return outputs
        last = hidden.shape[1] - 1
        hidden[:, last, :] = cv
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    with _prefill_hook(layer_module, hook):
        yield


# ---------------------------------------------------------------------------
# Context-vector extraction (shared between i2cl and tv)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_context_vector(model_wrapper, tokenizer, demo_examples: List[str],
                           cfg: Dict) -> Dict:
    """
    Extract and aggregate context vectors from individual demo examples.
    Returns context_vector_dict  {layer: {module: tensor(D,)}}
    """
    all_latent_dicts = []
    for example in demo_examples:
        with model_wrapper.extract_latent():
            tok = tokenizer(example, return_tensors="pt").to(model_wrapper.device)
            model_wrapper.model(**tok, use_cache=False)
        all_latent_dicts.append(model_wrapper.latent_dict)
        model_wrapper.reset_latent_dict()

    return model_wrapper.get_context_vector(all_latent_dicts, cfg)


# ---------------------------------------------------------------------------
# M2 task-vector extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_m2(model_wrapper, tokenizer, demo: str, train_queries: List[str],
               batch_size: int = 1) -> torch.Tensor:
    """Return mean Δ = h_ICL(last) − h_ZS(last) over train_queries."""
    deltas, model = [], model_wrapper.model
    for i in tqdm(range(0, len(train_queries), batch_size),
                  desc="Extract M2", leave=False):
        bq = train_queries[i: i + batch_size]

        icl_tok = tokenizer([demo + q for q in bq],
                            return_tensors="pt", padding=True).to(model.device)
        h_icl = um.extract_label_position_hidden(
            model(**icl_tok, output_hidden_states=True, use_cache=False).hidden_states[-1],
            icl_tok["attention_mask"]
        )
        zs_tok = tokenizer(bq, return_tensors="pt", padding=True).to(model.device)
        h_zs = um.extract_label_position_hidden(
            model(**zs_tok, output_hidden_states=True, use_cache=False).hidden_states[-1],
            zs_tok["attention_mask"]
        )
        deltas.append((h_icl - h_zs).cpu())
        del icl_tok, zs_tok
        torch.cuda.empty_cache()

    return torch.cat(deltas).mean(dim=0)


@torch.no_grad()
def extract_m2_adaptive(model_wrapper, tokenizer, demo: str,
                        train_queries: List[str], ridge_lambda: float = 0.01,
                        batch_size: int = 1, use_bias: bool = False) -> torch.Tensor:
    """Fit W: Δ ≈ W h_ZS (+ b if use_bias).
    Returns W (D, D) if use_bias=False, or W_aug (D, D+1) if use_bias=True."""
    features, targets, model = [], [], model_wrapper.model
    for i in tqdm(range(0, len(train_queries), batch_size),
                  desc="Extract M2-Adaptive", leave=False):
        bq = train_queries[i: i + batch_size]

        icl_tok = tokenizer([demo + q for q in bq],
                            return_tensors="pt", padding=True).to(model.device)
        h_icl = um.extract_label_position_hidden(
            model(**icl_tok, output_hidden_states=True, use_cache=False).hidden_states[-1],
            icl_tok["attention_mask"]
        )
        zs_tok = tokenizer(bq, return_tensors="pt", padding=True).to(model.device)
        h_zs = um.extract_label_position_hidden(
            model(**zs_tok, output_hidden_states=True, use_cache=False).hidden_states[-1],
            zs_tok["attention_mask"]
        )
        features.append(h_zs.cpu());  targets.append((h_icl - h_zs).cpu())
        del icl_tok, zs_tok
        torch.cuda.empty_cache()

    F_mat = torch.cat(features).T;  T_mat = torch.cat(targets).T
    if use_bias:
        # Augment features with bias row: fits Δ ≈ W @ h_zs + b → returns (D, D+1)
        ones = torch.ones(1, F_mat.shape[1], dtype=F_mat.dtype)
        F_mat = torch.cat([F_mat, ones], dim=0)
    return um.ridge_regression(T_mat, F_mat, ridge_lambda,
                               model_wrapper.device).cpu()


# ---------------------------------------------------------------------------
# FV (Function Vector) extraction and injection
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_fv(model_wrapper, tokenizer, demo: str, train_queries: List[str],
               model_config, n_top_heads: int = 10,
               batch_size: int = 1) -> Optional[torch.Tensor]:
    """
    Extract Function Vector from mean attention head activations.

    For each train query, runs (demo + query) through the model and records
    the input to o_proj (pre-projection head outputs) at the last token.
    Averages across queries, then builds the FV via compute_universal_function_vector.

    Returns fv_vector (D,) or None if the model is not in the universal-head list.
    """
    fv_cfg = model_config.fv
    attn_hook_names = fv_cfg["attn_hook_names"]   # o_proj hook names
    n_heads   = fv_cfg["n_heads"]
    head_dim  = fv_cfg["resid_dim"] // n_heads
    model     = model_wrapper.model

    all_activations: List[torch.Tensor] = []  # each: (n_layers, n_heads, head_dim)

    for i in tqdm(range(0, len(train_queries), batch_size),
                  desc="Extract FV", leave=False):
        bq = train_queries[i: i + batch_size]
        # Process one query at a time (batch_size=1) to get last-token activation
        for q in bq:
            inputs = tokenizer(demo + q, return_tensors="pt").to(model.device)
            with TraceDict(model, layers=attn_hook_names,
                           retain_input=True, retain_output=False) as td:
                model(**inputs, use_cache=False)

            layer_acts = []
            for hook_name in attn_hook_names:
                inp = td[hook_name].input
                if isinstance(inp, tuple):
                    inp = inp[0]
                # inp: (batch=1, seq, hidden) → last token → split by head
                x = inp[0, -1, :].view(n_heads, head_dim).cpu()
                layer_acts.append(x)
            all_activations.append(torch.stack(layer_acts, dim=0))  # (L, H, D_h)

    mean_activations = torch.stack(all_activations, dim=0).mean(dim=0)  # (L, H, D_h)

    try:
        fv, _ = compute_universal_function_vector(
            mean_activations, model, model_config, n_top_heads
        )
        return fv.reshape(-1).cpu()   # (resid_dim,)
    except (KeyError, UnboundLocalError):
        print(f"[FV] Model '{fv_cfg['name_or_path']}' not in universal-head list → FV skipped.")
        return None


@contextmanager
def inject_fv_prefill(model_wrapper, fv_vector: torch.Tensor, target_layer: int):
    """Add FV to residual stream at last token position, prefill only."""
    device = model_wrapper.device
    fv = fv_vector.to(device)
    layer_module = model_wrapper._get_nested_attr(
        model_wrapper._get_arribute_path(target_layer, "hidden")
    )

    def hook(_mod, _inp, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.shape[1] <= 1:
            return outputs
        hidden[:, -1, :] = hidden[:, -1, :] + fv
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    with _prefill_hook(layer_module, hook):
        yield


@torch.no_grad()
def select_best_layer_fv(model_wrapper, tokenizer, fv_vector: torch.Tensor,
                          val_queries: List[str], val_ys: List[float],
                          bs: int = 1) -> int:
    """Pick the layer with lowest val MSE using inject_fv_prefill."""
    best_layer, best_mse = 0, float("inf")
    for layer in range(model_wrapper.num_layers):
        with inject_fv_prefill(model_wrapper, fv_vector, layer):
            preds = generate_predictions(
                model_wrapper.model, tokenizer, val_queries, batch_size=bs
            )
        mse = regression_metrics(preds, val_ys)["mse"]
        print(f"  [FV] layer {layer:3d}  val MSE={mse:.4f}")
        if mse < best_mse:
            best_mse, best_layer = mse, layer
    print(f"  [FV] → best layer = {best_layer}  (MSE={best_mse:.4f})")
    return best_layer


# ---------------------------------------------------------------------------
# SV (State Vector) – proper implementation via SVEvaluator
#
# 원래 SV 방식 그대로:
#   1. SVEvaluator.read_activation()으로 각 demo 예제의
#      proj_tokens ("\ny: ") 위치에서 attention 상태 추출
#   2. demo 예제별 task vector를 평균
#   3. SVEvaluator.generate_write()로 test query의 "\ny: " 위치에 주입
#      (use_cache=False로 generation 전체에 지속 주입)
#
# 주의: generate_write()가 use_cache=False를 사용하므로
#       layer 선택용 val 평가가 느림 → n_sv_val(기본 10)로 제한
# ---------------------------------------------------------------------------

from sv_utils.TVframework import SVEvaluator as _SVEvaluator

# 회귀용 포맷: "x: [...]\ny: {y:.4f}\n"
_SV_FORMAT = {"proj_tokens": "\ny: ", "eos": "\n", "system": ""}


def _build_sv_evaluator(model_wrapper, tokenizer) -> "_SVEvaluator":
    """SVEvaluator를 기존 모델/토크나이저로 초기화 (모델 재로딩 없음)."""
    device_idx = (model_wrapper.device.index
                  if hasattr(model_wrapper.device, "index") else 0)
    return _SVEvaluator(
        model_path="",
        model=model_wrapper.model,
        tokenizer=tokenizer,
        devices=str(device_idx),
    )


@torch.no_grad()
def extract_sv(sv_evaluator: "_SVEvaluator",
               ctx_xs, ctx_ys, dummy_test_x,
               n_layers: int) -> Dict[int, torch.Tensor]:
    """
    demo 예제의 proj_tokens("\ny: ") 위치에서 attention 상태 추출.

    SVEvaluator.read_activation() 반환값:
      raw_tv[layer] = {'dev_0': (n_proj_toks, hidden), ..., 'test': ...}
    각 layer별 dev 예제를 평균하여 반환: {layer: (n_proj_toks, hidden)}
    """
    demon = [
        (f"x: [{', '.join(f'{v:.4f}' for v in x)}]", f"{y:.4f}")
        for x, y in zip(ctx_xs, ctx_ys)
    ]
    dummy_q = f"x: [{', '.join(f'{v:.4f}' for v in dummy_test_x)}]"

    raw_tv, _ = sv_evaluator.read_activation(
        dummy_q, demon,
        layer_indices=list(range(n_layers)),
        format_dict=_SV_FORMAT,
    )

    avg_tv: Dict[int, torch.Tensor] = {}
    n_shot = len(demon)
    for layer in range(n_layers):
        dev_vecs = [raw_tv[layer][f"dev_{r}"] for r in range(n_shot)]
        avg_tv[layer] = torch.stack(dev_vecs, dim=0).mean(dim=0)  # (n_proj_toks, D)
    return avg_tv


def _sv_generate(sv_evaluator: "_SVEvaluator",
                 xs, sv_task_vector: Dict, target_layer: int) -> List[float]:
    """SVEvaluator.generate_write()로 SV 주입 후 float 예측 생성."""
    layer_tv = {target_layer: sv_task_vector[target_layer]}
    preds = []
    for x in xs:
        query = f"x: [{', '.join(f'{v:.4f}' for v in x)}]"
        _, gen_str = sv_evaluator.generate_write(
            query,
            demon_list=[],
            task_vector=layer_tv,
            format_dict=_SV_FORMAT,
            intervention_mode="add#0#1",
            add_to="atten",
            write_pos=["project"],
        )
        preds.append(parse_float(gen_str))
    return preds


@torch.no_grad()
def select_best_layer_sv(sv_evaluator: "_SVEvaluator",
                          sv_task_vector: Dict,
                          val_xs, val_ys: List[float],
                          n_layers: int,
                          n_sv_val: int = 10) -> int:
    """
    val 예측 MSE가 가장 낮은 layer 선택.
    generate_write(use_cache=False) 기반이라 느리므로 n_sv_val로 val set 제한.
    """
    # val 예제 중 앞 n_sv_val개만 사용
    xs_sub  = val_xs[:n_sv_val]
    ys_sub  = val_ys[:n_sv_val]

    best_layer, best_mse = 0, float("inf")
    for layer in range(n_layers):
        preds = _sv_generate(sv_evaluator, xs_sub, sv_task_vector, layer)
        mse   = regression_metrics(preds, ys_sub)["mse"]
        print(f"  [SV] layer {layer:3d}  val MSE={mse:.4f}")
        if mse < best_mse:
            best_mse, best_layer = mse, layer
    print(f"  [SV] → best layer = {best_layer}  (MSE={best_mse:.4f})")
    return best_layer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(preds: List[float], targets: List[float]) -> dict:
    p, t = np.array(preds), np.array(targets)
    mse  = float(np.mean((p - t) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(p - t)))}


def ridge_oracle(ctx_xs: np.ndarray, ctx_ys: np.ndarray,
                 test_xs: np.ndarray, lam: float = 0.1) -> List[float]:
    """Fit ridge regression on context examples, predict on test_xs.

    For linear tasks (y = w^T x) with enough shots this should give near-zero MSE,
    confirming that the data generation is correct.
    """
    X = ctx_xs                                          # (n_shot, d)
    y = ctx_ys                                          # (n_shot,)
    w_hat = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return list(test_xs @ w_hat)


# ---------------------------------------------------------------------------
# Layer selection for TV (uses val MSE instead of accuracy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def select_best_layer_tv(model_wrapper, tokenizer, context_vector_dict: Dict,
                         tv_cfg: Dict, val_queries: List[str],
                         val_ys: List[float], bs: int = 1) -> int:
    """Pick the layer with lowest MSE on val queries using replace_tv_prefill."""
    best_layer, best_mse = 0, float("inf")
    for layer in range(model_wrapper.num_layers):
        if layer not in context_vector_dict:
            continue
        with replace_tv_prefill(model_wrapper, context_vector_dict, layer, tv_cfg):
            preds = generate_predictions(
                model_wrapper.model, tokenizer, val_queries, batch_size=bs
            )
        mse = regression_metrics(preds, val_ys)["mse"]
        print(f"  layer {layer:3d}  val MSE={mse:.4f}")
        if mse < best_mse:
            best_mse, best_layer = mse, layer
    print(f"  → best layer = {best_layer}  (MSE={best_mse:.4f})")
    return best_layer


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
    model_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dummy dataset just for d and task_fn
    dataset_cls = REGRESSION_DATASETS[args.dataset_name]
    dummy_ds = dataset_cls(
        split="train", d=cfg["d"], num_samples=10,
        task_name=args.dataset_name, seed=cfg["seed"],
    )
    d = dummy_ds.d

    # ── Experiment dir ──────────────────────────────────────────────────────
    save_dir = os.path.join(
        cfg["exp_name"], args.model_name.replace("/", "_"), args.dataset_name
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Method flags ────────────────────────────────────────────────────────
    run_zero_shot    = cfg.get("run_zero_shot",    True)
    run_few_shot     = cfg.get("run_few_shot",     True)
    run_ridge_oracle = cfg.get("run_ridge_oracle", True)
    run_i2cl_default = cfg.get("run_i2cl_default", False)
    run_tv           = cfg.get("run_tv",           False)
    run_fv           = cfg.get("run_fv",           False)
    run_sv           = cfg.get("run_sv",           False)
    run_m2           = cfg.get("run_m2",           True)
    run_m2_adaptive  = cfg.get("run_m2_adaptive",  True)
    # run_i2cl_train not implemented for regression

    n_shot        = cfg["num_shot"]
    n_train       = cfg["num_train_queries"]
    n_val         = cfg.get("num_val_queries", 50)
    n_test        = cfg["test_data_num"]
    _rl           = cfg["ridge_lambda"]
    ridge_lambdas = _rl if isinstance(_rl, list) else [_rl]
    bs            = cfg.get("bs", 1)

    # ── Result storage ──────────────────────────────────────────────────────
    result = {}
    for key, flag in [("zero_shot",    run_zero_shot),
                      ("few_shot",     run_few_shot),
                      ("ridge_oracle", run_ridge_oracle),
                      ("i2cl_default", run_i2cl_default),
                      ("tv",           run_tv),
                      ("fv",           run_fv),
                      ("sv",           run_sv),
                      ("m2",           run_m2)]:
        if flag:
            result[key] = []
    if run_m2_adaptive:
        for lam in ridge_lambdas:
            result[f"m2_adaptive_lam{lam}"] = []

    # ── Pre-initialise i2cl strength (layer & coef structure) ───────────────
    if run_i2cl_default:
        i2cl_cfg = {**I2CL_CFG}
        model_wrapper.init_strength(i2cl_cfg, cali_train=False)

    for run_id in tqdm(range(cfg["run_num"]), desc="Runs"):
        run_seed = cfg["seed"] + run_id
        print(f"\n{'='*60}\nRun {run_id+1}/{cfg['run_num']}\n{'='*60}")

        # 1. Sample w for this run (prior depends on dataset class)
        rng_w = np.random.default_rng(run_seed)
        w = dummy_ds.sample_w_for_run(d, rng_w)

        # 2. Context demonstration
        ctx_xs, ctx_ys = dummy_ds.regenerate_with_w(w, n=n_shot, seed=run_seed + 1000)
        demo = build_demo(ctx_xs, ctx_ys)
        # Individual examples for i2cl extraction
        split_demo_list = [
            f"x: [{', '.join(f'{v:.4f}' for v in x)}]\ny: {y:.4f}"
            for x, y in zip(ctx_xs, ctx_ys)
        ]

        # 3. Test queries
        test_xs, test_ys = dummy_ds.regenerate_with_w(w, n=n_test, seed=run_seed + 3000)
        test_queries = [build_query(x) for x in test_xs]
        true_ys = list(test_ys)

        # 4. Val queries (for TV layer selection)
        val_xs, val_ys = dummy_ds.regenerate_with_w(w, n=n_val, seed=run_seed + 4000)
        val_queries = [build_query(x) for x in val_xs]
        val_ys_list = list(val_ys)

        # ── Zero-shot ───────────────────────────────────────────────────
        if run_zero_shot:
            preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
            result["zero_shot"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] Zero-shot    MSE={result['zero_shot'][-1]['mse']:.4f}")

        # ── Few-shot ────────────────────────────────────────────────────
        if run_few_shot:
            preds = generate_predictions(model, tokenizer,
                                         [demo + q for q in test_queries], batch_size=bs)
            result["few_shot"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] Few-shot     MSE={result['few_shot'][-1]['mse']:.4f}")

        # ── Ridge Oracle ─────────────────────────────────────────────────
        if run_ridge_oracle:
            oracle_lam = cfg.get("ridge_oracle_lambda", 0.1)
            preds = ridge_oracle(ctx_xs, ctx_ys, test_xs, lam=oracle_lam)
            result["ridge_oracle"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] Ridge Oracle MSE={result['ridge_oracle'][-1]['mse']:.4f}")

        # ── I2CL default ────────────────────────────────────────────────
        if run_i2cl_default:
            cv_dict = extract_context_vector(
                model_wrapper, tokenizer, split_demo_list, i2cl_cfg
            )
            with inject_i2cl_prefill(model_wrapper, cv_dict, i2cl_cfg):
                preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
            result["i2cl_default"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] I2CL-default MSE={result['i2cl_default'][-1]['mse']:.4f}")

        # ── ICL Task Vector (TV) ────────────────────────────────────────
        if run_tv:
            tv_cfg = {**TV_CFG}
            cv_dict = extract_context_vector(
                model_wrapper, tokenizer, [demo], tv_cfg
            )
            print(f"[Run {run_id}] TV: selecting best layer via val MSE...")
            best_layer = select_best_layer_tv(
                model_wrapper, tokenizer, cv_dict, tv_cfg,
                val_queries, val_ys_list, bs=bs
            )
            with replace_tv_prefill(model_wrapper, cv_dict, best_layer, tv_cfg):
                preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
            result["tv"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] TV           MSE={result['tv'][-1]['mse']:.4f}")

        # ── Train queries (shared by FV / SV / M2 / M2-Adaptive) ────────
        if run_fv or run_sv or run_m2 or run_m2_adaptive:
            train_xs, _ = dummy_ds.regenerate_with_w(w, n=n_train, seed=run_seed + 2000)
            train_queries = [build_query(x) for x in train_xs]

        # ── FV (Function Vector) ─────────────────────────────────────────
        if run_fv:
            fv_vec = extract_fv(
                model_wrapper, tokenizer, demo, train_queries,
                model_config, n_top_heads=cfg.get("n_top_heads", 10), batch_size=bs
            )
            if fv_vec is not None:
                print(f"[Run {run_id}] FV: selecting best layer via val MSE...")
                best_fv_layer = select_best_layer_fv(
                    model_wrapper, tokenizer, fv_vec, val_queries, val_ys_list, bs=bs
                )
                with inject_fv_prefill(model_wrapper, fv_vec, best_fv_layer):
                    preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
                result["fv"].append(regression_metrics(preds, true_ys))
            else:
                result["fv"].append({"mse": float("nan"), "rmse": float("nan"), "mae": float("nan")})
            print(f"[Run {run_id}] FV           MSE={result['fv'][-1]['mse']:.4f}")

        # ── SV (State Vector) ────────────────────────────────────────────
        # SVEvaluator 기반 proper SV: proj_tokens("\ny: ") 위치 attention 주입
        # generate_write(use_cache=False) 사용 → layer 선택 시 n_sv_val로 제한
        if run_sv:
            sv_evaluator = _build_sv_evaluator(model_wrapper, tokenizer)
            sv_task_vector = extract_sv(
                sv_evaluator, ctx_xs, ctx_ys, test_xs[0], model_wrapper.num_layers
            )
            print(f"[Run {run_id}] SV: selecting best layer via val MSE (n_sv_val={cfg.get('n_sv_val', 10)})...")
            best_sv_layer = select_best_layer_sv(
                sv_evaluator, sv_task_vector,
                val_xs, val_ys_list,
                model_wrapper.num_layers,
                n_sv_val=cfg.get("n_sv_val", 10),
            )
            preds = _sv_generate(sv_evaluator, test_xs, sv_task_vector, best_sv_layer)
            result["sv"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] SV           MSE={result['sv'][-1]['mse']:.4f}")

        if run_m2:  # train_queries already built above
            delta = extract_m2(model_wrapper, tokenizer, demo, train_queries, batch_size=bs)
            with inject_m2_prefill(model_wrapper, delta=delta):
                preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
            result["m2"].append(regression_metrics(preds, true_ys))
            print(f"[Run {run_id}] M2           MSE={result['m2'][-1]['mse']:.4f}")

        if run_m2_adaptive:
            for lam in ridge_lambdas:
                W = extract_m2_adaptive(model_wrapper, tokenizer, demo, train_queries,
                                        ridge_lambda=lam, batch_size=bs,
                                        use_bias=cfg.get("m2_adaptive_use_bias", False))
                with inject_m2_prefill(model_wrapper, W=W):
                    preds = generate_predictions(model, tokenizer, test_queries, batch_size=bs)
                key = f"m2_adaptive_lam{lam}"
                result[key].append(regression_metrics(preds, true_ys))
                print(f"[Run {run_id}] M2-Adaptive(lam={lam})  MSE={result[key][-1]['mse']:.4f}")

    # ── Aggregate & save ────────────────────────────────────────────────────
    summary = {}
    for method, runs in result.items():
        mse_vals  = [r["mse"]  for r in runs]
        rmse_vals = [r["rmse"] for r in runs]
        mae_vals  = [r["mae"]  for r in runs]
        summary[method] = {
            "mse_mean":  float(np.mean(mse_vals)),
            "mse_std":   float(np.std(mse_vals)),
            "rmse_mean": float(np.mean(rmse_vals)),
            "mae_mean":  float(np.mean(mae_vals)),
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
    parser.add_argument("--config",  required=True,  help="path to config .py file")
    parser.add_argument("--model",   required=True,  dest="model_name")
    parser.add_argument("--dataset", required=True,  dest="dataset_name",
                        choices=list(REGRESSION_DATASETS.keys()))
    parser.add_argument("--gpu",     default="0")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("cfg", args.config)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args.config = mod.config

    main(args)

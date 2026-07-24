"""
Math generation experiment runner.

Evaluates on open-ended math tasks (GSM8K, MultiArith):
  - zero_shot      : no context examples
  - few_shot       : n_shot in-context examples prepended to each query
  - m2             : constant task vector (mean Δ) at last hidden layer
  - m2_adaptive    : query-adaptive linear map W at last hidden layer

Metric: top-1 accuracy (numerical exact match after extracting first number).

Usage (run from repo root, with PYTHONPATH=repo root):
  python run/run_math_gen.py --config_path config/config_math_gen.py --gpu 0 --dataset gsm8k
  python run/run_math_gen.py --config_path config/config_math_gen.py --gpu 0 --dataset multiarith

Ported from ICLTV/I2CL (run_math_gen.py). Only the import paths were adapted to
this repo's core.utils / core.wrapper_base / our_datasets layout; the M2 /
M2-Adaptive extraction and prefill-only injection logic is unchanged.
"""

import argparse
import gc
import itertools
import json
import os
import random
from contextlib import contextmanager
import multiprocessing
from multiprocessing import Process, Queue
from typing import List, Optional

import torch
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

import core.utils.utils as utils
import core.utils.utils_method as um
from core.wrapper_base import LlamaWrapper as M2AdaptiveWrapper


class StopStringCriteria(StoppingCriteria):
    """Stop generation once all sequences in the batch contain stop_string."""
    def __init__(self, tokenizer, stop_string: str, prompt_len: int):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores, **_kwargs) -> bool:
        for seq in input_ids:
            text = self.tokenizer.decode(seq[self.prompt_len:], skip_special_tokens=True)
            if self.stop_string not in text:
                return False
        return True


# ---------------------------------------------------------------------------
# Dataset loader (generation-style, not BaseTask)
# ---------------------------------------------------------------------------

def load_math_dataset(dataset_name, split, max_data_num=None, seed=42, is_instruct=True):
    if dataset_name == 'gsm8k':
        from our_datasets.gsm8k import GSM8KDataset
        return GSM8KDataset(split=split, max_data_num=max_data_num, seed=seed)
    elif dataset_name == 'multiarith':
        from our_datasets.multiarith import MultiArithDataset
        return MultiArithDataset(split=split, max_data_num=max_data_num, seed=seed, is_instruct=is_instruct)
    else:
        raise ValueError(f"Unknown math dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 16,
    batch_size: int = 1,
    dataset=None,
    return_raw: bool = False,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[str]:
    preds = []
    raw_texts = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        prompt_len = enc["input_ids"].shape[1]
        stopping_criteria = StoppingCriteriaList([
            StopStringCriteria(tokenizer, '\n\nQuestion:', prompt_len)
        ])
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        if do_sample:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['top_p'] = top_p
        out_ids = model.generate(**enc, **gen_kwargs)
        for j, ids in enumerate(out_ids):
            text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
            parsed = dataset.parse_prediction(text) if dataset else text.strip()
            preds.append(parsed)
            raw_texts.append(text)
    if return_raw:
        return preds, raw_texts
    return preds


# ---------------------------------------------------------------------------
# Task vector extraction (prefill-only hooks for generation)
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
                        batch_size: int = 1) -> torch.Tensor:
    """Fit W: Δ ≈ W h_ZS."""
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
        features.append(h_zs.cpu())
        targets.append((h_icl - h_zs).cpu())
        del icl_tok, zs_tok
        torch.cuda.empty_cache()

    F_mat = torch.cat(features).T
    T_mat = torch.cat(targets).T
    return um.ridge_regression(T_mat, F_mat, ridge_lambda,
                               model_wrapper.device).cpu()


# ---------------------------------------------------------------------------
# Prefill-only injection (hook is a no-op during KV-cache decode steps)
# ---------------------------------------------------------------------------

@contextmanager
def inject_m2_prefill(model_wrapper, delta: Optional[torch.Tensor] = None,
                      W: Optional[torch.Tensor] = None):
    assert (delta is None) != (W is None), "Provide exactly one of delta or W."
    device = model_wrapper.device
    delta = delta.to(device) if delta is not None else None
    W = W.to(device) if W is not None else None

    layer_idx = model_wrapper.num_layers - 1
    layer_module = model_wrapper._get_nested_attr(
        model_wrapper._get_arribute_path(layer_idx, "hidden")
    )

    def hook(_module, _inputs, outputs):
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        if hidden.shape[1] <= 1:   # KV-cache decode step → skip
            return outputs
        last = hidden.shape[1] - 1
        if delta is not None:
            hidden[:, last, :] = hidden[:, last, :] + delta
        else:
            h = hidden[:, last, :]
            hidden[:, last, :] = h + (W @ h.T).T
        if isinstance(outputs, tuple):
            return (hidden,) + outputs[1:]
        return hidden

    handle = layer_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def math_metrics(preds: List[str], targets: List[str], dataset) -> dict:
    n = len(targets)
    correct = sum(dataset.compare(p, t) for p, t in zip(preds, targets))
    return {"accuracy": correct / n if n > 0 else 0.0,
            "n_correct": correct, "n_total": n}


def _log_demo_preview(demo: str, n: int = 2):
    print(f"\n[DEMO PREVIEW] (first {n} example(s))")
    examples = [e for e in demo.split('\n\n') if e.strip()]
    for ex in examples[:n]:
        print(ex)
        print()


def _log_tv_prompts(demo: str, train_queries: List[str], n: int = 2):
    """Print full task-vector training prompts (demo + train query)."""
    print(f"\n[TV TRAIN PROMPTS] (first {min(n, len(train_queries))} samples)")
    sep = '-' * 60
    for q in train_queries[:n]:
        print(sep)
        print(demo + q)
    print(sep)


def _log_fs_prompts(demo: str, test_queries: List[str], n: int = 2):
    """Print full few-shot inference prompts (demo + test query)."""
    print(f"\n[FEW-SHOT PROMPTS] (first {min(n, len(test_queries))} samples)")
    sep = '-' * 60
    for q in test_queries[:n]:
        print(sep)
        print(demo + q)
    print(sep)


def _log_predictions(method, queries, preds, targets, dataset, raw_texts=None, n=3):
    print(f"\n[PREDICTIONS:{method}] (first {min(n, len(queries))} samples)")
    for i, (q, g, p) in enumerate(zip(queries[:n], targets[:n], preds[:n])):
        match = "✓" if dataset.compare(p, g) else "✗"
        print(f"  {match}  gold={repr(g):10s}  pred={repr(p):10s}  prompt={repr(q[:50])}")
        if raw_texts is not None:
            print(f"       raw_output={repr(raw_texts[i])}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = args.config
    utils.set_seed(cfg['seed'])
    device = utils.set_device(args.gpu)

    save_dir = os.path.join(cfg['exp_name'],
                            args.model_name.replace('/', '_'),
                            args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Math Generation: {args.model_name} on {args.dataset_name}")
    print(f"{'='*60}\n")

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, device, output_hidden_states=True,
        load_in_8bit=cfg.get('load_in_8bit', False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, device)

    # Load datasets
    is_instruct = 'instruct' in args.model_name.lower()
    
    print(f"[Model type] {'Instruct' if is_instruct else 'Base'} → MultiArith query suffix")
    train_ds = load_math_dataset(args.dataset_name, split='train', seed=cfg['seed'], is_instruct=is_instruct)
    val_ds   = load_math_dataset(args.dataset_name, split='validation',
                                  max_data_num=cfg.get('val_data_num'), seed=cfg['seed'], is_instruct=is_instruct)
    test_ds  = load_math_dataset(args.dataset_name, split='test',
                                  max_data_num=cfg.get('test_data_num'), seed=cfg['seed'], is_instruct=is_instruct)

    n_shot  = cfg['num_shot']
    n_train = cfg['num_train_queries']
    bs      = cfg.get('bs', 1)
    max_new = cfg.get('max_new_tokens', getattr(train_ds, 'max_new_tokens', 16))
    gen_kw  = {
        'do_sample':   cfg.get('do_sample', False),
        'temperature': cfg.get('temperature', 1.0),
        'top_p':       cfg.get('top_p', 1.0),
    }
    _rl     = cfg['ridge_lambda']
    ridge_lambdas = _rl if isinstance(_rl, list) else [_rl]

    result = {'zero_shot': [], 'few_shot': [], 'm2': []}
    for lam in ridge_lambdas:
        result[f'm2_adaptive_lam{lam}'] = []

    for run_id in tqdm(range(cfg['run_num']), desc="Runs"):
        run_seed = cfg['seed'] + run_id
        utils.set_seed(run_seed)
        print(f"\n{'='*60}\nRun {run_id+1}/{cfg['run_num']}\n{'='*60}")

        # Sample demo and train queries from train split
        all_train = train_ds.all_data[:]
        random.shuffle(all_train)
        demo_items   = all_train[:n_shot]
        train_items  = all_train[n_shot: n_shot + n_train]

        demo = train_ds.build_demo(demo_items)
        train_queries = [train_ds.get_query(item) for item in train_items]

        if run_id == 0:
            _log_demo_preview(demo, n=2)
            _log_tv_prompts(demo, train_queries, n=2)

        # Build test and val prompts
        test_queries = [test_ds.get_query(item) for item in test_ds.all_data]
        test_targets = [test_ds.get_answer(item) for item in test_ds.all_data]
        val_queries  = [val_ds.get_query(item)  for item in val_ds.all_data]
        val_targets  = [val_ds.get_answer(item) for item in val_ds.all_data]

        # Zero-shot
        preds, raw = generate_predictions(model, tokenizer, test_queries,
                                          max_new_tokens=max_new, batch_size=bs,
                                          dataset=test_ds, return_raw=True, **gen_kw)
        result['zero_shot'].append(math_metrics(preds, test_targets, test_ds))
        print(f"[Run {run_id}] Zero-shot    acc={result['zero_shot'][-1]['accuracy']:.3f}")
        _log_predictions('zero_shot', test_queries, preds, test_targets, test_ds, raw_texts=raw)

        # Few-shot
        fs_prompts = [demo + q for q in test_queries]
        if run_id == 0:
            _log_fs_prompts(demo, test_queries, n=2)
        preds, raw = generate_predictions(model, tokenizer, fs_prompts,
                                          max_new_tokens=max_new, batch_size=bs,
                                          dataset=test_ds, return_raw=True, **gen_kw)
        result['few_shot'].append(math_metrics(preds, test_targets, test_ds))
        print(f"[Run {run_id}] Few-shot     acc={result['few_shot'][-1]['accuracy']:.3f}")
        _log_predictions('few_shot', test_queries, preds, test_targets, test_ds, raw_texts=raw)

        # M2
        delta = extract_m2(model_wrapper, tokenizer, demo, train_queries, batch_size=bs)
        with inject_m2_prefill(model_wrapper, delta=delta):
            preds, raw = generate_predictions(model, tokenizer, test_queries,
                                              max_new_tokens=max_new, batch_size=bs,
                                              dataset=test_ds, return_raw=True, **gen_kw)
        result['m2'].append(math_metrics(preds, test_targets, test_ds))
        print(f"[Run {run_id}] M2           acc={result['m2'][-1]['accuracy']:.3f}")
        _log_predictions('m2', test_queries, preds, test_targets, test_ds, raw_texts=raw)

        # M2-Adaptive
        for lam in ridge_lambdas:
            W = extract_m2_adaptive(model_wrapper, tokenizer, demo, train_queries,
                                    ridge_lambda=lam, batch_size=bs)
            with inject_m2_prefill(model_wrapper, W=W):
                preds, raw = generate_predictions(model, tokenizer, test_queries,
                                                  max_new_tokens=max_new, batch_size=bs,
                                                  dataset=test_ds, return_raw=True, **gen_kw)
            key = f'm2_adaptive_lam{lam}'
            result[key].append(math_metrics(preds, test_targets, test_ds))
            print(f"[Run {run_id}] M2-Adaptive(lam={lam}) acc={result[key][-1]['accuracy']:.3f}")
            _log_predictions(key, test_queries, preds, test_targets, test_ds, raw_texts=raw)

        with open(os.path.join(save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result, f, indent=2)

    # Summary
    print(f"\n===== Summary =====")
    summary = {}
    for method, runs in result.items():
        accs = [r['accuracy'] for r in runs]
        mean = sum(accs) / len(accs) if accs else 0.0
        std  = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5 if accs else 0.0
        summary[method] = {'mean': mean, 'std': std, 'runs': accs}
        print(f"  {method:<30s}  acc={mean:.3f} ± {std:.3f}")

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    del model_wrapper, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_math_gen.py')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    return parser.parse_args()


def run_task(gpu_id, cfg, task_queue):
    import argparse as _ap
    while not task_queue.empty():
        model_name, dataset_name = task_queue.get()
        input_args = _ap.Namespace(
            model_name=model_name,
            dataset_name=dataset_name,
            gpu=gpu_id,
            config=cfg,
        )
        try:
            main(input_args)
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = get_args()
    config = utils.load_config(args.config_path)

    models   = [args.model]   if args.model   else config['models']
    datasets = [args.dataset] if args.dataset else config['datasets']

    combinations = list(itertools.product(models, datasets))
    task_queue = Queue()
    for combo in combinations:
        task_queue.put(combo)

    processes = [Process(target=run_task, args=(gpu_id, config, task_queue))
                 for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All math generation tasks completed.")

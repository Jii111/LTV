"""
Error-focused experiment runner for ICL vs M2 family.

Evaluates:
- ICL (few-shot) label hidden states
- M2 (constant Δ) corrected label hidden states
- M2 Adaptive (ridge, closed-form) corrected label hidden states
- M2 MLP (2-layer ReLU, square) corrected label hidden states

Metrics:
- Test-set MSE between ICL label hidden and each method's corrected hidden
- Optional accuracy metrics are left untouched in Evaluator; this script focuses on MSE.

MLP training:
- 256 anchor queries (configurable), batch size 8, 20 epochs
- LR 1e-3, cosine scheduler, warmup_ratio=0.1
- Eval/log every 0.5 epoch; logs saved to JSON and plotted

All tunables live in configs/config_error_exp.py (or another supplied config).
"""

import argparse
import copy
import gc
import itertools
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from tqdm import tqdm

import evaluator as ev
import my_datasets as md
import utils
import utils_method as um
from wrapper_m2 import M2AdaptiveWrapper, M2Wrapper


# ---------------------------
# Small utilities
# ---------------------------

def build_train_queries(train_dataset, max_queries, exclude_indices=None):
    total = len(train_dataset.all_data)
    if total == 0:
        return [], []
    if exclude_indices is None:
        exclude_indices = set()
    candidate_indices = [idx for idx in range(total) if idx not in exclude_indices]
    if len(candidate_indices) == 0:
        return [], []
    sample_size = min(max_queries, len(candidate_indices))
    query_indices = random.sample(candidate_indices, sample_size)
    queries = []
    for idx in query_indices:
        ques_str, _, _ = train_dataset.apply_template(train_dataset.all_data[idx])
        queries.append(ques_str)
    return queries, query_indices


def mse_between(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error between two tensors."""
    with torch.no_grad():
        device = a.device
        return F.mse_loss(
            a.to(device=device, dtype=torch.float32),
            b.to(device=device, dtype=torch.float32),
            reduction='mean'
        ).item()


# ---------------------------
# Data for MLP training
# ---------------------------

class HiddenDeltaDataset(Dataset):
    """Pairs zero-shot label hidden with target delta."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        return {
            "inputs": self.features[idx].float(),
            "labels": self.targets[idx].float(),
        }


class M2MLP(nn.Module):
    """2-layer square MLP with ReLU activation."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, inputs=None, labels=None):
        x = inputs
        delta = self.fc2(self.act(self.fc1(x)))
        loss = None
        if labels is not None:
            loss = F.mse_loss(delta, labels)
        return {"loss": loss, "logits": delta}


def collect_zero_and_delta(
    model,
    tokenizer,
    demo: str,
    queries: List[str],
    device: torch.device,
    batch_size: int,
    to_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect zero-shot label hiddens and ICL-zero deltas for given queries.
    Returns (features, targets) on CPU in the specified dtype.
    """
    feature_list = []
    target_list = []
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    for batch in tqdm(batches, desc="Collect MLP data", disable=len(batches) == 1):
        # ICL hidden
        icl_inputs = [demo + q for q in batch]
        icl_tokens = tokenizer(icl_inputs, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.no_grad():
            icl_out = model(**icl_tokens, output_hidden_states=True, use_cache=False)
            icl_label = um.extract_label_position_hidden(icl_out.hidden_states[-1], icl_tokens["attention_mask"])

        # Zero hidden
        zero_tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.no_grad():
            zero_out = model(**zero_tokens, output_hidden_states=True, use_cache=False)
            zero_label = um.extract_label_position_hidden(zero_out.hidden_states[-1], zero_tokens["attention_mask"])

        delta = icl_label - zero_label
        feature_list.append(zero_label.to(dtype=to_dtype).cpu())
        target_list.append(delta.to(dtype=to_dtype).cpu())

        del icl_tokens, icl_out, zero_tokens, zero_out, icl_label, zero_label, delta
        torch.cuda.empty_cache()

    features = torch.cat(feature_list, dim=0) if feature_list else torch.zeros(1, model.config.hidden_size)
    targets = torch.cat(target_list, dim=0) if target_list else torch.zeros_like(features)
    return features, targets


def summarize_run_mse(mse_runs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean over runs for each mse key."""
    if not mse_runs:
        return {}
    keys = list(next(iter(mse_runs.values())).keys())
    summary = {}
    for k in keys:
        vals = [v[k] for v in mse_runs.values()]
        summary[k] = float(np.mean(vals))
    return summary


def summarize_run_acc(acc_runs: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Mean over runs for each acc key."""
    if not acc_runs:
        return {}
    keys = list(next(iter(acc_runs.values())).keys())
    summary = {}
    for k in keys:
        vals = [v.get(k) for v in acc_runs.values() if v.get(k) is not None]
        if vals:
            summary[k] = float(np.mean(vals))
    return summary


def update_aggregate(args, dataset_summary: Dict[str, any]):
    """
    Persist dataset-level summary and aggregate mean/std across datasets.
    """
    agg_dir = os.path.join(args.config['exp_name'], args.model_name, "_aggregate")
    os.makedirs(agg_dir, exist_ok=True)
    agg_path = os.path.join(agg_dir, "aggregate.json")

    if os.path.exists(agg_path):
        with open(agg_path, 'r') as f:
            aggregate = json.load(f)
    else:
        aggregate = {}

    aggregate[args.dataset_name] = dataset_summary

    # Compute aggregate mean/std over datasets for MSE
    mse_keys = ['icl_vs_m2', 'icl_vs_m2_adaptive', 'icl_vs_m2_mlp']
    all_vals = {k: [] for k in mse_keys}
    for ds, entry in aggregate.items():
        mse_summary = entry.get('mse_summary', {})
        for k in mse_keys:
            if k in mse_summary:
                all_vals[k].append(mse_summary[k])
    agg_stats = {}
    for k, vals in all_vals.items():
        if vals:
            agg_stats[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals))
            }
    aggregate['_overall_mse'] = agg_stats

    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=4)

    # Plot aggregate bar with error bars if data present
    if agg_stats:
        plot_cfg = args.config.get('plot', {})
        colors = plot_cfg.get('colors', {})
        font_family = plot_cfg.get('font_family', None)
        if font_family:
            plt.rcParams['font.family'] = font_family
        labels = ['Static', 'Task Matrix', 'MLP']
        keys = ['icl_vs_m2', 'icl_vs_m2_adaptive', 'icl_vs_m2_mlp']
        means = [agg_stats[k]['mean'] if k in agg_stats else 0.0 for k in keys]
        stds = [agg_stats[k]['std'] if k in agg_stats else 0.0 for k in keys]

        colors_ordered = [
            colors.get('m2', '#1f77b4'),
            colors.get('m2_adaptive', '#ff7f0e'),
            colors.get('mlp', '#2ca02c'),
        ]

        plt.figure(figsize=(7, 4.5))
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=stds, color=colors_ordered, alpha=0.8, capsize=4)
        plt.xticks(x, labels, fontsize=12)
        plt.ylabel('Test MSE vs ICL (mean ± std across datasets)', fontsize=12)
        plt.title('Benchmark Aggregate', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        bar_png = os.path.join(agg_dir, "aggregate_mse.png")
        bar_svg = os.path.join(agg_dir, "aggregate_mse.svg")
        plt.savefig(bar_png, dpi=150)
        plt.savefig(bar_svg, dpi=150)
        plt.close()
        aggregate['_aggregate_plots'] = {'mse_png': bar_png, 'mse_svg': bar_svg}
        with open(agg_path, 'w') as f:
            json.dump(aggregate, f, indent=4)


def run_mlp_training(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    eval_features: torch.Tensor,
    eval_targets: torch.Tensor,
    dim: int,
    cfg: dict,
    save_dir: str,
):
    """Train M2 MLP with Trainer and return trained model plus logs."""
    train_dataset = HiddenDeltaDataset(train_features, train_targets)
    eval_dataset = HiddenDeltaDataset(eval_features, eval_targets)
    steps_per_epoch = math.ceil(len(train_dataset) / cfg['batch_size'])
    eval_steps = max(1, int(steps_per_epoch * cfg['eval_interval_epochs']))

    model = M2MLP(dim)

    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "mlp_ckpt"),
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['epochs'],
        learning_rate=cfg['lr'],
        warmup_ratio=cfg['warmup_ratio'],
        lr_scheduler_type="cosine",
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        report_to=[],
        save_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )

    mlp_logs = []

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = torch.tensor(preds, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        mse = F.mse_loss(preds, labels, reduction='mean').item()
        return {"mse": mse}

    class LoggingCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            mlp_logs.append(
                {
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                    "mse": metrics.get("eval_mse"),
                }
            )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=None,
        callbacks=[LoggingCallback()],
    )
    # Initial eval before training (epoch 0)
    initial_metrics = trainer.evaluate()
    mlp_logs.append(
        {
            "global_step": 0,
            "epoch": 0.0,
            "mse": initial_metrics.get("eval_mse"),
        }
    )
    trainer.train()
    return trainer.model.float(), mlp_logs


# ---------------------------
# Main experiment
# ---------------------------

def main(args):
    print(f"[Init] Starting run: model={args.model_name}, dataset={args.dataset_name}")
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    utils.init_exp_path(args, args.config['exp_name'])

    print("[Load] Loading model/tokenizer...")
    model, tokenizer, model_config, _ = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True
    )
    print("[Load] Model/tokenizer loaded.")

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    m2_wrapper = M2Wrapper(model, tokenizer, model_config, args.device)
    m2_adaptive_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, args.device)

    print("[Data] Loading datasets...")
    train_dataset = md.get_dataset(args.dataset_name, split='train', max_data_num=None,
                                   seed=args.config['seed'])
    val_dataset = md.get_dataset(args.dataset_name, split='validation',
                                 max_data_num=args.config['val_data_num'],
                                 sample_mode=args.config['sample_method'],
                                 seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test',
                                  max_data_num=args.config['test_data_num'],
                                  sample_mode=args.config['sample_method'],
                                  seed=args.config['seed'])
    print(f"[Data] Sizes -> train: {len(train_dataset.all_data)}, val: {len(val_dataset.all_data)}, test: {len(test_dataset.all_data)}")

    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    args.shot_num = args.config['shot_per_class']

    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])

    result_dict = {
        'meta': {
            'model': args.model_name,
            'dataset': args.dataset_name,
            'config_path': args.config_path if hasattr(args, 'config_path') else None,
        },
        'demon': {},
        'mse': {},
        'mlp_logs': {},
        'paths': {},
    }

    run_progress = tqdm(range(args.config['run_num']), desc="Overall Progress", position=0)
    for run_id in run_progress:
        run_name = f'run_{run_id}'
        args.run_name = run_name
        run_progress.set_description(f"Run {run_id + 1}/{args.config['run_num']}")
        print(f"\n[Run] ===== {run_name} =====")
        utils.set_seed(args.config['seed'] + run_id)

        # Demonstration
        print("[Demo] Generating few-shot demonstration...")
        demon, _, demon_indices = train_dataset.gen_few_shot_demonstration(
            tokenizer=tokenizer,
            shot_num=args.shot_num,
            max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
            add_extra_query=args.config['add_extra_query'],
            example_separator=args.config['example_separator'],
            return_data_index=True,
            seed=args.config['demo_seed'] + run_id,
            index_info=None
        )
        if args.config['add_extra_query']:
            first_anchor = train_dataset.get_dmonstration_template()['format'][0]
            baseline_demon = demon[:demon.rfind(first_anchor)] if first_anchor in demon else demon
        else:
            baseline_demon = demon
        result_dict['demon'][run_name] = demon

        # Baseline ICL hidden (test)
        print("[Eval] Collecting ICL hidden states on test...")
        icl_metrics, icl_hidden = test_evaluator.evaluate(
            base_wrapper, tokenizer, demonstration=baseline_demon,
            use_cache=args.config['use_cache'],
            return_label_hidden=True
        )
        icl_hidden = icl_hidden.to(dtype=model.dtype if hasattr(model, 'dtype') else torch.float32)

        # Zero-shot hidden (needed for MLP inference)
        print("[Eval] Collecting zero-shot hidden states on test...")
        zero_metrics, zero_hidden = test_evaluator.evaluate(
            base_wrapper, tokenizer, demonstration='',
            use_cache=args.config['use_cache'],
            return_label_hidden=True
        )
        zero_hidden = zero_hidden.to(dtype=model.dtype if hasattr(model, 'dtype') else torch.float32)

        # Train queries for task vectors / MLP
        exclude_demo = set(demon_indices) if demon_indices is not None else set()
        train_queries, _ = build_train_queries(
            train_dataset, args.config['num_train_queries_m2'], exclude_indices=exclude_demo
        )
        print(f"[Train Queries] Collected {len(train_queries)} anchors (excluded {len(exclude_demo)} demo indices)")

        # M2 constant vector (precompute once)
        print("[M2] Extracting constant task vector...")
        task_vector = m2_wrapper.extract_m2_task_vector(
            demo=baseline_demon,
            train_queries=train_queries,
            tokenizer=tokenizer,
            batch_size=args.config['extraction_batch_size'],
            verbose=True
        )

        # M2 Adaptive (closed-form) precompute once
        print("[M2 Adaptive] Extracting adaptive task matrix...")
        adaptive_matrix = m2_adaptive_wrapper.extract_adaptive_task_vector(
            demo=baseline_demon,
            train_queries=train_queries,
            tokenizer=tokenizer,
            batch_size=args.config['extraction_batch_size'],
            ridge_lambda=args.config['ridge_lambda'],
            verbose=True
        )

        # Evaluate M2 corrected hidden
        print("[Eval] Evaluating M2 on test...")
        with m2_wrapper.inject_m2_task_vector(task_vector):
            m2_metrics, m2_hidden = test_evaluator.evaluate(
                m2_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache'],
                return_label_hidden=True
            )
        m2_hidden = m2_hidden.to(dtype=model.dtype if hasattr(model, 'dtype') else torch.float32)

        # Evaluate M2 Adaptive corrected hidden
        print("[Eval] Evaluating M2 Adaptive on test...")
        with m2_adaptive_wrapper.inject_adaptive_task_vector(adaptive_matrix):
            m2a_metrics, m2a_hidden = test_evaluator.evaluate(
                m2_adaptive_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache'],
                return_label_hidden=True
            )
        m2a_hidden = m2a_hidden.to(dtype=model.dtype if hasattr(model, 'dtype') else torch.float32)

        mlp_cfg = args.config['mlp']
        if mlp_cfg.get('enabled', True):
            # Collect MLP training data
            mlp_query_num = min(mlp_cfg.get('num_train_queries', len(train_queries)), len(train_queries))
            mlp_queries = train_queries[:mlp_query_num]
            print(f"[MLP] Collecting train anchors ({mlp_query_num}) for MLP...")
            torch.cuda.empty_cache()
            mlp_features, mlp_targets = collect_zero_and_delta(
                model, tokenizer, baseline_demon, mlp_queries, args.device,
                args.config['extraction_batch_size'],
                to_dtype=model.dtype if hasattr(model, 'dtype') else torch.bfloat16
            )
            # Eval data for MLP: use test set zero/ICL difference
            print("[MLP] Preparing eval data from test hidden states...")
            mlp_eval_features = zero_hidden.detach().to(dtype=mlp_features.dtype, device='cpu')
            mlp_eval_targets = (icl_hidden - zero_hidden).detach().to(dtype=mlp_features.dtype, device='cpu')

            # Train M2 MLP
            print("[MLP] Training MLP...")
            mlp_model, mlp_logs = run_mlp_training(
                mlp_features, mlp_targets,
                mlp_eval_features, mlp_eval_targets,
                icl_hidden.size(-1), mlp_cfg, args.save_dir
            )
            result_dict['mlp_logs'][run_name] = mlp_logs

            # Apply MLP to zero-shot hidden for test
            mlp_model.eval()
            mlp_model.to(args.device, dtype=mlp_features.dtype)
            with torch.no_grad():
                mlp_delta = []
                bs = mlp_cfg['batch_size']
                for i in range(0, zero_hidden.size(0), bs):
                    batch = zero_hidden[i:i + bs].to(
                        args.device, dtype=mlp_model.fc1.weight.dtype
                    )
                    out = mlp_model(inputs=batch)
                    mlp_delta.append(out['logits'].cpu())
                mlp_delta = torch.cat(mlp_delta, dim=0).float()
                m2mlp_hidden = zero_hidden.float() + mlp_delta
        else:
            mlp_logs = []
            m2mlp_hidden = zero_hidden.clone()

        # Compute MSEs (ICL vs each corrected hidden)
        print("[MSE] Computing test MSEs...")
        mse_dict = {
            'icl_vs_m2': mse_between(icl_hidden, m2_hidden),
            'icl_vs_m2_adaptive': mse_between(icl_hidden, m2a_hidden),
            'icl_vs_m2_mlp': mse_between(icl_hidden, m2mlp_hidden),
        }
        result_dict['mse'][run_name] = mse_dict
        result_dict['acc'] = result_dict.get('acc', {})
        result_dict['acc'][run_name] = {
            'icl': icl_metrics.get('acc'),
            'zero': zero_metrics.get('acc'),
            'm2': m2_metrics.get('acc'),
            'm2_adaptive': m2a_metrics.get('acc'),
        }

        # Plot curves: Static (M2), Task Matrix (M2 Adaptive), MLP (curve with error bars)
        if mlp_logs:
            plot_cfg = args.config.get('plot', {})
            colors = plot_cfg.get('colors', {})
            font_family = plot_cfg.get('font_family', None)
            if font_family:
                plt.rcParams['font.family'] = font_family

            # Aggregate per epoch for mean/ std (multiple evals per epoch)
            from collections import defaultdict
            bucket = defaultdict(list)
            for log in mlp_logs:
                if log.get('epoch') is not None and log.get('mse') is not None:
                    bucket[log['epoch']].append(log['mse'])
            epochs = sorted(bucket.keys())
            means = [sum(bucket[e]) / len(bucket[e]) for e in epochs]
            stds = [
                (0.0 if len(bucket[e]) == 1 else float(torch.tensor(bucket[e]).std().item()))
                for e in epochs
            ]

            plt.figure(figsize=(8, 5))
            # Static line (M2)
            plt.axhline(
                y=mse_dict['icl_vs_m2'],
                color=colors.get('m2', '#1f77b4'),
                linestyle='--',
                label=f"Static (M2): {mse_dict['icl_vs_m2']:.4f}",
            )
            # Task Matrix line (M2 Adaptive)
            plt.axhline(
                y=mse_dict['icl_vs_m2_adaptive'],
                color=colors.get('m2_adaptive', '#ff7f0e'),
                linestyle='-.',
                label=f"Task Matrix (Adaptive): {mse_dict['icl_vs_m2_adaptive']:.4f}",
            )
            # MLP curve
            plt.errorbar(
                epochs,
                means,
                yerr=stds,
                fmt='-o',
                color=colors.get('mlp', '#2ca02c'),
                ecolor=colors.get('mlp', '#2ca02c'),
                capsize=3,
                label='MLP (eval on test Δ)',
            )

            plt.xlabel('Epoch', fontsize=13)
            plt.ylabel('Eval MSE', fontsize=13)
            plt.title(f"Test MSE vs. ICL ({args.dataset_name})", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            curve_path_png = os.path.join(args.save_dir, f"{run_name}_mlp_curve.png")
            curve_path_svg = os.path.join(args.save_dir, f"{run_name}_mlp_curve.svg")
            plt.tight_layout()
            plt.savefig(curve_path_png, dpi=150)
            plt.savefig(curve_path_svg, dpi=150)
            plt.close()
            result_dict['paths'][run_name] = {'mlp_curve_png': curve_path_png, 'mlp_curve_svg': curve_path_svg}

        # Persist results after each run
        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)

        # Dataset-level summary and aggregate across benchmarks
        dataset_summary = {
            'mse_summary': summarize_run_mse(result_dict['mse']),
            'acc_summary': summarize_run_acc(result_dict['acc']),
        }
        update_aggregate(args, dataset_summary)

    del base_wrapper, m2_wrapper, m2_adaptive_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_error_exp.py')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    combinations = list(itertools.product(config['models'], config['datasets']))
    task_queue = mp.Queue()
    for combo in combinations:
        task_queue.put(combo)

    def run_task(gpu_id, cfg):
        while not task_queue.empty():
            model_name, dataset_name = task_queue.get()
            print(f"Running error exp: {model_name} on {dataset_name} with GPU {gpu_id}")

            input_args = argparse.Namespace()
            cur_config = copy.deepcopy(cfg)
            input_args.model_name = model_name
            input_args.dataset_name = dataset_name
            input_args.gpu = gpu_id
            input_args.config = cur_config
            input_args.config_path = args.config_path if hasattr(args, 'config_path') else None

            try:
                main(input_args)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}")
                time.sleep(3)

    processes = [mp.Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All error exp tasks completed.")

"""
Comparison experiment: LTV (Adaptive Task Vector) vs MLP (2, 4, 8, 16 layers).

Measures accuracy (acc, macro_f1) and **fair** timing for each method.
- LTV inference time includes the full zero-shot forward pass (via hook injection).
- MLP inference time includes the zero-shot forward pass + MLP forward pass.

Output: JSON per (model, dataset) combination.
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
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from tqdm import tqdm

import our_datasets as md
import core.evaluator as ev
import core.metric as metric
import core.utils.utils as utils
import core.utils.utils_method as um
from core.wrapper_ltv import LTVWrapper


# ------------------------------------------------------------------
# MLP utilities (ported from ICLTV with fair timing)
# ------------------------------------------------------------------

class HiddenDeltaDataset(Dataset):
    """Pairs zero-shot label hidden with target delta for MLP training."""

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


class ResidualBlock(nn.Module):
    """Two-layer MLP block: Linear -> ReLU -> Linear."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class M2MLP(nn.Module):
    """Variable-depth square MLP with residual connections and LayerNorm every 2 layers."""

    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 2, "Must have at least 2 layers"
        assert num_layers % 2 == 0, "num_layers must be even for residual blocks"
        num_blocks = num_layers // 2
        self.blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(num_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_blocks)])

    def forward(self, inputs=None, labels=None):
        x = inputs
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x))
        delta = x
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
    """Collect zero-shot label hiddens and ICL-zero deltas for MLP training."""
    feature_list, target_list = [], []
    batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    for batch in tqdm(batches, desc="Collect MLP data", disable=len(batches) == 1):
        icl_inputs = [demo + q for q in batch]
        icl_tokens = tokenizer(icl_inputs, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.no_grad():
            icl_out = model(**icl_tokens, output_hidden_states=True, use_cache=False)
            icl_label = um.extract_label_position_hidden(icl_out.hidden_states[-1], icl_tokens["attention_mask"])

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


def run_mlp_training(
    train_features, train_targets, eval_features, eval_targets,
    dim, num_layers, cfg, save_dir,
):
    """Train M2 MLP with HuggingFace Trainer and return trained model."""
    train_dataset = HiddenDeltaDataset(train_features, train_targets)
    eval_dataset = HiddenDeltaDataset(eval_features, eval_targets)
    steps_per_epoch = math.ceil(len(train_dataset) / cfg['batch_size'])
    eval_steps = max(1, int(steps_per_epoch * cfg['eval_interval_epochs']))

    model = M2MLP(dim, num_layers=num_layers)

    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, f"mlp_{num_layers}layer_ckpt"),
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['epochs'],
        learning_rate=cfg['lr'],
        weight_decay=cfg.get('weight_decay', 0.0),
        warmup_ratio=cfg['warmup_ratio'],
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        report_to=[],
        save_strategy="no",
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = torch.tensor(preds, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        mse = F.mse_loss(preds, labels, reduction='mean').item()
        return {"mse": mse}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.model.float()


def run_zero_shot_hidden_extraction(
    model, tokenizer, test_dataset, device, batch_size=1,
):
    """Run zero-shot forward on test set, return hidden states and elapsed time."""
    all_hiddens = []
    all_inputs = []
    for data in test_dataset.all_data:
        ques_str, _, _ = test_dataset.apply_template(data)
        all_inputs.append(ques_str)

    t0 = time.time()
    batches = [all_inputs[i:i + batch_size] for i in range(0, len(all_inputs), batch_size)]
    for batch in batches:
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.no_grad():
            out = model(**tokens, output_hidden_states=True, use_cache=False)
            hidden = um.extract_label_position_hidden(out.hidden_states[-1], tokens["attention_mask"])
            all_hiddens.append(hidden.detach().cpu())
        del tokens, out, hidden
        torch.cuda.empty_cache()
    zero_infer_time = time.time() - t0

    zero_hidden = torch.cat(all_hiddens, dim=0)
    return zero_hidden, zero_infer_time


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

@torch.no_grad()
def main(args):
    cfg = args.config
    compute_L_mse = cfg.get('compute_L_mse', False)
    print(f"[Init] {args.model_name} on {args.dataset_name}")
    utils.set_seed(cfg['seed'])
    args.device = utils.set_device(args.gpu)
    utils.init_exp_path(args, cfg['exp_name'])

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True,
        load_in_8bit=cfg.get('load_in_8bit', False),
    )

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    ltv_wrapper = LTVWrapper(model, tokenizer, model_config, args.device)

    train_dataset = md.get_dataset(args.dataset_name, split='train', max_data_num=None, seed=cfg['seed'])
    test_dataset = md.get_dataset(
        args.dataset_name, split='test',
        max_data_num=cfg['test_data_num'],
        sample_mode=cfg['sample_method'],
        seed=cfg['seed'],
    )

    args.shot_num = cfg['num_shot']
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    test_evaluator = ev.Evaluator(test_dataset, batch_size=cfg['bs'])
    num_test = len(test_dataset.all_data)

    result_dict = {
        'meta': {
            'model': args.model_name,
            'dataset': args.dataset_name,
            'ridge_lambda': cfg['ridge_lambda'],
            'mlp_layers': cfg['mlp_layers'],
            'num_train_queries': cfg['num_train_queries'],
        },
    }

    for run_id in range(cfg['run_num']):
        run_name = f'run_{run_id}'
        print(f"\n[Run] ===== {run_name} =====")
        utils.set_seed(cfg['seed'] + run_id)
        run_result = {}

        # Demonstration
        demon, _, demon_indices = train_dataset.gen_few_shot_demonstration(
            tokenizer=tokenizer,
            shot_num=args.shot_num,
            max_demonstration_tok_len=args.test_max_token,
            add_extra_query=cfg['add_extra_query'],
            example_separator=cfg['example_separator'],
            return_data_index=True,
            seed=cfg['demo_seed'] + run_id,
            index_info=None,
        )
        if cfg['add_extra_query']:
            first_anchor = train_dataset.get_dmonstration_template()['format'][0]
            baseline_demon = demon[:demon.rfind(first_anchor)] if first_anchor in demon else demon
        else:
            baseline_demon = demon

        # ====== ICL baseline ======
        print("[Eval] ICL baseline...")
        icl_metrics = test_evaluator.evaluate(
            base_wrapper, tokenizer, demonstration=baseline_demon,
            use_cache=cfg['use_cache'],
        )
        if isinstance(icl_metrics, tuple):
            icl_metrics = icl_metrics[0]
        run_result['icl_baseline'] = {
            'acc': icl_metrics.get('acc'),
            'macro_f1': icl_metrics.get('macro_f1'),
        }

        # ====== Zero-shot baseline + hidden extraction (timed) ======
        print("[Eval] Zero-shot baseline + hidden extraction...")
        zero_metrics = test_evaluator.evaluate(
            base_wrapper, tokenizer, demonstration='',
            use_cache=cfg['use_cache'],
        )
        if isinstance(zero_metrics, tuple):
            zero_metrics = zero_metrics[0]

        # Separately measure zero-shot forward time for fair MLP comparison
        zero_hidden, zero_infer_time = run_zero_shot_hidden_extraction(
            model, tokenizer, test_dataset, args.device, batch_size=cfg['bs'],
        )
        zero_hidden = zero_hidden.to(dtype=model.dtype if hasattr(model, 'dtype') else torch.float32)

        run_result['zero_baseline'] = {
            'acc': zero_metrics.get('acc'),
            'macro_f1': zero_metrics.get('macro_f1'),
            'infer_time_total_sec': round(zero_infer_time, 3),
        }

        # ====== ICL hidden (for MLP training targets) ======
        print("[Eval] ICL hidden extraction for MLP targets...")
        # We need ICL hidden states for computing deltas
        icl_hidden_list = []
        all_inputs_icl = []
        for data in test_dataset.all_data:
            ques_str, _, _ = test_dataset.apply_template(data)
            all_inputs_icl.append(baseline_demon + ques_str)

        icl_batches = [all_inputs_icl[i:i + cfg['bs']] for i in range(0, len(all_inputs_icl), cfg['bs'])]
        for batch in icl_batches:
            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(args.device)
            with torch.no_grad():
                out = model(**tokens, output_hidden_states=True, use_cache=False)
                hidden = um.extract_label_position_hidden(out.hidden_states[-1], tokens["attention_mask"])
                icl_hidden_list.append(hidden.detach().cpu())
            del tokens, out, hidden
            torch.cuda.empty_cache()
        icl_hidden = torch.cat(icl_hidden_list, dim=0).to(dtype=zero_hidden.dtype)

        if compute_L_mse:
            # f = 0 reference: E_x ||h_icl - h_zs||^2 (no task vector applied).
            run_result['zero_baseline'].update(metric.compute_L_mse(icl_hidden, zero_hidden))

        # Train queries
        exclude_demo = set(demon_indices) if demon_indices is not None else set()
        train_queries, _ = utils.build_train_queries(
            train_dataset, cfg['num_train_queries'], exclude_indices=exclude_demo,
        )
        print(f"[Train Queries] {len(train_queries)} anchors")

        # ====== LTV Adaptive ======
        print("[LTV] Extracting adaptive task vector...")
        t0 = time.time()
        adaptive_vectors = ltv_wrapper.extract_adaptive_task_vector(
            demo=baseline_demon,
            train_queries=train_queries,
            tokenizer=tokenizer,
            batch_size=cfg['extraction_batch_size'],
            ridge_lambda=cfg['ridge_lambda'],
            verbose=True,
        )
        extract_time = time.time() - t0

        print("[LTV] Evaluating...")
        ltv_hidden = None
        t0 = time.time()
        with ltv_wrapper.inject_adaptive_task_vector(adaptive_vectors):
            ltv_out = test_evaluator.evaluate(
                ltv_wrapper, tokenizer, demonstration='',
                use_cache=cfg['use_cache'],
                return_hidden=compute_L_mse,
            )
        infer_time = time.time() - t0
        if compute_L_mse:
            ltv_metrics, ltv_hidden = ltv_out
        else:
            ltv_metrics = ltv_out
        if isinstance(ltv_metrics, tuple):
            ltv_metrics = ltv_metrics[0]

        run_result['ltv_adaptive'] = {
            'acc': ltv_metrics.get('acc'),
            'macro_f1': ltv_metrics.get('macro_f1'),
            'extract_time_sec': round(extract_time, 3),
            'infer_time_per_sample_sec': round(infer_time / num_test, 6),
        }
        if ltv_hidden is not None:
            # h_tv captured from the actual injected forward (same pass as the
            # LTV metrics above); h_icl from the extraction pass earlier.
            run_result['ltv_adaptive'].update(metric.compute_L_mse(icl_hidden, ltv_hidden))

        # ====== MLP variants ======
        mlp_cfg = cfg['mlp']
        if mlp_cfg.get('enabled', True):
            mlp_query_num = min(mlp_cfg.get('num_train_queries', len(train_queries)), len(train_queries))
            mlp_queries = train_queries[:mlp_query_num]
            print(f"[MLP] Collecting training data ({mlp_query_num} queries)...")
            torch.cuda.empty_cache()

            t0 = time.time()
            mlp_features, mlp_targets = collect_zero_and_delta(
                model, tokenizer, baseline_demon, mlp_queries, args.device,
                cfg['extraction_batch_size'],
                to_dtype=model.dtype if hasattr(model, 'dtype') else torch.bfloat16,
            )
            data_collect_time = time.time() - t0

            # Eval data for MLP training
            mlp_eval_features = zero_hidden.detach().to(dtype=mlp_features.dtype, device='cpu')
            mlp_eval_targets = (icl_hidden - zero_hidden).detach().to(dtype=mlp_features.dtype, device='cpu')

            for num_layers in cfg['mlp_layers']:
                key = f'mlp_{num_layers}layer'
                print(f"[MLP {num_layers}L] Training...")

                # Layer별 weight decay override
                mlp_cfg_run = dict(mlp_cfg)
                if 'weight_decay_per_layer' in mlp_cfg and num_layers in mlp_cfg['weight_decay_per_layer']:
                    mlp_cfg_run['weight_decay'] = mlp_cfg['weight_decay_per_layer'][num_layers]

                t0 = time.time()
                # main() runs under no_grad for inference-heavy sections,
                # but MLP fitting needs autograd enabled.
                with torch.enable_grad():
                    mlp_model = run_mlp_training(
                        mlp_features, mlp_targets,
                        mlp_eval_features, mlp_eval_targets,
                        icl_hidden.size(-1), num_layers, mlp_cfg_run, args.save_dir,
                    )
                train_time = time.time() - t0

                mlp_model.eval()
                mlp_model.to(args.device, dtype=mlp_features.dtype)

                # ---- Fair inference: zero-shot forward + MLP forward ----
                print(f"[MLP {num_layers}L] Inference (zero-shot + MLP)...")
                t0 = time.time()

                # Step 1: Zero-shot forward pass on test set (same cost as LTV)
                mlp_zero_hiddens = []
                all_test_inputs = []
                for data in test_dataset.all_data:
                    ques_str, _, _ = test_dataset.apply_template(data)
                    all_test_inputs.append(ques_str)

                test_batches = [all_test_inputs[i:i + cfg['bs']] for i in range(0, len(all_test_inputs), cfg['bs'])]
                for batch in test_batches:
                    tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(args.device)
                    with torch.no_grad():
                        out = model(**tokens, output_hidden_states=True, use_cache=False)
                        h = um.extract_label_position_hidden(out.hidden_states[-1], tokens["attention_mask"])
                        mlp_zero_hiddens.append(h.detach().cpu())
                    del tokens, out, h
                    torch.cuda.empty_cache()
                mlp_zero_hidden = torch.cat(mlp_zero_hiddens, dim=0)

                # Step 2: MLP forward pass
                with torch.no_grad():
                    mlp_delta = []
                    bs = mlp_cfg['batch_size']
                    for i in range(0, mlp_zero_hidden.size(0), bs):
                        batch_h = mlp_zero_hidden[i:i + bs].to(args.device, dtype=mlp_model.blocks[0].fc1.weight.dtype)
                        out = mlp_model(inputs=batch_h)
                        mlp_delta.append(out['logits'].cpu())
                    mlp_delta = torch.cat(mlp_delta, dim=0).float()
                    m2mlp_hidden = mlp_zero_hidden.float() + mlp_delta

                mlp_infer_time = time.time() - t0  # includes zero-shot + MLP

                # Evaluate via LM head
                lm_head = model.lm_head if hasattr(model, 'lm_head') else None
                if lm_head is not None:
                    with torch.no_grad():
                        corrected_logits = []
                        for i in range(0, m2mlp_hidden.size(0), bs):
                            batch_hidden = m2mlp_hidden[i:i + bs].to(args.device, dtype=model.dtype)
                            batch_logits = lm_head(batch_hidden)
                            corrected_logits.append(batch_logits.detach().cpu())
                        corrected_logits = torch.cat(corrected_logits, dim=0)

                    all_labels = []
                    for data in test_dataset.all_data:
                        _, _, label = test_dataset.apply_template(data)
                        all_labels.append(label)

                    mlp_metrics = test_evaluator.evaluate_logits(
                        [corrected_logits], all_labels, tokenizer,
                        model.config._name_or_path,
                    )
                    if isinstance(mlp_metrics, tuple):
                        mlp_metrics = mlp_metrics[0]
                else:
                    mlp_metrics = {'acc': None, 'macro_f1': None}

                run_result[key] = {
                    'acc': mlp_metrics.get('acc'),
                    'macro_f1': mlp_metrics.get('macro_f1'),
                    'weight_decay': mlp_cfg_run.get('weight_decay', 0.0),
                    'data_collect_time_sec': round(data_collect_time, 3),
                    'train_time_sec': round(train_time, 3),
                    'infer_time_per_sample_sec': round(mlp_infer_time / num_test, 6),
                }
                if compute_L_mse:
                    # h_tv for the MLP variant is h_zs + MLP(h_zs), fed to the LM head.
                    run_result[key].update(metric.compute_L_mse(icl_hidden, m2mlp_hidden))

                del mlp_model
                torch.cuda.empty_cache()

        result_dict[run_name] = run_result

        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)

    del base_wrapper, ltv_wrapper, model, tokenizer
    del train_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Done] Results saved to {args.save_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config_comparison_exp.py')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    gpu_id = config['gpus'][0]
    combinations = list(itertools.product(config['models'], config['datasets']))

    for model_name, dataset_name in combinations:
        print(f"Running comparison: {model_name} on {dataset_name} (GPU {gpu_id})")

        input_args = argparse.Namespace()
        input_args.model_name = model_name
        input_args.dataset_name = dataset_name
        input_args.gpu = gpu_id
        input_args.config = copy.deepcopy(config)

        try:
            main(input_args)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"CUDA memory cleared for GPU {gpu_id}")
            time.sleep(3)

    print("All comparison experiments completed.")

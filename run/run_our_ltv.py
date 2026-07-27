"""
Unified baseline runner that evaluates:
1. Zero-shot ICL
2. Few-shot ICL
3. LTV (adaptive task vector)

Sampling (train/test splits + demonstrations) follows the I2CL/task-vector
setup so every method uses exactly the same data.
"""

import argparse
import copy
import gc
import itertools
import json
import os
import time
import random
import re
import torch
from tqdm import tqdm
import our_datasets as md
import core.evaluator as ev
import core.metric as metric
import core.utils.utils as utils
import core.utils.utils_method as um
from core.wrapper_ltv import LTVWrapper
from core.wrapper_learned_tv import LearnedTVWrapper
from core.wrapper_learnable_tv import LearnableTVWrapper
from core.wrapper_loreft import LoReFTWrapper

task_queue = None

def print_header(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(title)
    print(f"{line}\n")


def print_result(label: str, result) -> None:
    print(f"{label}: {result}\n")


def fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def print_eta(prefix: str, done: int, total: int, start_time: float) -> None:
    """Elapsed / per-unit / ETA line after each completed unit of work."""
    elapsed = time.time() - start_time
    per = elapsed / max(done, 1)
    remain = per * (total - done)
    finish = time.strftime("%H:%M", time.localtime(time.time() + remain))
    print(f"[ETA] {prefix}: {done}/{total} done | elapsed {fmt_secs(elapsed)} "
          f"| {fmt_secs(per)}/unit | remaining {fmt_secs(remain)} (~{finish})")


def init_result_dict() -> dict:
    result_dict = {
        'test_result': {
            'zero_shot': [], 'few_shot': [], 'ltv': [], 'learned_tv': [], 'learnable_tv': [],
            'loreft': []
        },
        'time': {'ltv': [], 'learned_tv': [], 'learnable_tv': [], 'loreft': []}
    }
    return result_dict

@torch.no_grad()
def main(args):
    cfg = args.config
    run_num = cfg['run_num']
    seed = cfg['seed']
    num_shot = cfg['num_shot']
    use_cache = cfg['use_cache']
    return_logits = cfg['return_logits']
    compute_L_mse = cfg.get('compute_L_mse', False)
    if compute_L_mse and not return_logits:
        # The 4-way unpacks below (and d_NTP) require logits alongside hiddens.
        print("[Warn] compute_L_mse=True requires logits; forcing return_logits=True")
        return_logits = True
    load_in_8bit = cfg['load_in_8bit']
    save_logits = cfg.get('save_logits', False)
    extraction_batch_size = cfg['extraction_batch_size']
    ridge_lambdas = cfg['ridge_lambda']
    num_train_queries = cfg['num_train_queries']

    utils.set_seed(seed)
    args.device = utils.set_device(args.gpu)
    args.metric = cfg['metric']
    utils.init_exp_path(args, cfg['exp_name'])
    print_header(f"Baseline Suite: {args.model_name} on {args.dataset_name}")

    train_dataset = md.get_dataset(
        args.dataset_name, split='train', max_data_num=None,
        seed=seed
        )
    test_dataset = md.get_dataset(
        args.dataset_name, split='test',
        max_data_num=cfg['test_data_num'],
        sample_mode=cfg['sample_method'],
        seed=seed
        )

    args.shot_num = num_shot
    
    model, tokenizer, model_config = utils.load_model_tokenizer(
            args.model_name, args.device, output_hidden_states=True, load_in_8bit=load_in_8bit
        )

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    test_evaluator = ev.Evaluator(test_dataset, batch_size=cfg['bs'])
     

    result_dict = init_result_dict()

    cv_save_dict = {}

    # Zero-shot label hiddens/logits (demo-independent): extracted once at
    # run 0, reused across runs as the f=0 reference for L_mse / d_NTP.
    test_zero_hidden = None
    test_zero_logits = test_zero_labels = None

    suite_start = time.time()
    for run_id in tqdm(range(run_num), desc="Overall Progress", position=0):
        args.run_name = f'run_{run_id} : {time.time()}'
        print_header(f"Run {run_id + 1}/{run_num}: {args.run_name}")

        utils.set_seed(seed + run_id)
        # shared train queries cache per num_queries

        demon, split_demon, demon_indices = train_dataset.gen_few_shot_demonstration(
            tokenizer=tokenizer,
            shot_num=args.shot_num,
            max_demonstration_tok_len=args.test_max_token,
            add_extra_query=cfg['add_extra_query'],
            example_separator=cfg['example_separator'],
            return_data_index=True,
            seed=cfg['demo_seed'] + run_id,
            index_info=None
        )

        if cfg['add_extra_query']:
            first_anchor = train_dataset.get_dmonstration_template()['format'][0]
            baseline_demon = demon[:demon.rfind(first_anchor)] if first_anchor in demon else demon
        else:
            baseline_demon = demon

        # Zero-shot baseline
        if run_id == 0 and cfg['run_baseline']:
            print("Evaluating zero-shot baseline...")
            zero_out = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=use_cache,
                return_logits=return_logits,
                return_hidden=compute_L_mse,
                desc="Eval zero-shot"
            )
            if return_logits and compute_L_mse:
                test_zero, test_zero_logits, test_zero_labels, test_zero_hidden = zero_out
            elif return_logits:
                test_zero, test_zero_logits, test_zero_labels = zero_out
            elif compute_L_mse:
                test_zero, test_zero_hidden = zero_out
            else:
                test_zero = zero_out
            result_dict['test_result']['zero_shot'].append(test_zero)
            print_result("Test zero-shot", test_zero)

        # Few-shot baseline
        test_few_logits = test_few_labels = test_few_hidden = None
        if cfg['run_baseline']:
            print("Evaluating few-shot baseline...")
            few_out = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=use_cache,
                return_logits=return_logits,
                return_hidden=compute_L_mse,
                desc="Eval few-shot ICL"
            )
            if compute_L_mse:
                test_few, test_few_logits, test_few_labels, test_few_hidden = few_out
            else:
                test_few, test_few_logits, test_few_labels = few_out
            result_dict['test_result']['few_shot'].append(test_few)
            print_result("Test few-shot", test_few)
            if save_logits and test_few_logits is not None:
                safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
                icl_dir = os.path.join(args.save_dir, "logits_icl")
                os.makedirs(icl_dir, exist_ok=True)
                icl_path = os.path.join(icl_dir, f"icl_{safe_run}.pt")
                print("DEBUG : ", test_few_logits.shape)
                torch.save(
                    {
                        "logits": test_few_logits,
                        "run_name": args.run_name,
                        "model_name": args.model_name,
                        "dataset_name": args.dataset_name,
                    },
                    icl_path,
                )

        if cfg['run_ltv']:
            ltv_test_dict = {}
            ltv_wrapper = LTVWrapper(model, tokenizer, model_config, args.device)
            
            for num_queries in num_train_queries:
                q_key = f"{num_queries}_queries"
                exclude_demo = set(demon_indices) if demon_indices is not None else set()
                train_queries, _ = utils.build_train_queries(
                    train_dataset, num_queries, exclude_indices=exclude_demo
                )
            
                for r_lambda in ridge_lambdas:
                    print(f"Using {len(train_queries)} training queries for task vector learning (ridge {r_lambda})")

                    lam_key = f"ridge_lambda_{r_lambda}"

                    ridge_lambda = r_lambda
                    print(f"LTV: extracting (λ={ridge_lambda})...")
                    adaptive_vectors = ltv_wrapper.extract_adaptive_task_vector(
                        demo=baseline_demon,
                        train_queries=train_queries,
                        tokenizer=tokenizer,
                        batch_size=extraction_batch_size,
                        ridge_lambda=ridge_lambda,
                        verbose=True
                    )
                    if cfg.get('save_task_vectors', False):
                        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
                        layer_idx = ltv_wrapper.num_layers - 1
                        save_path = os.path.join(
                            args.save_dir,
                            f"task_vectors_run{safe_run}_{q_key}_{lam_key}.pt",
                        )
                        um.save_task_vectors({layer_idx: adaptive_vectors}, save_path)

                    print("LTV: evaluating...")
                    test_ltv_hidden = None
                    ltv_start = time.time()
                    with ltv_wrapper.inject_adaptive_task_vector(adaptive_vectors):
                        ltv_out = test_evaluator.evaluate(
                            ltv_wrapper, tokenizer, demonstration='',
                            use_cache=use_cache,
                            return_logits=return_logits,
                            return_hidden=compute_L_mse,
                            desc=f"Eval LTV (λ={ridge_lambda})"
                        )
                    ltv_end = time.time()
                    if compute_L_mse:
                        test_ltv, test_ltv_logits, test_ltv_labels, test_ltv_hidden = ltv_out
                    else:
                        test_ltv, test_ltv_logits, test_ltv_labels = ltv_out
                    ltv_metrics = dict(test_ltv) if isinstance(test_ltv, dict) else {"result": test_ltv}
                    result_dict['time']['ltv'].append(ltv_end - ltv_start)
                    print_result("Test LTV", test_ltv)

                    if cfg.get('compute_d_NTP', False) and test_few_labels==test_ltv_labels:
                        mean_d_NTP_ltv = metric.compute_d_NTP(
                            test_few_logits, test_ltv_logits, is_qwen='Qwen' in args.model_name
                        )
                        ltv_metrics["d_NTP"] = mean_d_NTP_ltv
                        # Label-logit-space MSE on the same (N, K) tensors as d_NTP.
                        ltv_metrics.update(metric.compute_L_mse_logit(test_few_logits, test_ltv_logits))

                        # f = 0 references on the same tensors: KL / logit-MSE between
                        # this run's ICL distribution and the (demo-independent)
                        # zero-shot distribution. Together with d_NTP / L_mse_logit
                        # above, each cell carries the full zero -> LTV pair used by
                        # run/compare_zero_ltv.py.
                        if test_zero_logits is not None and test_zero_labels == test_few_labels:
                            ltv_metrics["d_NTP_zero_ref"] = metric.compute_d_NTP(
                                test_few_logits, test_zero_logits, is_qwen='Qwen' in args.model_name
                            )
                            zero_logit_ref = metric.compute_L_mse_logit(test_few_logits, test_zero_logits)
                            ltv_metrics["L_mse_logit_zero_ref"] = zero_logit_ref["L_mse_logit"]

                    # L_MSE (paper eq. 11): E_x ||h_icl - h_tv||^2 at the final-layer
                    # label position. h_tv is captured from the actual injected forward
                    # (the same pass that produced the LTV logits above).
                    if compute_L_mse and test_few_hidden is None:
                        print("[Warn] L_mse skipped: no ICL reference hidden (run_baseline=False?)")
                    if compute_L_mse and test_few_hidden is not None and test_ltv_hidden is not None \
                            and test_few_labels == test_ltv_labels:
                        ltv_metrics.update(metric.compute_L_mse(test_few_hidden, test_ltv_hidden))
                        if test_zero_hidden is not None:
                            # f = 0 reference: E_x ||h_icl - h_zs||^2 (no task vector).
                            zero_ref = metric.compute_L_mse(test_few_hidden, test_zero_hidden)
                            ltv_metrics["L_mse_zero_ref"] = zero_ref["L_mse"]
                        print_result("Test LTV L_mse", {k: v for k, v in ltv_metrics.items()
                                                        if k.startswith("L_mse")})

                    utils.nested_set(ltv_test_dict, [q_key, lam_key], ltv_metrics)
                    if save_logits and test_ltv_logits is not None:
                        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
                        tv_dir = os.path.join(args.save_dir, "logits_tv")
                        os.makedirs(tv_dir, exist_ok=True)
                        tv_path = os.path.join(tv_dir, f"tv_{safe_run}_{q_key}_{lam_key}.pt")
                        torch.save(
                            {
                                "logits": test_ltv_logits,
                                "run_name": args.run_name,
                                "model_name": args.model_name,
                                "dataset_name": args.dataset_name,
                                "q_key": q_key,
                                "lam_key": lam_key,
                            },
                            tv_path,
                        )

            result_dict['test_result']['ltv'].append(ltv_test_dict)
            del ltv_wrapper

        # ====== Learned-TV baseline (Yang et al., ICLR 2026) ======
        if cfg.get('run_learned_tv', False):
            lt_cfg = cfg.get('learned_tv', {})
            lt_wrapper = LearnedTVWrapper(model, tokenizer, model_config, args.device)

            exclude_demo = set(demon_indices) if demon_indices is not None else set()
            lt_queries, lt_indices = utils.build_train_queries(
                train_dataset, lt_cfg.get('num_train_queries', 256),
                exclude_indices=exclude_demo,
            )
            lt_labels = [train_dataset.apply_template(train_dataset.all_data[i])[2]
                         for i in lt_indices]
            options = train_dataset.get_dmonstration_template()['options']
            losses = lt_cfg.get('losses', ['lmse'])

            icl_targets = None
            if 'lmse' in losses:
                icl_targets = lt_wrapper.collect_icl_hidden(
                    baseline_demon, lt_queries, tokenizer,
                    batch_size=extraction_batch_size,
                )

            lt_entry = {}
            for loss_name in losses:
                print(f"Learned-TV ({loss_name}): training "
                      f"(layer={lt_cfg.get('layer', 'mid')}, lr={lt_cfg.get('lr', 1e-3)})...")
                lt_start = time.time()
                theta, train_info = lt_wrapper.train_learned_tv(
                    queries=lt_queries, labels=lt_labels, tokenizer=tokenizer,
                    model_name=args.model_name, options=options,
                    loss=loss_name, icl_targets=icl_targets,
                    layer=lt_cfg.get('layer', 'mid'),
                    lr=lt_cfg.get('lr', 1e-3),
                    weight_decay=lt_cfg.get('weight_decay', 0.01),
                    epochs=lt_cfg.get('epochs', 10),
                    samples_per_epoch=lt_cfg.get('samples_per_epoch', 100),
                    patience=lt_cfg.get('patience', 2),
                    val_ratio=lt_cfg.get('val_ratio', 0.2),
                    init_scale=lt_cfg.get('init_scale', 0.1),
                    seed=seed + run_id,
                )
                train_time = time.time() - lt_start

                print(f"Learned-TV ({loss_name}): evaluating...")
                test_lt_hidden = None
                with lt_wrapper.inject_learned_tv(theta, train_info['layer_idx']):
                    lt_out = test_evaluator.evaluate(
                        lt_wrapper, tokenizer, demonstration='',
                        use_cache=use_cache,
                        return_logits=return_logits,
                        return_hidden=compute_L_mse,
                        desc=f"Eval Learned-TV ({loss_name})"
                    )
                if compute_L_mse:
                    test_lt, test_lt_logits, test_lt_labels, test_lt_hidden = lt_out
                else:
                    test_lt, test_lt_logits, test_lt_labels = lt_out
                lt_metrics = dict(test_lt) if isinstance(test_lt, dict) else {"result": test_lt}
                lt_metrics.update({'loss': loss_name, 'train_time_sec': round(train_time, 3),
                                   **{f'train_{k}': v for k, v in train_info.items()}})
                result_dict['time']['learned_tv'].append(train_time)
                print_result(f"Test Learned-TV ({loss_name})", test_lt)

                # Same alignment metrics as the LTV block, vs the same ICL reference.
                if cfg.get('compute_d_NTP', False) and test_few_labels == test_lt_labels:
                    lt_metrics["d_NTP"] = metric.compute_d_NTP(
                        test_few_logits, test_lt_logits, is_qwen='Qwen' in args.model_name
                    )
                    lt_metrics.update(metric.compute_L_mse_logit(test_few_logits, test_lt_logits))
                    if test_zero_logits is not None and test_zero_labels == test_few_labels:
                        lt_metrics["d_NTP_zero_ref"] = metric.compute_d_NTP(
                            test_few_logits, test_zero_logits, is_qwen='Qwen' in args.model_name
                        )
                        lt_metrics["L_mse_logit_zero_ref"] = metric.compute_L_mse_logit(
                            test_few_logits, test_zero_logits)["L_mse_logit"]
                if compute_L_mse and test_few_hidden is not None and test_lt_hidden is not None \
                        and test_few_labels == test_lt_labels:
                    lt_metrics.update(metric.compute_L_mse(test_few_hidden, test_lt_hidden))
                    if test_zero_hidden is not None:
                        lt_metrics["L_mse_zero_ref"] = metric.compute_L_mse(
                            test_few_hidden, test_zero_hidden)["L_mse"]
                    print_result(f"Test Learned-TV ({loss_name}) L_mse",
                                 {k: v for k, v in lt_metrics.items() if k.startswith("L_mse")})

                lt_entry[loss_name] = lt_metrics

            result_dict['test_result']['learned_tv'].append(lt_entry)
            del lt_wrapper
            torch.cuda.empty_cache()

        # ====== Learnable-TV baseline (Saglam et al., ACL Findings 2025) ======
        if cfg.get('run_learnable_tv', False):
            la_cfg = cfg.get('learnable_tv', {})
            la_wrapper = LearnableTVWrapper(model, tokenizer, model_config, args.device)

            exclude_demo = set(demon_indices) if demon_indices is not None else set()
            la_queries, la_indices = utils.build_train_queries(
                train_dataset, la_cfg.get('num_train_queries', 256),
                exclude_indices=exclude_demo,
            )
            la_labels = [train_dataset.apply_template(train_dataset.all_data[i])[2]
                         for i in la_indices]
            options = train_dataset.get_dmonstration_template()['options']
            # Pool of structured demonstration examples for the label-shuffled CE prompts.
            la_pool = [train_dataset.apply_template(train_dataset.all_data[i])
                       for i in (demon_indices if demon_indices is not None else la_indices)]
            losses = la_cfg.get('losses', ['ce'])

            print("Learnable-TV: building per-layer head basis...")
            la_basis = la_wrapper.collect_head_basis(
                baseline_demon, la_queries, tokenizer, batch_size=extraction_batch_size,
            )
            la_icl_targets = None
            if 'lmse' in losses:
                la_icl_targets = la_wrapper.collect_icl_hidden(
                    baseline_demon, la_queries, tokenizer, batch_size=extraction_batch_size,
                )

            la_entry = {}
            for loss_name in losses:
                print(f"Learnable-TV ({loss_name}): training (lr={la_cfg.get('lr', 5e-5)})...")
                la_start = time.time()
                phi, train_info = la_wrapper.train_learnable_tv(
                    queries=la_queries, labels=la_labels, pool=la_pool, tokenizer=tokenizer,
                    model_name=args.model_name, options=options, basis=la_basis,
                    loss=loss_name, icl_targets=la_icl_targets,
                    k_shot=la_cfg.get('k_shot', 10),
                    lr=la_cfg.get('lr', 5e-5),
                    weight_decay=la_cfg.get('weight_decay', 0.0),
                    epochs=la_cfg.get('epochs', 10),
                    samples_per_epoch=la_cfg.get('samples_per_epoch', 100),
                    val_ratio=la_cfg.get('val_ratio', 0.2),
                    init=la_cfg.get('init', 'zero'),
                    example_separator=cfg['example_separator'],
                    seed=seed + run_id,
                )
                train_time = time.time() - la_start

                print(f"Learnable-TV ({loss_name}): evaluating...")
                test_la_hidden = None
                with la_wrapper.inject_learnable_tv(phi, la_basis):
                    la_out = test_evaluator.evaluate(
                        la_wrapper, tokenizer, demonstration='',
                        use_cache=use_cache,
                        return_logits=return_logits,
                        return_hidden=compute_L_mse,
                        desc=f"Eval Learnable-TV ({loss_name})"
                    )
                if compute_L_mse:
                    test_la, test_la_logits, test_la_labels, test_la_hidden = la_out
                else:
                    test_la, test_la_logits, test_la_labels = la_out
                la_metrics = dict(test_la) if isinstance(test_la, dict) else {"result": test_la}
                la_metrics.update({'loss': loss_name, 'train_time_sec': round(train_time, 3),
                                   **{f'train_{k}': v for k, v in train_info.items()}})
                result_dict['time']['learnable_tv'].append(train_time)
                print_result(f"Test Learnable-TV ({loss_name})", test_la)

                # Same alignment metrics as the LTV block, vs the same ICL reference.
                if cfg.get('compute_d_NTP', False) and test_few_labels == test_la_labels:
                    la_metrics["d_NTP"] = metric.compute_d_NTP(
                        test_few_logits, test_la_logits, is_qwen='Qwen' in args.model_name
                    )
                    la_metrics.update(metric.compute_L_mse_logit(test_few_logits, test_la_logits))
                    if test_zero_logits is not None and test_zero_labels == test_few_labels:
                        la_metrics["d_NTP_zero_ref"] = metric.compute_d_NTP(
                            test_few_logits, test_zero_logits, is_qwen='Qwen' in args.model_name
                        )
                        la_metrics["L_mse_logit_zero_ref"] = metric.compute_L_mse_logit(
                            test_few_logits, test_zero_logits)["L_mse_logit"]
                if compute_L_mse and test_few_hidden is not None and test_la_hidden is not None \
                        and test_few_labels == test_la_labels:
                    la_metrics.update(metric.compute_L_mse(test_few_hidden, test_la_hidden))
                    if test_zero_hidden is not None:
                        la_metrics["L_mse_zero_ref"] = metric.compute_L_mse(
                            test_few_hidden, test_zero_hidden)["L_mse"]
                    print_result(f"Test Learnable-TV ({loss_name}) L_mse",
                                 {k: v for k, v in la_metrics.items() if k.startswith("L_mse")})

                la_entry[loss_name] = la_metrics

            result_dict['test_result']['learnable_tv'].append(la_entry)
            del la_wrapper
            torch.cuda.empty_cache()

        # ====== LoReFT-style baseline (Wu et al., NeurIPS 2024) ======
        if cfg.get('run_loreft', False):
            lo_cfg = cfg.get('loreft', {})
            lo_wrapper = LoReFTWrapper(model, tokenizer, model_config, args.device)

            exclude_demo = set(demon_indices) if demon_indices is not None else set()
            lo_queries, lo_indices = utils.build_train_queries(
                train_dataset, lo_cfg.get('num_train_queries', 256),
                exclude_indices=exclude_demo,
            )
            lo_labels = [train_dataset.apply_template(train_dataset.all_data[i])[2]
                         for i in lo_indices]
            options = train_dataset.get_dmonstration_template()['options']
            losses = lo_cfg.get('losses', ['ce', 'lmse'])

            lo_icl_targets = None
            if 'lmse' in losses:
                lo_icl_targets = lo_wrapper.collect_icl_hidden(
                    baseline_demon, lo_queries, tokenizer,
                    batch_size=extraction_batch_size,
                )

            lo_entry = {}
            for loss_name in losses:
                print(f"LoReFT ({loss_name}): training "
                      f"(layers={lo_cfg.get('layers', 'mid')}, rank={lo_cfg.get('rank', 4)}, "
                      f"lr={lo_cfg.get('lr', 9e-4)})...")
                lo_start = time.time()
                lo_interv, train_info = lo_wrapper.train_loreft(
                    queries=lo_queries, labels=lo_labels, tokenizer=tokenizer,
                    model_name=args.model_name, options=options,
                    loss=loss_name, icl_targets=lo_icl_targets,
                    layers=lo_cfg.get('layers', 'mid'),
                    rank=lo_cfg.get('rank', 4),
                    lr=lo_cfg.get('lr', 9e-4),
                    weight_decay=lo_cfg.get('weight_decay', 0.0),
                    dropout=lo_cfg.get('dropout', 0.0),
                    warmup_ratio=lo_cfg.get('warmup_ratio', 0.1),
                    epochs=lo_cfg.get('epochs', 8),
                    samples_per_epoch=lo_cfg.get('samples_per_epoch', 100),
                    patience=lo_cfg.get('patience', 2),
                    val_ratio=lo_cfg.get('val_ratio', 0.2),
                    seed=seed + run_id,
                )
                train_time = time.time() - lo_start

                print(f"LoReFT ({loss_name}): evaluating...")
                test_lo_hidden = None
                with lo_wrapper.inject_loreft(lo_interv, train_info['layer_idxs']):
                    lo_out = test_evaluator.evaluate(
                        lo_wrapper, tokenizer, demonstration='',
                        use_cache=use_cache,
                        return_logits=return_logits,
                        return_hidden=compute_L_mse,
                        desc=f"Eval LoReFT ({loss_name})"
                    )
                if compute_L_mse:
                    test_lo, test_lo_logits, test_lo_labels, test_lo_hidden = lo_out
                else:
                    test_lo, test_lo_logits, test_lo_labels = lo_out
                lo_metrics = dict(test_lo) if isinstance(test_lo, dict) else {"result": test_lo}
                lo_metrics.update({'loss': loss_name, 'train_time_sec': round(train_time, 3),
                                   **{f'train_{k}': v for k, v in train_info.items()}})
                result_dict['time']['loreft'].append(train_time)
                print_result(f"Test LoReFT ({loss_name})", test_lo)

                # Same alignment metrics as the LTV block, vs the same ICL reference.
                if cfg.get('compute_d_NTP', False) and test_few_labels == test_lo_labels:
                    lo_metrics["d_NTP"] = metric.compute_d_NTP(
                        test_few_logits, test_lo_logits, is_qwen='Qwen' in args.model_name
                    )
                    lo_metrics.update(metric.compute_L_mse_logit(test_few_logits, test_lo_logits))
                    if test_zero_logits is not None and test_zero_labels == test_few_labels:
                        lo_metrics["d_NTP_zero_ref"] = metric.compute_d_NTP(
                            test_few_logits, test_zero_logits, is_qwen='Qwen' in args.model_name
                        )
                        lo_metrics["L_mse_logit_zero_ref"] = metric.compute_L_mse_logit(
                            test_few_logits, test_zero_logits)["L_mse_logit"]
                if compute_L_mse and test_few_hidden is not None and test_lo_hidden is not None \
                        and test_few_labels == test_lo_labels:
                    lo_metrics.update(metric.compute_L_mse(test_few_hidden, test_lo_hidden))
                    if test_zero_hidden is not None:
                        lo_metrics["L_mse_zero_ref"] = metric.compute_L_mse(
                            test_few_hidden, test_zero_hidden)["L_mse"]
                    print_result(f"Test LoReFT ({loss_name}) L_mse",
                                 {k: v for k, v in lo_metrics.items() if k.startswith("L_mse")})

                lo_entry[loss_name] = lo_metrics

            result_dict['test_result']['loreft'].append(lo_entry)
            del lo_wrapper
            torch.cuda.empty_cache()

        print_eta(f"{args.dataset_name} runs", run_id + 1, run_num, suite_start)

    with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
    del base_wrapper, model, tokenizer
    del train_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Baseline suite completed!")
    print(f"Results saved to: {args.save_dir}, time : {time.time()}")
    print(f"{'=' * 60}\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_baseline_all.py')
    
    # Basic / Runtime
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--return_logits", type=bool, default=True)
    parser.add_argument("--demo_seed", type=int, default=12)
    parser.add_argument("--run_baseline", type=bool, default=True)
    parser.add_argument("--metric", type=str, default="acc")  # acc | macro_f1
    parser.add_argument("--load_in_8bit", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=False)

    parser.add_argument("--num_train_queries", nargs="+", type=int, default=[256])
    parser.add_argument("--ridge_lambda", nargs="+", type=float, default=[1.0])
    parser.add_argument("--extraction_batch_size", type=int, default=1)
    parser.add_argument("--inference_batch_size", type=int, default=1)

    parser.add_argument("--target_layers", default=None)
    parser.add_argument("--module", nargs="+", default=["hidden"])
    parser.add_argument("--tok_pos", type=str, default="last")

    # Data settings
    parser.add_argument("--val_data_num", type=int, default=32)
    parser.add_argument("--test_data_num", type=int, default=500)
    parser.add_argument("--sample_method", type=str, default="uniform")
    parser.add_argument("--use_instruction", type=bool, default=False)
    parser.add_argument("--add_extra_query", type=bool, default=True)
    parser.add_argument("--example_separator", type=str, default="\n")

    # Evaluation
    parser.add_argument("--compute_d_NTP", type=bool, default=True)
    parser.add_argument("--save_task_vectors", type=bool, default=False)
    parser.add_argument("--evaluate_reconstruction", type=bool, default=False)    
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    gpu_id = config['gpus'][0]

    combinations = list(itertools.product(config['models'], config['datasets']))
    combos_start = time.time()
    for combo_i, (model_name, dataset_name) in enumerate(combinations):
        print(f"Running Baselines: {model_name} on {dataset_name} with GPU {gpu_id}")

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
            print_eta("model x dataset combinations", combo_i + 1, len(combinations), combos_start)
            time.sleep(3)

    print("All baseline tasks completed.")

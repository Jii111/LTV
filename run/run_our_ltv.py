"""
Unified baseline runner that evaluates:
1. Zero-shot ICL
2. Few-shot ICL
3. LTV (adaptive task vector)

Sampling (train/val/test splits + demonstrations) follows the I2CL/task-vector
setup so every method uses exactly the same data slices.
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

task_queue = None

def print_header(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(title)
    print(f"{line}\n")


def print_result(label: str, result) -> None:
    print(f"{label}: {result}\n")


def init_result_dict() -> dict:
    result_dict = {
        'demon': {},
        'test_result': {
            'zero_shot': [], 'few_shot': [], 'ltv': []
        },
        'time': {'ltv': []}
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

        result_dict['demon'][args.run_name] = demon

        # Zero-shot baseline
        if run_id == 0 and cfg['run_baseline']:
            print("Evaluating zero-shot baseline...")
            test_zero = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=use_cache
            )
            result_dict['test_result']['zero_shot'].append(test_zero)
            print_result("Test zero-shot", test_zero)

        # Few-shot baseline
        test_few_logits = test_few_labels = None
        if cfg['run_baseline']:
            print("Evaluating few-shot baseline...")
            test_few, test_few_logits, test_few_labels = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=use_cache,
                return_logits=return_logits
            )
            result_dict['test_result']['few_shot'].append(test_few)
            print_result("Test few-shot", test_few)
            if save_logits and test_few_logits is not None:
                safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
                icl_dir = os.path.join(args.save_dir, "logits_icl")
                os.makedirs(icl_dir, exist_ok=True)
                icl_path = os.path.join(icl_dir, f"icl_{safe_run}.pt")
                torch.save(
                    {
                        "logits": test_few_logits,
                        "labels": test_few_labels,
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
                    ltv_start = time.time()
                    with ltv_wrapper.inject_adaptive_task_vector(adaptive_vectors):
                        test_ltv, test_ltv_logits, test_ltv_labels = test_evaluator.evaluate(
                            ltv_wrapper, tokenizer, demonstration='',
                            use_cache=use_cache,
                            return_logits=return_logits
                        )
                    ltv_end = time.time()
                    ltv_metrics = dict(test_ltv) if isinstance(test_ltv, dict) else {"result": test_ltv}
                    result_dict['time']['ltv'].append(ltv_end - ltv_start)
                    print_result("Test LTV", test_ltv)

                    if cfg.get('compute_d_NTP', False) and test_few_logits is not None:
                        mean_d_NTP_ltv = metric.compute_d_NTP(
                            test_few_logits, test_ltv_logits, is_qwen='Qwen' in args.model_name
                        )
                        ltv_metrics["d_NTP"] = mean_d_NTP_ltv
                        ltv_metrics["labels"] = list(map(int, test_ltv_labels))

                    utils.nested_set(ltv_test_dict, [q_key, lam_key], ltv_metrics)
                    if save_logits and test_ltv_logits is not None:
                        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.run_name)
                        tv_dir = os.path.join(args.save_dir, "logits_tv")
                        os.makedirs(tv_dir, exist_ok=True)
                        tv_path = os.path.join(tv_dir, f"tv_{safe_run}_{q_key}_{lam_key}.pt")
                        torch.save(
                            {
                                "logits": test_ltv_logits,
                                "labels": test_ltv_labels,
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
    for model_name, dataset_name in combinations:
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
            time.sleep(3)

    print("All baseline tasks completed.")

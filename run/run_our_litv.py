"""
Unified baseline runner that evaluates:
1. Zero-shot ICL
2. Few-shot ICL
3. M1 (single-layer task vector)
4. M2 (mask-based multi-layer task vector)
5. Adaptive M2 (query-adaptive linear map)

Sampling (train/val/test splits + demonstrations) follows the I2CL/task-vector
setup so every method uses exactly the same data slices.
"""

import argparse
import copy
import gc
from sv_utils.TVeval import ICLVectorEvaluator
from sv_utils.TVframework import SVEvaluator
from sv_utils.utils import set_rand_seed
from fv_utils.compute_indirect_effect import compute_indirect_effect

import argparse
import copy
import json
import os
import sys
import time
import random
import re
import torch
from tqdm import tqdm
import evaluator as ev
import my_datasets as md
import utils
import utils_method as um
from fv_utils.extract_utils import *
from wrapper_m2 import M2AdaptiveWrapper, M2Wrapper

task_queue = None

def target_layer_selection(args, model_wrapper, tokenizer, evaluator, tv_type, context_vector, model_config=None):
    num_layers = model_wrapper.num_layers
    best_layer = 0
    best_metric = float('-inf')
    with torch.no_grad():
        for layer in range(num_layers):
            if tv_type == 'icl_tv':
                with model_wrapper.replace_latent(context_vector, [layer], args.config):
                    val_result = evaluator.evaluate(
                        model_wrapper, tokenizer, demonstration='',
                        use_cache=args.config['use_cache']
                    )
            elif tv_type == 'function_vector':
                val_result = evaluator.evaluate(
                    model_wrapper, tokenizer, demonstration='',
                    use_cache=args.config['use_cache'],
                    fv_vector=context_vector, edit_layer=layer, model_config=model_config
                )
            metric_value = val_result[args.metric]
            print(f"Layer {layer} validation {args.metric}: {metric_value:.4f}")
            if metric_value > best_metric:
                best_metric = metric_value
                best_layer = layer
    print(f"Selected layer {best_layer} (metric={best_metric:.4f})")
    return best_layer

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


def get_acc(entry):
    if isinstance(entry, tuple):
        data = entry[0]
        if isinstance(data, dict):
            return data.get("acc", -1)

    if isinstance(entry, dict):
        return entry.get("acc", -1)

    return -1
@torch.no_grad()
def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    print(f"\n{'=' * 60}")
    print(f"Baseline Suite: {args.model_name} on {args.dataset_name}")
    print(f"{'=' * 60}\n")

    train_dataset = md.get_dataset(
        args.dataset_name, split='train', max_data_num=None,
        seed=args.config['seed']
        )
    val_dataset = md.get_dataset(
        args.dataset_name, split='validation',
        max_data_num=args.config['val_data_num'],
        sample_mode=args.config['sample_method'],
        seed=args.config['seed']
        )
    test_dataset = md.get_dataset(
        args.dataset_name, split='test',
        max_data_num=args.config['test_data_num'],
        sample_mode=args.config['sample_method'],
        seed=args.config['seed']
        )

    args.shot_num = args.config['shot_per_class']
    
    model, tokenizer, model_config = utils.load_model_tokenizer(
            args.model_name, args.device, output_hidden_states=True
        )

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    
    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)

    # TODO: evaluator 정리
    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
     

    result_dict = {
        'demon': {},
        'test_result': {
            'zero_shot': [], 'few_shot': [], 'm2': [], 'm2_adaptive': []
        },
        'time': {'m2': [], 'm2_adaptive': []}
    }
    kl_dict = {'m2': {}, 'm2_adaptive': {}}

    cv_save_dict = {}

    for run_id in tqdm(range(args.config['run_num']), desc="Overall Progress", position=0):
        args.run_name = f'run_{run_id} : {time.time()}'
        print(f"\n{'=' * 60}")
        print(f"Run {run_id + 1}/{args.config['run_num']}: {args.run_name}")
        print(f"{'=' * 60}\n")

        utils.set_seed(args.config['seed'] + run_id)
        # shared train queries cache per num_queries

        demon, split_demon, demon_indices = train_dataset.gen_few_shot_demonstration(
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

        result_dict['demon'][args.run_name] = demon

        # Zero-shot baseline
        if run_id == 0 and args.config['run_baseline']:
            print("Evaluating zero-shot baseline...")
            test_zero = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            result_dict['test_result']['zero_shot'].append(test_zero)
            print(f"Test zero-shot: {test_zero}\n")

        # Few-shot baseline
        test_few_logits = test_few_labels = None
        if args.config['run_baseline']:
            print("Evaluating few-shot baseline...")
            test_few, test_few_logits, test_few_labels = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache'],
                return_logits=args.config['return_logits']
            )
            result_dict['test_result']['few_shot'].append(test_few)
            print(f"Test few-shot: {test_few}\n")

        if args.config['run_m2']:
            m2_test_dict = {}
            m2a_test_dict ={}
            m2_wrapper = M2Wrapper(model, tokenizer, model_config, args.device)
            m2_adaptive_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, args.device)
            
            for num_queries in args.config['num_train_queries']:
                q_key = f"{num_queries}_queries"
                exclude_demo = set(demon_indices) if demon_indices is not None else set()
                train_queries, _ = build_train_queries(
                    train_dataset, num_queries, exclude_indices=exclude_demo
                )
                print(f"Using {len(train_queries)} training queries for task vector learning (shared across ridge λ)")

                # M2 constant vectors
                print(f"M2: extracting constant task vector... on {num_queries}_queries")
                task_vector = m2_wrapper.extract_m2_task_vector(
                    demo=baseline_demon,
                    train_queries=train_queries,
                    tokenizer=tokenizer,
                    batch_size=args.config['extraction_batch_size'],
                    verbose=True
                )

                if args.config.get('save_task_vectors', False):
                    tv_path = os.path.join(args.save_dir, f'{args.run_name}_m2_task_vector.pt')
                    um.save_task_vectors({m2_wrapper.num_layers - 1: task_vector}, tv_path)
                    
                print(f"M2: evaluating constant vector... on {num_queries}_queries")
                m2_start = time.time()
                with m2_wrapper.inject_m2_task_vector(task_vector):
                    test_m2, test_m2_logits, test_m2_labels = test_evaluator.evaluate(
                        m2_wrapper, tokenizer, demonstration='',
                        use_cache=args.config['use_cache'],
                        return_logits=args.config['return_logits']
                    )
                m2_end = time.time()
                m2_test_dict[q_key] = test_m2
                print(f"Test M2({q_key}): {test_m2}\n")
                result_dict['time']['m2'].append(m2_end - m2_start)

                if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                    mean_kl = utils.compute_kl_divergence(
                        test_few_logits, test_m2_logits, is_qwen='Qwen' in args.model_name
                    )
                    print(f"KL divergence (Few-shot vs M2): {mean_kl:.4f}")
                    print(">>> test_m2_logits")
                    print("  ",test_m2_logits)
                    print(">>> mean_kl")
                    print("  ",mean_kl)
                    utils.nested_set(kl_dict,
                            ['m2', args.run_name, q_key],
                            {"mean_kl": mean_kl,
                            "labels": list(map(int, test_m2_labels))})

                for r_lambda in args.config['ridge_lambda']:
                    lam_key = f"ridge_lambda_{r_lambda}"

                    # Adaptive M2
                    ridge_lambda = r_lambda
                    print(f"M2-Adaptive: extracting (λ={ridge_lambda})...")
                    adaptive_vectors = m2_adaptive_wrapper.extract_adaptive_task_vector(
                        demo=baseline_demon,
                        train_queries=train_queries,
                        tokenizer=tokenizer,
                        batch_size=args.config['extraction_batch_size'],
                        ridge_lambda=ridge_lambda,
                        verbose=True
                    )

                    print("M2-Adaptive: evaluating...")
                    m2a_start = time.time()
                    with m2_adaptive_wrapper.inject_adaptive_task_vector(adaptive_vectors):
                        test_m2a, test_m2a_logits, test_m2a_labels = test_evaluator.evaluate(
                            m2_adaptive_wrapper, tokenizer, demonstration='',
                            use_cache=args.config['use_cache'],
                            return_logits=args.config['return_logits']
                        )
                    m2a_end = time.time()
                    if args.config.get('save_m2a_figures', False):
                        h_list = [d["h"] for d in m2_adaptive_wrapper.injected_deltas if "h" in d]
                        delta_list = [d["delta"] for d in m2_adaptive_wrapper.injected_deltas if "delta" in d]
                        if h_list and delta_list and len(h_list) == len(delta_list):
                            h_states = torch.stack(h_list)
                            deltas = torch.stack(delta_list)
                            raw_prefix = f"{args.run_name}_{q_key}_{lam_key}"
                            safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_prefix)
                            utils.save_m2a_figures(
                                h_states,
                                deltas,
                                args.save_dir,
                                safe_prefix,
                                add_permuted=args.config.get('plot_permuted_control', False),
                            )
                    utils.nested_set(m2a_test_dict, [q_key, lam_key], test_m2a)
                    result_dict['time']['m2_adaptive'].append(m2a_end - m2a_start)
                    print(f"Test M2-Adaptive: {test_m2a}\n")

                    if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                        mean_kl_a = utils.compute_kl_divergence(
                            test_few_logits, test_m2a_logits, is_qwen='Qwen' in args.model_name
                        )

                        utils.nested_set(kl_dict,
                            ['m2_adaptive', args.run_name, q_key, lam_key],
                            {"mean_kl": mean_kl_a,
                            "labels": list(map(int, test_m2a_labels))})

            result_dict['test_result']['m2'].append(m2_test_dict)
            result_dict['test_result']['m2_adaptive'].append(m2a_test_dict)
            del m2_wrapper, m2_adaptive_wrapper
            
    with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
    if kl_dict['m2'] or kl_dict['m2_adaptive']:
        with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
            json.dump(kl_dict, f, indent=4)


    del base_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Baseline suite completed!")
    print(f"Results saved to: {args.save_dir}, time : {time.time()}")
    print(f"{'=' * 60}\n")

# TODO
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
    parser.add_argument("--compute_kl_divergence", type=bool, default=True)
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

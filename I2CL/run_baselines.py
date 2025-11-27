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
import itertools
import json
import os
import random
import time
from multiprocessing import Process, Queue

import torch
from tqdm import tqdm

import evaluator as ev
import my_datasets as md
import utils
import utils_method as um
from wrapper_m2 import M2AdaptiveWrapper, M2Wrapper


def target_layer_selection(args, model_wrapper, tokenizer, evaluator, context_vector_dict):
    num_layers = model_wrapper.num_layers
    best_layer = 0
    best_metric = float('-inf')
    with torch.no_grad():
        for layer in range(num_layers):
            with model_wrapper.replace_latent(context_vector_dict, [layer], args.config):
                val_result = evaluator.evaluate(
                    model_wrapper, tokenizer,
                    demonstration='',
                    use_cache=args.config['use_cache']
                )
                metric_value = val_result[args.metric]
                print(f"[M1] Layer {layer} validation {args.metric}: {metric_value:.4f}")
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_layer = layer
    print(f"[M1] Selected layer {best_layer} (metric={best_metric:.4f})")
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


def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    print(f"\n{'=' * 60}")
    print(f"Baseline Suite: {args.model_name} on {args.dataset_name}")
    print(f"{'=' * 60}\n")

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True
    )

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    m2_wrapper = M2Wrapper(model, tokenizer, model_config, args.device)
    m2_adaptive_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, args.device)

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

    train_evaluator = ev.Evaluator(train_dataset, batch_size=args.config['bs'])
    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    args.shot_num = args.config['shot_per_class']

    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])

    result_dict = {
        'demon': {},
        'best_replace_layer': {},
        'test_result': {'zero_shot': [], 'few_shot': [], 'm1': [], 'm2': [], 'm2_adaptive': []},
        'val_result': {'zero_shot': [], 'few_shot': [], 'm1': [], 'm2': [], 'm2_adaptive': []},
        'time': {'m1': [], 'm2': [], 'm2_adaptive': []}
    }
    kl_dict = {'m1': {}, 'm2': {}, 'm2_adaptive': {}}

    run_progress = tqdm(range(args.config['run_num']), desc="Overall Progress", position=0)
    for run_id in run_progress:
        run_name = f'run_{run_id}'
        args.run_name = run_name
        run_progress.set_description(f"Run {run_id + 1}/{args.config['run_num']}")
        print(f"\n{'=' * 60}")
        print(f"Run {run_id + 1}/{args.config['run_num']}: {run_name}")
        print(f"{'=' * 60}\n")

        utils.set_seed(args.config['seed'] + run_id)
        # shared train queries cache per num_queries
        m2_val_dict = {}; m2_test_dict = {}; m2a_val_dict = {}; m2a_test_dict = {}

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

        # Zero-shot baseline
        if run_id == 0 and args.config['run_baseline']:
            print("Evaluating zero-shot baseline...")
            val_zero = val_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            test_zero = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            result_dict['val_result']['zero_shot'].append(val_zero)
            result_dict['test_result']['zero_shot'].append(test_zero)
            print(f"Val zero-shot: {val_zero}")
            print(f"Test zero-shot: {test_zero}\n")

        # Few-shot baseline
        test_few_logits = test_few_labels = None
        if args.config['run_baseline']:
            print("Evaluating few-shot baseline...")
            val_few = val_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache']
            )
            test_few, test_few_logits, test_few_labels = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache'],
                return_logits=True,
                logits_mode=args.config['logits_mode']
            )
            result_dict['val_result']['few_shot'].append(val_few)
            result_dict['test_result']['few_shot'].append(test_few)
            print(f"Val few-shot: {val_few}")
            print(f"Test few-shot: {test_few}\n")

        # M1: standard task vector
        print("Evaluating M1 task vector...")
        all_latent_dicts = []
        with torch.no_grad():
            with base_wrapper.extract_latent():
                demon_token = tokenizer(demon, return_tensors='pt').to(args.device)
                _ = model(**demon_token)
            all_latent_dicts.append(base_wrapper.latent_dict)
            base_wrapper.reset_latent_dict()

        context_vector_dict = base_wrapper.get_context_vector(all_latent_dicts, args.config)
        best_layer = target_layer_selection(
            args, base_wrapper, tokenizer, val_evaluator, context_vector_dict
        )
        result_dict['best_replace_layer'][run_name] = best_layer

        m1_start = time.time()
        with base_wrapper.replace_latent(context_vector_dict, [best_layer], args.config):
            val_m1 = val_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            test_m1, test_m1_logits, test_m1_labels = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache'],
                return_logits=True,
                logits_mode=args.config['logits_mode']
            )
        m1_end = time.time()
        result_dict['val_result']['m1'].append(val_m1)
        result_dict['test_result']['m1'].append(test_m1)
        result_dict['time']['m1'].append(m1_end - m1_start)
        print(f"Val M1: {val_m1}")
        print(f"Test M1: {test_m1}\n")

        if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
            mean_kl_m1, kl_values_m1 = utils.compute_kl_divergence(
                test_few_logits, test_m1_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs M1): {mean_kl_m1:.4f}")
            kl_dict['m1'][run_name] = {
                "mean_kl": mean_kl_m1,
                "kl_values": kl_values_m1.tolist(),
                "labels": list(map(int, test_m1_labels))
            }

        for num_queries in args.config['num_train_queries']:
            q_key = f"{num_queries}_queries"
            exclude_demo = set(demon_indices) if demon_indices is not None else set()
            train_queries, _ = build_train_queries(
                train_dataset, num_queries, exclude_indices=exclude_demo
            )
            print(f"Using {len(train_queries)} training queries for task vector learning (shared across ridge λ)")

            for r_lambda in args.config['ridge_lambda']:
                lam_key = f"ridge_lambda_{r_lambda}"

                # M2 constant vectors
                print("M2: extracting constant task vector...")
                task_vector = m2_wrapper.extract_m2_task_vector(
                    demo=baseline_demon,
                    train_queries=train_queries,
                    tokenizer=tokenizer,
                    batch_size=args.config['extraction_batch_size'],
                    verbose=True
                )

                if args.config.get('save_task_vectors', False):
                    tv_path = os.path.join(args.save_dir, f'{run_name}_m2_task_vector.pt')
                    um.save_task_vectors({m2_wrapper.num_layers - 1: task_vector}, tv_path)

                print("M2: evaluating constant vector...")
                m2_start = time.time()
                with m2_wrapper.inject_m2_task_vector(task_vector):
                    val_m2 = val_evaluator.evaluate(
                        m2_wrapper, tokenizer, demonstration='',
                        use_cache=args.config['use_cache']
                    )
                    test_m2, test_m2_logits, test_m2_labels = test_evaluator.evaluate(
                        m2_wrapper, tokenizer, demonstration='',
                        use_cache=args.config['use_cache'],
                        return_logits=True,
                        logits_mode=args.config['logits_mode']
                    )
                m2_end = time.time()
                utils.nested_set(m2_val_dict, [q_key, lam_key], val_m2)
                utils.nested_set(m2_test_dict, [q_key, lam_key], test_m2)
                result_dict['time']['m2'].append(m2_end - m2_start)
                print(f"Val M2({q_key}, {lam_key}): {val_m2}")
                print(f"Test M2({q_key}, {lam_key}): {test_m2}\n")

                if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                    mean_kl, kl_values = utils.compute_kl_divergence(
                        test_few_logits, test_m2_logits, is_qwen='Qwen' in args.model_name
                    )
                    print(f"KL divergence (Few-shot vs M2): {mean_kl:.4f}")
                    kl_dict['m2'][run_name] = {
                        "mean_kl": mean_kl,
                        "kl_values": kl_values.tolist(),
                        "labels": list(map(int, test_m2_labels))
                    }
                    utils.plot_kl_hist(
                        kl_values, mean_kl,
                        os.path.join(args.save_dir, f"{run_name}_m2_kl.png")
                    )

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
                    val_m2a = val_evaluator.evaluate(
                        m2_adaptive_wrapper, tokenizer, demonstration='',
                        use_cache=args.config['use_cache']
                    )
                    test_m2a, test_m2a_logits, test_m2a_labels = test_evaluator.evaluate(
                        m2_adaptive_wrapper, tokenizer, demonstration='',
                        use_cache=args.config['use_cache'],
                        return_logits=True,
                        logits_mode=args.config['logits_mode']
                    )
                m2a_end = time.time()
                utils.nested_set(m2a_val_dict, [q_key, lam_key], val_m2a)
                utils.nested_set(m2a_test_dict, [q_key, lam_key], test_m2a)
                result_dict['time']['m2_adaptive'].append(m2a_end - m2a_start)
                print(f"Val M2-Adaptive: {val_m2a}")
                print(f"Test M2-Adaptive: {test_m2a}\n")

                if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                    mean_kl_a, kl_values_a = utils.compute_kl_divergence(
                        test_few_logits, test_m2a_logits, is_qwen='Qwen' in args.model_name
                    )
                    print(f"KL divergence (Few-shot vs M2-Adaptive): {mean_kl_a:.4f}")
                    kl_dict['m2_adaptive'][run_name] = {
                        "mean_kl": mean_kl_a,
                        "kl_values": kl_values_a.tolist(),
                        "labels": list(map(int, test_m2a_labels))
                    }
                    utils.plot_kl_hist(
                        kl_values_a, mean_kl_a,
                        os.path.join(args.save_dir, f"{run_name}_m2_adaptive_kl.png")
                    )

        result_dict['val_result']['m2'].append(m2_val_dict)
        result_dict['test_result']['m2'].append(m2_test_dict)
        result_dict['val_result']['m2_adaptive'].append(m2a_val_dict)
        result_dict['test_result']['m2_adaptive'].append(m2a_test_dict)

        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)
        if kl_dict['m1'] or kl_dict['m2'] or kl_dict['m2_adaptive']:
            with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                json.dump(kl_dict, f, indent=4)

    del base_wrapper, m2_wrapper, m2_adaptive_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Baseline suite completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'=' * 60}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_m2.py')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    combinations = list(itertools.product(config['models'], config['datasets']))
    task_queue = Queue()
    for combo in combinations:
        task_queue.put(combo)

    def run_task(gpu_id, cfg):
        while not task_queue.empty():
            model_name, dataset_name = task_queue.get()
            print(f"Running Baselines: {model_name} on {dataset_name} with GPU {gpu_id}")

            input_args = argparse.Namespace()
            cur_config = copy.deepcopy(cfg)
            input_args.model_name = model_name
            input_args.dataset_name = dataset_name
            input_args.gpu = gpu_id
            input_args.config = cur_config

            try:
                main(input_args)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}")
                time.sleep(3)

    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All baseline tasks completed.")

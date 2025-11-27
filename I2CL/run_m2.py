"""
M2 Runner: constant final-layer task vector (Δ = h_ICL - h_Zero at label).
"""

import argparse
import copy
import gc
import itertools
import json
import os
import random
from multiprocessing import Process, Queue

import torch
from tqdm import tqdm

import evaluator as ev
import my_datasets as md
import utils
import utils_method as um
from wrapper_m2 import M2Wrapper


def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    print(f"\n{'=' * 60}")
    print(f"M2 (constant Δ) : {args.model_name} on {args.dataset_name}")
    print(f"{'=' * 60}\n")

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True
    )
    model_wrapper = M2Wrapper(model, tokenizer, model_config, args.device)

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

    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)
    args.shot_num = args.config['shot_per_class']

    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])

    result_dict = {
        'demon': {},
        'test_result': {'zero_shot': [], 'few_shot': [], 'm2': []},
        'val_result': {'zero_shot': [], 'few_shot': [], 'm2': []},
        'time': {}
    }
    kl_dict = {}

    run_progress = tqdm(range(args.config['run_num']), desc="Overall Progress", position=0)
    for run_id in run_progress:
        run_name = f'run_{run_id}'
        args.run_name = run_name
        run_progress.set_description(f"Run {run_id + 1}/{args.config['run_num']}")
        print(f"\n{'=' * 60}")
        print(f"Run {run_id + 1}/{args.config['run_num']}: {run_name}")
        print(f"{'=' * 60}\n")

        run_seed = args.config['seed'] + run_id
        utils.set_seed(run_seed)

        # Zero-shot baseline (first run only)
        if run_id == 0 and args.config['run_baseline']:
            print("Evaluating zero-shot baseline...")
            val_zero = val_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            test_zero = test_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            result_dict['val_result']['zero_shot'].append(val_zero)
            result_dict['test_result']['zero_shot'].append(test_zero)
            print(f"Val zero-shot: {val_zero}")
            print(f"Test zero-shot: {test_zero}\n")

        # Sample demonstration
        idx_path = args.save_dir.replace("m2", "i2cl") + "/demon_idx.json"
        if os.path.exists(idx_path):
            with open(idx_path, "r") as f:
                index_dict = json.load(f)
            demon_data_index = index_dict.get(run_name, None)
        else:
            demon_data_index = None

        demon, _, _ = train_dataset.gen_few_shot_demonstration(
            tokenizer=tokenizer,
            shot_num=args.shot_num,
            max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
            add_extra_query=args.config['add_extra_query'],
            example_separator=args.config['example_separator'],
            return_data_index=True,
            seed=args.config['demo_seed'] + run_id,
            index_info=demon_data_index
        )

        baseline_demon = demon
        if args.config['add_extra_query']:
            first_format_anchor = train_dataset.get_dmonstration_template()['format'][0]
            if first_format_anchor in demon:
                baseline_demon = demon[:demon.rfind(first_format_anchor)]

        print(f"Demonstration (len tokens={len(tokenizer(demon)['input_ids'])}): {demon[:200]}...\n")
        result_dict['demon'][run_name] = demon

        # Few-shot baseline
        if args.config['run_baseline']:
            print("Evaluating few-shot ICL baseline...")
            val_few = val_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache']
            )
            test_few, test_few_logits, test_few_labels = test_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache'],
                return_logits=True,
                logits_mode=args.config['logits_mode']
            )
            result_dict['val_result']['few_shot'].append(val_few)
            result_dict['test_result']['few_shot'].append(test_few)
            print(f"Val few-shot: {val_few}")
            print(f"Test few-shot: {test_few}\n")
        else:
            test_few_logits = test_few_labels = None

        # Train anchors
        num_train_queries = args.config.get('num_train_queries', 25)
        train_indices = random.sample(range(len(train_dataset.all_data)),
                                      min(num_train_queries, len(train_dataset.all_data)))
        train_queries = []
        for idx in train_indices:
            ques_str, _, _ = train_dataset.apply_template(train_dataset.all_data[idx])
            train_queries.append(ques_str)
        print(f"Using {len(train_queries)} training queries for task vector learning")

        # Extract and save task vector
        task_vector = model_wrapper.extract_m2_task_vector(
            demo=baseline_demon,
            train_queries=train_queries,
            tokenizer=tokenizer,
            batch_size=args.config['extraction_batch_size'],
            verbose=True
        )
        if args.config['save_task_vectors']:
            tv_path = args.save_dir + f'/{run_name}_task_vector.pt'
            um.save_task_vectors({model_wrapper.num_layers - 1: task_vector}, tv_path)  # keep format similar

        # Evaluate with injection
        print("M2: Evaluating with task vector injection...")
        with model_wrapper.inject_m2_task_vector(task_vector):
            val_m2 = val_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            test_m2, test_m2_logits, test_m2_labels = test_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache'],
                return_logits=True,
                logits_mode=args.config['logits_mode']
            )

        result_dict['val_result']['m2'].append(val_m2)
        result_dict['test_result']['m2'].append(test_m2)
        print(f"Val M2: {val_m2}")
        print(f"Test M2: {test_m2}\n")

        if args.config['compute_kl_divergence'] and test_few_logits is not None:
            mean_kl, kl_values = utils.compute_kl_divergence(
                test_few_logits, test_m2_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (ICL vs M2): {mean_kl:.4f}\n")
            kl_dict[run_name] = {
                "mean_kl": mean_kl,
                "kl_values": kl_values.tolist(),
                "labels": list(map(int, test_m2_labels))
            }
            with open(args.save_dir + '/kl_divergence.json', 'w') as f:
                json.dump(kl_dict, f, indent=4)

        # Save results after each run
        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

    del model_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"M2 Experiment Completed!")
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
    for combine in combinations:
        task_queue.put(combine)

    def run_task(gpu_id, config):
        while not task_queue.empty():
            model_name, dataset_name = task_queue.get()
            print(f"Running M2: {model_name} on {dataset_name} with GPU {gpu_id}")

            input_args = argparse.Namespace()
            cur_config = copy.deepcopy(config)
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

    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All M2 tasks completed.")

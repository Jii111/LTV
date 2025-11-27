"""
M2-Adaptive Runner: closed-form ridge on final-layer hidden (Δ = h_ICL - h_Zero).
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
from wrapper_m2 import M2AdaptiveWrapper


def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    print(f"\n{'=' * 60}")
    print(f"M2-Adaptive (ridge) : {args.model_name} on {args.dataset_name}")
    print(f"Ridge λ: {args.config['ridge_lambda']}")
    print(f"{'=' * 60}\n")

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True
    )
    model_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, args.device)

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
        'test_result': {'zero_shot': [], 'few_shot': [], 'm2_adaptive': []},
        'val_result': {'zero_shot': [], 'few_shot': [], 'm2_adaptive': []},
    }

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

        idx_path = args.save_dir.replace("m2_adaptive", "i2cl") + "/demon_idx.json"
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
            first_anchor = train_dataset.get_dmonstration_template()['format'][0]
            if first_anchor in demon:
                baseline_demon = demon[:demon.rfind(first_anchor)]

        result_dict['demon'][run_name] = demon

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

        if args.config['run_baseline']:
            print("Evaluating few-shot ICL baseline...")
            val_few = val_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache']
            )
            test_few = test_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration=baseline_demon,
                use_cache=args.config['use_cache']
            )
            result_dict['val_result']['few_shot'].append(val_few)
            result_dict['test_result']['few_shot'].append(test_few)

        print("M2-Adaptive: collecting anchors...")
        num_train_queries = args.config.get('num_train_queries', 25)
        train_indices = random.sample(range(len(train_dataset.all_data)),
                                      min(num_train_queries, len(train_dataset.all_data)))
        train_queries = []
        for idx in train_indices:
            ques_str, _, _ = train_dataset.apply_template(train_dataset.all_data[idx])
            train_queries.append(ques_str)
        print(f"Using {len(train_queries)} training queries for task vector learning")

        adaptive_vectors = model_wrapper.extract_adaptive_task_vector(
            demo=baseline_demon,
            train_queries=train_queries,
            tokenizer=tokenizer,
            batch_size=args.config['extraction_batch_size'],
            ridge_lambda=args.config['ridge_lambda'],
            verbose=True
        )

        print("M2-Adaptive: evaluating...")
        with model_wrapper.inject_adaptive_task_vector(adaptive_vectors):
            val_m2a = val_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
            test_m2a = test_evaluator.evaluate(
                model_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache']
            )
        result_dict['val_result']['m2_adaptive'].append(val_m2a)
        result_dict['test_result']['m2_adaptive'].append(test_m2a)

        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

    del model_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_m2_adaptive.py')
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
            print(f"Running M2-Adaptive: {model_name} on {dataset_name} with GPU {gpu_id}")

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

    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    print("All M2-Adaptive tasks completed.")

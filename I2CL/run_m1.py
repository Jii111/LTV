"""
M1 (Hendel-style) runner aligned with other experiment scripts.
"""

import argparse
import copy
import gc
import itertools
import json
import os
import time
from multiprocessing import Process, Queue

import torch

import evaluator as ev
import my_datasets as md
import utils


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
                print(f"Layer {layer} validation metric: {metric_value:.4f}")
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_layer = layer
    print(f"[M1] Selected layer {best_layer} (metric={best_metric:.4f})")
    return best_layer


def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    model, tokenizer, model_config = utils.load_custom_model_tokenizer(
        args.model_name, args.device, method='M2'
    )
    model_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )

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
        'split_demon': {},
        'best_replace_layer': {},
        'test_result': {'zero_shot': [], 'few_shot': [], 'm1': []},
        'val_result': {'zero_shot': [], 'few_shot': [], 'm1': []},
        'time': {'evaluate': []},
    }

    idx_path = args.save_dir.replace("taskvector", "i2cl") + "/demon_idx.json"
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            cached_idx = json.load(f)
    else:
        cached_idx = None

    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        utils.set_seed(args.config['seed'] + run_id)
        print(f"\n========== M1 Run {run_id + 1}/{args.config['run_num']} ==========\n")

        demon_data_index = None
        if cached_idx is not None:
            demon_data_index = cached_idx.get(run_name, None)

        demon, split_demon, sampled_idx = train_dataset.gen_few_shot_demonstration(
            tokenizer=tokenizer,
            shot_num=args.shot_num,
            max_demonstration_tok_len=min(args.val_max_token, args.test_max_token),
            add_extra_query=args.config['add_extra_query'],
            example_separator=args.config['example_separator'],
            return_data_index=True,
            seed=args.config['demo_seed'] + run_id,
            index_info=demon_data_index
        )

        if args.config['add_extra_query']:
            first_format_anchor = train_dataset.get_dmonstration_template()['format'][0]
            baseline_demon = demon[:demon.rfind(first_format_anchor)] if first_format_anchor in demon else demon
        else:
            baseline_demon = demon

        result_dict['demon'][run_name] = [demon]
        result_dict['split_demon'][run_name] = split_demon

        if run_id == 0 and args.config['run_baseline']:
            val_zero = val_evaluator.evaluate(
                model_wrapper, tokenizer,
                demonstration='',
                use_cache=args.config['use_cache']
            )
            test_zero = test_evaluator.evaluate(
                model_wrapper, tokenizer,
                demonstration='',
                use_cache=args.config['use_cache']
            )
            result_dict['val_result']['zero_shot'].append(val_zero)
            result_dict['test_result']['zero_shot'].append(test_zero)

        if args.config['run_baseline']:
            val_few = val_evaluator.evaluate(
                model_wrapper, tokenizer,
                demonstration=baseline_demon,
                use_cache=args.config['use_cache']
            )
            test_few = test_evaluator.evaluate(
                model_wrapper, tokenizer,
                demonstration=baseline_demon,
                use_cache=args.config['use_cache']
            )
            result_dict['val_result']['few_shot'].append(val_few)
            result_dict['test_result']['few_shot'].append(test_few)

        # Extract latent representations
        all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                demon_token = tokenizer(demon, return_tensors='pt').to(args.device)
                _ = model(**demon_token)
            all_latent_dicts.append(model_wrapper.latent_dict)
            model_wrapper.reset_latent_dict()

        context_vector_dict = model_wrapper.get_context_vector(all_latent_dicts, args.config)

        best_layer = target_layer_selection(
            args, model_wrapper, tokenizer, val_evaluator, context_vector_dict
        )
        result_dict['best_replace_layer'][run_name] = best_layer

        s_t = time.time()
        with model_wrapper.replace_latent(context_vector_dict, [best_layer], args.config):
            val_ours = val_evaluator.evaluate(
                model_wrapper, tokenizer,
                demonstration='',
                use_cache=args.config['use_cache']
            )
            test_ours = test_evaluator.evaluate(
                model_wrapper, tokenizer,
                demonstration='',
                use_cache=args.config['use_cache']
            )
        e_t = time.time()
        result_dict['val_result']['m1'].append(val_ours)
        result_dict['test_result']['m1'].append(test_ours)
        result_dict['time']['evaluate'].append(e_t - s_t)

        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

    del model_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_m1.py')
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
            print(f"Running M1: {model_name} on {dataset_name} with GPU {gpu_id}")

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

    print("All M1 tasks completed.")

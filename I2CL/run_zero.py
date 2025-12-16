"""
Zero-shot runner: evaluates only zero-shot performance over (model, dataset) combos.
"""

import argparse
import copy
import gc
import itertools
import json
import time

import torch

import evaluator as ev
import my_datasets as md
import utils


def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    print(f"\n{'=' * 60}")
    print(f"Zero-shot: {args.model_name} on {args.dataset_name}")
    print(f"{'=' * 60}\n")

    model, tokenizer, model_config = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True
    )
    model_wrapper = utils.get_model_wrapper(args.model_name, model, tokenizer, model_config, args.device)

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

    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])

    result_dict = {
        'val_result': {'zero_shot': []},
        'test_result': {'zero_shot': []},
    }

    for run_id in range(args.config['run_num']):
        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f"\n{'=' * 60}")
        print(f"Run {run_id + 1}/{args.config['run_num']}: {run_name}")
        print(f"{'=' * 60}\n")

        utils.set_seed(args.config['seed'] + run_id)

        if args.config['run_baseline']:
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
            print(f"Test zero-shot: {test_zero}")

        with open(args.save_dir + '/result_dict.json', 'w') as f:
            json.dump(result_dict, f, indent=4)

    del model_wrapper, model, tokenizer
    del val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Zero-shot run completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'=' * 60}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/Aconfig_zero.py')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    gpu_id = config['gpus'][0]
    combinations = list(itertools.product(config['models'], config['datasets']))
    for model_name, dataset_name in combinations:
        print(f"Running Zero-shot: {model_name} on {dataset_name} with GPU {gpu_id}")

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

    print("All zero-shot tasks completed.")

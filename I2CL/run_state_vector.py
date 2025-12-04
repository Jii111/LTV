"""
State Vector baseline runner.
Validation selects the best single layer to avoid storing all layers,
and inference uses that layer only. Models load in 8bit by default to reduce memory.
"""

import argparse
import copy
import gc
import itertools
import json
import os
import random
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import my_datasets as md
import utils
from sv_utils.TVeval import ICLVectorEvaluator
from sv_utils.TVframework import Evaluator


def main(args):
    utils.set_seed(args.config['seed'])
    args.device = utils.set_device(args.gpu)
    args.metric = args.config['metric']
    utils.init_exp_path(args, args.config['exp_name'])

    print(f"\n{'=' * 60}")
    print(f"State Vector Baseline: {args.model_name} on {args.dataset_name}")
    print(f"{'=' * 60}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

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
    args.format_dict = {'eos': args.config['eos'], 'proj_tokens': args.config['proj_tokens']}
    metric = {'top_k': {'max_top': 1}}

    result_dict = {
        'demon': {},
        'test_result': {'state vector': []},
        'time': {'state vector': []},
        'best_layer': []
    }

    for run_id in tqdm(range(args.config['run_num']), desc="Overall Progress", position=0):
        args.run_name = f'run_{run_id}'
        print(f"\n{'=' * 60}")
        print(f"Run {run_id + 1}/{args.config['run_num']}: {args.run_name}")
        print(f"{'=' * 60}\n")

        utils.set_seed(args.config['seed'] + run_id)

        args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
        args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)

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
        result_dict['demon'][args.run_name] = demon

        print("Preparing State Vector datasets...")
        dev_data, dummy_test, valid_data, test_data = utils.convert_to_svdataset(
            split_demon, train_dataset, demon_indices, val_dataset, test_dataset, tokenizer, run_id, args
        )
        _ = dummy_test  # unused

        val_eval_data = copy.deepcopy(valid_data)
        for entry in val_eval_data:
            entry['demon'] = dev_data

        nshot, fshot = ('zs', False)
        run_name = f"{nshot}_raw"
        optimizer_weight = [[1, 2, 4, 8, 16, 0]]
        optimizer_config = {"fix-one-step": {"lr": [1], "weight": optimizer_weight}}
        label_info = train_dataset.get_dmonstration_template()['options']

        sv_8bit = args.config.get('load_in_8bit', True)
        svevaluator = ICLVectorEvaluator(
            metric,
            Evaluator(
                args.model_name,
                devices=str(args.gpu),
                load_int8=sv_8bit,
                load_float16=not sv_8bit
            )
        )

        best_layer = 0
        best_acc = -1
        max_layer = args.config.get('eval_edit_layer', None)
        if max_layer is None:
            max_layer = svevaluator.evaluator.model.config.num_hidden_layers - 1

        for layer_idx in range(max_layer + 1):
            topk_val, _ = svevaluator.single_atv_test(
                dummy_queries=[valid_data[0]],
                dev_data=[dev_data],
                test_data=val_eval_data,
                class_texts=label_info,
                layer_indices=[layer_idx],
                optimizer_config=optimizer_config,
                fs_eval=fshot,
                shuffle_labels=False,
                intervention_mode='add#0#1',
                add_to='atten',
                question_prompt=args.config['question_prompt'],
                format_dict=args.format_dict,
                return_logits=False
            )
            opt_key = next(iter(topk_val))
            layer_acc = topk_val[opt_key][0]
            print(f"[State Vector] val acc at layer {layer_idx}: {layer_acc:.4f}")
            if layer_acc > best_acc:
                best_acc = layer_acc
                best_layer = layer_idx

        print(f"[State Vector] Selected layer {best_layer} (val acc={best_acc:.4f})")
        result_dict['best_layer'].append(best_layer)

        sv_start = time.time()
        test_sv, _, _, acc = svevaluator.single_atv_test(
            dummy_queries=[valid_data[0]],
            dev_data=[dev_data],
            test_data=test_data,
            class_texts=label_info,
            layer_indices=[best_layer],
            optimizer_config=optimizer_config,
            fs_eval=fshot,
            shuffle_labels=False,
            intervention_mode='add#0#1',
            add_to='atten',
            question_prompt=args.config['question_prompt'],
            format_dict=args.format_dict,
            return_logits=False
        )
        sv_end = time.time()
        for k, v in acc.items():
            print(run_name + '_' + k, v[0])

        result_dict['test_result']['state vector'].append(test_sv)
        result_dict['time']['state vector'].append(sv_end - sv_start)
        print(f"Test State Vector: {test_sv}\n")

        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)

        del svevaluator
        gc.collect()
        torch.cuda.empty_cache()

    del train_dataset, val_dataset, test_dataset, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("State Vector baseline completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'=' * 60}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_state_vector.py')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = utils.load_config(args.config_path)

    gpu_id = config['gpus'][0]

    combinations = list(itertools.product(config['models'], config['datasets']))
    for model_name, dataset_name in combinations:
        print(f"Running State Vector: {model_name} on {dataset_name} with GPU {gpu_id}")

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

    print("All State Vector tasks completed.")

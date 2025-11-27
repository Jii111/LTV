import argparse
import gc
import itertools
import json
import os
import time
from multiprocessing import Process, Queue

import numpy as np
import torch

import evaluator as ev
import my_datasets as md
import utils


def main(args):
    # set global seed
    utils.set_seed(args.seed)
    # set device
    args.device = utils.set_device(args.gpu)
    # set metric used
    args.metric = args.metric

    # load tokenizer and model with PEFT adapter (soft prompts)
    model, tokenizer, model_config = \
        utils.load_peft_model_tokenizer(args.model_name, args.peft_name, args.device)

    # get model_wrapper
    model_wrapper = utils.get_model_wrapper(
        args.model_name, model,
        tokenizer, model_config,
        args.device
    )

    # ===== DEBUGGING: Check PEFT configuration =====
    print("\n" + "=" * 60)
    print("DEBUGGING: PEFT Model Configuration")
    print("=" * 60)

    # Check if model is PeftModel
    from peft import PeftModel
    print(f"Is PeftModel: {isinstance(model, PeftModel)}")

    if isinstance(model, PeftModel):
        # Get PEFT config
        peft_config = model.peft_config['default']
        print(f"\nPEFT Type: {peft_config.peft_type}")
        print(f"Task Type: {peft_config.task_type}")
        print(f"Num Virtual Tokens: {peft_config.num_virtual_tokens}")
        print(f"Num Transformer Submodules: {getattr(peft_config, 'num_transformer_submodules', 'N/A')}")
        print(f"Token Dim: {getattr(peft_config, 'token_dim', 'N/A')}")

        # Check active adapters
        print(f"\nActive Adapters: {model.active_adapters}")
        print(f"Adapters: {list(model.peft_config.keys())}")

        # Check if model is in training or eval mode
        print(f"Model Training Mode: {model.training}")

        # Check prompt encoder (for prompt tuning)
        if hasattr(model, 'prompt_encoder'):
            print(f"\nPrompt Encoder exists: True")
            for name, module in model.named_modules():
                if 'prompt' in name.lower():
                    print(f"  {name}: {type(module)}")

    # Check tokenizer settings
    print(f"\nTokenizer Padding Side: {tokenizer.padding_side}")
    print(f"Tokenizer Pad Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print("=" * 60 + "\n")

    # init result_dict
    exp_name = args.exp_name if hasattr(args, 'exp_name') else 'eval_prompt_tuned'
    peft_short_name = args.peft_name.split('/')[-1] if '/' in args.peft_name else args.peft_name
    exp_dir = f"{exp_name}/{args.model_name.split('/')[-1]}/{args.dataset_name}/{peft_short_name}"
    args.save_dir = exp_dir
    result_dict = {
        'test_result': {'soft_prompt_zero_shot': []},
        'time': {'evaluate': []},
        'yaml_config': args.yaml_path,
    }

    run_num = args.run_num if hasattr(args, 'run_num') else 1
    for run_id in range(run_num):

        run_name = f'run_{run_id}'
        args.run_name = run_name
        print(f'Run {run_name}')

        # Set seed for this run
        run_seed = args.seed + run_id
        utils.set_seed(run_seed)

        test_data_num = args.test_data_num if hasattr(args, 'test_data_num') else None
        sample_method = args.sample_method if hasattr(args, 'sample_method') else 'random'

        test_dataset = md.get_dataset(
            args.dataset_name, split='test',
            max_data_num=test_data_num,
            sample_mode=sample_method,
            seed=args.seed + run_id
        )

        # build test evaluator
        bs = args.batch_size if hasattr(args, 'batch_size') else 8
        test_evaluator = ev.Evaluator(test_dataset, batch_size=bs)

        # Soft prompt + zero-shot evaluation
        print("Evaluating with soft prompts (zero-shot)...")
        s_t = time.time()
        test_result, test_logits, test_labels = test_evaluator.evaluate(
            model_wrapper, tokenizer,
            demonstration='',  # No demonstration - zero-shot
            use_cache=args.use_cache,
            return_logits=args.return_logits,
            logits_mode=args.logits_mode
        )
        e_t = time.time()

        print(f'Test result (Soft Prompt + Zero-shot): {test_result}')
        print(f'Evaluation time: {e_t - s_t:.2f}s\n')

        result_dict['test_result']['soft_prompt_zero_shot'].append(test_result)
        result_dict['time']['evaluate'].append(e_t - s_t)

        # save result_dict after each run
        file_path = f"{args.save_dir}_result_dict.json"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(result_dict, f, indent=4)

    # Calculate average results across runs
    print("\n" + "=" * 50)
    print("Final Results Across All Runs:")
    print("=" * 50)
    avg_acc = np.mean([r['acc'] for r in result_dict['test_result']['soft_prompt_zero_shot']])
    avg_f1 = np.mean([r['macro_f1'] for r in result_dict['test_result']['soft_prompt_zero_shot']])
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Average Macro F1: {avg_f1:.4f}")
    print(f"Average Evaluation Time: {np.mean(result_dict['time']['evaluate']):.2f}s")

    # delete all variables
    del model_wrapper, model, tokenizer, test_dataset, test_evaluator
    del result_dict


# get args
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate PEFT models from YAML config')
    parser.add_argument(
        '--config_path', type=str, default='configs/yamls/config_prompt_tuned.yaml',
        help='Path to evaluation config YAML file'
        )
    return parser.parse_args()


if __name__ == "__main__":
    # get args
    cmd_args = get_args()

    # Load config
    config = utils.load_yaml(cmd_args.config_path)

    # Generate combinations: models x zip(datasets, peft_names)
    models = config['models']
    datasets = config['datasets']
    peft_names = config['peft_names']

    # Validate datasets and peft_names lengths match
    if len(datasets) != len(peft_names):
        raise ValueError(
            f"Length mismatch: datasets({len(datasets)}), peft_names({len(peft_names)}). "
            "They must have the same length for zip pairing."
        )

    # Create combinations: each model with each (dataset, peft_name) pair
    dataset_peft_pairs = list(zip(datasets, peft_names))
    combinations = list(itertools.product(models, dataset_peft_pairs))
    print(f"Found {len(combinations)} evaluation combinations")

    # Queue to hold tasks
    task_queue = Queue()
    for combine in combinations:
        task_queue.put(combine)


    def run_task(gpu_id, config):
        while not task_queue.empty():
            try:
                model_name, (dataset_name, peft_name) = task_queue.get(block=False)
            except:
                break

            print(f"\n{'=' * 70}")
            print(f"Model: {model_name}")
            print(f"Dataset: {dataset_name}")
            print(f"PEFT: {peft_name}")
            print(f"GPU: {gpu_id}")
            print(f"{'=' * 70}\n")

            # Create args namespace
            input_args = argparse.Namespace()
            input_args.config_path = cmd_args.config_path
            input_args.model_name = model_name
            input_args.dataset_name = dataset_name
            input_args.peft_name = peft_name
            input_args.seed = config.get('seed', 2025)
            input_args.gpu = gpu_id
            input_args.run_num = config.get('run_num', 1)
            input_args.test_data_num = config.get('test_data_num', None)
            input_args.batch_size = config.get('bs', 1)
            input_args.exp_name = config.get('exp_name', 'exps')
            input_args.sample_method = config.get('sample_method', 'uniform')
            input_args.metric = config.get('metric', 'acc')
            input_args.use_cache = config.get('use_cache', False)
            input_args.return_logits = config.get('return_logits', True)
            input_args.logits_mode = config.get('logits_mode', 'first')
            input_args.yaml_path = cmd_args.config_path

            try:
                main(input_args)
            except Exception as e:
                print(f"ERROR evaluating {peft_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up CUDA memory after each task
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}")
                time.sleep(5)


    # Create a process for each GPU
    gpus = [int(gpu) for gpu in config['gpus']]
    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in gpus]
    # Start all processes
    for p in processes:
        p.start()
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print("\nAll evaluation tasks completed!")

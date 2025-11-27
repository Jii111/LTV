import gc
import json
import copy
import time
import random
import argparse
import itertools
import torch
import numpy as np
from multiprocessing import Process, Queue

import utils
import my_datasets as md
import evaluator as ev


def main(args):
    # set global seed
    utils.set_seed(args.config['seed'])
    # set device
    args.device = utils.set_device(args.gpu)
    # set metric used
    args.metric = args.config['metric']
    # get save dir
    utils.init_exp_path(args, args.config['exp_name'])

    # load tokenizer and model
    model, tokenizer, model_config = \
    utils.load_model_tokenizer(args.model_name, args.device)
    
    # get model_wrapper
    model_wrapper = utils.get_model_wrapper(args.model_name, model, 
                                            tokenizer, model_config, 
                                            args.device)
    
    # load datasets
    train_dataset = md.get_dataset(args.dataset_name, split='train',
                                   max_data_num=None, seed=args.config['seed'])
    holdout_dataset = md.get_dataset(args.dataset_name, split='validation', 
                                     max_data_num=args.config['val_data_num'],
                                     sample_mode=args.config['sample_method'],
                                     seed=args.config['seed'])
    test_dataset = md.get_dataset(args.dataset_name, split='test', 
                                  max_data_num=None,
                                  sample_mode=args.config['sample_method'],
                                  seed=args.config['seed'])

    args.shot_num = args.config['shot_per_class']

    # build evaluators
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs']) 
    holdout_evaluator = ev.Evaluator(holdout_dataset, batch_size=args.config['bs'])
    # init result_dict
    result_dict = {'demon': {},
                   'split_demon': {},
                   'test_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 
                   'linear_coef': {},
                   'time': {'calibrate': [], 'evaluate': []},
                   }
    cv_save_dict = {}
    kl_dict = {}; index_dict = {} # ✅ 수정
    
        # zero-shot baseline
        test_zeroshot_result = test_evaluator.evaluate(model_wrapper, tokenizer, demonstration='',
                                                        use_cache=args.config['use_cache'])
        result_dict['test_result']['zero_shot'].append(test_zeroshot_result)
        print(f'Test zero-shot result: {test_zeroshot_result}\n')

        # sample demonstration
        count = 0
        temp_demon_list, temp_result_list = [], []
        while True:
            demon, split_demon, demon_data_index = \
            train_dataset.gen_few_shot_demonstration(tokenizer=tokenizer, shot_num=args.shot_num, 
                                                     max_demonstration_tok_len=args.test_max_token,
                                                     add_extra_query=args.config['add_extra_query'],
                                                     example_separator=args.config['example_separator'],
                                                     gen_example_method = args.config['gen_example_method'],
                                                     return_data_index=True, seed=args.config['demo_seed'] + run_id)
            temp_demon_list.append((demon, split_demon, demon_data_index))

            if args.config['demo_sample_method'] == 'random':
                break
            else:
                tem_val_result = holdout_evaluator.evaluate(model_wrapper, tokenizer, 
                                                            demonstration=demon,
                                                            use_cache=args.config['use_cache'])
                temp_result = tem_val_result[args.metric]
                temp_result_list.append(temp_result)
            if count > 20:
                if args.config['demo_sample_method'] == 'deficient':
                    demon, split_demon, demon_data_index = temp_demon_list[np.argmin(temp_result_list)]
                else:
                    raise ValueError('Invalid demo_sample_method!')
                break
            count += 1

        baseline_demon = demon
        query_demon = None
        index_dict[run_name] = list(map(int, demon_data_index))
            
        print(f'Demonstration:\n{demon}\n')
        print(f'Baseline demonstration:\n{baseline_demon}\n')
        print(f'Query demonstration:\n{query_demon}\n')
        
        # few-shot baseline
        test_fewshot_result, test_fewshot_logits, test_fewshot_labels = test_evaluator.evaluate(model_wrapper, tokenizer, 
                                                        demonstration=baseline_demon, 
                                                        use_cache=args.config['use_cache'], return_logits=args.config['return_logits'], logits_mode=args.config['logits_mode']) # ✅ 수정
        result_dict['test_result']['few_shot'].append(test_fewshot_result)
        print(f'Test few-shot result: {test_fewshot_result}\n')

        # generate demon_list
        demon_list = [demon]
        split_demon_list = split_demon
        result_dict['demon'][run_name] = demon_list

        with open(args.save_dir + '/cv_save_dict.json', 'w') as f:
            json.dump(cv_save_dict, f, indent=4)

        print(f"(Test query count: {len(test_fewshot_labels)})")

        with open(args.save_dir + f"/demon_idx.json", "w") as f:
                json.dump(index_dict, f, indent=4)
        
    # delete all variables
    del model_wrapper, model, tokenizer, train_dataset, cali_dataset, test_dataset, holdout_dataset
    del test_evaluator, holdout_evaluator
    del result_dict, context_vector_dict, demon_list, kl_dict
            

# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_i2cl.py', help='path to config file')
    return parser.parse_args()


if __name__ == "__main__":
    # get args
    args = get_args()
    # load config
    config = utils.load_config(args.config_path)
    # Generate all combinations of models and datasets
    combinations = list(itertools.product(config['models'], config['datasets']))
    # Queue to hold tasks
    task_queue = Queue()
    for combine in combinations:
        task_queue.put(combine)

    def run_task(gpu_id, config):
        while not task_queue.empty():
            model_name, dataset_name = task_queue.get()
            print(f"Running {model_name} on {dataset_name} with GPU {gpu_id}")
            input_args = argparse.Namespace()
            cur_config = copy.deepcopy(config)
            input_args.model_name = model_name
            input_args.dataset_name = dataset_name
            input_args.gpu = gpu_id
            input_args.config = cur_config
            try:
                main(input_args)
            finally:
                # Clean up CUDA memory after each task
                gc.collect()
                torch.cuda.empty_cache()
                print(f"CUDA memory cleared for GPU {gpu_id}") 
                time.sleep(5)

    # Create a process for each GPU
    processes = [Process(target=run_task, args=(gpu_id, config)) for gpu_id in config['gpus']]
    # Start all processes
    for p in processes:
        p.start()
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print("All tasks completed.")
    
    

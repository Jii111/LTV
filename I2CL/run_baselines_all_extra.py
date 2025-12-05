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
from sv_utils.TVframework import Evaluator
from sv_utils.utils import set_rand_seed

import argparse
import copy
import json
import os
import sys
import time
import random
import torch
from tqdm import tqdm
import evaluator as ev
import my_datasets as md
import utils
import utils_method as um
from fv_utils.extract_utils import *
from wrapper_m2 import M2AdaptiveWrapper, M2Wrapper

task_queue = None


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
                print(f"[ICL TV] Layer {layer} validation {args.metric}: {metric_value:.4f}")
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_layer = layer
    print(f"[ICL TV] Selected layer {best_layer} (metric={best_metric:.4f})")
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
    args.format_dict = {'eos': args.config['eos'], 'proj_tokens': args.config['proj_tokens']}
    metric = {'top_k': {'max_top': 1}}
    
    model, tokenizer, model_config, model_config_fv = utils.load_model_tokenizer(
            args.model_name, args.device, output_hidden_states=True
        )

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    m2_wrapper = M2Wrapper(model, tokenizer, model_config, args.device)
    m2_adaptive_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, args.device)
    
    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)

    # TODO: evaluator 정리
    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])
     

    result_dict = {
        'demon': {},
        'test_result': {
            'zero_shot': [], 'few_shot': [], 'i2cl_default': [], 'i2cl_train': [], 'ICLTV': [], 'fv': [], 'state vector': [], 'm2': [], 'm2_adaptive': []
        },
        'best_replace_layer': {'i2cl_default': [],'i2cl_train': [], 'ICLTV': [], 'fv': []},
        'i2cl_linear_coef(default)': {},
        'i2cl_linear_coef(train)': {},
        'time': {'i2cl_default': [], 'i2cl_train': [], 'ICLTV': [], 'fv': [], 'state vector': [], 'm2': [], 'm2_adaptive': []}
    }
    kl_dict = {'i2cl_default': {}, 'i2cl_train': {},'ICLTV': {}, 'fv': {},  'state vector': {},'m2': {}, 'm2_adaptive': {}}

    cv_save_dict = {}

    for run_id in tqdm(range(args.config['run_num']), desc="Overall Progress", position=0):
        args.run_name = f'run_{run_id}'
        print(f"\n{'=' * 60}")
        print(f"Run {run_id + 1}/{args.config['run_num']}: {args.run_name}")
        print(f"{'=' * 60}\n")

        utils.set_seed(args.config['seed'] + run_id)
        # shared train queries cache per num_queries
        m2_test_dict = {}; m2a_test_dict = {}

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
                return_logits=True,
                logits_mode=args.config['logits_mode']
            )
            result_dict['test_result']['few_shot'].append(test_few)
            print(f"Test few-shot: {test_few}\n")

        '''
        ################## I2CL baseline ##################
        print("Evaluating ICL TV baseline...")
        i2cl_start = time.time()
        temp_demon_list, temp_result_list = [], []
        temp_demon_list.append((demon, split_demon, demon_indices))
        cali_dataset = copy.deepcopy(train_dataset)
        cali_dataset.all_data = [train_dataset.all_data[i] for i in demon_indices]

        # 1. fix strength_params
        # TODO: 선택으로 바꾸기
        base_wrapper.init_strength(args.config,cali_train=False)
        result_dict['i2cl_linear_coef(default)'][args.run_name] = base_wrapper.linear_coef.tolist()

        # 2. extract latents 
        demon_list = [demon]
        split_demon_list = split_demon
        all_latent_dicts = []
        with torch.no_grad():
            if not args.config['split_demon']:
                target_demon_list = demon_list[0]
            else:
                target_demon_list = split_demon_list
            for cur_demon in target_demon_list:
                with base_wrapper.extract_latent():
                    demon_token = tokenizer(cur_demon, return_tensors='pt').to(args.device)
                    _ = model(**demon_token)
                all_latent_dicts.append(base_wrapper.latent_dict)
                base_wrapper.reset_latent_dict()

        # 3. generate context vector 
        context_vector_dict = base_wrapper.get_context_vector(all_latent_dicts, args.config)
        del all_latent_dicts
        
        # 4. evaluate i2cl
        with torch.no_grad():
            with base_wrapper.inject_latent(
                    context_vector_dict, args.config,
                    base_wrapper.linear_coef
                    ):
                test_i2cl, test_i2cl_logits, test_i2cl_labels = test_evaluator.evaluate(
                    base_wrapper, tokenizer, demonstration='',
                    use_cache=args.config['use_cache'], return_logits=args.config['return_logits'],
                    logits_mode=args.config['logits_mode']
                    )
        i2cl_end = time.time()

        result_dict['test_result']['i2cl_default'].append(test_i2cl)
        result_dict['time']['i2cl_default'].append(i2cl_end - i2cl_start)
        print(f"Test I2CL_default: {test_i2cl}\n")

        if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
            assert test_few_labels == test_i2cl_labels, "Label mismatch between few-shot and I2CL results!"
            mean_kl_i2cl, kl_values_i2cl = utils.compute_kl_divergence(
                test_few_logits, test_i2cl_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs I2CL_default): {mean_kl_i2cl:.4f}")
            kl_dict['i2cl_default'][args.run_name] = {
                "mean_kl": mean_kl_i2cl,
                "kl_values": kl_values_i2cl.tolist()
            }

        # 5. save context vector dict
        for layer, subdict in context_vector_dict.items():
            for module, activation in subdict.items():
                if 'Qwen' in args.model_name:
                    context_vector_dict[layer][module] = activation.to(torch.float32).cpu().numpy().tolist()
                else:
                    context_vector_dict[layer][module] = activation.cpu().numpy().tolist()
        cv_save_dict[args.run_name] = context_vector_dict

        with open(args.save_dir + '/i2cl_default_save_dict.json', 'w') as f:
            json.dump(cv_save_dict, f, indent=4)
        
        # 1. fix strength_params
        # TODO: 선택으로 바꾸기
        i2cl_start = time.time()
        base_wrapper.init_strength(args.config,cali_train=True)
    
        # 2. extract latents 
        demon_list = [demon]
        split_demon_list = split_demon
        all_latent_dicts = []
        with torch.no_grad():
            if not args.config['split_demon']:
                target_demon_list = demon_list[0]
            else:
                target_demon_list = split_demon_list
            for cur_demon in target_demon_list:
                with base_wrapper.extract_latent():
                    demon_token = tokenizer(cur_demon, return_tensors='pt').to(args.device)
                    _ = model(**demon_token)
                all_latent_dicts.append(base_wrapper.latent_dict)
                base_wrapper.reset_latent_dict()
             
        # 3. generate context vector    
        context_vector_dict = base_wrapper.get_context_vector(all_latent_dicts, args.config)
        del all_latent_dicts
        base_wrapper.calibrate_strength(context_vector_dict, cali_dataset, 
                                    args.config, save_dir=args.save_dir, 
                                    run_name=args.run_name)
        result_dict['i2cl_linear_coef(train)'][args.run_name] = base_wrapper.linear_coef.tolist()

        # 4. evaluate i2cl
        with torch.no_grad():
            with base_wrapper.inject_latent(
                    context_vector_dict, args.config,
                    base_wrapper.linear_coef
                    ):
                test_i2cl, test_i2cl_logits, test_i2cl_labels = test_evaluator.evaluate(
                    base_wrapper, tokenizer, demonstration='',
                    use_cache=args.config['use_cache'], return_logits=args.config['return_logits'],
                    logits_mode=args.config['logits_mode']
                    )

        i2cl_end = time.time()
        result_dict['test_result']['i2cl_train'].append(test_i2cl)
        result_dict['time']['i2cl_train'].append(i2cl_end - i2cl_start)
        print(f"Test I2CL_train: {test_i2cl}\n")

        if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
            assert test_few_labels == test_i2cl_labels, "Label mismatch between few-shot and I2CL results!"
            mean_kl_i2cl, kl_values_i2cl = utils.compute_kl_divergence(
                test_few_logits, test_i2cl_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs I2CL_train): {mean_kl_i2cl:.4f}")
            kl_dict['i2cl_train'][args.run_name] = {
                "mean_kl": mean_kl_i2cl,
                "kl_values": kl_values_i2cl.tolist()
            }

        ################## ICL task vector baseline ##################
        print("Evaluating ICL TV baseline...")
        m1_start = time.time()
        all_latent_dicts = []
        with torch.no_grad():
            with base_wrapper.extract_latent():
                demon_token = tokenizer(demon, return_tensors='pt').to(args.device)
                _ = model(**demon_token)
            all_latent_dicts.append(base_wrapper.latent_dict)
            base_wrapper.reset_latent_dict()

        context_vector_dict = base_wrapper.get_context_vector(all_latent_dicts, args.config)
        del all_latent_dicts
        best_layer = target_layer_selection(
            args, base_wrapper, tokenizer, val_evaluator, context_vector_dict
        )
        result_dict['best_replace_layer']['ICLTV'].append(best_layer)

        
        with base_wrapper.replace_latent(context_vector_dict, [best_layer], args.config):
            test_m1, test_m1_logits, test_m1_labels = test_evaluator.evaluate(
                base_wrapper, tokenizer, demonstration='',
                use_cache=args.config['use_cache'],
                return_logits=True,
                logits_mode=args.config['logits_mode']
            )
        m1_end = time.time()
        result_dict['test_result']['ICLTV'].append(test_m1)
        result_dict['time']['ICLTV'].append(m1_end - m1_start)
        print(f"Test ICL TV: {test_m1}\n")

        if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
            assert test_few_labels == test_m1_labels, "Label mismatch between few-shot and ICL TV results!"
            mean_kl_m1, kl_values_m1 = utils.compute_kl_divergence(
                test_few_logits, test_m1_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs ICL TV): {mean_kl_m1:.4f}")
            kl_dict['ICLTV'][args.run_name] = {
                "mean_kl": mean_kl_m1,
                "kl_values": kl_values_m1.tolist()
            }

        ################## FV baseline ##################
        print("Evaluating FV baseline...")
        fv_start = time.time()
        dataset_fv = {}
        fv_result_dict = {}
        args.n_mean_activations_trials = config['n_mean_activations_trials']
        args.n_top_heads = config['n_top_heads']
        args.prefixes = config['prefixes']
        args.separators = config['separators']
        args.revision = config['revision']

        # TODO: model_config 체크
        dataset_fv['train'] = convert_basetask_to_icldataset(train_dataset, demon_indices)
        dataset_fv['validation'] = convert_basetask_to_icldataset(val_dataset)
        dataset_fv['test'] = convert_basetask_to_icldataset(test_dataset)

        # 1. filter dataset to cases where model gets it correct
        fs_results_validation = n_shot_eval_no_intervention(
            dataset=dataset_fv, task_name=args.dataset_name, n_shots=args.shot_num, model=model,
            model_config=model_config_fv, tokenizer=tokenizer, compute_ppl=True, test_split='validation',
            prefixes=args.prefixes, separators=args.separators
            )
        filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
        utils.set_seed(args.config['seed']+run_id)
        fs_results = n_shot_eval_no_intervention(
            dataset=dataset_fv, task_name=args.dataset_name, n_shots=args.shot_num, model=model,
            model_config=model_config_fv, tokenizer=tokenizer, compute_ppl=True, prefixes=args.prefixes,
            separators=args.separators
            )
        filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

        # 2. compute mean_head_activations
        utils.set_seed(args.config['seed']+run_id)
        class_num = train_dataset.class_num
        mean_activations = get_mean_head_activations(
            dataset_fv, model=model, model_config=model_config_fv, tokenizer=tokenizer,
            n_icl_examples=args.shot_num // class_num,
            N_TRIALS=args.n_mean_activations_trials, prefixes=args.prefixes, separators=args.separators,
            filter_set=filter_set_validation
            )
        args.mean_activations_path = f'{args.save_dir}/{args.dataset_name}_mean_head_activations_{run_id}run.pt'
        #torch.save(mean_activations, args.mean_activations_path)

        # 3. load or re-compute indirect_effect values
        ## TODO : if not using universal_set
        fv, top_heads = compute_universal_function_vector(
            mean_activations, model, model_config=model_config_fv, n_top_heads=args.n_top_heads
            )
        '''
        # 4. evaluate FV
        if config['edit_layer'] == -2:
            if 'Qwen2.5-7B' in args.model_name:
                eval_edit_layer = 9
            elif 'Llama-3.1-8B' in args.model_name:
                eval_edit_layer = 11
            else: # in case for other model
                eval_edit_layer = model_config_fv['n_layers']//3 
        '''
        utils.set_seed(args.config['seed'])
        fv_results = {}
        fv_logits = {}
        fv_labels = {}
        if isinstance(eval_edit_layer, int):
            fv_results[eval_edit_layer], fv_logits[eval_edit_layer], fv_labels[eval_edit_layer] = n_shot_eval(
                dataset=dataset_fv, task_name=args.dataset_name, fv_vector=fv, edit_layer=eval_edit_layer,
                n_shots=0, prefixes=args.prefixes, separators=args.separators, test_type='test',
                model=model, model_config=model_config_fv, tokenizer=tokenizer, filter_set=filter_set,
                return_logits=True
                )
            fv_results_file_suffix = f'_fv_layer_{eval_edit_layer}_sweep.json'
        else:
            raise ValueError("Not allowed to sweep layers in this experiment")


        fv_results_file_name = make_valid_path_name(f'{args.save_dir}/' + fv_results_file_suffix)
        args.zs_results_file_name = fv_results_file_name
        fv_result_dict[args.run_name] = fv_results
        with open(fv_results_file_name, 'w') as results_file:
            json.dump(fv_result_dict, results_file, indent=2)

        best_layer = max(fv_results, key=lambda l: get_acc(fv_results[l]))

        test_fv, test_fv_logits, test_fv_labels = get_acc(fv_results[best_layer]), fv_logits[best_layer], fv_labels[
            best_layer]

        fv_end = time.time()
        result_dict['test_result']['fv'].append(test_fv)
        result_dict['best_replace_layer']['fv'].append(best_layer)
        result_dict['time']['fv'].append(fv_end - fv_start)
        print(f"Test FV: {test_fv}\n")

        if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
            assert test_few_labels == test_fv_labels, "Label mismatch between few-shot and FV results!"
            mean_kl_fv, kl_values_fv = utils.compute_kl_divergence(
                test_few_logits, test_fv_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs FV): {mean_kl_fv:.4f}")
            kl_dict['fv'][args.run_name] = {
                "mean_kl": mean_kl_fv,
                "kl_values": kl_values_fv.tolist()
            }
        '''
        
        print("Evaluating State Vector baseline...")
        sv_start = time.time()
            
        svevaluator = ICLVectorEvaluator(metric, Evaluator(model_path = args.model_name, model=model, tokenizer=tokenizer, devices="0")) 
        dev_data, dummy_test, valid_data, test_data = utils.convert_to_svdataset(split_demon, train_dataset, demon_indices, val_dataset, test_dataset, tokenizer, run_id, args)
        dummy_test = dummy_test[0] # TODO: 1 train query setting check
        print("test dataset len 비교")
        print(len(test_dataset))
        print(len(test_data)) 

        iv_result = {}

        for i in range(len(test_data)):
            test_data[i]['demon'] = []

        acc, _ = svevaluator.single_ICL_test(test_data, format_dict=args.format_dict)
        print("ZS acc : ", acc)

        for i in range(len(test_data)):
            test_data[i]['demon'] = dev_data
        acc, _ = svevaluator.single_ICL_test(test_data, format_dict=args.format_dict)
        print('ICL baseline acc : ', acc)
        print()
        
        nshot, fshot = ('zs', False)
            
        run_name = f"{nshot}_raw_layer{eval_edit_layer}"
        optimizer_weight = [[1, 2, 4, 8, 16, 0]]
        optimizer_config = {"fix-one-step": {"lr": [1], "weight": optimizer_weight}}
        label_info = train_dataset.get_dmonstration_template()['options']
        test_sv, test_sv_logits, test_sv_labels, acc = svevaluator.single_atv_test(dummy_queries=[valid_data[0]],
                                                dev_data=[dev_data],
                                                test_data=test_data, class_texts = label_info,
                                                layer_indices=list(range(eval_edit_layer + 1)),
                                                optimizer_config = optimizer_config,
                                                fs_eval=fshot,
                                                shuffle_labels=False,
                                                intervention_mode='add#0#1',
                                                add_to='atten',
                                                question_prompt=args.config['question_prompt'],
                                                format_dict=args.format_dict,
                                                return_logits=True)
        for k, v in acc.items():
            iv_result[run_name + '_' + k] = v[0]
            print(run_name + '_' + k, v[0])

        sv_end = time.time()
        result_dict['test_result']['state vector'].append(test_sv)
        result_dict['time']['state vector'].append(sv_end - sv_start)
        print(f"Test State Vector: {test_sv}\n")

        if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
            assert test_few_labels == test_sv_labels, "Label mismatch between few-shot and FV results!"
            mean_kl_sv, kl_values_sv = utils.compute_kl_divergence(
                test_few_logits, test_sv_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs FV): {mean_kl_sv:.4f}")
            kl_dict['state vector'][args.run_name] = {
                    "mean_kl": mean_kl_sv,
                    "kl_values": kl_values_sv.tolist()
                }
        
        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)
        if kl_dict['state vector']:
            with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                json.dump(kl_dict, f, indent=4)
                
        del svevaluator, mean_kl_sv, kl_values_sv, test_sv_logits
        gc.collect()
        torch.cuda.empty_cache()
        '''
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
                    return_logits=True,
                    logits_mode=args.config['logits_mode']
                )
            m2_end = time.time()
            m2_test_dict[q_key] = test_m2
            print(f"Test M2({q_key}): {test_m2}\n")
            result_dict['time']['m2'].append(m2_end - m2_start)

            if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                mean_kl, kl_values = utils.compute_kl_divergence(
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
                        "kl_values": kl_values.tolist(),
                        "labels": list(map(int, test_m2_labels))})
                utils.plot_kl_hist(
                    kl_values, mean_kl,
                    os.path.join(args.save_dir, f"{args.run_name}_m2_kl.png")
                )

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
                        return_logits=True,
                        logits_mode=args.config['logits_mode']
                    )
                m2a_end = time.time()
                utils.nested_set(m2a_test_dict, [q_key, lam_key], test_m2a)
                result_dict['time']['m2_adaptive'].append(m2a_end - m2a_start)
                print(f"Test M2-Adaptive: {test_m2a}\n")

                if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                    mean_kl_a, kl_values_a = utils.compute_kl_divergence(
                        test_few_logits, test_m2a_logits, is_qwen='Qwen' in args.model_name
                    )
                    print(f"KL divergence (Few-shot vs M2-Adaptive): {mean_kl_a:.4f}")
                    print(">>> test_m2a_logits")
                    print("  ",test_m2a_logits)
                    print(">>> mean_kl_a")
                    print("  ",mean_kl_a)
                    utils.nested_set(kl_dict,
                        ['m2_adaptive', args.run_name, q_key, lam_key],
                        {"mean_kl": mean_kl_a,
                        "kl_values": kl_values_a.tolist(),
                        "labels": list(map(int, test_m2a_labels))})

                    utils.plot_kl_hist(
                        kl_values_a, mean_kl_a,
                        os.path.join(args.save_dir, f"{args.run_name}_m2_adaptive_kl.png")
                    )

        result_dict['test_result']['m2'].append(m2_test_dict)
        result_dict['test_result']['m2_adaptive'].append(m2a_test_dict)

        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)
        if kl_dict['i2cl_default'] or kl_dict['i2cl_train'] or kl_dict['ICLTV'] or kl_dict['fv'] or kl_dict['m2'] or kl_dict['m2_adaptive']:
            with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                json.dump(kl_dict, f, indent=4)
'''

    del mean_activations, base_wrapper, m2_wrapper, m2_adaptive_wrapper, model, tokenizer
    del train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Baseline suite completed!")
    print(f"Results saved to: {args.save_dir}")
    print(f"{'=' * 60}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_baseline_all.py')
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
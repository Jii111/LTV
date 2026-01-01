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
            elif tv_type == 'state_vector':
                val_result = evaluator.evaluate(
                    model_wrapper, tokenizer, demonstration='',
                    use_cache=args.config['use_cache'],
                    sv_vector=context_vector, edit_layer=layer, model_config=model_config
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
    args.format_dict = {'eos': args.config['eos'], 'proj_tokens': args.config['proj_tokens']}
    metric = {'top_k': {'max_top': 1}}
    
    model, tokenizer, model_config = utils.load_model_tokenizer(
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


        ################## I2CL baseline ##################
        if args.config['run_i2cl']:
            print("Evaluating I2CL baseline...")
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
                        use_cache=args.config['use_cache'], return_logits=args.config['return_logits']
                        )
            i2cl_end = time.time()

            result_dict['test_result']['i2cl_default'].append(test_i2cl)
            result_dict['time']['i2cl_default'].append(i2cl_end - i2cl_start)
            print(f"Test I2CL_default: {test_i2cl}\n")

            if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                assert test_few_labels == test_i2cl_labels, "Label mismatch between few-shot and I2CL results!"
                mean_kl_i2cl = utils.compute_kl_divergence(
                    test_few_logits, test_i2cl_logits, is_qwen='Qwen' in args.model_name
                )
                print(f"KL divergence (Few-shot vs I2CL_default): {mean_kl_i2cl:.4f}")
                kl_dict['i2cl_default'][args.run_name] = {
                    "mean_kl": mean_kl_i2cl
                }

            # 5. save context vector dict
            for layer, subdict in context_vector_dict.items():
                for module, activation in subdict.items():
                    if 'Qwen' in args.model_name:
                        context_vector_dict[layer][module] = activation.to(torch.float32).cpu().numpy().tolist()
                    else:
                        context_vector_dict[layer][module] = activation.cpu().numpy().tolist()
            cv_save_dict[args.run_name] = context_vector_dict

            #with open(args.save_dir + '/i2cl_default_save_dict.json', 'w') as f:
            #    json.dump(cv_save_dict, f, indent=4)
            '''
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
                        use_cache=args.config['use_cache'], return_logits=args.config['return_logits']
                        )

            i2cl_end = time.time()
            result_dict['test_result']['i2cl_train'].append(test_i2cl)
            result_dict['time']['i2cl_train'].append(i2cl_end - i2cl_start)
            print(f"Test I2CL_train: {test_i2cl}\n")

            if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                assert test_few_labels == test_i2cl_labels, "Label mismatch between few-shot and I2CL results!"
                mean_kl_i2cl = utils.compute_kl_divergence(
                    test_few_logits, test_i2cl_logits, is_qwen='Qwen' in args.model_name
                )
                print(f"KL divergence (Few-shot vs I2CL_train): {mean_kl_i2cl:.4f}")
                kl_dict['i2cl_train'][args.run_name] = {
                    "mean_kl": mean_kl_i2cl
                }
            '''
        ################## ICL task vector baseline ##################
        if args.config['run_tv']:
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
                args, base_wrapper, tokenizer, val_evaluator, "icl_tv", context_vector_dict
            )
            result_dict['best_replace_layer']['ICLTV'].append(best_layer)

            
            with base_wrapper.replace_latent(context_vector_dict, [best_layer], args.config):
                test_m1, test_m1_logits, test_m1_labels = test_evaluator.evaluate(
                    base_wrapper, tokenizer, demonstration='',
                    use_cache=args.config['use_cache'],
                    return_logits=args.config['return_logits']
                )
            m1_end = time.time()
            result_dict['test_result']['ICLTV'].append(test_m1)
            result_dict['time']['ICLTV'].append(m1_end - m1_start)
            print(f"Test ICL TV: {test_m1}\n")

            if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                assert test_few_labels == test_m1_labels, "Label mismatch between few-shot and ICL TV results!"
                mean_kl_m1 = utils.compute_kl_divergence(
                    test_few_logits, test_m1_logits, is_qwen='Qwen' in args.model_name
                )
                print(f"KL divergence (Few-shot vs ICL TV): {mean_kl_m1:.4f}")
                kl_dict['ICLTV'][args.run_name] = {
                    "mean_kl": mean_kl_m1
                }

        ################## FV baseline ##################
        if args.config['run_fv']:
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
                model_config=model_config, tokenizer=tokenizer, compute_ppl=True, test_split='validation',
                prefixes=args.prefixes, separators=args.separators
                )
            filter_set_validation = np.where(np.array(fs_results_validation['clean_rank_list']) == 0)[0]
            utils.set_seed(args.config['seed']+run_id)
            fs_results = n_shot_eval_no_intervention(
                dataset=dataset_fv, task_name=args.dataset_name, n_shots=args.shot_num, model=model,
                model_config=model_config, tokenizer=tokenizer, compute_ppl=True, prefixes=args.prefixes,
                separators=args.separators
                )
            filter_set = np.where(np.array(fs_results['clean_rank_list']) == 0)[0]

            # 2. compute mean_head_activations
            utils.set_seed(args.config['seed']+run_id)
            class_num = train_dataset.class_num
            mean_activations = get_mean_head_activations(
                dataset_fv, model=model, model_config=model_config, tokenizer=tokenizer,
                n_icl_examples=args.shot_num,
                N_TRIALS=args.n_mean_activations_trials, prefixes=args.prefixes, separators=args.separators,
                filter_set=filter_set_validation
                )
            args.mean_activations_path = f'{args.save_dir}/{args.dataset_name}_mean_head_activations_{run_id}run.pt'
            #torch.save(mean_activations, args.mean_activations_path)
            fv_t2 = time.time()
            # 3. load or re-compute indirect_effect values
            if config['universal_fv'] == False:
                indirect_effect = compute_indirect_effect(
                    dataset=dataset_fv, mean_activations=mean_activations, model=model, model_config=model_config, tokenizer=tokenizer, last_token_only=True,
                    n_shots=args.shot_num, prefixes=args.prefixes, separators=args.separators, n_trials=config['n_indirect_effect_trials'],
                    filter_set=filter_set_validation
                    )
                fv, top_heads, _ = compute_function_vector(
                    mean_activations, indirect_effect, model, model_config=model_config, n_top_heads=args.n_top_heads
                    )
            else:
                fv, top_heads = compute_universal_function_vector(
                    mean_activations, model, model_config=model_config, n_top_heads=args.n_top_heads
                    )
            fv_t3 = time.time()

            # 4. evaluate FV
            if config['choose_layer'] == False:
                if 'Qwen2.5-7B' in args.model_name:
                    eval_edit_layer = 9
                elif 'Llama-3.1-8B' in args.model_name:
                    eval_edit_layer = 11
                else: # in case for other model
                    eval_edit_layer = model_config.fv['n_layers']//3 
            else:
                eval_edit_layer = target_layer_selection(
                    args, base_wrapper, tokenizer, val_evaluator, tv_type = "function_vector",
                    context_vector = fv, model_config=model_config
                )
            fv_t4 = time.time()

            utils.set_seed(args.config['seed'])
            fv_results = {}
            fv_logits = {}
            fv_labels = {}
            if isinstance(eval_edit_layer, int):
                fv_results, test_fv_logits, fv_answer_labels = test_evaluator.evaluate(
                    model_wrapper=base_wrapper, tokenizer=tokenizer,
                    fv_vector=fv, edit_layer=eval_edit_layer,model_config=model_config,
                    return_logits=args.config['return_logits']
                )
            else:
                raise ValueError("Not allowed to sweep layers in this experiment")

            fv_end = time.time()
            result_dict['test_result']['fv'].append(fv_results)
            result_dict['best_replace_layer']['fv'].append(eval_edit_layer)
            result_dict['time']['fv'].append(fv_end - fv_start)
            print(f"Test FV: {fv_results}\n")
            print(f'[Time Debug] FV mean activation time: {fv_t2 - fv_start:.2f}s, FV computation time: {fv_t3 - fv_t2:.2f}s, layer selection time: {fv_t4 - fv_t3:.2f}s, evaluation time: {fv_end - fv_t4:.2f}s\n')

            with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
                json.dump(result_dict, f, indent=4)
            if kl_dict['i2cl_default'] or kl_dict['i2cl_train'] or kl_dict['ICLTV'] or kl_dict['fv'] or kl_dict['m2'] or kl_dict['m2_adaptive']:
                with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                    json.dump(kl_dict, f, indent=4)
            
            if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                assert test_few_labels == fv_answer_labels, "Label mismatch between few-shot and ICL TV results!"
                mean_kl_fv = utils.compute_kl_divergence(
                    test_few_logits, test_fv_logits, is_qwen='Qwen' in args.model_name
                )
                print(f"KL divergence (Few-shot vs ICL TV): {mean_kl_fv:.4f}")
                kl_dict['fv'][args.run_name] = {
                    "mean_kl": mean_kl_fv
                }
                
        ################## SV baseline ##################
        if args.config['run_sv']:
            print("Evaluating State Vector baseline...")
            sv_start = time.time()
                
            _evaluator = SVEvaluator(model_path = args.model_name, model=model, tokenizer=tokenizer, devices="0")
            svevaluator = ICLVectorEvaluator(metric, _evaluator)  
            dev_data, dummy_test, valid_data, test_data = utils.convert_to_svdataset(split_demon, train_dataset, demon_indices, val_dataset, test_dataset, tokenizer, run_id, args)
            dummy_test = dummy_test[0] # TODO: 1 train query setting check
            
            iv_result = {}

            for i in range(len(test_data)):
                test_data[i]['demon'] = []

            for i in range(len(test_data)):
                test_data[i]['demon'] = dev_data
            
            nshot, fshot = ('zs', False)
                
            run_name = f"{nshot}_raw_layer{eval_edit_layer}"
            #optimizer_weight = [[1, 2, 4, 8, 16, 0]]
            #optimizer_config = {"fix-one-step": {"lr": [1], "weight": optimizer_weight}}
            optimizer_config = {"fixed": {"beta": [1, 1, 1, 1, 1, 1, 1]}}
            label_info = train_dataset.get_dmonstration_template()['options']
            test_sv, test_sv_logits, test_sv_labels, t = svevaluator.single_atv_test(dummy_queries=[valid_data[0]],
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
                                                    return_logits=args.config['return_logits'])

            sv_end = time.time()
            result_dict['test_result']['state vector'].append(test_sv)
            result_dict['time']['state vector'].append(sv_end - sv_start)
            print(f"Test State Vector: {test_sv}\n")

            if args.config.get('compute_kl_divergence', False) and test_few_logits is not None:
                assert test_few_labels == test_sv_labels, "Label mismatch between few-shot and FV results!"
                mean_kl_sv = utils.compute_kl_divergence(
                    test_few_logits, test_sv_logits, is_qwen='Qwen' in args.model_name
                )
                print(f"KL divergence (Few-shot vs SV): {mean_kl_sv:.4f}")
                kl_dict['state vector'][args.run_name] = {
                        "mean_kl": mean_kl_sv
                    }
            
            with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
                json.dump(result_dict, f, indent=4)
            if kl_dict['state vector']:
                with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                    json.dump(kl_dict, f, indent=4)
                    
            del svevaluator, mean_kl_sv, test_sv_logits, _evaluator

        m2_test_dict = {}
        m2a_test_dict ={}
        
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

        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)
        if kl_dict['i2cl_default'] or kl_dict['i2cl_train'] or kl_dict['ICLTV'] or kl_dict['fv'] or kl_dict['m2'] or kl_dict['m2_adaptive']:
            with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                json.dump(kl_dict, f, indent=4)

    del m2_wrapper, m2_adaptive_wrapper
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

    # SV baseline
    parser.add_argument("--question_prompt", type=str, default="{input}")
    parser.add_argument("--eos", type=str, default="\n\n")
    parser.add_argument("--proj_tokens", type=str, default="→")

    # I2CL baseline
    parser.add_argument("--init_value", nargs="+", type=float, default=[0.1, 1.0])

    parser.add_argument("--layer", type=str, default="all")
    parser.add_argument("--inject_method", type=str, default="linear")
    parser.add_argument("--inject_pos", type=str, default="all")
    parser.add_argument("--gen_cv_method", type=str, default="context")
    parser.add_argument("--post_fuse_method", type=str, default="mean")
    parser.add_argument("--split_demon", type=bool, default=True)
    parser.add_argument("--gen_example_method", type=str, default="normal")

    parser.add_argument("--add_noise", type=bool, default=True)
    parser.add_argument("--noise_scale", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optim", type=str, default="adamW")
    parser.add_argument("--grad_bs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--cali_example_method", type=str, default="normal")

    # Function Vector baseline
    parser.add_argument("--choose_layer", type=bool, default=False)
    parser.add_argument("--universal_fv", type=bool, default=True)
    parser.add_argument("--n_top_heads", type=int, default=10)
    parser.add_argument("--n_mean_activations_trials", type=int, default=20)

    parser.add_argument("--separators", type=dict,
                        default={"input": "Q:", "output": "A:", "instructions": ""})
    parser.add_argument("--prefixes", type=dict,
                        default={"input": "\n", "output": "\n\n", "instructions": ""})
    parser.add_argument("--revision", default=None)
    
    
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
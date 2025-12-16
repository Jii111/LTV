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

import evaluator as ev
import my_datasets as md
import utils
from fv_utils.extract_utils import *
from sv_utils.TVeval import ICLVectorEvaluator
from sv_utils.TVframework import Evaluator

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

    model, tokenizer, model_config, model_config_fv = utils.load_model_tokenizer(
        args.model_name, args.device, output_hidden_states=True
    )

    base_wrapper = utils.get_model_wrapper(
        args.model_name, model, tokenizer, model_config, args.device
    )
    # m2_wrapper = M2Wrapper(model, tokenizer, model_config, args.device)
    # m2_adaptive_wrapper = M2AdaptiveWrapper(model, tokenizer, model_config, args.device)

    args.val_max_token = val_dataset.get_max_demonstration_token_length(tokenizer)
    args.test_max_token = test_dataset.get_max_demonstration_token_length(tokenizer)

    # TODO: evaluator 정리
    val_evaluator = ev.Evaluator(val_dataset, batch_size=args.config['bs'])
    test_evaluator = ev.Evaluator(test_dataset, batch_size=args.config['bs'])

    result_dict = {
        'demon': {},
        'test_result': {
            'zero_shot': [], 'few_shot': [], 'i2cl_default': [], 'i2cl_train': [], 'ICLTV': [], 'fv': [],
            'state vector': [], 'm2': [], 'm2_adaptive': []
        },
        'best_replace_layer': {'i2cl_default': [], 'i2cl_train': [], 'ICLTV': [], 'fv': []},
        'i2cl_linear_coef(default)': {},
        'i2cl_linear_coef(train)': {},
        'time': {
            'i2cl_default': [], 'i2cl_train': [], 'ICLTV': [], 'fv': [], 'state vector': [], 'm2': [], 'm2_adaptive': []
        }
    }
    kl_dict = {
        'i2cl_default': {}, 'i2cl_train': {}, 'ICLTV': {}, 'fv': {}, 'state vector': {}, 'm2': {}, 'm2_adaptive': {}
    }

    cv_save_dict = {}

    for run_id in tqdm(range(args.config['run_num']), desc="Overall Progress", position=0):
        args.run_name = f'run_{run_id}'
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
                return_logits=args.config['return_logits'],
                logits_mode=args.config['logits_mode']
            )
            result_dict['test_result']['few_shot'].append(test_few)
            print(f"Test few-shot: {test_few}\n")

        # 4. evaluate FV
        if config['edit_layer'] == -2:
            if 'Qwen2.5-7B' in args.model_name:
                eval_edit_layer = 9
            elif 'Llama-3.1-8B' in args.model_name:
                eval_edit_layer = 11
            else:  # in case for other model
                eval_edit_layer = model_config_fv['n_layers'] // 3

        print("Evaluating State Vector baseline...")
        sv_start = time.time()

        _evaluator = Evaluator(model_path=args.model_name, model=model, tokenizer=tokenizer, devices="0")
        svevaluator = ICLVectorEvaluator(metric, _evaluator)
        dev_data, dummy_test, valid_data, test_data = utils.convert_to_svdataset(
            split_demon, train_dataset, demon_indices, val_dataset, test_dataset, tokenizer, run_id, args
            )
        dummy_test = dummy_test[0]  # TODO: 1 train query setting check
        print("test dataset len 비교")
        print(len(test_dataset))
        print(len(test_data))
        print(dummy_test)
        print(valid_data)

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
        test_sv, test_sv_logits, test_sv_labels, acc = svevaluator.single_atv_test(
            dummy_queries=[valid_data[0]],
            dev_data=[dev_data],
            test_data=test_data, class_texts=label_info,
            layer_indices=list(range(eval_edit_layer + 1)),
            optimizer_config=optimizer_config,
            fs_eval=fshot,
            shuffle_labels=False,
            intervention_mode='add#0#1',
            add_to='atten',
            question_prompt=args.config['question_prompt'],
            format_dict=args.format_dict,
            return_logits=args.config['return_logits']
            )
        for k, v in acc.items():
            iv_result[run_name + '_' + k] = v[0]
            print(run_name + '_' + k, v[0])

        sv_end = time.time()
        result_dict['test_result']['state vector'].append(test_sv)
        result_dict['time']['state vector'].append(sv_end - sv_start)
        print(f"Test State Vector: {test_sv}\n")

        if args.config.get('return_logits', False) and test_few_logits is not None:
            assert test_few_labels == test_sv_labels, "Label mismatch between few-shot and FV results!"
            mean_kl_sv = utils.compute_kl_divergence(
                test_few_logits, test_sv_logits, is_qwen='Qwen' in args.model_name
            )
            print(f"KL divergence (Few-shot vs FV): {mean_kl_sv:.4f}")
            kl_dict['state vector'][args.run_name] = {
                "mean_kl": mean_kl_sv
            }

        with open(os.path.join(args.save_dir, 'result_dict.json'), 'w') as f:
            json.dump(result_dict, f, indent=4)
        if kl_dict['state vector']:
            with open(os.path.join(args.save_dir, 'kl_divergence.json'), 'w') as f:
                json.dump(kl_dict, f, indent=4)

        del svevaluator, mean_kl_sv, test_sv_logits, _evaluator
        gc.collect()
        torch.cuda.empty_cache()

    # del m2_wrapper, m2_adaptive_wrapper
    del base_wrapper, model, tokenizer
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

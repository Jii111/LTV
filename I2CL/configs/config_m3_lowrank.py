config = {}

config['exp_name'] = 'exps/m3_lowrank'
config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']
config['datasets'] = ['agnews']
config['shot_per_class'] = 30
config['bs'] = 1

config['return_logits'] = True
config['logits_mode'] = 'first'
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'
config['load_in_8bit'] = True
config['use_cache'] = False

config['method'] = 'M3-LR'
config['all_layers'] = True
config['demo_masking'] = True
config['extract_head_outputs'] = True

config['ridge_lambda'] = 0.01
config['low_rank_r'] = 64

config['num_train_queries'] = 25
config['extraction_batch_size'] = 1
config['inference_batch_size'] = 1

config['target_layers'] = None
config['module'] = ['hidden']
config['tok_pos'] = 'last'

config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

config['compute_kl_divergence'] = True
config['save_task_vectors'] = False
config['evaluate_reconstruction'] = False
config['normalize_features'] = True
config['feature_norm_method'] = 'l2'

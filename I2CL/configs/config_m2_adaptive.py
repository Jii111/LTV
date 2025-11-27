config = {}

config['exp_name'] = 'exps/m2_adaptive'
config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']
config['datasets'] = ['agnews']
config['shot_per_class'] = 30
config['bs'] = 1

config['return_logits'] = False
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'
config['load_in_8bit'] = True
config['use_cache'] = False

config['num_train_queries'] = 25
config['extraction_batch_size'] = 1
config['ridge_lambda'] = 0.01

config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

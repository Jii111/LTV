config = {}

config['exp_name'] = 'exps/zero_shot'
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B']
config['datasets'] = ['mr']
config['shot_per_class'] = 0
config['bs'] = 1

config['return_logits'] = False
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'
config['load_in_8bit'] = True
config['use_cache'] = False

config['val_data_num'] = 32
config['test_data_num'] = 8000
config['sample_method'] = 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = False
config['example_separator'] = '\n'

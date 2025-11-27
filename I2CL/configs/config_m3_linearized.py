config = {}

config['exp_name'] = 'exps/m3_linearized'
config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']
config['datasets'] = ['agnews']
config['shot_per_class'] = 30
config['bs'] = 1

config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'
config['use_cache'] = False
config['load_in_8bit'] = True

config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

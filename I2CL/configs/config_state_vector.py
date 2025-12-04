config = {}

# Experiment name and hardware
config['exp_name'] = 'exps/state_vector'
config['gpus'] = ['0']

# Model / dataset
config['models'] = ['meta-llama/Llama-3.1-8B']
config['datasets'] = ['sst2']

# Data & sampling
config['shot_per_class'] = 30
config['val_data_num'] = 32
config['test_data_num'] = 8000
config['sample_method'] = 'uniform'
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Seeds & runs
config['run_num'] = 5
config['seed'] = 42
config['demo_seed'] = 12

# SV settings
config['metric'] = 'acc'
config['question_prompt'] = "{input}"
config["eos"] = "\n\n"
config["proj_tokens"] = "→"
config['eval_edit_layer'] = None  # if None, sweep all layers
config['load_in_8bit'] = True

# Misc
config['use_cache'] = False

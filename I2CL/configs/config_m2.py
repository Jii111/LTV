config = {}

### Main Configuration ###
config['exp_name'] = 'exps/m2_expconfig2'  # M2 experiment directory
config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']
config['datasets'] = ['sst2'] 
config['shot_per_class'] = 30 # number of shots (NOT per class)
config['bs'] = 1  # batch size
###

# Experiment settings
config['return_logits'] = True
config['logits_mode'] = 'first'
config['run_num'] = 5
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['load_in_8bit'] = True
config['use_cache'] = False

# M2-specific settings
config['method'] = 'M2'
config['all_layers'] = True  # Use all layers for task vector extraction
config['demo_masking'] = True  # Enable demo masking

# Task vector extraction
config['num_train_queries'] = [32, 64, 128, 256]  # Number of training queries for task vector learning
config['ridge_lambda'] = [0.01, 0.1, 1.0, 5.0, 10.0] 
config['extraction_batch_size'] = 1
config['inference_batch_size'] = 1

# Which layers to inject (None = all layers)
config['target_layers'] = None  # None means all layers

# Module to use for extraction/injection
config['module'] = ['hidden']  # Use hidden states (full layer output)
config['tok_pos'] = 'last'  # Token position for extraction

# Data settings
config['val_data_num'] = 32
config['test_data_num'] = None
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['compute_kl_divergence'] = True
config['save_task_vectors'] = False
config['evaluate_reconstruction'] = False  # Evaluate reconstruction quality on val set
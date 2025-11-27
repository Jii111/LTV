config = {}

### Main Configuration ###
config['exp_name'] = 'exps/m3'  # M3 experiment directory
config['gpus'] = ['0']
config['models'] = ['Qwen/Qwen2.5-7B']
config['datasets'] = ['agnews']  # 'agnews', 'dbpedia', 'hate_speech18', 'mr', 'sst2', 'sst5', 'trec'
config['shot_per_class'] = 30  # number of shots
config['bs'] = 1  # batch size
###

# Experiment settings
config['return_logits'] = True
config['logits_mode'] = 'first'
config['run_num'] = 1
config['seed'] = 42
config['demo_seed'] = 12
config['run_baseline'] = True
config['metric'] = 'acc'  # 'acc', 'macro_f1'
config['load_in_8bit'] = True
config['use_cache'] = False

# M3-specific settings
config['method'] = 'M3'
config['all_layers'] = True  # Use all layers
config['demo_masking'] = True  # Enable demo masking
config['extract_head_outputs'] = True  # Extract MHA head outputs for features

# Ridge regression
config['ridge_lambda'] = 0.01  # Regularization coefficient

# Task vector extraction
config['num_train_queries'] = 25  # Number of training queries for task vector learning
config['extraction_batch_size'] = 1
config['inference_batch_size'] = 1

# Which layers to inject (None = all layers)
config['target_layers'] = None

# Module to use
config['module'] = ['hidden']
config['tok_pos'] = 'last'

# Data settings
config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['compute_kl_divergence'] = True
config['save_task_vectors'] = True
config['evaluate_reconstruction'] = False
config['analyze_task_vectors'] = False  # Analyze task vector (matrix) statistics

# Feature normalization (optional)
config['normalize_features'] = True  # Set to True to normalize features before regression
config['feature_norm_method'] = 'l2'  # 'l2' or 'standardize'

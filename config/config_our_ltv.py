config = {}

### Main Configuration ###
config['exp_name'] = 'exps/ltv'  
config['gpus'] = ['0']
config['models'] = ['meta-llama/Llama-3.1-8B'] # 'meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'Qwen/Qwen3-8B', 'Qwen/Qwen2.5-7B'
config['datasets'] = ['sst2','sst5','mr','subj','trec','hate_speech18','agnews','dbpedia'] 
config['num_shot'] = 30 # number of shots (NOT per class)
config['bs'] = 8  # batch size

config['run_num'] = 5
config['seed'] = 42

# Run LTV (adaptive task vector)
config['run_ltv'] = True
config['run_ltv'] = True

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
 
# Task vector extraction
config['num_train_queries'] = [256]  # Number of training queries for task vector learning
config['ridge_lambda'] = [5.0] 
config['extraction_batch_size'] = 1
config['inference_batch_size'] = 1

# Which layers to inject (None = all layers)
config['target_layers'] = None  # None means all layers

# Module to use for extraction/injection
config['module'] = ['hidden']  # Use hidden states (full layer output)
config['tok_pos'] = 'last'  # Token position for extraction

# Data settings
config['val_data_num'] = 32
config['test_data_num'] = 500
config['sample_method'] = 'uniform'  # 'random', 'uniform'
config['use_instruction'] = False
config['add_extra_query'] = True
config['example_separator'] = '\n'

# Evaluation settings
config['compute_d_NTP'] = True
config['save_task_vectors'] = False
config['evaluate_reconstruction'] = False  # Evaluate reconstruction quality on val set

config['layer'] = 'all' # all, early, mid, late
config['tok_pos'] = 'last'  # 'random', 'first', 'last'
config['inject_method'] = 'linear'  # 'linear', 'constraint', 'add'
config['inject_pos'] = 'all'  # 'all', 'first', last', 'random'
config['module'] = ['mlp', 'attn']  # 'mlp', 'attn', 'hidden'
config['gen_cv_method'] = 'context'  # 'context', 'noise'
config['post_fuse_method'] = 'mean'  # 'mean', 'pca'
config['split_demon'] = True  # split demonstraiton into seperate examples
config['gen_example_method'] = 'normal' 
